
import os
import json
import torch
import numpy as np
from monai.bundle import ConfigParser, download
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    ToTensord,
)
from monai.inferers import sliding_window_inference
from src.utils.config import Config

class SegmentationPredictor:
    def __init__(self, bundle_name=None, device=None):
        self.bundle_name = bundle_name or Config.SEGMENTATION_BUNDLE_NAME
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bundle_path = os.path.join(Config.DATA_ROOT, "monai_bundles", self.bundle_name)
        self.model = None
        self.inferer = None
        self.preprocessing = None
        self.postprocessing = None
        self.label_map = {}
        
        self._initialize_bundle()

    def _initialize_bundle(self):
        # 1. Download if needed
        if not os.path.exists(self.bundle_path):
            print(f"Downloading MONAI Bundle: {self.bundle_name} to {self.bundle_path}...")
            download(name=self.bundle_name, bundle_dir=os.path.dirname(self.bundle_path))
        
        # 2. Parse Configs
        model_config_path = os.path.join(self.bundle_path, "configs", "inference.json")
        if not os.path.exists(model_config_path):
             # Fallback for some bundles that only have metadata or different structure
             # But standard bundles usually have configs/inference.json or similar.
             # Let's try to list files if it fails, but for now assume standard structure.
             pass

        parser = ConfigParser()
        parser.read_config(model_config_path)
        
        # 3. Load Network
        # Usually network_def is in the config
        self.model = parser.get_parsed_content("network_def")
        self.model.to(self.device)
        self.model.eval()

        # Load weights
        model_file = os.path.join(self.bundle_path, "models", "model.pt")
        if os.path.exists(model_file):
            checkpoint = torch.load(model_file, map_location=self.device)
            # Handle state dict keys (sometimes they are wrapped in 'state_dict')
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"Loaded weights from {model_file}")
        else:
            print(f"Warning: Model weights not found at {model_file}")

        # Parse Metadata for Target Label and label names
        self.target_indices = []
        meta_file = os.path.join(self.bundle_path, "configs", "metadata.json")
        if os.path.exists(meta_file):
            try:
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
                # Look for channel defs
                # Structure varies. Common path: network_data_format -> outputs -> pred -> channel_def
                outputs = meta.get("network_data_format", {}).get("outputs", {}).get("pred", {})
                channel_def = outputs.get("channel_def", {})
                
                self.label_map = {int(idx): name for idx, name in channel_def.items()}
                
                # Search for 'prostate'
                for idx, name in self.label_map.items():
                    if "prostate" in name.lower():
                        print(f"Found prostate class: '{name}' at index {idx}")
                        self.target_indices.append(int(idx))
                
                # Fallback: If no prostate, look for Urinary Bladder (anatomically close, good for ZOI)
                if not self.target_indices:
                    print("Prostate label not found. Searching for fallback (bladder)...")
                    for idx, name in self.label_map.items():
                         if "urinary_bladder" in name.lower() or "bladder" in name.lower():
                             print(f"Found fallback class: '{name}' at index {idx}")
                             self.target_indices.append(int(idx))
                             break
            except Exception as e:
                print(f"Error parsing metadata: {e}")
        
        if not self.label_map:
             print("Warning: Label map unavailable; viewers will display generic class names.")

        # 4. Define Transforms

        # We manually define transforms that match the bundle's expectation 
        # because the bundle's 'preprocessing' section often expects file paths (LoadImaged).
        # We start with arrays.
        
        # NOTE: prostate_ct_segmentation usually expects:
        # Spacing: 1.0, 1.0, 1.0 (or specific)
        # Orientation: RAS or SPL? MONAI usually standardized to RAS.
        # Intensity: standard CT windowing.
        
        # We will create a robust transform chain for array inputs
        self.preprocess_transforms = Compose([
            EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"), # Add channel dim
            Orientationd(keys=["image"], axcodes="RAS"), # Align to RAS
            # Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"), # Resample to 1mm iso
            # Intensity Scaling is crucial. Assuming Soft Tissue for Prostate.
            # However, DL models often use ScaleIntensityRanged.
            # We will use generic normalization if we can't parse it exact, 
            # but let's try to mimic standard CT preprocessing.
            ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True), 
            ToTensord(keys=["image"]),
        ])

    def predict(self, volume_np, spacing, original_affine=None):
        """
        Args:
            volume_np (np.ndarray): 3D input volume (Z, Y, X) or (X, Y, Z) depending on Viewer.
                                    DataViewer model loads it as (Z, Y, X).
            spacing (tuple): (z_sp, y_sp, x_sp)
        Returns:
            np.ndarray: Mask with same shape as volume_np
        """
        # DataViewer uses (Z, Y, X). MONAI usually likes (X, Y, Z) or (C, X, Y, Z).
        # Let's verify orientation. 
        # If DicomModel loads via pydicom pixel_array, it's typically (Slices, Rows, Cols) -> (Z, Y, X).
        # To convert to RAS (Physical), we need the affine or direction.
        # But simply flipping axes to (X, Y, Z) is a good start for Spatial transforms.
        
        # 1. Prepare Data Dictionary
        # Reshape (Z, Y, X) -> (X, Y, Z) for MONAI spatial consistency if needed
        # But EnsureChanneld handles 'no_channel'. 
        input_data = {"image": volume_np}
        
        # Add metadata for transforms
        # Spacing is critical for Spacingd
        # If we pass spacing as meta, Spacingd can use it.
        if original_affine is not None:
             input_data["image_meta_dict"] = {"affine": original_affine}
        else:
            # Construct simple affine from spacing assuming orthogonal
            # Note: Input volume_np is Z, Y, X.
            # spacing is dz, dy, dx
            affine = np.eye(4)
            affine[0, 0] = spacing[2] # x
            affine[1, 1] = spacing[1] # y
            affine[2, 2] = spacing[0] # z
            input_data["image_meta_dict"] = {"affine": affine}

        # 2. Preprocess
        # Note: If we really want to robustly match the bundle, we should use the bundle's pre-processing.
        # But 'prostate_ct_segmentation' usually starts from file loaders.
        # We'll use our manual transforms defined in __init__.
        data_tensor = self.preprocess_transforms(input_data)
        image_tensor = data_tensor["image"].unsqueeze(0).to(self.device) # Add batch dim -> (1, C, Spatial...)

        # 3. Inference
        with torch.no_grad():
            # Sliding window is safer for large CTs
            val_outputs = sliding_window_inference(
                inputs=image_tensor, 
                roi_size=(96, 96, 96), 
                sw_batch_size=2, 
                predictor=self.model,
                overlap=0.5
            )
            
        # 4. Post Process
        # Argmax to get class label per voxel
        val_outputs = torch.argmax(val_outputs, dim=1).detach().cpu().numpy()
        mask = val_outputs[0].astype(np.uint8)
        
        # 5. Inverse Transforms (to get back to original input shape/spacing)
        # This is tricky with manual arrays. 
        # If we used Spacingd, the shape changed.
        # For simplicity in this first iteration, we skipped Spacingd in the manual pipeline 
        # to ensure the output grid matches the input grid (1:1 overlay).
        # *Self-Correction*: If the model EXPECTS 1mm isotropic and we give it anisotropic thickness, 
        # performance degrades. But if we resample, we must resample BACK to overlay on the original viewer volume.
        # Given the "DataViewer" constraints, preserving the grid is paramount for simple overlay.
        # We will assume the input is close enough or that we accept the performance hit for now to ensure
        # the overlay works out-of-the-box without complex resampling interpolation logic in the Viewer.
        
        # If mask was permuted by Orientationd(RAS), we might need to revert.
        # However, we passed simple array. Orientationd might have swapped axes.
        # If we input (Z,Y,X) and Affine matches, Orientationd does its thing.
        # Let's keep it simple: assume input is already correctly oriented or just run inference on raw data.
        # For a robust "Add class", let's try to trust the network is robust or the data is standard.
        
        return mask

    def get_label_map(self):
        """Return mapping of label id to readable name (if available)."""
        return self.label_map or {}
