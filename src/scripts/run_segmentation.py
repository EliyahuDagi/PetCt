
import sys
import os
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.DataViewer.model import DicomModel
from src.utils.segmentation_utils import SegmentationPredictor
from src.utils.config import Config

def run_batch_segmentation(data_dir, bundle_name):
    # 1. Setup
    data_path = Path(data_dir)
    output_dir_name = Config.SEGMENTATION_DIR_NAME
    
    # 2. Initialize Predictor
    print(f"Initializing Predictor with bundle: {bundle_name}...")
    predictor = SegmentationPredictor(bundle_name=bundle_name)
    
    # 3. Scan for patients
    # We assume 'data_dir' contains patient folders
    patient_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    print(f"Found {len(patient_dirs)} patient directories in {data_dir}")

    # 4. Process Loop
    for patient_dir in tqdm(patient_dirs, desc="Processing Patients"):
        try:
            # Check if segmentation already exists
            seg_dir = patient_dir / output_dir_name
            seg_file = seg_dir / "mask.npy"
            if seg_file.exists():
                # print(f"Skipping {patient_dir.name}, mask exists.")
                continue

            # Load Data
            model = DicomModel()
            # DicomModel usually needs to load the series.
            # load_patient_data takes a path string
            # It populates self.ct_volume
            model.load_patient_data(str(patient_dir))
            
            if model.ct_volume is None:
                print(f"No CT volume found for {patient_dir.name}")
                continue
                
            # Get Metadata for predictor
            spacing = model.get_voxel_spacing() # (dr, dc, ds) usually -> need to check model.py impl
            
            # Predict
            # model.ct_volume is (Z, Y, X)
            mask = predictor.predict(model.ct_volume, spacing)
            label_map = predictor.get_label_map()
            
            # Save
            seg_dir.mkdir(exist_ok=True)
            np.save(seg_file, mask)
            if label_map:
                labels_path = seg_dir / "labels.json"
                with open(labels_path, "w") as f:
                    json.dump({int(k): v for k, v in label_map.items()}, f, indent=2)
            
        except Exception as e:
            print(f"Error processing {patient_dir.name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--bundle", type=str, default=None, help="MONAI bundle name")
    args = parser.parse_args()
    
    bundle = args.bundle if args.bundle else Config.SEGMENTATION_BUNDLE_NAME
    run_batch_segmentation(args.data_dir, bundle)
