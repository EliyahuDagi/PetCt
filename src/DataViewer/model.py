import os
import json
from typing import Optional

import numpy as np
import pydicom

from src.utils.config import Config
from src.utils.geometry import VolumeGeometry
from src.utils.prostate_locator import locate_prostate_bbox
from src.utils.segmentors import DiskSegmentor, PetBoxSegmentor, SegmentationResult

class DicomModel:
    def __init__(self):
        self.patient_list = []
        self.current_patient_index = -1
        self.ct_volume = None
        self.pet_volume = None
        self.pet_nac_volume = None
        self.segmentation_mask = None
        self.segmentation_labels = {}
        self.segmentation_available_classes = []
        self.ct_metadata = None
        self.pet_metadata = None
        self.pet_nac_metadata = None
        self.segmentation_source_name = Config.DEFAULT_SEGMENTATION_SOURCE
        self.segmentation_bbox_mm = None
        self.segmentation_bbox_vox = None
        self.segmentation_method = ""
        self.segmentation_debug = None

    def load_dataset(self, data_path):
        """
        Scans the data_path for subdirectories (patients).
        """
        if not os.path.exists(data_path):
            raise ValueError(f"Path not found: {data_path}")
        
        # Heuristic: allow selecting either a dataset root (many patients) or a single
        # patient folder (e.g., one that contains DICOM/, SECTRA/, Segmentation/).
        subdirs = [
            os.path.join(data_path, d)
            for d in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, d))
        ]
        subdir_names = {os.path.basename(d).lower() for d in subdirs}
        patient_markers = {"dicom", "ct", "pt", "pet", "segmentation", "sectra"}

        looks_like_single_patient = bool(subdir_names & patient_markers)

        if looks_like_single_patient:
            # Treat the selected folder as the patient root so we find Segmentation next to DICOM.
            self.patient_list = [data_path]
        else:
            # Assume each subdir is a patient or study
            self.patient_list = sorted(subdirs)

        self.current_patient_index = -1
        print(f"Found {len(self.patient_list)} patients/studies.")

    def has_next(self):
        return self.current_patient_index < len(self.patient_list) - 1

    def has_prev(self):
        return self.current_patient_index > 0

    def load_next_patient(self, segmentation_name=None):
        if self.has_next():
            self.current_patient_index += 1
            return self.load_patient_data(
                self.patient_list[self.current_patient_index], segmentation_name
            )
        return False

    def load_prev_patient(self, segmentation_name=None):
        if self.has_prev():
            self.current_patient_index -= 1
            return self.load_patient_data(
                self.patient_list[self.current_patient_index], segmentation_name
            )
        return False

    def load_patient_data(self, patient_path, segmentation_name=None, load_ct=True, run_segmentation=True):
        """
        Loads CT and PET series from the patient folder.
        Assumes structure: patient_path/CT/ and patient_path/PT/ or similar.

        ``load_ct=False`` skips the (costly) full CT series read and
        ``run_segmentation=False`` skips the segmentor. Both default True to
        preserve viewer behavior; the training PET pipelines pass False since
        they use PET only and never need a segmentation.
        """
        print(f"Loading data from {patient_path}")
        # Reset volumes
        self.ct_volume = None
        self.pet_volume = None
        self.pet_nac_volume = None
        self.pet_nac_metadata = None
        
        # Collect CT/PT candidates and prefer the one with the most slices
        ct_candidates = {}
        pet_candidates = []
        
        for root, dirs, files in os.walk(patient_path):
            # Check if this folder has DICOM files (ignore non-image markers)
            candidates = [f for f in files if f not in ['DICOMDIR', 'README.TXT', 'CONTENT.XML']]
            if not candidates:
                continue

            # Check a few files to confirm DICOM and modality
            valid_dicom_path = None
            for cand in candidates[:5]:  # Check first 5 to be safe/fast
                try:
                    p = os.path.join(root, cand)
                    pydicom.dcmread(p, stop_before_pixels=True)
                    valid_dicom_path = p
                    break
                except:
                    continue
            if not valid_dicom_path:
                continue

            try:
                ds = pydicom.dcmread(valid_dicom_path)
                modality = getattr(ds, 'Modality', '')
                slice_count = len(candidates)
                if modality == 'CT':
                    # Keep the largest CT series (most slices)
                    if slice_count > ct_candidates.get(root, 0):
                        ct_candidates[root] = slice_count
                elif modality == 'PT':
                    pet_candidates.append({
                        "path": root,
                        "slice_count": slice_count,
                        "series_number": getattr(ds, "SeriesNumber", None),
                        "series_description": getattr(ds, "SeriesDescription", ""),
                        "image_type": getattr(ds, "ImageType", None),
                        "ac_class": self._classify_pet_series(ds),
                    })
            except Exception as e:
                print(f"Error reading DICOM header in {root}: {e}")

        ct_path = max(ct_candidates, key=ct_candidates.get) if ct_candidates else None
        pet_ac_path = None
        pet_nac_path = None
        pet_fallback_path = None

        if pet_candidates:
            def _pick_best(candidates):
                return max(
                    candidates,
                    key=lambda c: (c.get("slice_count", 0), c.get("series_number") or -1)
                )

            ac_candidates = [c for c in pet_candidates if c.get("ac_class") == "ac"]
            nac_candidates = [c for c in pet_candidates if c.get("ac_class") == "nac"]
            unknown_candidates = [c for c in pet_candidates if c.get("ac_class") is None]

            if ac_candidates:
                pet_ac_path = _pick_best(ac_candidates).get("path")
            if nac_candidates:
                pet_nac_path = _pick_best(nac_candidates).get("path")
            if unknown_candidates:
                pet_fallback_path = _pick_best(unknown_candidates).get("path")
            if not pet_ac_path:
                pet_ac_path = pet_fallback_path

        # Skipping the CT read (load_ct=False) is the main speedup for training:
        # the CT series is by far the largest volume and PET stages never use it.
        if ct_path and load_ct:
            self.ct_volume, self.ct_metadata = self._load_series(ct_path)

        if pet_ac_path:
            self.pet_volume, self.pet_metadata = self._load_series(pet_ac_path)
        if pet_nac_path:
            self.pet_nac_volume, self.pet_nac_metadata = self._load_series(pet_nac_path)

        # SUV factor uses the AC series when available, else fallback to NAC
        self.suv_factor = self._compute_suv_factor(self.pet_metadata or self.pet_nac_metadata)

        # Handle Missing Modalities simply. Include NAC PET so a NAC-only (or
        # CT-skipped) load is not rejected when it still has usable PET data.
        if self.ct_volume is None and self.pet_volume is None and self.pet_nac_volume is None:
             raise ValueError("No CT or PET data found in patient directory.")
             
        # Reset segmentation state
        self.segmentation_mask = None
        self.segmentation_labels = {}
        self.segmentation_available_classes = []
        self.segmentation_debug = None
        self.segmentation_bbox_mm = None
        self.segmentation_bbox_vox = None
        self.segmentation_method = ""

        # Load Segmentation if exists or run locator
        if segmentation_name:
            self.segmentation_source_name = segmentation_name
        elif not getattr(self, "segmentation_source_name", None):
            self.segmentation_source_name = Config.DEFAULT_SEGMENTATION_SOURCE

        # Segmentation is a viewer-only feature; training loads skip it.
        if run_segmentation:
            self._run_segmentor(patient_path, self.segmentation_source_name)

        return True

    def reload_segmentation_for_current(self, segmentation_name):
        if self.current_patient_index < 0 or self.current_patient_index >= len(self.patient_list):
            return False
        if segmentation_name:
            self.segmentation_source_name = segmentation_name
        patient_path = self.patient_list[self.current_patient_index]
        self._run_segmentor(patient_path, self.segmentation_source_name)
        return True

    def _load_segmentation(self, patient_path, segmentation_name):
        base_dir = os.path.join(patient_path, Config.SEGMENTATION_DIR_NAME)
        candidates = []
        if segmentation_name:
            candidates.append(os.path.join(base_dir, segmentation_name))
        candidates.append(base_dir)

        for seg_dir in candidates:
            seg_path = os.path.join(seg_dir, "mask.npy")
            if not os.path.exists(seg_path):
                continue

            try:
                seg_label = os.path.basename(seg_dir)
                print(f"Loading segmentation from {seg_path} (source: {seg_label})")
                mask = np.load(seg_path)
                if self.ct_volume is not None and mask.shape != self.ct_volume.shape:
                    print(
                        f"Warning: Segmentation shape {mask.shape} != CT shape {self.ct_volume.shape}"
                    )
                    continue

                labels_path = os.path.join(seg_dir, "labels.json")
                labels = self._read_segmentation_labels(labels_path)
                return SegmentationResult(
                    mask=mask,
                    labels=labels,
                    method=f"disk:{seg_label}",
                )
            except Exception as e:
                print(f"Error loading segmentation from {seg_path}: {e}")
                continue
        return None

    def _apply_segmentation_result(self, result: SegmentationResult):
        if result is None:
            return
        self.segmentation_mask = result.mask
        self.segmentation_labels = result.labels or {}
        self.segmentation_bbox_mm = result.bbox_mm
        self.segmentation_bbox_vox = result.bbox_vox
        self.segmentation_method = result.method or ""
        self.segmentation_debug = getattr(result, "debug", None)
        if self.segmentation_mask is not None:
            self._update_available_segmentation_classes()
        else:
            self.segmentation_available_classes = []

    def _run_segmentor(self, patient_path, segmentation_name):
        method_key = Config.SEGMENTATION_METHOD_MAP.get(segmentation_name, "disk")

        if method_key == "pet_box":
            pet_spacing, pet_origin = self._get_pet_spacing_origin()
            seg = PetBoxSegmentor().segment(
                patient_path,
                self.ct_volume,
                self.pet_volume,
                pet_spacing,
                pet_origin,
            )
            self._apply_segmentation_result(seg)
            return

        # Default: load from disk
        seg = DiskSegmentor(segmentation_name, Config.SEGMENTATION_DIR_NAME).segment(
            patient_path,
            self.ct_volume,
            self.pet_volume,
            self.get_voxel_spacing(),
            self.get_origin(),
        )
        self._apply_segmentation_result(seg)


    def _load_series(self, series_path):
        """
        Reads a DICOM series and returns a 3D numpy array + list of datasets.
        """
        files = [os.path.join(series_path, f) for f in os.listdir(series_path) 
                 if f not in ['DICOMDIR', 'README.TXT', 'CONTENT.XML']]
        
        slices = []
        for f in files:
            try:
                ds = pydicom.dcmread(f)
                # Ensure it has image data
                if hasattr(ds, 'pixel_array') and hasattr(ds, 'ImagePositionPatient'):
                    slices.append(ds)
            except:
                continue
                
        if not slices:
            return None, None

        # Sort by ImagePositionPatient Z coordinate (usually index 2)
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        
        # Stack pixel data
        # Handle Rescale Slope/Intercept if present to get Hunsfield Units or Bq/ml
        images = []
        for s in slices:
            img = s.pixel_array.astype(np.float32)
            slope = getattr(s, 'RescaleSlope', 1)
            intercept = getattr(s, 'RescaleIntercept', 0)
            img = img * slope + intercept
            images.append(img)
            
        return np.stack(images), slices

    def _compute_suv_factor(self, metadata):
        if not metadata:
            return 1.0
        try:
            ds = metadata[0]
            if getattr(ds, 'Modality', '') != 'PT':
                return 1.0
            # SUV Formula: pixel(Bq/ml) * weight(kg) * 1000(g/kg) / dose(Bq)
            weight_kg = float(getattr(ds, 'PatientWeight', 75.0))
            dose_bq = 1.0
            if hasattr(ds, 'RadiopharmaceuticalInformationSequence'):
                seq = ds.RadiopharmaceuticalInformationSequence[0]
                dose_bq = float(getattr(seq, 'RadionuclideTotalDose', 1.0))
            if dose_bq > 0:
                return (weight_kg * 1000) / dose_bq
        except Exception as e:
            print(f"Error calculating SUV factor: {e}")
        return 1.0

    def _classify_pet_series(self, ds):
        """Classify PET series as AC, NAC, or unknown using DICOM tags only.

        The DICOM ``CorrectedImage`` tag (0028,0051) is authoritative: it lists
        the corrections applied to the series. ``ATTN`` present -> attenuation
        corrected ("ac"); present without ``ATTN`` -> not attenuation corrected
        ("nac"). Only when the tag is absent do we fall back to the
        SeriesDescription/ImageType token heuristic (which misses cases like
        "PET WB-uncorrected" that tokenize to "WB-UNCORRECTED").
        """
        corrected = ds.get((0x0028, 0x0051))
        if corrected is not None and corrected.value:
            attrs = corrected.value
            if isinstance(attrs, str):
                attrs = [attrs]
            attrs = [str(a).upper() for a in attrs]
            if "ATTN" in attrs:
                return "ac"
            return "nac"

        desc = str(getattr(ds, "SeriesDescription", "")).upper()
        img_type = getattr(ds, "ImageType", None)

        def _tokens(val):
            if val is None:
                return []
            if isinstance(val, (list, tuple)):
                raw = [str(v) for v in val]
            else:
                raw = str(val).replace("\\", " ").replace("/", " ").split()
            return [t.strip().upper() for t in raw if t]

        desc_tokens = _tokens(desc)
        img_tokens = _tokens(img_type)
        all_tokens = set(desc_tokens + img_tokens)

        if "NAC" in all_tokens or "NON-AC" in all_tokens or "NONAC" in all_tokens or "UNCORRECTED" in all_tokens:
            return "nac"
        if "AC" in all_tokens or "ATTENUATION" in all_tokens or "CORRECTED" in all_tokens:
            return "ac"

        return None

    def get_suv_factor(self):
        return getattr(self, 'suv_factor', 1.0)


    def get_slice_count(self, orientation='AXIAL'):
        volume = self.ct_volume if self.ct_volume is not None else self.pet_volume
        if volume is None: 
            return 0
        
        if orientation == 'AXIAL':     # Z-axis
            return volume.shape[0]
        elif orientation == 'CORONAL': # Y-axis
            return volume.shape[1]
        elif orientation == 'SAGITTAL':# X-axis
            return volume.shape[2]
        return 0

    def get_voxel_spacing(self):
        """ Returns (dz, dy, dx) """
        if not self.ct_metadata:
            return (1.0, 1.0, 1.0)
        
        try:
            ds = self.ct_metadata[0]
            dy, dx = ds.PixelSpacing
            # Estimate dz
            if len(self.ct_metadata) > 1:
                z1 = float(self.ct_metadata[0].ImagePositionPatient[2])
                z2 = float(self.ct_metadata[1].ImagePositionPatient[2])
                dz = abs(z2 - z1)
            else:
                dz = getattr(ds, 'SliceThickness', 1.0)
            return (float(dz), float(dy), float(dx))
        except:
            return (1.0, 1.0, 1.0)

    def get_origin(self):
        """ Returns (z, y, x) origin from the first slice of CT """
        if not self.ct_metadata: return (0,0,0)
        try:
            pos = self.ct_metadata[0].ImagePositionPatient
            return (float(pos[2]), float(pos[1]), float(pos[0]))
        except:
            return (0,0,0)

    def get_bounds(self, orientation='AXIAL'):
         """ 
         Returns extents for the view [x_start, x_end, y_end, y_start] (Matplotlib convention).
         Ideally returns (left, right, bottom, top) in physical coordinates.
         """
         # Priority to CT geometry
         if self.ct_volume is not None:
              vol = self.ct_volume
              dz, dy, dx = self.get_voxel_spacing()
              oz, oy, ox = self.get_origin()
         else:
              return [0, 1, 1, 0] # Fallback
         
         if orientation == 'AXIAL':
              # X axis: ox to ox + X*dx
              # Y axis: oy to oy + Y*dy
              width = vol.shape[2] * dx
              height = vol.shape[1] * dy
              # Matplotlib imshow extent [left, right, bottom, top]
              # Image coords: row 0 is top usually. origin='upper'.
              # Physical Y often increases downwards or upwards relative to array?
              # Standard DICOM ImagePositionPatient is top-left voxel.
              # So top (y start in image space) is oy.
              # Bottom (y end in image space) is oy + height (assuming standard scanning direction).
              # [left, right, bottom, top]
              return [ox, ox + width, oy + height, oy] 
              
              # If we want alignment between CT and PET, we MUST use absolute coords?
              # Yes.
              # Axial: X is X (Right-Left), Y is Y (Ant-Post).
              # We return [ox, ox + width, oy + height, oy] ... wait y axis direction?
              # Let's try [ox, ox + width, oy + height, oy] 
              # Depending on row-order. DICOM rows usually scan Y top to bottom.
              # let's assume standard [left, right, bottom, top]
              # return [ox, ox + width, oy + height, oy]
         
         elif orientation == 'CORONAL':
              # X Axis: X. Y Axis: Z.
              width = vol.shape[2] * dx
              height = vol.shape[0] * dz 
              # Z increases usually head to feet or feet to head.
              # We want 'Top' of screen to be Head.
              # If Z increases feet->head (typical HFS), higher Z is top.
              # Matplotlib origin='upper' (pixel 0,0 at top-left).
              # We need to map pixel grid to coords.
              # Let's use 0-based for now but separate scaling for CT/PET?
              # No, if we want them to align, we rely on the fact that we sliced them roughly same way.
              # But user says PET covers more.
              # If we use physical coords:
              # CT Z range: 1000 to 1250.
              # PET Z range: 0 to 1800.
              # If we use extent=[0, W, 1250, 1000] for CT and [0, W, 1800, 0] for PET:
              # They will align on Y axis correctly!
              
              # Z limits
              z_start = oz
              z_end = oz + (vol.shape[0] * dz) # Assumes constant spacing and specific direction
              # If direction is reversed (z decreasing), handle
              # Check dz sign? We used abs() in voxel spacing.
              # Let's assume standard behavior for MVP.
              
              return [ox, ox + width, z_end, z_start] # Z usually Y axis effectively. 
         
         elif orientation == 'SAGITTAL':
             # X Axis: Y. Y Axis: Z.
             width = vol.shape[1] * dy
             height = vol.shape[0] * dz
             
             z_start = oz
             z_end = oz + (vol.shape[0] * dz)
             
             return [oy, oy + width, z_end, z_start]

    def _get_bounds_for_metadata(self, volume, metadata, orientation='AXIAL'):
         if volume is None or not metadata:
             return None
         try:
             ds = metadata[0]
             dy, dx = ds.PixelSpacing
             if len(metadata) > 1:
                 z1 = float(metadata[0].ImagePositionPatient[2])
                 z2 = float(metadata[1].ImagePositionPatient[2])
                 dz = abs(z2 - z1)
                 oz = z1
             else:
                 dz = getattr(ds, 'SliceThickness', 1.0)
                 oz = float(ds.ImagePositionPatient[2])

             ox = float(ds.ImagePositionPatient[0])
             oy = float(ds.ImagePositionPatient[1])

             vol = volume

             if orientation == 'AXIAL':
                 width = vol.shape[2] * dx
                 height = vol.shape[1] * dy
                 return [ox, ox + width, oy + height, oy]
             if orientation == 'CORONAL':
                 width = vol.shape[2] * dx
                 height = vol.shape[0] * dz
                 z_end = oz + (vol.shape[0] * dz)
                 return [ox, ox + width, z_end, oz]
             if orientation == 'SAGITTAL':
                 width = vol.shape[1] * dy
                 height = vol.shape[0] * dz
                 z_end = oz + (vol.shape[0] * dz)
                 return [oy, oy + width, z_end, oz]
         except Exception:
             return None

    def get_pet_bounds(self, orientation='AXIAL'):
         """Returns extents for PET AC (legacy PET fields)."""
         return self._get_bounds_for_metadata(self.pet_volume, self.pet_metadata, orientation)

    def get_pet_nac_bounds(self, orientation='AXIAL'):
         """Returns extents for PET NAC if available."""
         return self._get_bounds_for_metadata(self.pet_nac_volume, self.pet_nac_metadata, orientation)

    def _get_pet_spacing_origin(self):
         """Returns (spacing, origin) for PET if available, else falls back to CT spacing/origin."""
         if not self.pet_metadata:
             return self.get_voxel_spacing(), self.get_origin()
         try:
             ds = self.pet_metadata[0]
             dy, dx = ds.PixelSpacing
             if len(self.pet_metadata) > 1:
                 z1 = float(self.pet_metadata[0].ImagePositionPatient[2])
                 z2 = float(self.pet_metadata[1].ImagePositionPatient[2])
                 dz = abs(z2 - z1)
                 oz = z1
             else:
                 dz = getattr(ds, 'SliceThickness', 1.0)
                 oz = float(ds.ImagePositionPatient[2])
             ox = float(ds.ImagePositionPatient[0])
             oy = float(ds.ImagePositionPatient[1])
             return (float(dz), float(dy), float(dx)), (oz, oy, ox)
         except Exception:
             return self.get_voxel_spacing(), self.get_origin()

    def get_segmentation_bounds(self, orientation='AXIAL', label='all_classes'):
         """
         Returns physical bounds [xmin, xmax, ymax, ymin] of the segmentation mask
         for zooming the view.
         """
         if self.segmentation_mask is None:
             return None

         mask = self._get_binary_mask_for_label(label)
         if mask is None:
              return None

         # Find indices
         indices = np.argwhere(mask > 0)
         if indices.size == 0:
             return None
         
         # indices are (z, y, x)
         z_min, y_min, x_min = indices.min(axis=0)
         z_max, y_max, x_max = indices.max(axis=0)
         
         dz, dy, dx = self.get_voxel_spacing()
         oz, oy, ox = self.get_origin()
         
         # Add some padding (e.g., 20mm)
         pad = 20.0
         
         if orientation == 'AXIAL':
             # X (x indices), Y (y indices)
             x1 = ox + x_min * dx - pad
             x2 = ox + x_max * dx + pad
             y1 = oy + y_min * dy - pad
             y2 = oy + y_max * dy + pad
             # y increases? In get_bounds assume standard. 
             # Return [left, right, bottom, top]
             return [x1, x2, y2, y1]
             
         elif orientation == 'CORONAL':
             # X (x indices), Y (z indices effectively)
             x1 = ox + x_min * dx - pad
             x2 = ox + x_max * dx + pad
             z1 = oz + z_min * dz - pad
             z2 = oz + z_max * dz + pad
             # Z usually corresponds to Y axis on screen
             return [x1, x2, z2, z1]
             
         elif orientation == 'SAGITTAL':
             # X (y indices), Y (z indices)
             y1 = oy + y_min * dy - pad
             y2 = oy + y_max * dy + pad
             z1 = oz + z_min * dz - pad
             z2 = oz + z_max * dz + pad
             return [y1, y2, z2, z1]
             
         return None

    def has_segmentation_label(self, label='all_classes'):
         """Check if the requested label has any voxels in the current mask."""
         if self.segmentation_mask is None:
             return False
         if label in (None, 'all_classes'):
             return np.any(self.segmentation_mask)
         try:
             label_id = int(label)
         except (ValueError, TypeError):
             return False
         return np.any(self.segmentation_mask == label_id)

    def _bbox_mm_to_bounds(self, bbox_mm, orientation='AXIAL'):
         if not bbox_mm:
             return None
         x1, x2 = bbox_mm.get("x", [None, None])
         y1, y2 = bbox_mm.get("y", [None, None])
         z1, z2 = bbox_mm.get("z", [None, None])
         if None in (x1, x2, y1, y2, z1, z2):
             return None
         if orientation == 'AXIAL':
             return [x1, x2, y2, y1]
         elif orientation == 'CORONAL':
             return [x1, x2, z2, z1]
         elif orientation == 'SAGITTAL':
             return [y1, y2, z2, z1]
         return None

    def locate_prostate_roi(self, orientation='AXIAL', force_locator: bool = False, debug_override: Optional[bool] = None):
        """Estimate prostate center slice and bounds using bladder + PET heuristics."""
        # If a segmentor already produced a bbox (e.g., PET_BOX), use it directly
        if self.segmentation_bbox_mm and not force_locator:
            bounds = self._bbox_mm_to_bounds(self.segmentation_bbox_mm, orientation)
            center_slice = None

            spacing = None
            origin = None
            shape_for_geom = None
            if self.ct_volume is not None and self.ct_metadata:
                spacing = self.get_voxel_spacing()
                origin = self.get_origin()
                shape_for_geom = self.ct_volume.shape
            elif (self.segmentation_method or "").startswith("pet_box") and self.pet_volume is not None:
                spacing, origin = self._get_pet_spacing_origin()
                shape_for_geom = self.pet_volume.shape

            if spacing and origin and shape_for_geom is not None:
                geom = VolumeGeometry(spacing, origin, shape_for_geom)
                z1, z2 = self.segmentation_bbox_mm.get("z", (0.0, 0.0))
                y1, y2 = self.segmentation_bbox_mm.get("y", (0.0, 0.0))
                x1, x2 = self.segmentation_bbox_mm.get("x", (0.0, 0.0))
                center_mm = np.array([(z1 + z2) / 2.0, (y1 + y2) / 2.0, (x1 + x2) / 2.0], dtype=float)
                center_vox = geom.mm_to_vox(center_mm)
                if orientation == 'AXIAL':
                    center_slice = int(round(center_vox[0]))
                elif orientation == 'CORONAL':
                    center_slice = int(round(center_vox[1]))
                elif orientation == 'SAGITTAL':
                    center_slice = int(round(center_vox[2]))

            if center_slice is None and self.segmentation_bbox_vox:
                if orientation == 'AXIAL':
                    z_lo, z_hi = self.segmentation_bbox_vox.get("z", (0, 0))
                    center_slice = (z_lo + z_hi) // 2
                elif orientation == 'CORONAL':
                    y_lo, y_hi = self.segmentation_bbox_vox.get("y", (0, 0))
                    center_slice = (y_lo + y_hi) // 2
                elif orientation == 'SAGITTAL':
                    x_lo, x_hi = self.segmentation_bbox_vox.get("x", (0, 0))
                    center_slice = (x_lo + x_hi) // 2

            if center_slice is not None:
                max_slices = self.get_slice_count(orientation)
                if max_slices > 0:
                    center_slice = max(0, min(center_slice, max_slices - 1))

            return {
                "center_slice": center_slice,
                "bounds": bounds,
                "method": self.segmentation_method or "segmentor_bbox",
                "bbox_vox": self.segmentation_bbox_vox,
                "bbox_mm": self.segmentation_bbox_mm,
            }

        if self.segmentation_mask is None and self.pet_volume is None:
            return None

        try:
            pet_spacing, pet_origin = self._get_pet_spacing_origin()
            ct_spacing = self.get_voxel_spacing()
            ct_origin = self.get_origin()
            debug_flag = bool(getattr(Config, 'PROSTATE_LOCATOR_DEBUG', False))
            if debug_override is not None:
                debug_flag = bool(debug_override)
            bbox = locate_prostate_bbox(
                mask=self.segmentation_mask,
                labels=self.segmentation_labels or {},
                spacing=ct_spacing,
                origin=ct_origin,
                pet_volume=self.pet_volume,
                pet_spacing=pet_spacing,
                pet_origin=pet_origin,
                ct_volume=self.ct_volume,
                ct_spacing=ct_spacing,
                ct_origin=ct_origin,
                debug=debug_flag,
            )
        except Exception as e:
            print(f"Prostate locator error: {e}")
            return None

        if not bbox:
            return None

        center_vox = bbox.get("center_vox")
        if not center_vox:
            return None

        if orientation == 'AXIAL':
            center_slice = int(center_vox[0])
        elif orientation == 'CORONAL':
            center_slice = int(center_vox[1])
        elif orientation == 'SAGITTAL':
            center_slice = int(center_vox[2])
        else:
            center_slice = int(center_vox[0])

        # Keep slice index within valid range
        max_slices = self.get_slice_count(orientation)
        if max_slices > 0:
            center_slice = max(0, min(center_slice, max_slices - 1))

        bounds = self._bbox_mm_to_bounds(bbox.get("bbox_mm"), orientation)

        return {
            "center_slice": center_slice,
            "bounds": bounds,
            "method": bbox.get("method"),
            "bbox_vox": bbox.get("bbox_vox"),
            "bbox_mm": bbox.get("bbox_mm"),
            "center_vox": bbox.get("center_vox"),
            "debug": bbox.get("debug"),
        }
         
    def get_segmentation_center_slice(self, orientation='AXIAL', label='all_classes'):
         """ Returns the slice index of the center of the segmentation """
         mask = self._get_binary_mask_for_label(label)
         if mask is None: return 0
         indices = np.argwhere(mask > 0)
         if indices.size == 0: return 0
         
         min_idx = indices.min(axis=0)
         max_idx = indices.max(axis=0)
         center = (min_idx + max_idx) // 2
         
         if orientation == 'AXIAL': return center[0]
         elif orientation == 'CORONAL': return center[1]
         elif orientation == 'SAGITTAL': return center[2]
         return 0

    def get_images(self, slice_index, orientation='AXIAL'):
        """
        Returns (ct_image, pet_image, seg_image) for a given slice index and orientation.
        """
        ct_img = None
        pet_img = None
        seg_img = None
        
        # Helper to slice volume
        def get_slice(vol, idx, mode):
            if vol is None: return None
            if mode == 'AXIAL':
                idx = min(idx, vol.shape[0]-1)
                return vol[idx, :, :]
            elif mode == 'CORONAL':
                idx = min(idx, vol.shape[1]-1)
                # Volume is (Z, Y, X). Coronal is (Z, X) at fixed Y.
                return np.flipud(vol[:, idx, :]) 
            elif mode == 'SAGITTAL':
                idx = min(idx, vol.shape[2]-1)
                # Volume is (Z, Y, X). Sagittal is (Z, Y) at fixed X.
                return np.flipud(vol[:, :, idx])
            return None

        ct_img = get_slice(self.ct_volume, slice_index, orientation)
        
        # Segmentation Logic (matches CT geometry 1:1)
        if self.segmentation_mask is not None:
             seg_img = get_slice(self.segmentation_mask, slice_index, orientation)
        
        # PET Logic with orientation is tricky because PET might have different Z-spacing.
        pet_img = self._get_pet_slice_for_volume(
            self.pet_volume,
            self.pet_metadata,
            slice_index,
            orientation,
            get_slice,
        )

        return ct_img, pet_img, seg_img

    def get_images_with_nac(self, slice_index, orientation='AXIAL'):
        """Returns (ct_image, pet_ac_image, pet_nac_image, seg_image)."""
        ct_img = None
        seg_img = None

        def get_slice(vol, idx, mode):
            if vol is None:
                return None
            if mode == 'AXIAL':
                idx = min(idx, vol.shape[0] - 1)
                return vol[idx, :, :]
            if mode == 'CORONAL':
                idx = min(idx, vol.shape[1] - 1)
                return np.flipud(vol[:, idx, :])
            if mode == 'SAGITTAL':
                idx = min(idx, vol.shape[2] - 1)
                return np.flipud(vol[:, :, idx])
            return None

        ct_img = get_slice(self.ct_volume, slice_index, orientation)
        if self.segmentation_mask is not None:
            seg_img = get_slice(self.segmentation_mask, slice_index, orientation)

        pet_ac_img = self._get_pet_slice_for_volume(
            self.pet_volume,
            self.pet_metadata,
            slice_index,
            orientation,
            get_slice,
        )
        pet_nac_img = self._get_pet_slice_for_volume(
            self.pet_nac_volume,
            self.pet_nac_metadata,
            slice_index,
            orientation,
            get_slice,
        )

        return ct_img, pet_ac_img, pet_nac_img, seg_img

    def _get_pet_slice_for_volume(self, volume, metadata, slice_index, orientation, get_slice):
        if volume is None or metadata is None:
            return None
        if self.ct_volume is not None and volume.shape == self.ct_volume.shape:
            return get_slice(volume, slice_index, orientation)

        if orientation == 'AXIAL':
            if self.ct_volume is not None and self.ct_metadata:
                ct_z = float(self.ct_metadata[min(slice_index, len(self.ct_metadata) - 1)].ImagePositionPatient[2])
                best_idx = 0
                min_dist = float('inf')
                for i, s in enumerate(metadata):
                    z = float(s.ImagePositionPatient[2])
                    dist = abs(z - ct_z)
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = i
                return volume[best_idx]
            return get_slice(volume, slice_index, orientation)

        if self.ct_volume is not None:
            ct_max = 0
            pet_max = 0
            if orientation == 'CORONAL':
                ct_max = self.ct_volume.shape[1]
                pet_max = volume.shape[1]
            elif orientation == 'SAGITTAL':
                ct_max = self.ct_volume.shape[2]
                pet_max = volume.shape[2]

            if ct_max > 0 and pet_max > 0:
                ratio = pet_max / ct_max
                pet_idx = int(slice_index * ratio)
                return get_slice(volume, pet_idx, orientation)

        return get_slice(volume, slice_index, orientation)

    def _read_segmentation_labels(self, labels_path):
        if not labels_path or not os.path.exists(labels_path):
            return {}
        try:
            with open(labels_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return {int(k): v for k, v in data.items()}
        except Exception as e:
            print(f"Error loading segmentation labels: {e}")
            return {}

    def _load_segmentation_labels(self, labels_path):
        self.segmentation_labels = self._read_segmentation_labels(labels_path)

    def _update_available_segmentation_classes(self):
        if self.segmentation_mask is None:
            self.segmentation_available_classes = []
            return
        unique_vals = np.unique(self.segmentation_mask)
        self.segmentation_available_classes = [int(val) for val in unique_vals if val != 0]

    def _get_binary_mask_for_label(self, label):
        if self.segmentation_mask is None:
            return None
        if label in (None, 'all_classes'):
            return (self.segmentation_mask > 0).astype(np.uint8)
        try:
            label_id = int(label)
        except (ValueError, TypeError):
            return (self.segmentation_mask > 0).astype(np.uint8)
        return (self.segmentation_mask == label_id).astype(np.uint8)

    def get_segmentation_label_map(self):
        return self.segmentation_labels

    def get_available_segmentation_classes(self):
        classes = []
        for label_id in self.segmentation_available_classes:
            display = self.segmentation_labels.get(label_id, f"Label {label_id}")
            classes.append({"id": label_id, "name": display})
        return classes
