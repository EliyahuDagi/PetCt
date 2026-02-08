import os
import pydicom
import numpy as np

class DicomModel:
    def __init__(self):
        self.patient_list = []
        self.current_patient_index = -1
        self.ct_volume = None
        self.pet_volume = None
        self.ct_metadata = None
        self.pet_metadata = None

    def load_dataset(self, data_path):
        """
        Scans the data_path for subdirectories (patients).
        """
        if not os.path.exists(data_path):
            raise ValueError(f"Path not found: {data_path}")
        
        # Assume each subdir in data_path is a patient or study
        self.patient_list = [
            os.path.join(data_path, d) 
            for d in os.listdir(data_path) 
            if os.path.isdir(os.path.join(data_path, d))
        ]
        self.patient_list.sort()
        self.current_patient_index = -1
        print(f"Found {len(self.patient_list)} patients/studies.")

    def has_next(self):
        return self.current_patient_index < len(self.patient_list) - 1

    def has_prev(self):
        return self.current_patient_index > 0

    def load_next_patient(self):
        if self.has_next():
            self.current_patient_index += 1
            return self.load_patient_data(self.patient_list[self.current_patient_index])
        return False

    def load_prev_patient(self):
        if self.has_prev():
            self.current_patient_index -= 1
            return self.load_patient_data(self.patient_list[self.current_patient_index])
        return False

    def load_patient_data(self, patient_path):
        """
        Loads CT and PET series from the patient folder.
        Assumes structure: patient_path/CT/ and patient_path/PT/ or similar.
        """
        print(f"Loading data from {patient_path}")
        # Reset volumes
        self.ct_volume = None
        self.pet_volume = None
        
        # Simple heuristic to find CT and PT folders
        ct_path = None
        pet_path = None
        
        for root, dirs, files in os.walk(patient_path):
            # Check if this folder has DICOM files (ignore extension)
            candidates = [f for f in files if f not in ['DICOMDIR', 'README.TXT', 'CONTENT.XML']]
            if not candidates:
                continue
            
            # Check first few candidates to see if they are valid DICOMs
            valid_dicom_path = None
            for cand in candidates[:5]: # Check first 5 to be safe/fast
                try:
                    p = os.path.join(root, cand)
                    pydicom.dcmread(p, stop_before_pixels=True)
                    valid_dicom_path = p
                    break
                except:
                    continue
            
            if not valid_dicom_path:
                continue
            
            # Read DICOM to determine modality
            try:
                ds = pydicom.dcmread(valid_dicom_path)
                modality = getattr(ds, 'Modality', '')
                if modality == 'CT' and not ct_path:
                    ct_path = root
                elif modality == 'PT' and not pet_path:
                    pet_path = root
            except Exception as e:
                print(f"Error reading DICOM header in {root}: {e}")

        if ct_path:
            self.ct_volume, self.ct_metadata = self._load_series(ct_path)
        
        if pet_path:
            self.pet_volume, self.pet_metadata = self._load_series(pet_path)

        # Handle Missing Modalities simply
        if self.ct_volume is None and self.pet_volume is None:
             raise ValueError("No CT or PET data found in patient directory.")

        return True

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
        
        # Calculate SUV Factor if PET
        self.suv_factor = 1.0
        try:
             ds = slices[0]
             if getattr(ds, 'Modality', '') == 'PT':
                # SUV Formula: pixel(Bq/ml) * weight(kg) * 1000(g/kg) / dose(Bq)
                weight_kg = float(getattr(ds, 'PatientWeight', 75.0)) # Default 75kg if missing
                
                dose_bq = 1.0
                # Radiopharmaceutical Info usually in Sequence
                if hasattr(ds, 'RadiopharmaceuticalInformationSequence'):
                     seq = ds.RadiopharmaceuticalInformationSequence[0]
                     dose_bq = float(getattr(seq, 'RadionuclideTotalDose', 1.0))
                
                # Some scanners behave differently, but simplistic approach:
                if dose_bq > 0:
                     self.suv_factor = (weight_kg * 1000) / dose_bq
        except Exception as e:
             print(f"Error calculating SUV factor: {e}")
             self.suv_factor = 1.0

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

    def get_suv_factor(self):
        return getattr(self, 'suv_factor', 1.0)


    def get_slice_count(self, orientation='AXIAL'):
        volume = self.ct_volume if self.ct_volume is not None else self.pet_volume
        if volume is None: return 0
        
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
              # Physical Y often increases downwards in simple graphics, but in DICOM:
              # Y is Posterior to Anterior? No.
              # Let's just strictly assume standard orientation matches pixel grid first.
              return [0, width, height, 0] # Local coord system relative to Image Top-Left (0,0)
              
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

    def get_pet_bounds(self, orientation='AXIAL'):
         """ Returns extents for PET independent of CT """
         if self.pet_volume is None: return None
         
         # Need PET spacing/origin independent of CT
         # We need to actually read PET metadata properly.
         # _load_series returns (volume, metadata)
         if not self.pet_metadata: return None
         
         try:
             ds = self.pet_metadata[0]
             dy, dx = ds.PixelSpacing
             if len(self.pet_metadata) > 1:
                z1 = float(self.pet_metadata[0].ImagePositionPatient[2])
                z2 = float(self.pet_metadata[1].ImagePositionPatient[2])
                dz = abs(z2 - z1) # Assuming slice thickness/sorting
                oz = z1
             else:
                dz = getattr(ds, 'SliceThickness', 1.0)
                oz = float(ds.ImagePositionPatient[2])
                
             ox = float(ds.ImagePositionPatient[0])
             oy = float(ds.ImagePositionPatient[1])
             
             vol = self.pet_volume
             
             if orientation == 'AXIAL':
                  width = vol.shape[2] * dx
                  height = vol.shape[1] * dy
                  return [ox, ox + width, oy + height, oy] 
             elif orientation == 'CORONAL':
                  width = vol.shape[2] * dx
                  height = vol.shape[0] * dz
                  # Z direction
                  # Check if z1 < z2 or > z2 to determine start/end relative to top/bottom
                  # Usually we want Top = Head. 
                  # If z1 (slice 0) is Feet, and zN is Head.
                  # Matplotlib origin='upper' means Top is Y-min? No Y-max usually in plot but pixel 0.
                  # extent=[left, right, bottom, top]
                  z_end = oz + (vol.shape[0] * dz) # This is purely additive
                  # If we just treat Z as values:
                  return [ox, ox + width, z_end, oz] # Using Z values directly.
             elif orientation == 'SAGITTAL':
                  width = vol.shape[1] * dy
                  height = vol.shape[0] * dz
                  z_end = oz + (vol.shape[0] * dz)
                  return [oy, oy + width, z_end, oz]
         except:
             return None

    def get_images(self, slice_index, orientation='AXIAL'):
        """
        Returns (ct_image, pet_image) for a given slice index and orientation.
        """
        ct_img = None
        pet_img = None
        
        # Helper to slice volume
        def get_slice(vol, idx, mode):
            if vol is None: return None
            if mode == 'AXIAL':
                idx = min(idx, vol.shape[0]-1)
                return vol[idx, :, :]
            elif mode == 'CORONAL':
                idx = min(idx, vol.shape[1]-1)
                # Volume is (Z, Y, X). Coronal is (Z, X) at fixed Y.
                # We often want Z to be up-down. Matplotlib origin is usually top-left.
                # Standard Coronal: Top is Head (low Z? or high Z?), Bottom is Feet.
                # In standard numpy (Z, Y, X), Z index 0 is usually feet or head depending on scan.
                # Let's just return the raw slice for now, maybe flip Z for display.
                return np.flipud(vol[:, idx, :]) 
            elif mode == 'SAGITTAL':
                idx = min(idx, vol.shape[2]-1)
                # Volume is (Z, Y, X). Sagittal is (Z, Y) at fixed X.
                return np.flipud(vol[:, :, idx])
            return None

        ct_img = get_slice(self.ct_volume, slice_index, orientation)
        
        # PET Logic with orientation is tricky because PET might have different Z-spacing.
        # For MVP, we will only try to slice PET if it matches CT shape or we fail gracefully.
        # If geometry differs, MPR on PET natively requires volume resampling.
        # Check if basic shapes match
        if self.pet_volume is not None:
             if self.ct_volume is not None and self.ct_volume.shape == self.pet_volume.shape:
                  pet_img = get_slice(self.pet_volume, slice_index, orientation)
             else:
                  # Fallback or simple slice if independent
                  # If we are in Axial, we use the sophisticated Z-matching
                  if orientation == 'AXIAL':
                      # ... (existing Z-matching logic reused largely)
                      if self.ct_volume is not None:
                           # ... reusing old logic if desired, or just simple index for now to save time
                           # Let's stick to the sophisticated logic for Axial:
                           ct_z = float(self.ct_metadata[min(slice_index, len(self.ct_metadata)-1)].ImagePositionPatient[2])
                           best_idx = 0
                           min_dist = float('inf')
                           for i, s in enumerate(self.pet_metadata):
                               z = float(s.ImagePositionPatient[2])
                               dist = abs(z - ct_z)
                               if dist < min_dist:
                                   min_dist = dist
                                   best_idx = i
                           pet_img = self.pet_volume[best_idx]
                  else:
                       # Handle MPR for PET when shapes differ from CT
                       # We map the slice index from CT space to PET space
                       if self.ct_volume is not None:
                           # Determine Ratio
                           ct_max = 0
                           pet_max = 0
                           if orientation == 'CORONAL':
                               ct_max = self.ct_volume.shape[1]
                               pet_max = self.pet_volume.shape[1]
                           elif orientation == 'SAGITTAL':
                               ct_max = self.ct_volume.shape[2]
                               pet_max = self.pet_volume.shape[2]
                           
                           if ct_max > 0 and pet_max > 0:
                               ratio = pet_max / ct_max
                               pet_idx = int(slice_index * ratio)
                               pet_img = get_slice(self.pet_volume, pet_idx, orientation)
                           else:
                               pet_img = None
                       else:
                            # No CT, just slice PET directly (though slice_index might be wrong range?)
                            # Assuming slice_index is correct for whatever is driving the view
                            pet_img = get_slice(self.pet_volume, slice_index, orientation)

        return ct_img, pet_img
