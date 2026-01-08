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
            # Check if this folder has DICOM files
            dicom_files = [f for f in files if f.endswith('.dcm')]
            if not dicom_files:
                continue
            
            # Read first DICOM to determine modality
            try:
                ds = pydicom.dcmread(os.path.join(root, dicom_files[0]))
                modality = ds.Modality
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
        files = [os.path.join(series_path, f) for f in os.listdir(series_path) if f.endswith('.dcm')]
        slices = [pydicom.dcmread(f) for f in files]
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

    def get_slice_count(self):
        if self.ct_volume is not None:
            return self.ct_volume.shape[0]
        if self.pet_volume is not None:
            return self.pet_volume.shape[0]
        return 0

    def get_images(self, slice_index):
        """
        Returns (ct_image, pet_image, fused_image) for a given slice index.
        Note: Real fusion requires resampling if geometries differ. 
        For this simple viewer, we assume they are roughly aligned or simply fail gracefully.
        """
        ct_img = None
        pet_img = None
        
        # Get CT Slice
        if self.ct_volume is not None:
            max_idx = self.ct_volume.shape[0] - 1
            idx = min(slice_index, max_idx)
            ct_img = self.ct_volume[idx]

        # Get PET Slice
        # If PET has different resolution/slice thickness, simple index matching WON'T work perfectly.
        # But for an MVP viewer, we might assume pre-processed data or try nearest neighbor by Z-position.
        if self.pet_volume is not None:
            if self.ct_volume is not None:
                # Try to find PET slice closest in Z to current CT slice
                ct_z = float(self.ct_metadata[min(slice_index, len(self.ct_metadata)-1)].ImagePositionPatient[2])
                
                # Find closest PET slice
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
                # Just use index
                max_idx = self.pet_volume.shape[0] - 1
                idx = min(slice_index, max_idx)
                pet_img = self.pet_volume[idx]

        return ct_img, pet_img
