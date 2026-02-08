import os
import sys
import threading

# Add parent directory to path to allow importing modules from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.config import Config
try:
    from utils.gdrive_loader import GoogleDriveLoader
except ImportError as e:
    print(f"Could not import GoogleDriveLoader: {e}")
    GoogleDriveLoader = None

class Presenter:
    def __init__(self, model, view):
        self.model = model
        self.view = view
        self.current_slice = 0
        self.orientation = 'AXIAL'
        
        # Default Window/Level (Abdominal/Soft Tissue)
        self.wl = 40
        self.ww = 400
        
        # Connect View -> Presenter
        self.view.set_presenter(self)

    def set_orientation(self, mode):
        self.orientation = mode
        # Reset slice to middle of new dimension
        total = self.model.get_slice_count(self.orientation)
        self.view.set_max_slice(total)
        self.current_slice = total // 2
        self.set_slice(self.current_slice)

    def change_window_level(self, dx, dy):
        # Sensitivity factors
        self.ww += dx * 2
        self.wl -= dy * 2 # Moving mouse down usually decreases level (darker) or increases? 
                          # Standard: Down -> Lower Level (Darker usually if W is constant)? 
                          # Actually usually Drag Right -> Increase Width (Lower Contrast). Drag Down -> Decrease Level (Darker).
        
        # Ensure Width > 1
        if self.ww < 1: self.ww = 1
        
        # Redraw
        self.set_slice(self.current_slice)

    def change_slice(self, delta):
        total = self.model.get_slice_count()
        new_idx = self.current_slice + delta
        new_idx = max(0, min(new_idx, total - 1))
        
        # Update View Slider
        if new_idx != self.current_slice:
            self.view.slice_scale.set(new_idx)

    def load_dataset(self, path):
        try:
            self.model.load_dataset(path)
            self.view.set_current_patient_info(f"Loaded {len(self.model.patient_list)} items. Press Next.")
            # Automatically load first if available
            self.next_patient()
        except Exception as e:
            print(f"Error loading dataset: {e}")
            self.view.set_current_patient_info(f"Error: {e}")

    def load_from_gdrive(self, folder_id):
        if GoogleDriveLoader is None:
            self.view.set_current_patient_info("Google Drive Loader not available.")
            return

        def _download_and_load():
            try:
                # Determine download path
                download_path = os.path.join(Config.DATA_ROOT, "gdrive_downloads", folder_id)
                os.makedirs(download_path, exist_ok=True)
                
                # Update UI (this might need to be thread-safe depending on TKinter, 
                # but setting text is usually okay or handled via after())
                # For safety, we should ideally use view.after, but for simplicity here:
                print(f"Downloading from GDrive: {folder_id}...")
                
                loader = GoogleDriveLoader(
                     credentials_file=os.path.join(os.path.dirname(__file__), '../../credentials.json'),
                     token_file=os.path.join(os.path.dirname(__file__), '../../token.json')
                )
                loader.download_folder_recursive(folder_id, download_path)
                
                # Load the downloaded dataset
                # We need to call load_dataset on the main thread ideally
                self.view.after(0, lambda: self.load_dataset(download_path))
                
            except Exception as e:
                print(f"Error loading from GDrive: {e}")
                self.view.after(0, lambda: self.view.set_current_patient_info(f"GDrive Error: {e}"))

        self.view.set_current_patient_info(f"Downloading {folder_id}... please wait.")
        thread = threading.Thread(target=_download_and_load, daemon=True)
        thread.start()

    def next_patient(self):
        if self.model.has_next():
            success = self.model.load_next_patient()
            if success:
                self._update_view_after_patient_load()
        else:
            print("No next patient")

    def prev_patient(self):
        if self.model.has_prev():
            success = self.model.load_prev_patient()
            if success:
                self._update_view_after_patient_load()
        else:
            print("No prev patient")

    def _update_view_after_patient_load(self):
        # Update Patient Label
        idx = self.model.current_patient_index
        total = len(self.model.patient_list)
        name = os.path.basename(self.model.patient_list[idx])
        self.view.set_current_patient_info(f"Patient ({idx+1}/{total}): {name}")
        
        # Update Slice Slider
        total_slices = self.model.get_slice_count()
        self.view.set_max_slice(total_slices)
        
        # Reset to middle slice or 0
        self.current_slice = total_slices // 2
        
        # Draw
        self.set_slice(self.current_slice)

    def set_slice(self, slice_idx):
        self.current_slice = slice_idx
        ct, pet = self.model.get_images(slice_idx, self.orientation)
        
        # Calculate Aspect Ratio
        dz, dy, dx = self.model.get_voxel_spacing()
        aspect = 1.0
        if self.orientation == 'AXIAL':
            aspect = dy / dx 
        elif self.orientation == 'CORONAL':
            aspect = dz / dx
        elif self.orientation == 'SAGITTAL':
            aspect = dz / dy
        
        # Gather Metadata for Overlays
        meta = {
            'wl': self.wl,
            'ww': self.ww,
            'slice': slice_idx + 1 # 1-based index for display
        }
        
        try:
            # Metadata fetch is volume-dependent (only accurate for Axial really)
            # For MPR, we can just show generic info or current slice index
            if self.orientation == 'AXIAL' and self.model.ct_metadata and slice_idx < len(self.model.ct_metadata):
                ds = self.model.ct_metadata[slice_idx]
                meta['name'] = str(getattr(ds, 'PatientName', 'Unknown'))
                meta['id'] = str(getattr(ds, 'PatientID', 'N/A'))
                meta['thickness'] = getattr(ds, 'SliceThickness', 0)
                if hasattr(ds, 'ImagePositionPatient'):
                    meta['pos'] = ds.ImagePositionPatient[2] 
            else:
                 # Minimal info for MPR
                 meta['name'] = "MPR View"
        except Exception as e:
            print(f"Meta error: {e}")

        self.view.update_overlays(meta)
        
        # Determine Extents for proper alignment
        ct_extent = self.model.get_bounds(self.orientation)
        pet_extent = self.model.get_pet_bounds(self.orientation)
        
        # If PET extent is None, fallback or handle?
        # If CT extent is used, and PET extent differs, View handles it.
        
        self.view.update_images(ct, pet, slice_idx, self.wl, self.ww, aspect, ct_extent, pet_extent)

    def change_slice(self, delta):
        total = self.model.get_slice_count(self.orientation)
        new_idx = self.current_slice + delta
        new_idx = max(0, min(new_idx, total - 1))
        
        # Update View Slider
        if new_idx != self.current_slice:
            self.view.slice_scale.set(new_idx)
import os
