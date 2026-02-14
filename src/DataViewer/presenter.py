import os
import sys
import threading

# Path setup handled in main.py usually, but for standalone tests or if main didn't set it:
# We rely on src being importable. 
# If main.py sets root, then src.utils works.

from src.utils.config import Config
try:
    from src.utils.gdrive_loader import GoogleDriveLoader
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
        self.selected_segmentation_label = 'all_classes'
        
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
        self.selected_segmentation_label = 'all_classes'
        self._configure_segmentation_controls()
        
        # Draw
        self.set_slice(self.current_slice)

    def set_slice(self, slice_idx):
        self.current_slice = slice_idx
        ct, pet, seg = self.model.get_images(slice_idx, self.orientation)
        
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
        
        self.view.update_images(
            ct,
            pet,
            seg,
            slice_idx,
            self.wl,
            self.ww,
            aspect,
            ct_extent,
            pet_extent,
            segmentation_label=self.selected_segmentation_label
        )

    def toggle_segmentation_zoi(self):
         """ Centers and Zooms the view to the Prostate Segmentation """
         if self.model.segmentation_mask is None:
             print("No segmentation available.")
             return

         # 1. Determine optimal slice (Center Z)
         center_slice = self.model.get_segmentation_center_slice(self.orientation, self.selected_segmentation_label)
         self.change_slice(center_slice - self.current_slice)
         
         # 2. Get Bounding Box in Physical Coordinates
         bounds = self.model.get_segmentation_bounds(self.orientation, self.selected_segmentation_label)
         if bounds:
              # bounds is [xmin, xmax, ymax, ymin] or similar depending on impl
              # View expects specific instructions.
              # Let's pass bounds to view method.
              self.view.zoom_to_bounds(bounds)

    def set_segmentation_class(self, label_value):
        """Update which segmentation label should be displayed/used for ZOI."""
        self.selected_segmentation_label = label_value or 'all_classes'
        self.set_slice(self.current_slice)

    def _configure_segmentation_controls(self):
        classes = self.model.get_available_segmentation_classes()
        label_map = self.model.get_segmentation_label_map()
        self.view.set_segmentation_classes(classes, label_map, self.selected_segmentation_label)


    def change_slice(self, delta):
        total = self.model.get_slice_count(self.orientation)
        new_idx = self.current_slice + delta
        new_idx = max(0, min(new_idx, total - 1))
        
        # Update View Slider
        if new_idx != self.current_slice:
            self.view.slice_scale.set(new_idx)
import os
