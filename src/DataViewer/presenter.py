import os
import threading

import numpy as np

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
        # Viewer is fixed to TotalSegmentor segmentation.
        self.segmentation_source_name = "TotalSegmentor"
        self.show_zoi = False
        self.prostate_roi_cache = None

        # Connect View -> Presenter
        self.view.set_presenter(self)
        self.view.set_segmentation_sources([self.segmentation_source_name], self.segmentation_source_name)

    def set_orientation(self, mode):
        self.orientation = mode
        # Clear ROI cache as orientation changes projection
        self.prostate_roi_cache = None
        # Reset slice to middle of new dimension
        total = self.model.get_slice_count(self.orientation)
        self.view.set_max_slice(total)
        self.current_slice = total // 2

        # If ZOI is enabled, recompute ROI for the new orientation
        if getattr(self, 'show_zoi', False):
            self._calculate_roi()
        self.set_slice(self.current_slice)

    def change_window_level(self, dx, dy):
        # Sensitivity factors
        self.ww += dx * 2
        self.wl -= dy * 2 # Moving mouse down usually decreases level (darker) or increases? 
                          # Standard: Down -> Lower Level (Darker usually if W is constant)? 
                          # Actually usually Drag Right -> Increase Width (Lower Contrast). Drag Down -> Decrease Level (Darker).
        
        # Ensure Width > 1
        if self.ww < 1:
            self.ww = 1
        
        # Redraw
        self.set_slice(self.current_slice)

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
                err = str(e)
                self.view.after(0, lambda err=err: self.view.set_current_patient_info(f"GDrive Error: {err}"))

        self.view.set_current_patient_info(f"Downloading {folder_id}... please wait.")
        thread = threading.Thread(target=_download_and_load, daemon=True)
        thread.start()

    def next_patient(self):
        if self.model.has_next():
            success = self.model.load_next_patient(self.segmentation_source_name)
            if success:
                self._update_view_after_patient_load()
        else:
            print("No next patient")

    def prev_patient(self):
        if self.model.has_prev():
            success = self.model.load_prev_patient(self.segmentation_source_name)
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
        self.show_zoi = False
        self.prostate_roi_cache = None
        self._configure_segmentation_controls()

        # Keep segmentation source UI in sync
        self.view.set_segmentation_source_selection(self.segmentation_source_name)
        
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
        
        # Calculate ZOI box for current slice
        zoi_box = None
        if self.show_zoi and self.prostate_roi_cache:
            roi = self.prostate_roi_cache
            bbox_vox = roi.get("bbox_vox", {})
            bbox_mm = roi.get("bbox_mm")
            slice_range = None
            
            in_slice = False

            # Prefer physical bbox to handle differing CT/PET grids
            if bbox_mm:
                spacing, origin = self._get_view_spacing_origin()
                if spacing and origin:
                    dz, dy, dx = spacing
                    oz, oy, ox = origin
                    if self.orientation == 'AXIAL':
                        z_start = oz + slice_idx * dz
                        z_end = z_start + dz
                        z0, z1 = bbox_mm.get("z", (0.0, 0.0))
                        lo, hi = (min(z0, z1), max(z0, z1))
                        in_slice = (z_start <= hi) and (z_end >= lo)
                    elif self.orientation == 'CORONAL':
                        y_start = oy + slice_idx * dy
                        y_end = y_start + dy
                        y0, y1 = bbox_mm.get("y", (0.0, 0.0))
                        lo, hi = (min(y0, y1), max(y0, y1))
                        in_slice = (y_start <= hi) and (y_end >= lo)
                    elif self.orientation == 'SAGITTAL':
                        x_start = ox + slice_idx * dx
                        x_end = x_start + dx
                        x0, x1 = bbox_mm.get("x", (0.0, 0.0))
                        lo, hi = (min(x0, x1), max(x0, x1))
                        in_slice = (x_start <= hi) and (x_end >= lo)

            # Fallback to voxel range if available
            if not in_slice:
                if self.orientation == 'AXIAL':
                    slice_range = bbox_vox.get("z", [0, 0])
                elif self.orientation == 'CORONAL':
                    slice_range = bbox_vox.get("y", [0, 0])
                elif self.orientation == 'SAGITTAL':
                    slice_range = bbox_vox.get("x", [0, 0])
                if slice_range:
                    if min(slice_range) <= slice_idx <= max(slice_range):
                        in_slice = True

              # 2. If in slice, project 3D bbox into displayed 2D coordinates
            if in_slice:
                  # Match prostate_locator_debug display mapping for coronal/sagittal slices,
                  # where slice planes are flipud()-ed before display.
                  if bbox_mm:
                      zoi_box = self._bbox_mm_to_display_box(bbox_mm, ct_extent)

                  # Fallback for legacy ROIs that only provide 2D bounds.
                  if zoi_box is None:
                      bounds = roi.get("bounds")  # [left, right, bottom, top] in display coords
                      if bounds:
                          left, right, bottom, top = bounds
                          x = min(left, right)
                          y = min(top, bottom)
                          w = abs(right - left)
                          h = abs(bottom - top)
                          zoi_box = [x, y, w, h]

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
            segmentation_label=self.selected_segmentation_label,
            zoi_box=zoi_box
        )

    def toggle_segmentation_zoi(self):
         """Toggle visibility of the ZOI bbox on the current slice."""
         self.show_zoi = not getattr(self, 'show_zoi', False)
         
         if self.show_zoi:
             # Recalculate to keep ZOI consistent with current data/orientation.
             self._calculate_roi()
             roi = self.prostate_roi_cache or {}
             center_slice = roi.get("center_slice")
             if center_slice is not None:
                 try:
                     center_slice = int(center_slice)
                 except Exception:
                     center_slice = None
             if center_slice is not None:
                 self.current_slice = center_slice
                 try:
                     self.view.slice_scale.set(center_slice)
                 except Exception:
                     pass
         
         # Re-render current slice
         self.set_slice(self.current_slice)

    def _calculate_roi(self):
         if self.model.segmentation_mask is None and self.model.pet_volume is None:
             print("No segmentation or PET available.")
             return

         # Use the prostate locator algorithm used by the debug script.
         roi = self.model.locate_prostate_roi(self.orientation, force_locator=True, debug_override=True)
         self.prostate_roi_cache = roi
         if roi:
             print(f"ZOI method: {roi.get('method')}")
             if getattr(Config, 'PROSTATE_LOCATOR_DEBUG', False) and roi.get("debug") is not None:
                 try:
                     import json
                     print("Prostate locator debug:\n" + json.dumps(roi.get("debug"), indent=2))
                 except Exception:
                     print(f"Prostate locator debug: {roi.get('debug')}")

    def set_segmentation_class(self, label_value):
        """Update which segmentation label should be displayed/used for ZOI."""
        self.selected_segmentation_label = label_value or 'all_classes'
        self.prostate_roi_cache = None # Invalidate
        if self.show_zoi:
             self._calculate_roi()
        self.set_slice(self.current_slice)

    def _get_view_spacing_origin(self):
        """Return spacing/origin for the displayed geometry (CT preferred)."""
        if self.model.ct_volume is not None:
            return self.model.get_voxel_spacing(), self.model.get_origin()
        try:
            return self.model._get_pet_spacing_origin()  # type: ignore
        except Exception:
            return (None, None)

    def _flip_z_mm_for_display(self, z_mm, extent):
        if extent is None:
            return float(z_mm)
        z_bottom = float(extent[2])
        z_top = float(extent[3])
        return float(z_top + z_bottom - float(z_mm))

    def _bbox_mm_to_display_box(self, bbox_mm, ct_extent):
        """Project a 3D mm bbox into displayed 2D coordinates for the current orientation."""
        if not isinstance(bbox_mm, dict):
            return None

        try:
            if self.orientation == 'AXIAL':
                x0, x1 = bbox_mm.get("x", (None, None))
                y0, y1 = bbox_mm.get("y", (None, None))
                if None in (x0, x1, y0, y1):
                    return None
                x = min(float(x0), float(x1))
                y = min(float(y0), float(y1))
                w = abs(float(x1) - float(x0))
                h = abs(float(y1) - float(y0))
                return [x, y, w, h]

            if self.orientation == 'CORONAL':
                x0, x1 = bbox_mm.get("x", (None, None))
                z0, z1 = bbox_mm.get("z", (None, None))
                if None in (x0, x1, z0, z1):
                    return None
                z0_d = self._flip_z_mm_for_display(float(z0), ct_extent)
                z1_d = self._flip_z_mm_for_display(float(z1), ct_extent)
                x = min(float(x0), float(x1))
                y = min(z0_d, z1_d)
                w = abs(float(x1) - float(x0))
                h = abs(z1_d - z0_d)
                return [x, y, w, h]

            if self.orientation == 'SAGITTAL':
                y0, y1 = bbox_mm.get("y", (None, None))
                z0, z1 = bbox_mm.get("z", (None, None))
                if None in (y0, y1, z0, z1):
                    return None
                z0_d = self._flip_z_mm_for_display(float(z0), ct_extent)
                z1_d = self._flip_z_mm_for_display(float(z1), ct_extent)
                x = min(float(y0), float(y1))
                y = min(z0_d, z1_d)
                w = abs(float(y1) - float(y0))
                h = abs(z1_d - z0_d)
                return [x, y, w, h]
        except Exception:
            return None

        return None

    def set_segmentation_source(self, source_name):
        """Switch segmentation source (e.g., MONAI vs TotalSegmentor) and reload masks."""
        if not source_name:
            return
        self.segmentation_source_name = source_name
        reloaded = self.model.reload_segmentation_for_current(source_name)
        if reloaded:
            self.selected_segmentation_label = 'all_classes'
            self._configure_segmentation_controls()
            self.set_slice(self.current_slice)
        self.view.set_segmentation_source_selection(source_name)

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
