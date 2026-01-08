class Presenter:
    def __init__(self, model, view):
        self.model = model
        self.view = view
        self.current_slice = 0
        
        # Connect View -> Presenter
        self.view.set_presenter(self)

    def load_dataset(self, path):
        try:
            self.model.load_dataset(path)
            self.view.set_current_patient_info(f"Loaded {len(self.model.patient_list)} items. Press Next.")
            # Automatically load first if available
            self.next_patient()
        except Exception as e:
            print(f"Error loading dataset: {e}")
            self.view.set_current_patient_info(f"Error: {e}")

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
        ct, pet = self.model.get_images(slice_idx)
        self.view.update_images(ct, pet, slice_idx)
import os
