# DataViewer Architecture Summary

**Compact Context for LLM**
**Project:** PetCt Data Viewer (Python/Tkinter/Matplotlib/Pydicom)
**Architecture:** Model-View-Presenter (MVP)

---

## 1. File Structure & Responsibilities

### `main.py`
- **Role:** Entry point.
- **Flow:** Instantiates `DicomModel`, `MainView`, injects them into `Presenter`, starts `view.mainloop()`.

### `model.py` (`DicomModel`)
- **Role:** Data persistence, DICOM I/O, Logic.
- **Key Attributes:** 
  - `ct_volume`, `pet_volume`: 3D numpy arrays (Z, Y, X).
  - `patient_list`: List of available patient ID paths.
  - `suv_factor`: Computed scalar for raw->SUV conversion.
- **Key Methods:**
  - `load_series(path)`: Loads `.dcm` files, sorts by InstanceNumber/ImagePositionPatient.
  - `load_patient_data(path)`: Orchestrates loading CT and PET series for a patient.
  - `get_images(slice_idx, orientation)`: Returns sliced 2D arrays for CT and PET based on MPR orientation (AXIAL, CORONAL, SAGITTAL).
  - `get_suv_factor()`: Parses DICOM tags (0010,1030) PatientWeight and (0054,1102) RadionuclideTotalDose.
  - `get_ct_bounds()`, `get_pet_bounds()`: Returns physical extent `[left, right, bottom, top]` (mm) for alignment.

### `view.py` (`MainView`)
- **Role:** GUI (Tkinter), Rendering (Matplotlib), User Input.
- **Components:** 
  - 3 Axes: CT, PET, Fusion (Overlay).
  - Controls: Slider (Slice), Buttons (Patient Nav), Toolbar.
- **Key Methods:**
  - `update_images(ct_img, pet_img, ...)`: Renders arrays to `imshow`. Applies physical extents and `clim` (W/L).
  - `_on_scroll(event)`: Handles **Zoom** (Ctrl+Scroll) and Slice Change (Scroll). Syncs all axes.
  - `_on_mouse_move(event)`: Handles **Pan** (Middle-Click), **Window/Level** (Right-Click), and **Pixel Probe** (Hover).
  - `_get_pixel_value_at_location(artist, x, y)`: Maps physical coords (mm) back to array indices for probing values.
  - `_sync_zoom_pan(xlim, ylim)`: Ensures CT, PET, and Fusion zoom/pan together.

### `presenter.py` (`Presenter`)
- **Role:** Mediator. Handles business logic state (Slice Index, Window/Level, Orientation).
- **Key Methods:**
  - `change_slice(delta)`: Updates index, calls `_update_view()`.
  - `change_window_level(dx, dy)`: Adjusts W/L based on drag, calls `_update_view()`.
  - `set_orientation(mode)`: Switches MPR mode, resets slice max range.
  - `_update_view()`: Fetches images from Model, sends to View with metadata (extents, overlays).

---

## 2. Key Data Flows

### Loading Data
1. `View` triggers `on_load_click`.
2. `Presenter` calls `Model.load_dataset`.
3. `Model` scans directory, populates `patient_list`.
4. `Presenter` triggers `load_patient` -> `Model` parses DICOMs -> `Model` calcs `suv_factor`.

### Rendering a Slice
1. `Presenter` calls `model.get_images(slice_idx, orientation)`.
2. `Model` calculates correct 2D slice from 3D volumes.
3. `Model` calculates Physical Extents (`get_ct_bounds`, `get_pet_bounds`) to align different resolutions/FOVs (e.g. PET covers legs, CT only torso).
4. `Presenter` calls `view.update_images` with data and extents.
5. `View` aligns images using `ax.set_extent`.

### Interaction (Zoom/Probe)
- **Zoom/Pan:** Event in `view` -> `_sync_zoom_pan` updates `xlim`/`ylim` on all axes -> `view.canvas.draw()`.
- **Probe:** Mouse Move -> `_get_pixel_value_at_location` -> Logic: `(Mouse_mm - Origin) / Spacing = Index` -> Fetch Value -> `status_bar` update (CT in HU, PET in SUV).

## 3. Physical Alignment Logic
- CT and PET often have different resolutions and coverages.
- We use Matplotlib `extent=[x_start, x_end, y_end, y_start]` derived from DICOM `ImagePositionPatient` & `PixelSpacing`.
- This ensures visual alignment in Fusion even if arrays differ in size.

## 4. Current Context Notes
- **Features Active:** MPR (Axial/Sag/Cor), Fusion (Alpha Blend), SUV Display, HU Display, Sync Zoom/Pan.
- **Dependencies:** `pydicom`, `numpy`, `matplotlib`, `tkinter`.
The viewer expects a folder structure like this:
```
Dataset_Root/
  Patient_001/
    CT/
       ...dcm
    PT/
       ...dcm
  Patient_002/
    ...
```
Or any recursive structure where CT and PT DICOMs can be found within the patient folder.

## Usage
Run the main script:
```bash
python src/DataViewer/main.py
```
1. Click **Load Dataset Folder**.
2. Select the `data/sample_dicom` folder (or your dataset root).
3. Use the **Next Patient** button to load the first case.
4. Use the **Slice** slider to navigate the 3D volume.
