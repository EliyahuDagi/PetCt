# DataViewer

A simple MVP DICOM Viewer for PET/CT datasets.

## Features
- Load a dataset folder (containing subfolders for each patient).
- Iterate through patients using Next/Previous buttons.
- View CT, PET, and Fuse (Aligned) images side-by-side.
- Scroll through slices using the slider.

## Requirements
- Python 3.x
- `pydicom`
- `matplotlib`
- `numpy`
- `Pillow`
- `scipy`
- `tcia-utils` (for downloading samples, optional for viewer itself)

## proper Data Structure
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
