# Organ-Conditioned 3D PET/CT Segmentation

This project implements a 3D Swin-UNETR model that is conditioned on a learned "organ token". This allows the model to be trained on multiple datasets, where each dataset might only have labels for a specific organ (e.g., Prostate, Liver, Lung), while learning a shared representation.

## Key Features

1.  **Organ Token Injection**: A learnable embedding vector for each organ is injected into the encoder and decoder of the Swin-UNETR.
2.  **Organ-Balanced Sampling**: A `WeightedRandomSampler` ensures that each training batch contains a balanced mix of different organs.
3.  **Masked Dice Loss**: The loss function only penalizes the prediction for the organ present in the current sample, ignoring unlabelled organs.
4.  **Multi-Modal Input**: Handles stacked PET and CT volumes.

## Structure

-   `src/config.py`: Configuration (Organ IDs, Hyperparameters).
-   `src/dataset.py`: Custom Dataset and Transforms (DICOM loading, Intensity Norm).
-   `src/model.py`: `OrganAwareSwinUNETR` implementation.
-   `src/loss.py`: `OrganMaskedDiceLoss` implementation.
-   `src/train.py`: Main training loop.

## Usage

1.  **Install Dependencies**:
    ```bash
    pip install torch monai nibabel numpy
    ```

2.  **Run Training**:
    ```bash
    python src/train.py
    ```
    *Note: The script will generate dummy NIfTI data in `data/dummy_train` if no data is found.*

## Configuration

Modify `src/config.py` to change:
-   `ORGAN_MAP`: The list of organs and their IDs.
-   `IMG_SIZE`: Input patch size.
-   `DATA_ROOT`: Path to your real data.

## Data Format

The `PetCtDataset` expects a list of dictionaries:
```python
{
    "image": "path/to/ct.nii.gz",
    "pet": "path/to/pet.nii.gz",
    "label": "path/to/label.nii.gz",
    "organ": "LIVER" # Must match keys in ORGAN_MAP
}
```