
import os

class Config:
    # Data Paths (Placeholder paths)
    DATA_ROOT = "data"
    
    # Google Drive Configuration
    USE_GDRIVE = False
    GDRIVE_FOLDER_ID = "" # The folder ID on Google Drive to download/load from
    GDRIVE_CREDENTIALS_FILE = "credentials.json"
    GDRIVE_TOKEN_FILE = "token.json"
    
    # Organ Configuration
    # Mapping organ names to IDs and vice versa
    ORGAN_MAP = {
        "PROSTATE": 0,
        "LUNG": 1,
        "LIVER": 2,
        "LYMPH_NODE": 3,
        "KIDNEY": 4
    }
    ID_TO_ORGAN = {v: k for k, v in ORGAN_MAP.items()}
    NUM_ORGANS = len(ORGAN_MAP)

    # Model Hyperparameters
    IMG_SIZE = (96, 96, 96) # Input patch size
    IN_CHANNELS = 1 # PET/CT usually 2 channels if stacked, or 1 if fused. Assuming 1 for now or 2 (CT+PET). Let's use 2.
    OUT_CHANNELS = NUM_ORGANS + 1 # +1 for background
    FEATURE_SIZE = 48
    EMBEDDING_DIM = 64 # Dimension of the organ token
    
    # Training Hyperparameters
    BATCH_SIZE = 2
    LEARNING_RATE = 1e-4
    MAX_EPOCHS = 100
    VAL_INTERVAL = 2
    SW_BATCH_SIZE = 4 # Sliding window batch size for inference
    
    # Intensity Normalization
    # CT Windowing (Soft tissue)
    CT_WINDOW_LEVEL = 40
    CT_WINDOW_WIDTH = 400
    # PET SUV Scaling (approximate max SUV to clip)
    PET_SUV_MAX = 15.0

    # System
    DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    NUM_WORKERS = 4
    SEED = 42

    @staticmethod
    def get_organ_id(organ_name):
        return Config.ORGAN_MAP.get(organ_name.upper())
