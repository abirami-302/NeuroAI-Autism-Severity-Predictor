# =============================================================
# training/config.py  — Central Configuration
# No heavy dependencies at import time.
# =============================================================
import os

BASE_DIR           = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR        = os.path.join(BASE_DIR, "dataset")
RAW_DATA_DIR       = os.path.join(DATASET_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATASET_DIR, "processed")
SAMPLE_DATA_DIR    = os.path.join(DATASET_DIR, "sample")
SAVED_MODELS_DIR   = os.path.join(BASE_DIR, "saved_models")
FULL_MODEL_PATH    = os.path.join(SAVED_MODELS_DIR, "full_model_mock.npz")
CHECKPOINT_PATH    = os.path.join(SAVED_MODELS_DIR, "checkpoint.npz")
OUTPUTS_DIR        = os.path.join(BASE_DIR, "outputs")
HEATMAP_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "heatmaps")
LOG_DIR            = os.path.join(OUTPUTS_DIR, "logs")

IMAGE_SIZE     = (64, 64, 64)
IMAGE_CHANNELS = 1

NUM_CLASSES  = 3
CLASS_NAMES  = ["Mild", "Moderate", "Severe"]
CLASS_COLORS = {"Mild": "#2ECC71", "Moderate": "#F39C12", "Severe": "#E74C3C"}

CNN_BASE_FILTERS = 32
CNN_FEATURE_DIM  = 128
CNN_DROPOUT      = 0.3

NUM_ROI                = 10
CONNECTIVITY_THRESHOLD = 0.3
GNN_INPUT_DIM          = 16
GNN_HIDDEN_DIM         = 64
GNN_OUTPUT_DIM         = 32
GNN_NUM_LAYERS         = 3
GNN_DROPOUT            = 0.3

BATCH_SIZE          = 4
EPOCHS              = 30
LEARNING_RATE       = 1e-3
WEIGHT_DECAY        = 1e-4
LR_STEP_SIZE        = 10
LR_GAMMA            = 0.5
EARLY_STOP_PATIENCE = 7
TRAIN_SPLIT         = 0.70
VAL_SPLIT           = 0.15
TEST_SPLIT          = 0.15
RANDOM_SEED         = 42

NORMALIZE_METHOD = "zscore"
DENOISE_SIGMA    = 1.0
CLIP_PERCENTILE  = (1, 99)

GRADCAM_TARGET_LAYER = "features"
GRADCAM_COLORMAP     = "jet"
GRADCAM_ALPHA        = 0.5

def get_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

DEVICE = get_device()

LOG_LEVEL   = "INFO"
LOG_TO_FILE = True

ROI_NAMES = [
    "Prefrontal Cortex", "Amygdala", "Temporal Lobe",
    "Cerebellum", "Corpus Callosum", "Cingulate Cortex",
    "Fusiform Gyrus", "Insula", "Basal Ganglia", "Hippocampus",
]

KAGGLE_DATASET     = "birdy654/autistic-children-data-set-acds"
KAGGLE_ALT_DATASET = "fabdelja/autism-screening-for-toddlers"

def print_config():
    print("\n" + "="*55)
    print("  NeuroAI Configuration")
    print("="*55)
    print(f"  Device     : {DEVICE}")
    print(f"  Image Size : {IMAGE_SIZE}")
    print(f"  Classes    : {CLASS_NAMES}")
    print(f"  ROI Regions: {NUM_ROI}")
    print(f"  Base Dir   : {BASE_DIR}")
    print("="*55 + "\n")

if __name__ == "__main__":
    print_config()


CNN_WEIGHTS_PATH = "weights/cnn_best.pth"
GNN_WEIGHTS_PATH = "weights/gnn_best.pth"
MODEL_WEIGHTS_PATH = "weights/best_model.pth"