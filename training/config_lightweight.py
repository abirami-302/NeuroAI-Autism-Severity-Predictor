# =============================================================
# training/config_lightweight.py
#
# Laptop-friendly training configuration for the NeuroAI
# 3D MRI pipeline.  Keeps the FULL architecture intact
# (3D CNN → GNN → Classifier → Grad-CAM) but reduces
# spatial resolution and model capacity so that 50-100
# NIfTI samples can be trained on a 16 GB RAM / CPU laptop
# in under 60 minutes.
#
# HOW TO USE:
#   python training/train_lightweight.py
#   python training/train_lightweight.py --size 32   # 32×32×32
#   python training/train_lightweight.py --size 64   # 64×64×64 (default)
#
# ORIGINAL pipeline is UNCHANGED — this is a separate config.
# =============================================================

import os

# ── Inherit all paths from the main config ───────────────────
BASE_DIR           = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR        = os.path.join(BASE_DIR, "dataset")
RAW_DATA_DIR       = os.path.join(DATASET_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATASET_DIR, "processed")
SAMPLE_DATA_DIR    = os.path.join(DATASET_DIR, "sample")
SAVED_MODELS_DIR   = os.path.join(BASE_DIR, "saved_models")
OUTPUTS_DIR        = os.path.join(BASE_DIR, "outputs")
HEATMAP_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "heatmaps")
LOG_DIR            = os.path.join(OUTPUTS_DIR, "logs")

# ── Lightweight model save paths (separate from mock weights) ─
LW_MODEL_PATH      = os.path.join(SAVED_MODELS_DIR, "lightweight_model.pth")
LW_CHECKPOINT_PATH = os.path.join(SAVED_MODELS_DIR, "lightweight_checkpoint.pth")
LW_BEST_PATH       = os.path.join(SAVED_MODELS_DIR, "lightweight_best.pth")

# =============================================================
# VOLUME SIZE — key knob for laptop training
#
#   32×32×32 → very fast (~5 min/epoch), lower accuracy
#   64×64×64 → balanced (~15 min/epoch), good accuracy ← DEFAULT
#
# Brain structure is preserved at both resolutions for
# the 10-ROI grid used by the GNN.
# =============================================================
IMAGE_SIZE     = (64, 64, 64)   # Change to (32, 32, 32) if slow
IMAGE_CHANNELS = 1

# =============================================================
# CLASS SETTINGS  (identical to main config)
# =============================================================
NUM_CLASSES  = 3
CLASS_NAMES  = ["Mild", "Moderate", "Severe"]
CLASS_COLORS = {"Mild": "#2ECC71", "Moderate": "#F39C12", "Severe": "#E74C3C"}

# =============================================================
# LIGHTWEIGHT 3D CNN SETTINGS
# Fewer filters than the full model → 4× less memory
# =============================================================
CNN_BASE_FILTERS = 16       # Full model uses 32 → halved
CNN_FEATURE_DIM  = 64       # Full model uses 128 → halved
CNN_DROPOUT      = 0.4      # Slightly higher dropout for small datasets

# =============================================================
# GNN SETTINGS  (kept same — GNN is already lightweight)
# =============================================================
NUM_ROI                = 10
CONNECTIVITY_THRESHOLD = 0.3
GNN_INPUT_DIM          = 16
GNN_HIDDEN_DIM         = 32   # Full model uses 64 → halved
GNN_OUTPUT_DIM         = 16   # Full model uses 32 → halved
GNN_NUM_LAYERS         = 2    # Full model uses 3 → reduced
GNN_DROPOUT            = 0.3

ROI_NAMES = [
    "Prefrontal Cortex", "Amygdala", "Temporal Lobe",
    "Cerebellum", "Corpus Callosum", "Cingulate Cortex",
    "Fusiform Gyrus", "Insula", "Basal Ganglia", "Hippocampus",
]

# =============================================================
# TRAINING HYPERPARAMETERS — tuned for 50-100 samples on CPU
# =============================================================
BATCH_SIZE          = 2        # Small batch = less RAM per step
EPOCHS              = 40       # More epochs to compensate small data
LEARNING_RATE       = 5e-4     # Lower LR → more stable on tiny data
WEIGHT_DECAY        = 1e-3     # Stronger regularisation (small dataset)
LR_STEP_SIZE        = 8        # Decay LR every 8 epochs
LR_GAMMA            = 0.5
EARLY_STOP_PATIENCE = 10       # More patience on small data

# =============================================================
# DATASET SPLIT  (adjusted for small datasets)
# With 60 samples: 42 train / 9 val / 9 test
# With 90 samples: 63 train / 13 val / 14 test
# =============================================================
TRAIN_SPLIT  = 0.70
VAL_SPLIT    = 0.15
TEST_SPLIT   = 0.15
RANDOM_SEED  = 42

# =============================================================
# DATA AUGMENTATION — critical for small datasets
# Augmentation synthetically multiplies your training data
# =============================================================
AUGMENT_FLIP_PROB    = 0.5    # Random left-right flip
AUGMENT_ROTATE_PROB  = 0.3    # Random 90° rotation
AUGMENT_NOISE_PROB   = 0.3    # Add Gaussian noise
AUGMENT_NOISE_STD    = 0.02   # Noise standard deviation

# =============================================================
# PREPROCESSING  (identical to main config)
# =============================================================
NORMALIZE_METHOD = "zscore"
DENOISE_SIGMA    = 1.0
CLIP_PERCENTILE  = (1, 99)

# =============================================================
# GRAD-CAM  (identical to main config)
# =============================================================
GRADCAM_TARGET_LAYER = "features"
GRADCAM_COLORMAP     = "jet"
GRADCAM_ALPHA        = 0.5

# =============================================================
# DEVICE
# =============================================================
def get_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

DEVICE = get_device()

# =============================================================
# ESTIMATED TRAINING TIMES ON YOUR LAPTOP (16GB RAM, CPU)
# =============================================================
# IMAGE_SIZE=(32,32,32)  BATCH=2  60 samples → ~3-5  min/epoch
# IMAGE_SIZE=(64,64,64)  BATCH=2  60 samples → ~8-15 min/epoch
# Full 40 epochs at 64³ → approx 4-8 hours total
# Full 40 epochs at 32³ → approx 1-2 hours total
# =============================================================

def print_config():
    print("\n" + "="*60)
    print("  NeuroAI — Lightweight Training Config")
    print("="*60)
    print(f"  Device          : {DEVICE}")
    print(f"  Image Size      : {IMAGE_SIZE}  ← key setting")
    print(f"  CNN Filters     : {CNN_BASE_FILTERS}  (full=32)")
    print(f"  CNN Feature Dim : {CNN_FEATURE_DIM}   (full=128)")
    print(f"  GNN Hidden      : {GNN_HIDDEN_DIM}   (full=64)")
    print(f"  Batch Size      : {BATCH_SIZE}")
    print(f"  Epochs          : {EPOCHS}")
    print(f"  Learning Rate   : {LEARNING_RATE}")
    print(f"  Early Stop      : {EARLY_STOP_PATIENCE} epochs patience")
    print(f"  Augmentation    : flip={AUGMENT_FLIP_PROB}, "
          f"noise={AUGMENT_NOISE_PROB}")
    print("="*60)
    print(f"\n  Tip: Use --size 32 for faster training on slow CPUs")
    print(f"  Tip: Use --size 64 for better accuracy\n")

if __name__ == "__main__":
    print_config()
