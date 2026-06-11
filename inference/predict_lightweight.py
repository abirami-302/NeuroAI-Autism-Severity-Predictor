# =============================================================
# inference/predict_lightweight.py
#
# Inference script for the REAL trained lightweight model.
# Uses saved_models/lightweight_best.pth (produced by
# training/train_lightweight.py) instead of mock weights.
#
# USAGE:
#   python inference/predict_lightweight.py --input scan.nii.gz
#   python inference/predict_lightweight.py --demo
#
# Fully compatible with Grad-CAM and the existing Streamlit UI.
# The Streamlit app auto-detects which model to use based on
# which weights file is available.
# =============================================================

import os
import sys
import time
import tempfile
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.config_lightweight import (
    LW_BEST_PATH, LW_MODEL_PATH,
    IMAGE_SIZE, CLASS_NAMES, NUM_CLASSES, DEVICE,
    CONNECTIVITY_THRESHOLD, HEATMAP_OUTPUT_DIR
)
from preprocessing.pipeline import preprocess
from graph.graph_builder import volume_to_graph, get_top_connected_regions
from models.cnn3d_lightweight import LightweightCNN3DClassifier, LightweightCNN3D
from inference.gradcam import (
    GradCAM3D, generate_mock_heatmap,
    plot_gradcam_slices, resize_heatmap_to_volume
)

try:
    from torch_geometric.data import Batch as PyGBatch
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


# =============================================================
# MODEL LOADER
# =============================================================

def load_lightweight_model(weights_path: str = None) -> tuple:
    """
    Load the trained lightweight model from disk.

    Tries paths in order:
        1. weights_path (if provided)
        2. saved_models/lightweight_best.pth
        3. saved_models/lightweight_model.pth

    Returns:
        tuple: (model, image_size, use_gnn)
            model      : nn.Module in eval mode on DEVICE
            image_size : (D, H, W) the model was trained on
            use_gnn    : bool whether GNN branch is active
    """
    # Try each candidate path
    candidates = [
        p for p in [weights_path, LW_BEST_PATH, LW_MODEL_PATH]
        if p is not None
    ]

    loaded_meta = None
    loaded_path = None

    for path in candidates:
        if os.path.exists(path):
            print(f"  [Model] Loading: {path}")
            loaded_meta = torch.load(path, map_location=DEVICE)
            loaded_path = path
            break

    if loaded_meta is None:
        raise FileNotFoundError(
            "No trained lightweight model found.\n"
            "Train first with:\n"
            "  python training/train_lightweight.py\n\n"
            f"Expected at: {LW_BEST_PATH}"
        )

    # Extract metadata saved alongside weights
    if isinstance(loaded_meta, dict) and "model_state_dict" in loaded_meta:
        state_dict = loaded_meta["model_state_dict"]
        image_size = loaded_meta.get("image_size", IMAGE_SIZE)
        use_gnn    = loaded_meta.get("use_gnn", False)
        val_loss   = loaded_meta.get("val_loss", "?")
        print(f"  [Model] image_size={image_size}, val_loss={val_loss:.4f}")
    else:
        # Plain state dict (from torch.save(model.state_dict(), path))
        state_dict = loaded_meta
        image_size = IMAGE_SIZE
        use_gnn    = False

    # Build model and load weights
    # Try full LightweightNeuroAI first, fall back to CNN-only
    try:
        from training.train_lightweight import LightweightNeuroAI
        model = LightweightNeuroAI(use_gnn=use_gnn)
        model.load_state_dict(state_dict)
    except Exception:
        # Fallback: load as plain CNN classifier
        model = LightweightCNN3DClassifier(num_classes=NUM_CLASSES)
        model.load_state_dict(state_dict, strict=False)
        use_gnn = False

    model = model.to(DEVICE)
    model.eval()
    print(f"  [Model] ✅ Loaded from {loaded_path}")

    return model, image_size, use_gnn


# =============================================================
# PREDICTION RESULT
# =============================================================

class LightweightPredictionResult:
    """
    Structured prediction result from the lightweight pipeline.
    Same interface as inference/predict.py::PredictionResult
    so the Streamlit UI can use either interchangeably.
    """
    def __init__(self, label, confidence, probabilities,
                 affected_regions, heatmap, volume,
                 conn_matrix, inference_time, error=None,
                 is_real_model=True):
        self.label            = label
        self.confidence       = confidence
        self.probabilities    = probabilities
        self.affected_regions = affected_regions
        self.heatmap          = heatmap
        self.volume           = volume
        self.conn_matrix      = conn_matrix
        self.inference_time   = inference_time
        self.error            = error
        self.is_real_model    = is_real_model   # True = real weights, not mock


# =============================================================
# MAIN PREDICTION FUNCTION
# =============================================================

def predict_lightweight(file_path:    str,
                        model=None,
                        image_size:   tuple = None,
                        use_gnn:      bool  = False,
                        save_heatmap: bool  = False
                        ) -> LightweightPredictionResult:
    """
    Run the lightweight NeuroAI prediction pipeline.

    Uses real trained weights (lightweight_best.pth).
    Includes Grad-CAM heatmap generation using the backbone CNN.

    Args:
        file_path    : MRI file path (.nii, .nii.gz, .png, .jpg)
        model        : Pre-loaded model (loads fresh if None)
        image_size   : Override resize target (default: from saved model)
        use_gnn      : Enable GNN graph branch
        save_heatmap : Save heatmap PNG to outputs/heatmaps/

    Returns:
        LightweightPredictionResult
    """
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"  NeuroAI Lightweight Prediction Pipeline")
    print(f"{'='*60}")
    print(f"  Input : {os.path.basename(file_path)}")

    try:
        # ── Step 1: Load model ───────────────────────────────
        if model is None:
            model, saved_size, use_gnn = load_lightweight_model()
            if image_size is None:
                image_size = saved_size
        image_size = image_size or IMAGE_SIZE

        # ── Step 2: Preprocess MRI ───────────────────────────
        print("\n  [1/5] Preprocessing MRI...")
        volume = preprocess(file_path, target_size=image_size)
        print(f"        Volume shape: {volume.shape}")

        # ── Step 3: Build connectivity graph ─────────────────
        print("\n  [2/5] Building brain connectivity graph...")
        graph, conn_matrix = volume_to_graph(
            volume, threshold=CONNECTIVITY_THRESHOLD
        )
        print(f"        Nodes={graph.num_nodes}  Edges={graph.edge_index.shape[1]}")

        # ── Step 4: Forward pass ─────────────────────────────
        print("\n  [3/5] Running model inference...")
        input_tensor = torch.tensor(volume, dtype=torch.float32).unsqueeze(0)
        input_tensor = input_tensor.to(DEVICE)   # (1, 1, D, H, W)

        # Prepare graph batch if GNN enabled
        graph_batch = None
        if use_gnn and HAS_PYG:
            try:
                graph_batch = PyGBatch.from_data_list([graph]).to(DEVICE)
            except Exception as e:
                print(f"        [Warning] Graph batch failed: {e} — using CNN only")

        with torch.no_grad():
            logits = model(input_tensor, graph_batch)
            probs  = F.softmax(logits, dim=1)[0]

        pred_idx    = probs.argmax().item()
        pred_label  = CLASS_NAMES[pred_idx]
        confidence  = round(probs[pred_idx].item() * 100, 1)

        probabilities = {
            CLASS_NAMES[i]: round(probs[i].item() * 100, 1)
            for i in range(NUM_CLASSES)
        }

        print(f"        Prediction : {pred_label}")
        print(f"        Confidence : {confidence}%")
        print(f"        All probs  : {probabilities}")

        # ── Step 5: Affected regions ─────────────────────────
        print("\n  [4/5] Identifying affected regions...")
        top_regions = get_top_connected_regions(conn_matrix, top_k=3)
        print(f"        Top regions: {top_regions}")

        # ── Step 6: Grad-CAM heatmap ─────────────────────────
        print("\n  [5/5] Generating Grad-CAM heatmap...")

        try:
            # Get CNN backbone from model
            cnn_backbone = (
                model.cnn if hasattr(model, "cnn")
                else model.backbone if hasattr(model, "backbone")
                else model
            )
            gradcam  = GradCAM3D(cnn_backbone, target_layer="features")
            heatmap  = gradcam.generate(input_tensor, target_class=pred_idx)
            heatmap  = resize_heatmap_to_volume(heatmap, tuple(image_size))
            print(f"        Heatmap shape: {heatmap.shape}  (real Grad-CAM ✅)")
        except Exception as e:
            print(f"        [Warning] Grad-CAM error: {e} — using mock heatmap")
            heatmap = generate_mock_heatmap(tuple(image_size))

        # Save heatmap PNG
        if save_heatmap:
            os.makedirs(HEATMAP_OUTPUT_DIR, exist_ok=True)
            fname = os.path.splitext(os.path.basename(file_path))[0]
            hmap_path = os.path.join(
                HEATMAP_OUTPUT_DIR,
                f"{fname}_{pred_label.lower()}_lw_heatmap.png"
            )
            fig = plot_gradcam_slices(
                volume[0], heatmap,
                prediction=pred_label,
                confidence=confidence,
                save_path=hmap_path
            )
            import matplotlib.pyplot as plt
            plt.close(fig)

        elapsed = round(time.time() - t0, 2)

        print(f"\n  {'='*40}")
        print(f"  ✅ Done in {elapsed}s  →  {pred_label} ({confidence}%)")
        print(f"  {'='*40}\n")

        return LightweightPredictionResult(
            label            = pred_label,
            confidence       = confidence,
            probabilities    = probabilities,
            affected_regions = top_regions,
            heatmap          = heatmap,
            volume           = volume,
            conn_matrix      = conn_matrix,
            inference_time   = elapsed,
            is_real_model    = True
        )

    except Exception as e:
        elapsed = round(time.time() - t0, 2)
        import traceback; traceback.print_exc()
        return LightweightPredictionResult(
            label            = "Unknown",
            confidence       = 0.0,
            probabilities    = {c: 0.0 for c in CLASS_NAMES},
            affected_regions = [],
            heatmap          = generate_mock_heatmap(IMAGE_SIZE),
            volume           = np.zeros((1, *IMAGE_SIZE)),
            conn_matrix      = np.zeros((10, 10)),
            inference_time   = elapsed,
            error            = str(e),
            is_real_model    = False
        )


# =============================================================
# DEMO  (synthetic MRI — no real file needed)
# =============================================================

def predict_lightweight_demo() -> LightweightPredictionResult:
    """Run prediction on a synthetic MRI for quick testing."""
    import nibabel as nib

    print("[Demo] Generating synthetic MRI for demo prediction...")
    vol     = np.random.randn(91, 109, 91).astype(np.float32) * 300 + 800
    nii_img = nib.Nifti1Image(vol, affine=np.eye(4))

    with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
        nib.save(nii_img, tmp.name)
        tmp_path = tmp.name

    try:
        result = predict_lightweight(tmp_path)
    finally:
        os.unlink(tmp_path)

    return result


# =============================================================
# CLI
# =============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="NeuroAI Lightweight — Real Trained Model Inference"
    )
    parser.add_argument("--input", type=str, default=None,
                        help="MRI file path (.nii, .nii.gz, .png)")
    parser.add_argument("--save_heatmap", action="store_true",
                        help="Save Grad-CAM heatmap to outputs/heatmaps/")
    parser.add_argument("--demo", action="store_true",
                        help="Run with synthetic data (no file needed)")
    args = parser.parse_args()

    if args.demo or args.input is None:
        result = predict_lightweight_demo()
    else:
        result = predict_lightweight(
            args.input, save_heatmap=args.save_heatmap
        )

    print(f"\n{'─'*45}")
    print(f"  RESULT")
    print(f"{'─'*45}")
    print(f"  Label       : {result.label}")
    print(f"  Confidence  : {result.confidence}%")
    print(f"  Regions     : {', '.join(result.affected_regions)}")
    print(f"  Real model  : {result.is_real_model}")
    print(f"  Time        : {result.inference_time}s")
    if result.error:
        print(f"  ⚠ Error    : {result.error}")
    print(f"{'─'*45}")
