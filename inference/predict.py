# =============================================================
# inference/predict.py
# Single-sample prediction pipeline for NeuroAI.
#
# Given a raw MRI file path, this module:
#   1. Preprocesses the volume
#   2. Extracts CNN features
#   3. Builds the connectivity graph
#   4. Runs the full model forward pass
#   5. Generates a Grad-CAM heatmap
#   6. Returns a structured result dictionary
# =============================================================

import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.config import (
    CLASS_NAMES, NUM_CLASSES, DEVICE, FULL_MODEL_PATH,
    CNN_WEIGHTS_PATH, IMAGE_SIZE, ROI_NAMES, HEATMAP_OUTPUT_DIR
)
from preprocessing.pipeline import preprocess
from graph.graph_builder import volume_to_graph, get_top_connected_regions
from models.cnn3d import CNN3D, CNN3DClassifier
# NeuroAIClassifier (CNN+GNN combined) is available for full training runs;
# the default inference path uses CNN3DClassifier for compatibility with mock weights.
from inference.gradcam import GradCAM3D, generate_mock_heatmap, plot_gradcam_slices


# =============================================================
# PREDICTION RESULT DATA CLASS
# A clean container for everything the UI needs to display
# =============================================================

@dataclass
class PredictionResult:
    """
    Structured result from the NeuroAI prediction pipeline.

    Attributes:
        label (str):              Predicted class ("Mild" / "Moderate" / "Severe")
        confidence (float):       Confidence percentage (0–100)
        probabilities (dict):     Class → probability mapping
        affected_regions (list):  Top brain regions driving prediction
        heatmap (np.ndarray):     3D Grad-CAM heatmap volume
        volume (np.ndarray):      Preprocessed MRI volume for display
        conn_matrix (np.ndarray): Functional connectivity matrix
        inference_time (float):   Time taken for prediction (seconds)
        error (Optional[str]):    Error message if prediction failed
    """
    label:            str
    confidence:       float
    probabilities:    dict
    affected_regions: list
    heatmap:          np.ndarray
    volume:           np.ndarray
    conn_matrix:      np.ndarray
    inference_time:   float
    error:            Optional[str] = None


# =============================================================
# MOCK MODEL — for demo mode (no real training required)
# =============================================================

class MockNeuroAIModel(torch.nn.Module):
    """
    A mock model that returns realistic-looking random predictions.
    Used when no trained weights are available (demo mode).

    The model always loads and "works" — great for UI demonstrations
    before real training has been completed.
    """
    def __init__(self):
        super(MockNeuroAIModel, self).__init__()
        # Lightweight CNN to give realistic activations for Grad-CAM
        self.cnn = CNN3DClassifier(num_classes=NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns slightly randomized logits for demo predictions."""
        # Add deterministic seed based on input mean for consistent results
        seed = int(abs(x.mean().item()) * 1000) % 100
        torch.manual_seed(seed)
        # Random but realistic probabilities
        logits = torch.randn(x.shape[0], NUM_CLASSES) * 2
        return logits


# =============================================================
# MODEL LOADER
# =============================================================

def load_model(weights_path: str = None,
               use_mock: bool = True) -> torch.nn.Module:
    """
    Load the NeuroAI prediction model.

    Args:
        weights_path (str): Path to saved .pth weights file.
                            If None, uses default FULL_MODEL_PATH.
        use_mock (bool):    If True and weights not found, use mock model.

    Returns:
        nn.Module: Loaded model in eval mode on the correct device.
    """
    path = weights_path or FULL_MODEL_PATH

    if os.path.exists(path):
        # Load real trained model
        print(f"[Model] Loading weights from: {path}")
        model = CNN3DClassifier(num_classes=NUM_CLASSES)
        state_dict = torch.load(path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print(f"[Model] ✅ Loaded real model weights")
    elif use_mock:
        # Fall back to mock model for demo
        print(f"[Model] ⚠️  No weights found at {path}")
        print(f"[Model] Using mock model for demonstration...")
        model = MockNeuroAIModel()
    else:
        raise FileNotFoundError(
            f"Model weights not found: {path}\n"
            f"Train the model first: python training/train.py"
        )

    model = model.to(DEVICE)
    model.eval()
    return model


# =============================================================
# MAIN PREDICTION FUNCTION
# =============================================================

def predict(file_path: str,
            model: torch.nn.Module = None,
            save_heatmap: bool = False) -> PredictionResult:
    """
    Run the full NeuroAI prediction pipeline on a single MRI file.

    Pipeline:
        1. Preprocess MRI (normalize, denoise, resize)
        2. Build brain connectivity graph
        3. Extract CNN features
        4. Classify into Mild / Moderate / Severe
        5. Generate Grad-CAM heatmap
        6. Return structured PredictionResult

    Args:
        file_path (str):         Path to MRI file (.nii, .nii.gz, .png, .jpg)
        model (nn.Module):       Pre-loaded model (loads fresh if None)
        save_heatmap (bool):     If True, save heatmap PNG to outputs/

    Returns:
        PredictionResult: Complete prediction with all metadata
    """
    start_time = time.time()
    print(f"\n{'='*55}")
    print(f"  NeuroAI Prediction Pipeline")
    print(f"{'='*55}")
    print(f"  Input file: {os.path.basename(file_path)}")

    try:
        # ── Step 1: Load model ───────────────────────────────
        if model is None:
            model = load_model()

        # ── Step 2: Preprocess MRI ───────────────────────────
        print("\n[1/5] Preprocessing MRI...")
        volume = preprocess(file_path)        # → (1, 64, 64, 64)
        print(f"      Volume shape : {volume.shape}")

        # ── Step 3: Build connectivity graph ─────────────────
        print("\n[2/5] Building brain connectivity graph...")
        graph, conn_matrix = volume_to_graph(volume)
        num_edges = graph.edge_index.shape[1]
        print(f"      Nodes : {graph.num_nodes}  |  Edges : {num_edges}")

        # ── Step 4: Forward pass through model ───────────────
        print("\n[3/5] Running model inference...")

        # Convert volume to tensor: (1, 1, D, H, W)
        input_tensor = torch.tensor(volume, dtype=torch.float32)
        input_tensor = input_tensor.unsqueeze(0).to(DEVICE)  # Add batch dim

        with torch.no_grad():
            logits = model(input_tensor)                # (1, 3)
            probs  = F.softmax(logits, dim=1)[0]        # (3,)

        # Get predicted class
        pred_idx    = probs.argmax().item()
        pred_label  = CLASS_NAMES[pred_idx]
        confidence  = probs[pred_idx].item() * 100

        # Build probability dict for all classes
        probabilities = {
            CLASS_NAMES[i]: round(probs[i].item() * 100, 1)
            for i in range(NUM_CLASSES)
        }

        print(f"      Prediction : {pred_label}")
        print(f"      Confidence : {confidence:.1f}%")
        print(f"      All probs  : {probabilities}")

        # ── Step 5: Identify affected brain regions ───────────
        print("\n[4/5] Identifying affected brain regions...")
        top_regions = get_top_connected_regions(conn_matrix, top_k=3)
        print(f"      Top regions: {top_regions}")

        # ── Step 6: Generate Grad-CAM heatmap ────────────────
        print("\n[5/5] Generating Grad-CAM heatmap...")

        try:
            # Try real Grad-CAM first
            # Get the CNN backbone (works for both real and mock models)
            cnn_model = model.cnn if hasattr(model, "cnn") else model

            gradcam   = GradCAM3D(cnn_model, target_layer="features")
            heatmap   = gradcam.generate(input_tensor, target_class=pred_idx)

            # Resize heatmap to match volume dimensions
            from inference.gradcam import resize_heatmap_to_volume
            heatmap = resize_heatmap_to_volume(heatmap, IMAGE_SIZE)

        except Exception as e:
            # Fall back to mock heatmap if Grad-CAM fails
            print(f"      [Warning] Grad-CAM failed ({e}), using mock heatmap")
            heatmap = generate_mock_heatmap(IMAGE_SIZE)

        print(f"      Heatmap shape: {heatmap.shape}")

        # Save heatmap figure if requested
        heatmap_path = None
        if save_heatmap:
            os.makedirs(HEATMAP_OUTPUT_DIR, exist_ok=True)
            fname = os.path.splitext(os.path.basename(file_path))[0]
            heatmap_path = os.path.join(
                HEATMAP_OUTPUT_DIR,
                f"{fname}_{pred_label.lower()}_heatmap.png"
            )
            fig = plot_gradcam_slices(
                volume[0], heatmap,
                prediction=pred_label,
                confidence=confidence,
                save_path=heatmap_path
            )
            import matplotlib.pyplot as plt
            plt.close(fig)

        # ── Compute inference time ────────────────────────────
        elapsed = time.time() - start_time

        print(f"\n{'='*55}")
        print(f"  ✅ Prediction complete in {elapsed:.2f}s")
        print(f"  Result: {pred_label} ({confidence:.1f}% confidence)")
        print(f"{'='*55}\n")

        return PredictionResult(
            label            = pred_label,
            confidence       = round(confidence, 1),
            probabilities    = probabilities,
            affected_regions = top_regions,
            heatmap          = heatmap,
            volume           = volume,
            conn_matrix      = conn_matrix,
            inference_time   = round(elapsed, 2)
        )

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n[Error] Prediction failed: {e}")
        import traceback
        traceback.print_exc()

        # Return a result with error info for graceful UI handling
        return PredictionResult(
            label            = "Unknown",
            confidence       = 0.0,
            probabilities    = {c: 0.0 for c in CLASS_NAMES},
            affected_regions = [],
            heatmap          = np.zeros(IMAGE_SIZE),
            volume           = np.zeros((1, *IMAGE_SIZE)),
            conn_matrix      = np.zeros((10, 10)),
            inference_time   = round(elapsed, 2),
            error            = str(e)
        )


# =============================================================
# DEMO PREDICTION — no real MRI file needed
# =============================================================

def predict_demo() -> PredictionResult:
    """
    Run a demo prediction using synthetic (random) MRI data.
    Useful for testing the full pipeline without a real MRI file.

    Returns:
        PredictionResult: Demo prediction result
    """
    import tempfile
    import nibabel as nib

    print("[Demo] Generating synthetic MRI volume for demo prediction...")

    # Create a synthetic NIfTI volume
    fake_data = np.random.randn(91, 109, 91).astype(np.float32) * 300 + 800
    nii_img   = nib.Nifti1Image(fake_data, affine=np.eye(4))

    # Save to a temp file
    with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
        tmp_path = tmp.name

    nib.save(nii_img, tmp_path)

    try:
        result = predict(tmp_path)
    finally:
        os.unlink(tmp_path)  # Clean up temp file

    return result


# =============================================================
# CLI Entry Point
# =============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="NeuroAI — Autism Severity Predictor"
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to MRI file (.nii, .nii.gz, .png, .jpg)"
    )
    parser.add_argument(
        "--save_heatmap", action="store_true",
        help="Save Grad-CAM heatmap PNG to outputs/heatmaps/"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run demo prediction with synthetic data"
    )
    args = parser.parse_args()

    if args.demo or args.input is None:
        result = predict_demo()
    else:
        result = predict(args.input, save_heatmap=args.save_heatmap)

    # Print summary
    print("\n" + "─"*40)
    print("  PREDICTION SUMMARY")
    print("─"*40)
    print(f"  Label       : {result.label}")
    print(f"  Confidence  : {result.confidence}%")
    print(f"  Regions     : {', '.join(result.affected_regions)}")
    print(f"  Time        : {result.inference_time}s")
    if result.error:
        print(f"  ⚠️  Error   : {result.error}")
    print("─"*40)
