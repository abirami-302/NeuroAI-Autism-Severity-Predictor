# =============================================================
# main.py — NeuroAI Entry Point
#
# Usage:
#   python main.py --mode demo        # Demo prediction
#   python main.py --mode preprocess  # Preprocess dataset
#   python main.py --mode train       # Train model
#   python main.py --mode predict --input path/to/mri.nii
#   python main.py --mode ui          # Launch Streamlit UI
#   python main.py --mode weights     # Generate mock weights
# =============================================================

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def run_demo():
    from inference.predict import predict_demo
    result = predict_demo()
    print(f"\n{'─'*40}")
    print(f"  DEMO RESULT")
    print(f"{'─'*40}")
    print(f"  Label      : {result.label}")
    print(f"  Confidence : {result.confidence}%")
    print(f"  Regions    : {', '.join(result.affected_regions)}")
    print(f"{'─'*40}\n")


def run_preprocess():
    from preprocessing.pipeline import preprocess_dataset
    from training.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
    print(f"Preprocessing files in: {RAW_DATA_DIR}")
    preprocess_dataset(RAW_DATA_DIR, PROCESSED_DATA_DIR)


def run_train():
    from training.train import train
    train()


def run_predict(input_path):
    from inference.predict import predict
    result = predict(input_path, save_heatmap=True)
    print(f"\nResult: {result.label} ({result.confidence}%)")
    print(f"Regions: {result.affected_regions}")


def generate_mock_weights():
    """Save mock model weights so the UI runs immediately without training."""
    import torch
    from models.cnn3d import CNN3DClassifier
    from training.config import SAVED_MODELS_DIR, FULL_MODEL_PATH, NUM_CLASSES

    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    print("Generating mock model weights...")

    model = CNN3DClassifier(num_classes=NUM_CLASSES)
    torch.save(model.state_dict(), FULL_MODEL_PATH)
    print(f"✅ Mock weights saved → {FULL_MODEL_PATH}")


def launch_ui():
    import subprocess
    ui_path = os.path.join(os.path.dirname(__file__), "ui", "app.py")
    print(f"Launching Streamlit UI: {ui_path}")
    subprocess.run(["streamlit", "run", ui_path])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuroAI — Autism Severity Predictor")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["demo", "preprocess", "train", "predict", "ui", "weights"],
        default="demo",
        help="Operation mode"
    )
    parser.add_argument("--input", type=str, default=None,
                        help="Path to MRI file (for --mode predict)")
    args = parser.parse_args()

    if args.mode == "demo":
        run_demo()
    elif args.mode == "preprocess":
        run_preprocess()
    elif args.mode == "train":
        run_train()
    elif args.mode == "predict":
        if not args.input:
            print("Error: --input required for predict mode")
            sys.exit(1)
        run_predict(args.input)
    elif args.mode == "ui":
        launch_ui()
    elif args.mode == "weights":
        generate_mock_weights()
