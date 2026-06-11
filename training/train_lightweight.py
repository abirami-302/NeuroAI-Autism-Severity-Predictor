# =============================================================
# training/train_lightweight.py
#
# Lightweight Training Script for NeuroAI.
# Trains the FULL 3D CNN → GNN → Classifier pipeline on
# small NIfTI MRI datasets (50-100 samples) on a CPU laptop.
#
# USAGE:
#   python training/train_lightweight.py
#   python training/train_lightweight.py --size 32
#   python training/train_lightweight.py --size 64 --epochs 30
#   python training/train_lightweight.py --data dataset/raw
#
# WHAT THIS DOES:
#   1. Loads real .nii / .nii.gz MRI files from dataset/raw/
#   2. Applies preprocessing + augmentation automatically
#   3. Trains LightweightCNN3D + BrainGNN jointly
#   4. Saves best model to saved_models/lightweight_best.pth
#   5. Plots training curves + confusion matrix
#   6. Prints full classification report
# =============================================================

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — safe on Windows/CPU
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Lightweight config (separate from original) ──────────────
from training.config_lightweight import (
    RAW_DATA_DIR, SAVED_MODELS_DIR, LOG_DIR,
    LW_MODEL_PATH, LW_CHECKPOINT_PATH, LW_BEST_PATH,
    IMAGE_SIZE, NUM_CLASSES, CLASS_NAMES, DEVICE,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    LR_STEP_SIZE, LR_GAMMA, EARLY_STOP_PATIENCE,
    TRAIN_SPLIT, VAL_SPLIT, RANDOM_SEED,
    CNN_FEATURE_DIM, GNN_INPUT_DIM, GNN_HIDDEN_DIM,
    GNN_OUTPUT_DIM, GNN_NUM_LAYERS, GNN_DROPOUT,
    CONNECTIVITY_THRESHOLD, NUM_ROI,
    print_config
)

# ── Models — lightweight CNN + existing GNN ──────────────────
from models.cnn3d_lightweight import LightweightCNN3DClassifier, model_summary

# ── Dataset loader with augmentation ─────────────────────────
from dataset.dataloader import build_dataloaders, dataset_summary

# ── Graph pipeline ────────────────────────────────────────────
from graph.graph_builder import volume_to_graph

# ── PyTorch Geometric batch helper ───────────────────────────
try:
    from torch_geometric.data import Batch as PyGBatch
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("  [Warning] torch_geometric not found — GNN branch disabled.")
    print("            Install with: pip install torch-geometric")

# ── Loss & metrics ───────────────────────────────────────────
from training.loss import FocalLoss, get_loss_fn
from training.metrics import MetricTracker, full_report


# =============================================================
# COMBINED LIGHTWEIGHT MODEL (CNN + optional GNN)
# =============================================================

class LightweightNeuroAI(nn.Module):
    """
    Full lightweight NeuroAI pipeline:
        LightweightCNN3D → (optional BrainGNN) → Classifier

    The GNN branch is enabled only if torch_geometric is installed.
    If not, falls back to CNN-only classification.

    This means the model still works out of the box even if
    PyG installation had issues.
    """

    def __init__(self, use_gnn: bool = True):
        super().__init__()
        self.use_gnn = use_gnn and HAS_PYG

        # Import lightweight CNN backbone
        from models.cnn3d_lightweight import LightweightCNN3D
        self.cnn = LightweightCNN3D()  # → (B, 64)

        if self.use_gnn:
            from models.gnn import BrainGNN
            # Instantiate GNN with lightweight dims
            self.gnn = BrainGNN(
                input_dim  = GNN_INPUT_DIM,
                hidden_dim = GNN_HIDDEN_DIM,
                output_dim = GNN_OUTPUT_DIM,
                num_layers = GNN_NUM_LAYERS,
                dropout    = GNN_DROPOUT
            )
            combined_dim = CNN_FEATURE_DIM + GNN_OUTPUT_DIM   # 64 + 16 = 80
        else:
            combined_dim = CNN_FEATURE_DIM                     # 64 only

        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, NUM_CLASSES)
        )

        mode = "CNN + GNN" if self.use_gnn else "CNN only"
        print(f"  [Model] LightweightNeuroAI ({mode})")

    def forward(self, volumes: torch.Tensor,
                graphs=None) -> torch.Tensor:
        """
        Args:
            volumes: (B, 1, D, H, W)
            graphs:  PyG Batch object (optional, for GNN branch)
        Returns:
            (B, NUM_CLASSES) logits
        """
        # CNN branch — always runs
        cnn_feats = self.cnn(volumes)   # (B, 64)

        if self.use_gnn and graphs is not None:
            # GNN branch
            gnn_embed = self.gnn(
                graphs.x, graphs.edge_index, graphs.batch
            )                           # (B, 16)
            combined = torch.cat([cnn_feats, gnn_embed], dim=1)  # (B, 80)
        else:
            combined = cnn_feats        # (B, 64)

        return self.classifier(combined)


# =============================================================
# GRAPH BATCH BUILDER
# Converts a batch of MRI volumes → PyG graph batch
# =============================================================

def build_graph_batch(volumes: torch.Tensor) -> object:
    """
    Convert a batch of MRI volumes to a PyG graph batch
    using the existing graph_builder pipeline.

    Args:
        volumes: (B, 1, D, H, W) tensor

    Returns:
        PyG Batch object or None if PyG unavailable
    """
    if not HAS_PYG:
        return None

    graphs = []
    vol_np = volumes.detach().cpu().numpy()

    for i in range(vol_np.shape[0]):
        try:
            graph, _ = volume_to_graph(
                vol_np[i],
                threshold=CONNECTIVITY_THRESHOLD
            )
            graphs.append(graph)
        except Exception:
            # If graph build fails for one sample, skip GNN for whole batch
            return None

    try:
        return PyGBatch.from_data_list(graphs)
    except Exception:
        return None


# =============================================================
# EARLY STOPPING
# =============================================================

class EarlyStopping:
    def __init__(self, patience: int = EARLY_STOP_PATIENCE,
                 min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter   = 0

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter  += 1
        return self.counter >= self.patience


# =============================================================
# TRAINING & VALIDATION LOOPS
# =============================================================

def train_one_epoch(model, loader, criterion,
                    optimizer, device, use_gnn) -> tuple:
    """Run one full training epoch. Returns (avg_loss, avg_acc)."""
    model.train()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0

    for volumes, labels in loader:
        volumes = volumes.to(device)
        labels  = labels.to(device)

        # Build graph batch from volumes (CPU-safe)
        graphs = build_graph_batch(volumes) if use_gnn else None
        if graphs is not None:
            graphs = graphs.to(device)

        optimizer.zero_grad()
        logits = model(volumes, graphs)
        loss   = criterion(logits, labels)
        loss.backward()

        # Gradient clipping prevents exploding gradients on small data
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        preds       = logits.argmax(dim=1)
        total_loss += loss.item()
        total_acc  += (preds == labels).float().mean().item()
        n_batches  += 1

    return total_loss / n_batches, total_acc / n_batches


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_gnn) -> tuple:
    """Evaluate model on val/test loader. Returns (loss, acc, preds, labels)."""
    model.eval()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0
    all_preds, all_labels = [], []

    for volumes, labels in loader:
        volumes = volumes.to(device)
        labels  = labels.to(device)

        graphs = build_graph_batch(volumes) if use_gnn else None
        if graphs is not None:
            graphs = graphs.to(device)

        logits = model(volumes, graphs)
        loss   = criterion(logits, labels)
        preds  = logits.argmax(dim=1)

        total_loss += loss.item()
        total_acc  += (preds == labels).float().mean().item()
        n_batches  += 1

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(n_batches, 1)
    avg_acc  = total_acc  / max(n_batches, 1)
    return avg_loss, avg_acc, np.array(all_preds), np.array(all_labels)


# =============================================================
# PLOTTING HELPERS
# =============================================================

def save_training_curves(history: dict, save_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#1a1a2e")
    epochs = range(1, len(history["train_loss"]) + 1)

    for ax, metric, title in zip(
        axes,
        [("train_loss", "val_loss"), ("train_acc", "val_acc")],
        ["Loss Curves", "Accuracy Curves"]
    ):
        ax.plot(epochs, history[metric[0]], "b-o", label="Train", markersize=3)
        ax.plot(epochs, history[metric[1]], "r-o", label="Val",   markersize=3)
        ax.set_title(title, color="white")
        ax.set_xlabel("Epoch", color="white")
        ax.legend()
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#444")

    plt.tight_layout()
    path = os.path.join(save_dir, "lightweight_training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    print(f"  [Plot] Training curves → {path}")


def save_confusion_matrix(preds: np.ndarray,
                           labels: np.ndarray,
                           save_dir: str) -> None:
    cm  = confusion_matrix(labels, preds, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title("Confusion Matrix — Lightweight Model")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    path = os.path.join(save_dir, "lightweight_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] Confusion matrix → {path}")


# =============================================================
# MAIN TRAINING FUNCTION
# =============================================================

def train(data_dir:   str   = None,
          image_size: tuple = None,
          epochs:     int   = None,
          batch_size: int   = None,
          use_gnn:    bool  = True) -> dict:
    """
    Full lightweight training pipeline.

    Args:
        data_dir   : Directory with raw MRI files (default: config)
        image_size : (D, H, W) volume size (default: config)
        epochs     : Number of training epochs (default: config)
        batch_size : Samples per batch (default: config)
        use_gnn    : Enable GNN branch (default: True)

    Returns:
        dict: Training history
    """
    # ── Apply config defaults ─────────────────────────────────
    data_dir   = data_dir   or RAW_DATA_DIR
    image_size = image_size or IMAGE_SIZE
    epochs     = epochs     or EPOCHS
    batch_size = batch_size or BATCH_SIZE

    # ── Print config ─────────────────────────────────────────
    print_config()
    print(f"  Data dir   : {data_dir}")
    print(f"  Image size : {image_size}")
    print(f"  Use GNN    : {use_gnn and HAS_PYG}")
    print(f"  Device     : {DEVICE}\n")

    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # ── Dataset & DataLoaders ────────────────────────────────
    dataset_summary(data_dir)

    train_loader, val_loader, test_loader, class_weights, dataset = \
        build_dataloaders(
            data_dir    = data_dir,
            target_size = image_size,
            batch_size  = batch_size,
            train_split = TRAIN_SPLIT,
            val_split   = VAL_SPLIT,
            random_seed = RANDOM_SEED,
            cache       = True   # 16 GB RAM → cache all volumes
        )

    # ── Model ────────────────────────────────────────────────
    model = LightweightNeuroAI(use_gnn=use_gnn).to(DEVICE)
    model_summary(model)

    # ── Loss: weighted focal loss handles class imbalance ────
    criterion = FocalLoss(gamma=2.0)
    # Alternatively use weighted CE:
    # criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    # ── Optimiser + scheduler ────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA
    )
    early_stopper = EarlyStopping(patience=EARLY_STOP_PATIENCE)

    # ── Training loop ────────────────────────────────────────
    history     = {"train_loss":[], "val_loss":[],
                   "train_acc":[], "val_acc":[]}
    best_val_loss = float("inf")

    print(f"\n{'='*60}")
    print(f"  Starting Training — {epochs} epochs")
    print(f"{'='*60}\n")

    for epoch in range(epochs):
        t0 = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE,
            use_gnn=use_gnn and HAS_PYG
        )

        # Validate
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, DEVICE,
            use_gnn=use_gnn and HAS_PYG
        )

        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Current LR
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"  Epoch [{epoch+1:>3}/{epochs}]  "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.3f}  |  "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.3f}  |  "
            f"LR: {current_lr:.2e}  |  {elapsed:.0f}s"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), LW_BEST_PATH)
            print(f"             ✅ Best model saved (val_loss={val_loss:.4f})")

        # Periodic checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                "epoch":      epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss":   val_loss,
                "image_size": image_size,
            }, LW_CHECKPOINT_PATH)

        # Early stopping
        if early_stopper.step(val_loss):
            print(f"\n  ⏹  Early stopping at epoch {epoch+1} "
                  f"(no improvement for {EARLY_STOP_PATIENCE} epochs)")
            break

    # ── Final test evaluation ─────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Final Test Set Evaluation")
    print(f"{'='*60}")

    # Load best saved weights
    model.load_state_dict(torch.load(LW_BEST_PATH, map_location=DEVICE))

    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, DEVICE,
        use_gnn=use_gnn and HAS_PYG
    )

    print(f"\n  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_acc:.4f}  ({test_acc*100:.1f}%)")
    print(f"\n  Classification Report:")
    print(classification_report(
        test_labels, test_preds,
        target_names=CLASS_NAMES,
        zero_division=0
    ))

    # ── Save plots ────────────────────────────────────────────
    save_training_curves(history, LOG_DIR)
    if len(test_labels) > 0:
        save_confusion_matrix(test_preds, test_labels, LOG_DIR)

    # ── Save final model (also save image_size for inference) ─
    torch.save({
        "model_state_dict": model.state_dict(),
        "image_size":  image_size,
        "use_gnn":     use_gnn and HAS_PYG,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "val_loss":    best_val_loss,
    }, LW_MODEL_PATH)

    print(f"\n  ✅ Training complete!")
    print(f"  Best val loss : {best_val_loss:.4f}")
    print(f"  Model saved   : {LW_BEST_PATH}")
    print(f"  Full info     : {LW_MODEL_PATH}")
    print(f"  Logs + plots  : {LOG_DIR}\n")

    return history


# =============================================================
# CLI
# =============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NeuroAI Lightweight Training — 3D MRI on CPU laptop"
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to dataset dir (default: dataset/raw/)"
    )
    parser.add_argument(
        "--size", type=int, default=64, choices=[32, 64],
        help="Volume size: 32 (faster) or 64 (better accuracy)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Number of training epochs (default: from config)"
    )
    parser.add_argument(
        "--batch", type=int, default=None,
        help="Batch size (default: 2)"
    )
    parser.add_argument(
        "--no-gnn", action="store_true",
        help="Disable GNN branch (CNN-only, faster training)"
    )
    args = parser.parse_args()

    size = (args.size, args.size, args.size)

    train(
        data_dir   = args.data,
        image_size = size,
        epochs     = args.epochs,
        batch_size = args.batch,
        use_gnn    = not args.no_gnn,
    )
