# =============================================================
# training/train.py
# Main model training script for NeuroAI.
#
# Usage:
#   python training/train.py                     # Default settings
#   python training/train.py --epochs 50         # Custom epochs
#   python training/train.py --resume checkpoint.pth  # Resume
# =============================================================

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.config import (
    PROCESSED_DATA_DIR, SAMPLE_DATA_DIR, SAVED_MODELS_DIR,
    IMAGE_SIZE, NUM_CLASSES, CLASS_NAMES, BATCH_SIZE, EPOCHS,
    LEARNING_RATE, WEIGHT_DECAY, LR_STEP_SIZE, LR_GAMMA,
    EARLY_STOP_PATIENCE, TRAIN_SPLIT, VAL_SPLIT, RANDOM_SEED,
    DEVICE, FULL_MODEL_PATH, CHECKPOINT_PATH, LOG_DIR
)
from models.cnn3d import CNN3DClassifier
from utils.logger import get_logger

logger = get_logger("Training")


# =============================================================
# DATASET CLASS
# Loads preprocessed .npy volumes with integer class labels
# =============================================================

class MRIAutismDataset(Dataset):
    """
    PyTorch Dataset for preprocessed MRI autism data.

    Expects a directory of .npy files where each file is named:
        {subject_id}_{label}.npy
        e.g.:  sub001_mild.npy, sub002_severe.npy

    If no real data is found, automatically generates synthetic
    dummy data for immediate demo/testing.

    Args:
        data_dir (str):   Directory containing .npy MRI files
        augment (bool):   Apply random data augmentation
    """

    # Map label strings → integer indices
    LABEL_MAP = {"mild": 0, "moderate": 1, "severe": 2}

    def __init__(self, data_dir: str, augment: bool = False):
        self.data_dir = data_dir
        self.augment  = augment
        self.samples  = []   # List of (file_path, label_int)

        self._load_dataset()

    def _load_dataset(self):
        """Scan data_dir for .npy files and extract labels from filenames."""
        if not os.path.exists(self.data_dir):
            logger.warning(f"Data dir not found: {self.data_dir}. Using synthetic data.")
            self._generate_synthetic_data()
            return

        npy_files = [f for f in os.listdir(self.data_dir) if f.endswith(".npy")]

        if not npy_files:
            logger.warning(f"No .npy files in {self.data_dir}. Using synthetic data.")
            self._generate_synthetic_data()
            return

        for fname in npy_files:
            # Extract label from filename
            lower = fname.lower()
            label = None
            for key in self.LABEL_MAP:
                if key in lower:
                    label = self.LABEL_MAP[key]
                    break

            if label is None:
                logger.warning(f"Could not extract label from: {fname} — skipping")
                continue

            self.samples.append((os.path.join(self.data_dir, fname), label))

        logger.info(f"Loaded {len(self.samples)} samples from {self.data_dir}")

    def _generate_synthetic_data(self, num_samples: int = 60):
        """
        Generate synthetic MRI volumes for demo/testing.
        Creates num_samples random volumes split equally across 3 classes.
        """
        logger.info(f"Generating {num_samples} synthetic MRI samples...")
        os.makedirs(self.data_dir, exist_ok=True)

        per_class = num_samples // NUM_CLASSES

        for class_idx, class_name in enumerate(CLASS_NAMES):
            for i in range(per_class):
                # Add class-specific signal pattern to distinguish classes
                # This makes training actually converge on synthetic data
                volume = np.random.randn(1, *IMAGE_SIZE).astype(np.float32)
                # Class-specific mean offset to create learnable signal
                volume += class_idx * 0.5

                fname = f"synthetic_{class_name.lower()}_{i:03d}.npy"
                fpath = os.path.join(self.data_dir, fname)
                np.save(fpath, volume)
                self.samples.append((fpath, class_idx))

        logger.info(f"Synthetic dataset created: {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        """Load and return a single (volume, label) pair."""
        file_path, label = self.samples[idx]
        volume = np.load(file_path)               # Shape: (1, D, H, W)

        # Ensure correct shape
        if volume.ndim == 3:
            volume = volume[np.newaxis, ...]      # Add channel dim

        volume = torch.tensor(volume, dtype=torch.float32)

        # Optional: random augmentation (flip along depth axis)
        if self.augment and torch.rand(1).item() > 0.5:
            volume = torch.flip(volume, dims=[1])  # Flip D axis

        return volume, torch.tensor(label, dtype=torch.long)


# =============================================================
# TRAINING UTILITIES
# =============================================================

class EarlyStopping:
    """
    Stop training when validation loss doesn't improve for
    'patience' consecutive epochs.
    """
    def __init__(self, patience: int = EARLY_STOP_PATIENCE,
                 min_delta: float = 0.001):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def compute_accuracy(logits: torch.Tensor,
                     labels: torch.Tensor) -> float:
    """Compute batch accuracy from logits and true labels."""
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


# =============================================================
# SAVE AND LOAD CHECKPOINT
# =============================================================

def save_checkpoint(model, optimizer, epoch: int,
                    val_loss: float, path: str):
    """Save model + optimizer state to a checkpoint file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "epoch":      epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss":   val_loss,
    }, path)
    logger.info(f"Checkpoint saved → {path}")


def load_checkpoint(model, optimizer, path: str):
    """Load model + optimizer state from checkpoint."""
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    logger.info(f"Resumed from epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f}")
    return start_epoch


# =============================================================
# PLOT TRAINING CURVES
# =============================================================

def plot_training_curves(history: dict, save_path: str = None):
    """
    Plot loss and accuracy curves over training epochs.

    Args:
        history (dict): Keys: "train_loss", "val_loss",
                               "train_acc", "val_acc"
        save_path (str): Save figure to this path if given
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#1a1a2e")

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss plot
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=4)
    axes[0].plot(epochs, history["val_loss"],   "r-o", label="Val Loss",   markersize=4)
    axes[0].set_title("Loss Curves", color="white")
    axes[0].set_xlabel("Epoch", color="white")
    axes[0].set_ylabel("Loss", color="white")
    axes[0].legend()
    axes[0].set_facecolor("#0d1117")
    axes[0].tick_params(colors="white")

    # Accuracy plot
    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train Acc", markersize=4)
    axes[1].plot(epochs, history["val_acc"],   "r-o", label="Val Acc",   markersize=4)
    axes[1].set_title("Accuracy Curves", color="white")
    axes[1].set_xlabel("Epoch", color="white")
    axes[1].set_ylabel("Accuracy", color="white")
    axes[1].legend()
    axes[1].set_facecolor("#0d1117")
    axes[1].tick_params(colors="white")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        logger.info(f"Training curves saved → {save_path}")

    plt.close(fig)


# =============================================================
# MAIN TRAINING FUNCTION
# =============================================================

def train(data_dir: str   = None,
          epochs: int     = EPOCHS,
          batch_size: int = BATCH_SIZE,
          lr: float       = LEARNING_RATE,
          resume: str     = None):
    """
    Full training pipeline for NeuroAI.

    Args:
        data_dir (str):    Directory with preprocessed .npy files
        epochs (int):      Number of training epochs
        batch_size (int):  Samples per batch
        lr (float):        Learning rate
        resume (str):      Path to checkpoint to resume from
    """
    logger.info("="*55)
    logger.info("  NeuroAI Training Started")
    logger.info("="*55)
    logger.info(f"  Device     : {DEVICE}")
    logger.info(f"  Epochs     : {epochs}")
    logger.info(f"  Batch size : {batch_size}")
    logger.info(f"  LR         : {lr}")

    # ── Dataset & DataLoaders ────────────────────────────────
    data_dir = data_dir or PROCESSED_DATA_DIR
    dataset  = MRIAutismDataset(data_dir, augment=True)

    # Split into train / val / test
    n_total = len(dataset)
    n_train = int(n_total * TRAIN_SPLIT)
    n_val   = int(n_total * VAL_SPLIT)
    n_test  = n_total - n_train - n_val

    torch.manual_seed(RANDOM_SEED)
    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test]
    )

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=batch_size,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=batch_size,
                              shuffle=False, num_workers=0)

    logger.info(f"  Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

    # ── Model ────────────────────────────────────────────────
    model = CNN3DClassifier(num_classes=NUM_CLASSES).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model params: {total_params:,}")

    # ── Loss, Optimizer, Scheduler ───────────────────────────
    criterion = nn.CrossEntropyLoss()   # Handles multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA
    )
    early_stopper = EarlyStopping(patience=EARLY_STOP_PATIENCE)

    # ── Resume from checkpoint ───────────────────────────────
    start_epoch = 0
    if resume and os.path.exists(resume):
        start_epoch = load_checkpoint(model, optimizer, resume)

    # ── Training Loop ────────────────────────────────────────
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    history = {"train_loss": [], "val_loss": [],
               "train_acc":  [], "val_acc":  []}

    best_val_loss = float("inf")

    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        # ── Training phase ───────────────────────────────────
        model.train()
        train_loss, train_acc = 0.0, 0.0

        for batch_idx, (volumes, labels) in enumerate(train_loader):
            volumes = volumes.to(DEVICE)
            labels  = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(volumes)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc  += compute_accuracy(logits, labels)

        train_loss /= len(train_loader)
        train_acc  /= len(train_loader)

        # ── Validation phase ─────────────────────────────────
        model.eval()
        val_loss, val_acc = 0.0, 0.0

        with torch.no_grad():
            for volumes, labels in val_loader:
                volumes = volumes.to(DEVICE)
                labels  = labels.to(DEVICE)
                logits  = model(volumes)
                loss    = criterion(logits, labels)
                val_loss += loss.item()
                val_acc  += compute_accuracy(logits, labels)

        val_loss /= len(val_loader)
        val_acc  /= len(val_loader)

        scheduler.step()

        elapsed = time.time() - t0

        # ── Logging ──────────────────────────────────────────
        logger.info(
            f"Epoch [{epoch+1:>3}/{epochs}] "
            f"| Train Loss: {train_loss:.4f}  Acc: {train_acc:.3f} "
            f"| Val Loss: {val_loss:.4f}  Acc: {val_acc:.3f} "
            f"| {elapsed:.1f}s"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), FULL_MODEL_PATH)
            logger.info(f"  ✅ Best model saved → {FULL_MODEL_PATH}")

        # Periodic checkpoint
        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, optimizer, epoch, val_loss, CHECKPOINT_PATH)

        # Early stopping
        if early_stopper.step(val_loss):
            logger.info(f"  Early stopping triggered at epoch {epoch+1}")
            break

    # ── Final Evaluation on Test Set ─────────────────────────
    logger.info("\n" + "="*55)
    logger.info("  Final Test Set Evaluation")
    logger.info("="*55)

    # Load best saved model
    model.load_state_dict(torch.load(FULL_MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for volumes, labels in test_loader:
            volumes = volumes.to(DEVICE)
            logits  = model(volumes)
            preds   = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # Classification report
    print("\n" + classification_report(
        all_labels, all_preds,
        target_names=CLASS_NAMES
    ))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    cm_path = os.path.join(LOG_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"Confusion matrix saved → {cm_path}")

    # Training curves
    curves_path = os.path.join(LOG_DIR, "training_curves.png")
    plot_training_curves(history, save_path=curves_path)

    logger.info("\n  Training complete! 🎉")
    logger.info(f"  Best val loss : {best_val_loss:.4f}")
    logger.info(f"  Model saved   : {FULL_MODEL_PATH}")

    return history


# =============================================================
# CLI Entry Point
# =============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NeuroAI model")
    parser.add_argument("--data_dir",   type=str,   default=None)
    parser.add_argument("--epochs",     type=int,   default=EPOCHS)
    parser.add_argument("--batch_size", type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=LEARNING_RATE)
    parser.add_argument("--resume",     type=str,   default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    train(
        data_dir   = args.data_dir,
        epochs     = args.epochs,
        batch_size = args.batch_size,
        lr         = args.lr,
        resume     = args.resume
    )
