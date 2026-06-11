# =============================================================
# training/metrics.py
# Evaluation Metrics for NeuroAI Autism Severity Prediction.
#
# Provides:
#   - accuracy, per-class accuracy
#   - macro / weighted F1 score
#   - confusion matrix (raw + normalised)
#   - full classification report (dict)
#   - MetricTracker — accumulates stats across batches
# =============================================================

import numpy as np
import torch
import os
import sys
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, classification_report
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.config import CLASS_NAMES, NUM_CLASSES


# =============================================================
# BATCH-LEVEL HELPERS
# =============================================================

def batch_accuracy(logits: torch.Tensor,
                   labels: torch.Tensor) -> float:
    """
    Compute prediction accuracy for a single batch.

    Args:
        logits (torch.Tensor): Raw model output (B, C)
        labels (torch.Tensor): True class indices (B,)

    Returns:
        float: Accuracy in [0, 1]
    """
    preds = logits.argmax(dim=1)
    return float((preds == labels).float().mean().item())


def batch_to_numpy(logits: torch.Tensor,
                   labels: torch.Tensor):
    """Convert a batch of logits + labels to numpy int arrays."""
    preds  = logits.argmax(dim=1).detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    return preds, labels


# =============================================================
# EPOCH-LEVEL METRICS (from accumulated predictions)
# =============================================================

def compute_accuracy(all_preds: np.ndarray,
                     all_labels: np.ndarray) -> float:
    """Overall classification accuracy."""
    return float(accuracy_score(all_labels, all_preds))


def compute_f1(all_preds: np.ndarray,
               all_labels: np.ndarray,
               average: str = "macro") -> float:
    """
    Compute F1 score.

    Args:
        all_preds  (np.ndarray): Predicted class indices
        all_labels (np.ndarray): True class indices
        average (str): "macro" | "weighted" | "micro"

    Returns:
        float: F1 score
    """
    return float(f1_score(all_labels, all_preds,
                          average=average, zero_division=0))


def compute_precision_recall(all_preds: np.ndarray,
                              all_labels: np.ndarray,
                              average: str = "macro"):
    """
    Compute macro precision and recall.

    Returns:
        tuple: (precision, recall) both floats
    """
    p = float(precision_score(all_labels, all_preds,
                               average=average, zero_division=0))
    r = float(recall_score(all_labels, all_preds,
                            average=average, zero_division=0))
    return p, r


def compute_confusion_matrix(all_preds: np.ndarray,
                              all_labels: np.ndarray,
                              normalise: bool = False) -> np.ndarray:
    """
    Compute the confusion matrix.

    Args:
        all_preds  (np.ndarray): Predicted class indices
        all_labels (np.ndarray): True class indices
        normalise (bool):        If True, normalise by row (true labels)

    Returns:
        np.ndarray: Confusion matrix (NUM_CLASSES, NUM_CLASSES)
    """
    cm = confusion_matrix(all_labels, all_preds,
                          labels=list(range(NUM_CLASSES)))
    if normalise:
        row_sums = cm.sum(axis=1, keepdims=True).astype(float) + 1e-8
        cm = cm / row_sums
    return cm


def compute_per_class_accuracy(all_preds: np.ndarray,
                                all_labels: np.ndarray) -> dict:
    """
    Compute accuracy separately for each class.

    Returns:
        dict: {class_name: accuracy_float}
    """
    result = {}
    for i, name in enumerate(CLASS_NAMES):
        mask  = all_labels == i
        if mask.sum() == 0:
            result[name] = 0.0
        else:
            result[name] = float((all_preds[mask] == i).mean())
    return result


def full_report(all_preds: np.ndarray,
                all_labels: np.ndarray) -> dict:
    """
    Compute a comprehensive evaluation report.

    Returns:
        dict with keys:
            accuracy, f1_macro, f1_weighted,
            precision_macro, recall_macro,
            per_class_accuracy, confusion_matrix,
            sklearn_report (str)
    """
    acc       = compute_accuracy(all_preds, all_labels)
    f1_mac    = compute_f1(all_preds, all_labels, "macro")
    f1_wtd    = compute_f1(all_preds, all_labels, "weighted")
    prec, rec = compute_precision_recall(all_preds, all_labels)
    cm        = compute_confusion_matrix(all_preds, all_labels)
    pca       = compute_per_class_accuracy(all_preds, all_labels)
    sk_report = classification_report(
        all_labels, all_preds,
        target_names=CLASS_NAMES,
        zero_division=0
    )

    return {
        "accuracy":           round(acc,    4),
        "f1_macro":           round(f1_mac, 4),
        "f1_weighted":        round(f1_wtd, 4),
        "precision_macro":    round(prec,   4),
        "recall_macro":       round(rec,    4),
        "per_class_accuracy": {k: round(v, 4) for k, v in pca.items()},
        "confusion_matrix":   cm.tolist(),
        "sklearn_report":     sk_report,
    }


# =============================================================
# METRIC TRACKER
# Accumulates predictions batch-by-batch during a full epoch.
# =============================================================

class MetricTracker:
    """
    Accumulates batch predictions and labels across an epoch,
    then computes all metrics at epoch end.

    Usage:
        tracker = MetricTracker()

        for batch in dataloader:
            logits, labels = model(batch), batch.y
            tracker.update(logits, labels)

        report = tracker.compute()
        print(report["accuracy"])
        tracker.reset()
    """

    def __init__(self):
        self.all_preds  = []
        self.all_labels = []
        self.total_loss = 0.0
        self.n_batches  = 0

    def update(self, logits: torch.Tensor,
               labels: torch.Tensor,
               loss: float = 0.0):
        """
        Accumulate one batch.

        Args:
            logits (torch.Tensor): (B, C) model output
            labels (torch.Tensor): (B,) true class indices
            loss (float):          Batch loss value
        """
        preds, lbls = batch_to_numpy(logits, labels)
        self.all_preds.extend(preds.tolist())
        self.all_labels.extend(lbls.tolist())
        self.total_loss += loss
        self.n_batches  += 1

    def compute(self) -> dict:
        """
        Compute all metrics from accumulated predictions.

        Returns:
            dict: Complete evaluation report + mean_loss
        """
        preds  = np.array(self.all_preds,  dtype=int)
        labels = np.array(self.all_labels, dtype=int)

        report = full_report(preds, labels)
        report["mean_loss"] = round(
            self.total_loss / max(self.n_batches, 1), 6
        )
        return report

    def reset(self):
        """Clear all accumulated state."""
        self.all_preds  = []
        self.all_labels = []
        self.total_loss = 0.0
        self.n_batches  = 0

    @property
    def num_samples(self) -> int:
        return len(self.all_preds)


# =============================================================
# Self-test
# =============================================================
if __name__ == "__main__":
    print("Testing metrics module...")

    # Simulate 50 predictions
    np.random.seed(42)
    preds  = np.random.randint(0, 3, 50)
    labels = np.random.randint(0, 3, 50)

    print(f"  Accuracy        : {compute_accuracy(preds, labels):.4f}")
    print(f"  F1 (macro)      : {compute_f1(preds, labels):.4f}")
    print(f"  F1 (weighted)   : {compute_f1(preds, labels, 'weighted'):.4f}")
    p, r = compute_precision_recall(preds, labels)
    print(f"  Precision/Recall: {p:.4f} / {r:.4f}")
    print(f"  Per-class acc   : {compute_per_class_accuracy(preds, labels)}")
    print(f"\n  Confusion matrix:\n{compute_confusion_matrix(preds, labels)}")

    # MetricTracker test
    tracker = MetricTracker()
    logits  = torch.randn(8, 3)
    lbls    = torch.randint(0, 3, (8,))
    tracker.update(logits, lbls, loss=0.75)
    tracker.update(logits, lbls, loss=0.65)
    report  = tracker.compute()
    print(f"\n  Tracker accuracy : {report['accuracy']}")
    print(f"  Tracker mean_loss: {report['mean_loss']}")

    print("\n✅ Metrics tests passed!")
