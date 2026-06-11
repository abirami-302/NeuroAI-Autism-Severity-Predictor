# =============================================================
# training/loss.py
# Loss functions for NeuroAI multi-class classification.
#
# Provides:
#   - CrossEntropyLoss          (standard, balanced classes)
#   - FocalLoss                 (handles class imbalance well)
#   - LabelSmoothingLoss        (prevents overconfident predictions)
#   - get_loss_fn()             (factory based on config string)
# =============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.config import NUM_CLASSES


# =============================================================
# FOCAL LOSS
# Addresses class imbalance by down-weighting easy examples.
# Reference: Lin et al., 2017 "Focal Loss for Dense Object Detection"
# =============================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.

    Reduces the loss contribution from easy (well-classified)
    examples and focuses training on hard (misclassified) ones.
    Especially useful when severe autism cases are rare.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma (float):  Focusing parameter (0 = standard CE).
                        Higher gamma → more focus on hard examples.
        alpha (float or list): Class weight factor.
                        If float: uniform scaling.
                        If list:  per-class weights [mild, moderate, severe].
        reduction (str): "mean" | "sum" | "none"
    """
    def __init__(self,
                 gamma: float = 2.0,
                 alpha = None,
                 reduction: str = "mean"):
        super(FocalLoss, self).__init__()
        self.gamma     = gamma
        self.reduction = reduction

        # Register alpha as a buffer (moves with .to(device))
        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (list, np.ndarray)):
            self.register_buffer("alpha",
                                 torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = float(alpha)

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  (torch.Tensor): Raw model output (B, C)
            targets (torch.Tensor): True class indices (B,)

        Returns:
            torch.Tensor: Scalar focal loss value
        """
        # Compute standard cross-entropy per sample (no reduction)
        ce_loss = F.cross_entropy(logits, targets, reduction="none")  # (B,)

        # p_t = probability of the true class
        probs = F.softmax(logits, dim=1)                               # (B, C)
        p_t   = probs.gather(1, targets.unsqueeze(1)).squeeze(1)      # (B,)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1.0 - p_t) ** self.gamma

        # Apply alpha (class weight)
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha[targets]
            else:
                alpha_t = self.alpha
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * ce_loss   # (B,)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# =============================================================
# LABEL SMOOTHING LOSS
# Prevents the model from becoming overconfident.
# =============================================================

class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy loss with label smoothing.

    Instead of training toward hard 0/1 labels, targets are
    smoothed: true class → (1 - smoothing), others → smoothing/(C-1).

    Improves generalisation and calibration of confidence scores.

    Args:
        smoothing (float): Smoothing factor in [0, 1].
                           0.0 = standard CE; 0.1 is a common choice.
        num_classes (int): Number of output classes.
    """
    def __init__(self, smoothing: float = 0.1,
                 num_classes: int = NUM_CLASSES):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing   = smoothing
        self.num_classes = num_classes
        self.confidence  = 1.0 - smoothing

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  (torch.Tensor): (B, C)
            targets (torch.Tensor): (B,) integer class labels

        Returns:
            torch.Tensor: Scalar smoothed cross-entropy loss
        """
        log_probs = F.log_softmax(logits, dim=1)   # (B, C)

        # Build smooth target distribution
        with torch.no_grad():
            smooth_dist = torch.full_like(log_probs,
                                          self.smoothing / (self.num_classes - 1))
            smooth_dist.scatter_(1, targets.unsqueeze(1), self.confidence)

        # KL-divergence form of label-smoothed CE
        loss = -(smooth_dist * log_probs).sum(dim=1)
        return loss.mean()


# =============================================================
# WEIGHTED CROSS-ENTROPY
# Applies per-class weights to handle imbalanced datasets.
# =============================================================

def weighted_cross_entropy(logits: torch.Tensor,
                            targets: torch.Tensor,
                            class_counts: list = None) -> torch.Tensor:
    """
    Cross-entropy with inverse-frequency class weighting.

    Automatically computes weights as:
        weight_c = total_samples / (num_classes * count_c)

    Args:
        logits (torch.Tensor):      (B, C) raw model outputs
        targets (torch.Tensor):     (B,) class indices
        class_counts (list):        Sample counts per class [n_mild, n_mod, n_sev]
                                    If None, equal weights are used.

    Returns:
        torch.Tensor: Scalar weighted CE loss
    """
    if class_counts is None:
        return F.cross_entropy(logits, targets)

    total     = sum(class_counts)
    weights   = [total / (len(class_counts) * c + 1e-6) for c in class_counts]
    w_tensor  = torch.tensor(weights, dtype=torch.float32,
                             device=logits.device)
    return F.cross_entropy(logits, targets, weight=w_tensor)


# =============================================================
# LOSS FACTORY
# =============================================================

def get_loss_fn(name: str = "cross_entropy",
                gamma: float = 2.0,
                smoothing: float = 0.1,
                class_counts: list = None) -> nn.Module:
    """
    Factory function — returns the requested loss function.

    Args:
        name (str):    "cross_entropy" | "focal" | "label_smooth"
        gamma (float): Focal loss gamma parameter
        smoothing (float): Label smoothing epsilon
        class_counts (list): For weighted CE

    Returns:
        nn.Module: Loss function instance
    """
    name = name.lower().strip()

    if name in ("cross_entropy", "ce"):
        if class_counts:
            # Wrap weighted CE in a lambda-style module
            total   = sum(class_counts)
            weights = [total / (len(class_counts) * c + 1e-6)
                       for c in class_counts]
            return nn.CrossEntropyLoss(
                weight=torch.tensor(weights, dtype=torch.float32)
            )
        return nn.CrossEntropyLoss()

    elif name in ("focal", "focal_loss"):
        return FocalLoss(gamma=gamma)

    elif name in ("label_smooth", "label_smoothing", "ls"):
        return LabelSmoothingLoss(smoothing=smoothing)

    else:
        raise ValueError(
            f"Unknown loss: '{name}'. "
            f"Choose from: cross_entropy, focal, label_smooth"
        )


# =============================================================
# Self-test
# =============================================================
if __name__ == "__main__":
    print("Testing loss functions...")

    B, C = 8, 3
    logits  = torch.randn(B, C)
    targets = torch.randint(0, C, (B,))

    # Standard CE
    ce = nn.CrossEntropyLoss()(logits, targets)
    print(f"  CrossEntropy        : {ce.item():.4f}")

    # Focal Loss
    fl = FocalLoss(gamma=2.0)(logits, targets)
    print(f"  FocalLoss (γ=2)     : {fl.item():.4f}")

    # Label Smoothing
    ls = LabelSmoothingLoss(smoothing=0.1)(logits, targets)
    print(f"  LabelSmoothing (0.1): {ls.item():.4f}")

    # Weighted CE
    wce = weighted_cross_entropy(logits, targets, class_counts=[100, 60, 40])
    print(f"  WeightedCE          : {wce.item():.4f}")

    # Factory
    fn = get_loss_fn("focal", gamma=1.5)
    print(f"  Factory (focal)     : {fn(logits, targets).item():.4f}")

    print("\n✅ Loss function tests passed!")
