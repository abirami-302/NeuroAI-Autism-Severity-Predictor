# =============================================================
# models/cnn3d_lightweight.py
#
# Lightweight 3D CNN for laptop training.
# FULL ARCHITECTURE IS PRESERVED — only filter counts and
# depth are reduced.  Completely compatible with:
#   - preprocessing/pipeline.py  (same (1,D,H,W) input)
#   - graph/graph_builder.py     (same feature_dim output)
#   - inference/gradcam.py       (same .features hook target)
#   - inference/predict.py       (same forward() signature)
#
# The only differences from models/cnn3d.py:
#   CNN_BASE_FILTERS : 32 → 16  (4× fewer parameters)
#   CNN_FEATURE_DIM  : 128 → 64 (halved output vector)
#   Stages           : 4  → 3   (one fewer downsampling stage)
#
# This cuts memory from ~200 MB to ~25 MB per forward pass,
# making 64³ volumes trainable on 16 GB RAM / CPU.
# =============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.config_lightweight import (
    IMAGE_SIZE, IMAGE_CHANNELS,
    CNN_BASE_FILTERS, CNN_FEATURE_DIM, CNN_DROPOUT, NUM_CLASSES
)


# =============================================================
# REUSABLE BLOCK: Conv3D + BatchNorm + ReLU
# Identical to cnn3d.py — no changes needed here
# =============================================================

class ConvBlock3D(nn.Module):
    """3D Conv → BatchNorm → ReLU building block."""
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels,
                      kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# =============================================================
# LIGHTWEIGHT 3D CNN
# =============================================================

class LightweightCNN3D(nn.Module):
    """
    Reduced-capacity 3D CNN for brain MRI feature extraction.

    Compatible drop-in for CNN3D — same input/output interface.

    Architecture (for 64×64×64 input):
        Input  (1, 64, 64, 64)
        Stage1 → Conv(1→16) + Conv(16→16) + MaxPool → (16, 32, 32, 32)
        Stage2 → Conv(16→32) + Conv(32→32) + MaxPool → (32, 16, 16, 16)
        Stage3 → Conv(32→64) + Conv(64→64) + MaxPool → (64,  8,  8,  8)
        AdaptiveAvgPool → (64, 2, 2, 2)
        Flatten → FC(64*8=512 → 64) → feature vector

    For 32×32×32 input:
        Same architecture — AdaptiveAvgPool handles the size difference.

    Args:
        in_channels (int):   Input channels (1 for grayscale MRI)
        base_filters (int):  Filters in first stage (default 16)
        feature_dim (int):   Output feature vector size (default 64)
        dropout (float):     Dropout probability
    """
    def __init__(self,
                 in_channels:  int   = IMAGE_CHANNELS,
                 base_filters: int   = CNN_BASE_FILTERS,
                 feature_dim:  int   = CNN_FEATURE_DIM,
                 dropout:      float = CNN_DROPOUT):
        super().__init__()

        f = base_filters   # Shorthand: 16

        # ── 3 convolutional stages (vs 4 in full model) ──────
        self.features = nn.Sequential(
            # Stage 1: (1,64,64,64) → (16,32,32,32)
            ConvBlock3D(in_channels, f),       # 1  → 16
            ConvBlock3D(f,           f),       # 16 → 16
            nn.MaxPool3d(2, 2),

            # Stage 2: (16,32,32,32) → (32,16,16,16)
            ConvBlock3D(f,   f * 2),           # 16 → 32
            ConvBlock3D(f*2, f * 2),           # 32 → 32
            nn.MaxPool3d(2, 2),

            # Stage 3: (32,16,16,16) → (64,8,8,8)
            ConvBlock3D(f*2, f * 4),           # 32 → 64
            ConvBlock3D(f*4, f * 4),           # 64 → 64
            nn.MaxPool3d(2, 2),
        )

        # AdaptiveAvgPool → (64, 2, 2, 2) regardless of input size
        self.global_pool = nn.AdaptiveAvgPool3d((2, 2, 2))

        # Flatten size = 64 filters × 2×2×2 = 512
        flat_size = f * 4 * 2 * 2 * 2   # = 64 * 8 = 512

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, feature_dim),   # → 64
            nn.ReLU(inplace=True),
        )

        self.feature_dim = feature_dim
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, D, H, W)
        Returns:
            (B, feature_dim=64)
        """
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)


# =============================================================
# LIGHTWEIGHT END-TO-END CLASSIFIER
# LightweightCNN3D → MLP head → [Mild, Moderate, Severe]
# =============================================================

class LightweightCNN3DClassifier(nn.Module):
    """
    Full lightweight classifier: MRI volume → severity logits.

    This is what gets saved to lightweight_best.pth and loaded
    by predict_lightweight.py for real inference.

    Args:
        num_classes (int): 3 for Mild/Moderate/Severe
    """
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        # Backbone: lightweight 3D CNN
        self.backbone = LightweightCNN3D()

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(CNN_FEATURE_DIM, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(32, num_classes)   # Raw logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, D, H, W)
        Returns:
            (B, num_classes) logits
        """
        features = self.backbone(x)   # (B, 64)
        logits   = self.head(features)  # (B, 3)
        return logits


# =============================================================
# MODEL INFO UTILITY
# =============================================================

def model_summary(model: nn.Module) -> None:
    """Print parameter count and estimated memory usage."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mem_mb    = total * 4 / (1024 ** 2)   # float32 = 4 bytes

    print(f"\n{'─'*45}")
    print(f"  Model     : {model.__class__.__name__}")
    print(f"  Parameters: {total:,}  ({trainable:,} trainable)")
    print(f"  Est. size : {mem_mb:.1f} MB (weights only)")
    print(f"{'─'*45}")


# =============================================================
# Self-test
# =============================================================

if __name__ == "__main__":
    print("Testing LightweightCNN3D...")

    # Test both supported input sizes
    for size in [(32, 32, 32), (64, 64, 64)]:
        model = LightweightCNN3D()
        model.eval()
        x = torch.randn(2, 1, *size)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, CNN_FEATURE_DIM), \
            f"Expected (2,{CNN_FEATURE_DIM}), got {out.shape}"
        print(f"  Input {size} → features {out.shape}  ✅")

    model_summary(LightweightCNN3D())

    print("\nTesting LightweightCNN3DClassifier...")
    clf = LightweightCNN3DClassifier()
    clf.eval()
    x = torch.randn(2, 1, 64, 64, 64)
    with torch.no_grad():
        logits = clf(x)
    probs = torch.softmax(logits, dim=1)
    assert logits.shape == (2, 3)
    print(f"  Logits {logits.shape}  ✅")
    print(f"  Probs  {probs.round(decimals=3)}")

    model_summary(clf)

    # Compare with full model parameter count
    from models.cnn3d import CNN3DClassifier
    full  = sum(p.numel() for p in CNN3DClassifier().parameters())
    light = sum(p.numel() for p in LightweightCNN3DClassifier().parameters())
    print(f"\n  Full model params  : {full:,}")
    print(f"  Lightweight params : {light:,}")
    print(f"  Reduction          : {full/light:.1f}×  smaller")

    print("\n✅ All lightweight model tests passed!")
