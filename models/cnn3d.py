# =============================================================
# models/cnn3d.py — 3D CNN (NumPy mock implementation)
# Full PyTorch version activates automatically if torch is present.
# Falls back to a clean NumPy-based mock for demo/testing.
# =============================================================

import numpy as np
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.config import (IMAGE_CHANNELS, CNN_BASE_FILTERS,
                              CNN_FEATURE_DIM, CNN_DROPOUT, NUM_CLASSES)

# ── Try PyTorch first ─────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================
# NUMPY-BASED MOCK (always available, used for demo)
# =============================================================

class MockCNN3D:
    """
    Deterministic mock of 3D CNN feature extractor.
    Extracts hand-crafted spatial statistics from the MRI volume
    to simulate a trained CNN's feature vector output.
    """
    def __init__(self, feature_dim: int = CNN_FEATURE_DIM):
        self.feature_dim = feature_dim
        # Fixed random projection matrix (simulates learned weights)
        rng = np.random.RandomState(42)
        self.W = rng.randn(feature_dim, 512).astype(np.float32) * 0.1

    def extract_features(self, volume: np.ndarray) -> np.ndarray:
        """
        volume: (1, D, H, W) or (D, H, W)
        returns: (feature_dim,) feature vector
        """
        if volume.ndim == 4:
            vol = volume[0]
        else:
            vol = volume
        D, H, W = vol.shape

        # Extract 512 statistical features across spatial subdivisions
        feats = []
        for axis in range(3):
            slabs = np.array_split(vol, 8, axis=axis)
            for slab in slabs:
                f = slab.flatten()
                feats += [f.mean(), f.std(),
                          np.percentile(f, 25), np.percentile(f, 75)]
        raw = np.array(feats[:512], dtype=np.float32)

        # Normalize raw features
        raw = (raw - raw.mean()) / (raw.std() + 1e-8)

        # Project to feature_dim via fixed weights
        feat_vec = self.W @ raw                        # (feature_dim,)
        feat_vec = np.tanh(feat_vec)                   # non-linearity
        return feat_vec.astype(np.float32)

    def __call__(self, volume: np.ndarray) -> np.ndarray:
        return self.extract_features(volume)


class MockCNN3DClassifier:
    """
    Mock end-to-end CNN classifier.
    Returns deterministic class probabilities for a given volume.
    """
    def __init__(self, num_classes: int = NUM_CLASSES,
                 feature_dim: int = CNN_FEATURE_DIM):
        self.backbone   = MockCNN3D(feature_dim)
        self.num_classes = num_classes
        rng = np.random.RandomState(7)
        self.W2 = rng.randn(num_classes, feature_dim).astype(np.float32) * 0.3
        self.b2 = rng.randn(num_classes).astype(np.float32) * 0.1

    def forward(self, volume: np.ndarray) -> np.ndarray:
        """Returns (num_classes,) probability vector."""
        feat   = self.backbone(volume)               # (128,)
        logits = self.W2 @ feat + self.b2            # (3,)
        probs  = self._softmax(logits)
        return probs

    def _softmax(self, x):
        e = np.exp(x - x.max())
        return e / e.sum()

    def __call__(self, volume: np.ndarray) -> np.ndarray:
        return self.forward(volume)


# =============================================================
# PYTORCH VERSION (activated only when torch is installed)
# =============================================================

if TORCH_AVAILABLE:
    import torch.nn as nn

    class ConvBlock3D(nn.Module):
        def __init__(self, in_ch, out_ch, k=3, p=1):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, k, padding=p, bias=False),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True)
            )
        def forward(self, x): return self.block(x)

    class CNN3D(nn.Module):
        def __init__(self, in_channels=IMAGE_CHANNELS,
                     base_filters=CNN_BASE_FILTERS,
                     feature_dim=CNN_FEATURE_DIM,
                     dropout=CNN_DROPOUT):
            super().__init__()
            bf = base_filters
            self.features = nn.Sequential(
                ConvBlock3D(in_channels, bf),   ConvBlock3D(bf, bf),
                nn.MaxPool3d(2, 2),
                ConvBlock3D(bf, bf*2),          ConvBlock3D(bf*2, bf*2),
                nn.MaxPool3d(2, 2),
                ConvBlock3D(bf*2, bf*4),        ConvBlock3D(bf*4, bf*4),
                nn.MaxPool3d(2, 2),
                ConvBlock3D(bf*4, bf*8),        ConvBlock3D(bf*8, bf*8),
                nn.MaxPool3d(2, 2),
            )
            self.global_pool = nn.AdaptiveAvgPool3d((2, 2, 2))
            flat = bf * 8 * 8
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat, 512), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(512, feature_dim), nn.ReLU(),
            )
            self.feature_dim = feature_dim
        def forward(self, x):
            x = self.features(x)
            x = self.global_pool(x)
            return self.classifier(x)

    class CNN3DClassifier(nn.Module):
        def __init__(self, num_classes=NUM_CLASSES):
            super().__init__()
            self.backbone = CNN3D()
            self.head = nn.Sequential(
                nn.Linear(CNN_FEATURE_DIM, 64), nn.ReLU(),
                nn.Dropout(0.3), nn.Linear(64, num_classes)
            )
        def forward(self, x):
            return self.head(self.backbone(x))

else:
    # Aliases so imports don't break
    CNN3D           = MockCNN3D
    CNN3DClassifier = MockCNN3DClassifier


# ── Factory function used by the rest of the codebase ────────

def get_classifier(num_classes: int = NUM_CLASSES):
    """Return the best available classifier (PyTorch or Mock)."""
    if TORCH_AVAILABLE:
        return CNN3DClassifier(num_classes)
    return MockCNN3DClassifier(num_classes)


if __name__ == "__main__":
    print(f"Torch available: {TORCH_AVAILABLE}")
    clf = MockCNN3DClassifier()
    vol = np.random.randn(1, 64, 64, 64).astype(np.float32)
    probs = clf(vol)
    print(f"Probs: {probs}  sum={probs.sum():.4f}")
    print("✅ cnn3d.py OK")
