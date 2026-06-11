# =============================================================
# models/classifier.py
# Final Multi-Class Classification Head for NeuroAI.
#
# This module defines the MLP head that takes the fused
# feature vector (CNN + GNN) and produces class logits for:
#   0 → Mild Autism
#   1 → Moderate Autism
#   2 → Severe Autism
#
# Also provides the full end-to-end NeuroAIPipeline model
# that wires CNN3D → BrainGNN → ClassifierHead together.
# =============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.config import (
    CNN_FEATURE_DIM, GNN_OUTPUT_DIM, NUM_CLASSES, CLASS_NAMES
)


# =============================================================
# CLASSIFIER HEAD (MLP)
# Takes a concatenated feature vector and predicts severity.
# =============================================================

class ClassifierHead(nn.Module):
    """
    Multi-layer perceptron classification head.

    Input:  concatenated [CNN features | GNN embedding]
            default size = CNN_FEATURE_DIM + GNN_OUTPUT_DIM = 128 + 32 = 160
    Output: raw logits for [Mild, Moderate, Severe]

    Architecture:
        Linear(160 → 128) → ReLU → Dropout(0.4)
        Linear(128 → 64)  → ReLU → Dropout(0.3)
        Linear(64  → 32)  → ReLU
        Linear(32  → 3)         ← raw logits

    Args:
        input_dim (int):   Size of the concatenated feature vector
        hidden_dims (list): Hidden layer sizes
        num_classes (int): Output classes (default 3)
        dropout (float):   Dropout probability
    """
    def __init__(self,
                 input_dim:   int  = CNN_FEATURE_DIM + GNN_OUTPUT_DIM,
                 hidden_dims: list = None,
                 num_classes: int  = NUM_CLASSES,
                 dropout:     float = 0.35):
        super(ClassifierHead, self).__init__()

        hidden_dims = hidden_dims or [128, 64, 32]

        # Build MLP layers dynamically from hidden_dims list
        layers = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ]
            in_dim = h_dim

        # Final output layer — no activation (CrossEntropyLoss handles it)
        layers.append(nn.Linear(in_dim, num_classes))

        self.mlp = nn.Sequential(*layers)

        # Weight initialisation
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Fused features (B, input_dim)
        Returns:
            torch.Tensor: Raw logits (B, num_classes)
        """
        return self.mlp(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience method: return softmax probabilities.

        Args:
            x (torch.Tensor): Fused features (B, input_dim)
        Returns:
            torch.Tensor: Class probabilities (B, num_classes)
        """
        return F.softmax(self.forward(x), dim=1)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)


# =============================================================
# FULL PIPELINE MODEL: CNN3D + GNN + ClassifierHead
# One unified module for end-to-end training & inference.
# =============================================================

class NeuroAIPipeline(nn.Module):
    """
    Complete end-to-end NeuroAI model.

    Wires together:
        1. CNN3D        — 3D spatial feature extractor
        2. BrainGNN     — graph-based connectivity analyser
        3. ClassifierHead — MLP for multi-class prediction

    Input:
        mri_volume (torch.Tensor): (B, 1, D, H, W)
        graph_data (PyG Data):     brain connectivity graph

    Output:
        logits (torch.Tensor): (B, num_classes)

    Usage:
        model = NeuroAIPipeline()
        logits = model(mri_volume, graph_data)
        probs  = torch.softmax(logits, dim=1)
    """
    def __init__(self, num_classes: int = NUM_CLASSES):
        super(NeuroAIPipeline, self).__init__()

        from models.cnn3d import CNN3D
        from models.gnn   import BrainGNN

        # Three sub-modules
        self.cnn        = CNN3D()                          # → (B, 128)
        self.gnn        = BrainGNN()                       # → (B, 32)
        self.classifier = ClassifierHead(
            input_dim   = self.cnn.feature_dim + self.gnn.output_dim,
            num_classes = num_classes
        )

    def forward(self, mri_volume, graph_data) -> torch.Tensor:
        """
        Args:
            mri_volume (torch.Tensor): (B, 1, 64, 64, 64)
            graph_data (PyG Data):     .x, .edge_index, .batch

        Returns:
            torch.Tensor: (B, num_classes) logits
        """
        # 1. Extract spatial features from MRI volume
        cnn_feats = self.cnn(mri_volume)                  # (B, 128)

        # 2. Extract connectivity features from brain graph
        gnn_embed = self.gnn(
            graph_data.x,
            graph_data.edge_index,
            graph_data.batch
        )                                                  # (B, 32)

        # 3. Concatenate and classify
        fused  = torch.cat([cnn_feats, gnn_embed], dim=1) # (B, 160)
        logits = self.classifier(fused)                   # (B, 3)

        return logits


# =============================================================
# PREDICTION HELPER
# =============================================================

def decode_prediction(logits: torch.Tensor) -> dict:
    """
    Convert raw model logits into a human-readable prediction dict.

    Args:
        logits (torch.Tensor): (1, num_classes) from model forward pass

    Returns:
        dict with keys:
            label (str):         "Mild" / "Moderate" / "Severe"
            confidence (float):  0–100 percentage
            probabilities (dict): class → % mapping
    """
    probs      = F.softmax(logits, dim=1)[0]              # (num_classes,)
    pred_idx   = probs.argmax().item()
    pred_label = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx]) * 100

    return {
        "label":         pred_label,
        "confidence":    round(confidence, 1),
        "probabilities": {
            CLASS_NAMES[i]: round(float(probs[i]) * 100, 1)
            for i in range(len(CLASS_NAMES))
        }
    }


# =============================================================
# Self-test
# =============================================================
if __name__ == "__main__":
    print("Testing ClassifierHead...")
    head = ClassifierHead()
    head.eval()
    x = torch.randn(4, CNN_FEATURE_DIM + GNN_OUTPUT_DIM)
    with torch.no_grad():
        logits = head(x)
    print(f"  Logits shape : {logits.shape}")   # (4, 3)

    result = decode_prediction(logits[:1])
    print(f"  Prediction   : {result['label']} ({result['confidence']}%)")

    print("\nTesting NeuroAIPipeline...")
    from models.gnn import create_dummy_graph
    pipeline = NeuroAIPipeline()
    pipeline.eval()
    vol   = torch.randn(2, 1, 64, 64, 64)
    graph = create_dummy_graph(batch_size=2)
    with torch.no_grad():
        out = pipeline(vol, graph)
    print(f"  Pipeline output : {out.shape}")   # (2, 3)

    print("\n✅ Classifier tests passed!")
