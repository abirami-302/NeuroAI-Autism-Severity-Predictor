# =============================================================
# models/gnn.py
# Graph Neural Network (GCN) for Brain Connectivity Analysis.
#
# This GNN takes a functional connectivity graph where:
#   - Nodes = Brain ROI regions (e.g., Amygdala, Frontal Lobe)
#   - Edges = Correlation between region activity signals
#   - Node features = Signal statistics per region
#
# It learns which connectivity patterns are associated with
# Mild / Moderate / Severe autism.
# =============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# PyTorch Geometric imports
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.config import (
    NUM_ROI, GNN_INPUT_DIM, GNN_HIDDEN_DIM,
    GNN_OUTPUT_DIM, GNN_NUM_LAYERS, GNN_DROPOUT,
    NUM_CLASSES, CNN_FEATURE_DIM
)


# =============================================================
# GNN BACKBONE: Graph Convolutional Network
# =============================================================

class BrainGNN(nn.Module):
    """
    Graph Convolutional Network for brain functional connectivity.

    Architecture:
        Input node features (NUM_ROI, GNN_INPUT_DIM)
            → GCNConv Layer 1 → ReLU → Dropout
            → GCNConv Layer 2 → ReLU → Dropout
            → GCNConv Layer 3 → ReLU
            → Global Mean Pool (graph-level embedding)
            → Linear → output embedding (GNN_OUTPUT_DIM)

    Args:
        input_dim (int):  Node feature dimension (default: 16)
        hidden_dim (int): Hidden layer size (default: 64)
        output_dim (int): Output embedding size (default: 32)
        num_layers (int): Number of GCN layers (default: 3)
        dropout (float):  Dropout probability (default: 0.3)
    """
    def __init__(self,
                 input_dim:  int   = GNN_INPUT_DIM,
                 hidden_dim: int   = GNN_HIDDEN_DIM,
                 output_dim: int   = GNN_OUTPUT_DIM,
                 num_layers: int   = GNN_NUM_LAYERS,
                 dropout:    float = GNN_DROPOUT):
        super(BrainGNN, self).__init__()

        self.num_layers = num_layers
        self.dropout    = dropout

        # ── GCN Layers ───────────────────────────────────────
        # Build a list of GCNConv layers dynamically
        self.convs = nn.ModuleList()

        # First layer: input_dim → hidden_dim
        self.convs.append(GCNConv(input_dim, hidden_dim))

        # Middle layers: hidden_dim → hidden_dim
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Last layer: hidden_dim → output_dim
        self.convs.append(GCNConv(hidden_dim, output_dim))

        # ── Batch Normalization per layer ────────────────────
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1)
        ])

        # ── Output projection ────────────────────────────────
        # Maps pooled graph embedding to final output dim
        self.output_proj = nn.Linear(output_dim, output_dim)

        # Store output dim for downstream use
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GNN.

        Args:
            x (torch.Tensor):          Node features (N, input_dim)
            edge_index (torch.Tensor): Graph connectivity (2, E)
            batch (torch.Tensor):      Batch assignment for each node

        Returns:
            torch.Tensor: Graph-level embedding (B, output_dim)
        """
        # Pass through each GCN layer
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)      # Graph convolution

            if i < self.num_layers - 1:
                # Apply BatchNorm and ReLU for all but the last layer
                x = self.batch_norms[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # ── Graph-level pooling ──────────────────────────────
        # Aggregate all node features into a single graph embedding
        # Mean pooling: average of all node features
        graph_embed = global_mean_pool(x, batch)   # (B, output_dim)

        # Project to final output
        graph_embed = self.output_proj(graph_embed)

        return graph_embed


# =============================================================
# COMBINED MODEL: CNN3D Features + GNN → Autism Severity
# =============================================================

class NeuroAIClassifier(nn.Module):
    """
    Full NeuroAI classification model combining:
        1. CNN-derived MRI features (from CNN3D)
        2. GNN-derived connectivity features (from BrainGNN)

    Both feature vectors are concatenated and passed through
    a final MLP for multi-class prediction.

    Input:
        - mri_features:  (B, CNN_FEATURE_DIM=128) from CNN3D
        - graph data:    PyG Data/Batch object with node features + edges

    Output:
        - logits: (B, NUM_CLASSES=3) — raw scores for Mild/Moderate/Severe
    """
    def __init__(self,
                 cnn_feature_dim: int = CNN_FEATURE_DIM,
                 num_classes: int     = NUM_CLASSES):
        super(NeuroAIClassifier, self).__init__()

        # GNN processes the brain connectivity graph
        self.gnn = BrainGNN()

        # Combined feature size = CNN features + GNN embedding
        combined_dim = cnn_feature_dim + self.gnn.output_dim  # 128 + 32 = 160

        # ── Final MLP Classifier ─────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)   # Raw logits
        )

    def forward(self,
                mri_features: torch.Tensor,
                graph_data: Data) -> torch.Tensor:
        """
        Args:
            mri_features (torch.Tensor): (B, 128) from CNN3D
            graph_data (Data):           PyG graph with x, edge_index, batch

        Returns:
            torch.Tensor: (B, num_classes) logits
        """
        # Extract GNN embedding from brain connectivity graph
        gnn_embed = self.gnn(
            graph_data.x,
            graph_data.edge_index,
            graph_data.batch
        )                                # (B, 32)

        # Concatenate CNN and GNN features
        combined = torch.cat([mri_features, gnn_embed], dim=1)  # (B, 160)

        # Classify
        logits = self.classifier(combined)  # (B, 3)

        return logits


# =============================================================
# STANDALONE GNN CLASSIFIER (without CNN)
# Used when we only have connectivity features (no raw MRI)
# =============================================================

class GNNOnlyClassifier(nn.Module):
    """
    GNN-only classifier for brain connectivity graphs.
    Useful for ablation studies or when raw MRI is unavailable.
    """
    def __init__(self, num_classes: int = NUM_CLASSES):
        super(GNNOnlyClassifier, self).__init__()
        self.gnn = BrainGNN()
        self.head = nn.Sequential(
            nn.Linear(GNN_OUTPUT_DIM, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, graph_data: Data) -> torch.Tensor:
        embed  = self.gnn(graph_data.x, graph_data.edge_index, graph_data.batch)
        logits = self.head(embed)
        return logits


# =============================================================
# Utility: Create a dummy PyG graph for testing
# =============================================================

def create_dummy_graph(num_nodes: int = NUM_ROI,
                       input_dim: int  = GNN_INPUT_DIM,
                       batch_size: int = 2) -> Data:
    """
    Create a synthetic PyG graph batch for testing.

    Args:
        num_nodes (int):  Number of ROI nodes per graph
        input_dim (int):  Node feature dimension
        batch_size (int): Number of graphs in batch

    Returns:
        torch_geometric.data.Batch: Batched graph object
    """
    graphs = []
    for _ in range(batch_size):
        # Random node features
        x = torch.randn(num_nodes, input_dim)

        # Fully connected graph (every ROI connected to every other)
        # In practice, this is thresholded by connectivity strength
        edges = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edges.append([i, j])

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        graphs.append(Data(x=x, edge_index=edge_index))

    return Batch.from_data_list(graphs)


# =============================================================
# Self-test
# =============================================================
if __name__ == "__main__":
    print("Testing BrainGNN...")

    # Create dummy graph batch
    graph_batch = create_dummy_graph(batch_size=2)
    print(f"  Nodes per batch : {graph_batch.x.shape}")
    print(f"  Edge index shape: {graph_batch.edge_index.shape}")

    gnn = BrainGNN()
    gnn.eval()

    with torch.no_grad():
        embed = gnn(graph_batch.x, graph_batch.edge_index, graph_batch.batch)

    print(f"  GNN output shape: {embed.shape}")   # (2, 32)

    print("\nTesting NeuroAIClassifier (CNN + GNN)...")
    model = NeuroAIClassifier()
    model.eval()

    # Fake CNN features (batch of 2)
    mri_feats = torch.randn(2, CNN_FEATURE_DIM)

    with torch.no_grad():
        logits = model(mri_feats, graph_batch)

    probs = torch.softmax(logits, dim=1)
    print(f"  Logits shape    : {logits.shape}")   # (2, 3)
    print(f"  Probabilities   :\n{probs}")

    print("\n✅ GNN tests passed!")
