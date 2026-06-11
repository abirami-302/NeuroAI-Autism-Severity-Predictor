# =============================================================
# graph/graph_builder.py
# Brain Functional Connectivity Graph Construction.
#
# Pipeline:
#   MRI Volume → ROI Signal Extraction → Correlation Matrix
#   → Thresholded Adjacency Matrix → PyTorch Geometric Graph
#
# Each brain ROI becomes a node, and edges represent strong
# functional connectivity (correlation) between regions.
# =============================================================

import numpy as np
import torch
from torch_geometric.data import Data
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.config import (
    NUM_ROI, GNN_INPUT_DIM, CONNECTIVITY_THRESHOLD, ROI_NAMES, IMAGE_SIZE
)


# =============================================================
# STEP 1 — ROI SIGNAL EXTRACTION
# Divide the 3D MRI volume into ROI regions and extract
# a 1D signal (mean intensity time series) per region.
# =============================================================

def _skew_approx(arr: np.ndarray) -> float:
    """Simple skewness approximation: (mean - median) / std."""
    std = arr.std() + 1e-8
    return float((arr.mean() - np.median(arr)) / std)

# Attach helper to numpy namespace so it reads naturally inside the loop
np.skew_approx = _skew_approx


def extract_roi_signals(volume: np.ndarray,
                        num_roi: int = NUM_ROI) -> np.ndarray:
    """
    Extract average signal per brain ROI from an MRI volume.

    Strategy: Divide the 3D volume into a regular grid of
    num_roi cuboid regions and take the mean voxel value
    from each region as its representative signal.

    In a full clinical system this would use a proper brain
    atlas (e.g., AAL, Schaefer) for anatomically accurate ROIs.

    Args:
        volume (np.ndarray): Preprocessed volume, shape (1, D, H, W)
                             or (D, H, W)
        num_roi (int):       Number of ROI regions

    Returns:
        np.ndarray: ROI signal matrix, shape (num_roi, time_points)
                    For structural MRI: (num_roi, num_features)
    """
    # Remove channel dim if present: (1, D, H, W) → (D, H, W)
    if volume.ndim == 4:
        volume = volume[0]

    D, H, W = volume.shape

    # We'll extract GNN_INPUT_DIM features per ROI
    # by splitting the volume into spatial sub-volumes
    num_features = GNN_INPUT_DIM  # 16 features per ROI

    roi_signals = np.zeros((num_roi, num_features), dtype=np.float32)

    # Grid partition: divide D dimension into num_roi slabs
    slab_size = D // num_roi

    for i in range(num_roi):
        start = i * slab_size
        end   = start + slab_size if i < num_roi - 1 else D

        # Extract the i-th ROI slab
        roi_volume = volume[start:end, :, :]   # (slab, H, W)

        # Compute statistical features from this ROI
        flat = roi_volume.flatten()

        # Feature set: mean, std, min, max, percentiles, etc.
        features = [
            np.mean(flat),                      # F1: mean intensity
            np.std(flat),                       # F2: std deviation
            np.min(flat),                       # F3: minimum
            np.max(flat),                       # F4: maximum
            np.percentile(flat, 25),            # F5: 25th percentile
            np.percentile(flat, 50),            # F6: median
            np.percentile(flat, 75),            # F7: 75th percentile
            np.percentile(flat, 90),            # F8: 90th percentile
            np.sum(flat > 0) / len(flat),       # F9: fraction positive
            np.sum(flat > flat.mean()) / len(flat),  # F10: above mean ratio
            np.mean(np.abs(np.diff(flat[:100]))),    # F11: local variation
            np.var(flat),                       # F12: variance
            float(np.count_nonzero(flat)),      # F13: non-zero count
            np.mean(flat ** 2),                 # F14: mean squared
            np.skew_approx(flat),               # F15: skewness (custom)
            float(roi_volume.shape[0]),         # F16: slab depth
        ]

        roi_signals[i] = np.array(features[:num_features], dtype=np.float32)

    # Normalize ROI features to zero mean, unit variance
    for f in range(num_features):
        col = roi_signals[:, f]
        std = col.std() + 1e-8
        roi_signals[:, f] = (col - col.mean()) / std

    return roi_signals   # Shape: (num_roi, num_features)


# =============================================================
# STEP 2 — FUNCTIONAL CONNECTIVITY MATRIX
# Pearson correlation between all pairs of ROI signals.
# Result: (NUM_ROI × NUM_ROI) symmetric matrix
# =============================================================

def build_connectivity_matrix(roi_signals: np.ndarray) -> np.ndarray:
    """
    Build a functional connectivity matrix from ROI signals.

    Computes Pearson correlation between every pair of ROI
    feature vectors. Values near 1 indicate strong positive
    coupling; values near -1 indicate anti-correlation.

    Args:
        roi_signals (np.ndarray): Shape (num_roi, num_features)

    Returns:
        np.ndarray: Correlation matrix, shape (num_roi, num_roi)
                    Values in [-1, 1]
    """
    num_roi = roi_signals.shape[0]
    conn_matrix = np.zeros((num_roi, num_roi), dtype=np.float32)

    for i in range(num_roi):
        for j in range(num_roi):
            if i == j:
                conn_matrix[i, j] = 1.0   # Self-correlation = 1
            else:
                # Pearson correlation between ROI i and ROI j
                xi = roi_signals[i]
                xj = roi_signals[j]
                # np.corrcoef returns 2×2 matrix; we want [0,1] element
                corr = np.corrcoef(xi, xj)[0, 1]
                # Replace NaN (if std=0) with 0
                conn_matrix[i, j] = corr if not np.isnan(corr) else 0.0

    return conn_matrix


# =============================================================
# STEP 3 — ADJACENCY MATRIX
# Apply threshold to connectivity matrix to get binary edges
# =============================================================

def build_adjacency_matrix(conn_matrix: np.ndarray,
                            threshold: float = CONNECTIVITY_THRESHOLD
                            ) -> np.ndarray:
    """
    Convert correlation matrix to a thresholded adjacency matrix.

    An edge exists between ROI i and j if:
        |correlation(i,j)| > threshold

    Args:
        conn_matrix (np.ndarray): Correlation matrix (num_roi, num_roi)
        threshold (float):        Minimum absolute correlation for an edge

    Returns:
        np.ndarray: Binary adjacency matrix (num_roi, num_roi), dtype float32
    """
    # Use absolute correlation (both positive and negative are meaningful)
    adj = (np.abs(conn_matrix) > threshold).astype(np.float32)

    # Remove self-loops (diagonal = 0)
    np.fill_diagonal(adj, 0)

    return adj


# =============================================================
# STEP 4 — CONVERT TO PyTorch Geometric Graph
# =============================================================

def build_pyg_graph(roi_signals: np.ndarray,
                    adj_matrix: np.ndarray) -> Data:
    """
    Convert ROI signals and adjacency matrix to a
    PyTorch Geometric (PyG) Data object.

    PyG Data format:
        data.x          → Node feature matrix (num_roi, num_features)
        data.edge_index → Edge list in COO format (2, num_edges)
        data.edge_attr  → Edge weights (correlation values)
        data.num_nodes  → Number of nodes (= num_roi)

    Args:
        roi_signals (np.ndarray): Node features (num_roi, num_features)
        adj_matrix (np.ndarray):  Binary adjacency matrix (num_roi, num_roi)

    Returns:
        torch_geometric.data.Data: Ready-to-use PyG graph
    """
    # ── Node features ────────────────────────────────────────
    x = torch.tensor(roi_signals, dtype=torch.float)   # (num_roi, 16)

    # ── Edge list (COO format) ───────────────────────────────
    # Find all (i, j) pairs where adj_matrix[i,j] = 1
    rows, cols = np.where(adj_matrix > 0)
    edge_index = torch.tensor(
        np.stack([rows, cols], axis=0), dtype=torch.long
    )   # Shape: (2, num_edges)

    # ── Edge weights (absolute correlation values) ───────────
    # These tell the GNN how strong each connection is
    num_roi = roi_signals.shape[0]
    conn_matrix = build_connectivity_matrix(roi_signals)
    edge_weights = []
    for r, c in zip(rows, cols):
        edge_weights.append(abs(conn_matrix[r, c]))

    edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)

    # ── Build PyG Data object ────────────────────────────────
    graph = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=num_roi
    )

    return graph


# =============================================================
# FULL GRAPH PIPELINE
# Volume → ROI signals → Connectivity → Graph
# =============================================================

def volume_to_graph(volume: np.ndarray,
                    threshold: float = CONNECTIVITY_THRESHOLD) -> Data:
    """
    Convert a preprocessed MRI volume into a PyG brain graph.

    Full pipeline:
        1. Extract ROI signals (one per brain region)
        2. Compute Pearson correlation matrix
        3. Threshold → adjacency matrix
        4. Convert to PyTorch Geometric Data object

    Args:
        volume (np.ndarray): Preprocessed MRI (1, D, H, W) or (D, H, W)
        threshold (float):   Connectivity threshold

    Returns:
        torch_geometric.data.Data: Brain graph ready for GNN
    """
    # Step 1: ROI signals
    roi_signals = extract_roi_signals(volume)

    # Step 2: Connectivity matrix
    conn_matrix = build_connectivity_matrix(roi_signals)

    # Step 3: Adjacency matrix
    adj_matrix  = build_adjacency_matrix(conn_matrix, threshold=threshold)

    # Step 4: PyG graph
    graph = build_pyg_graph(roi_signals, adj_matrix)

    return graph, conn_matrix   # Return conn_matrix for visualization


def get_top_connected_regions(conn_matrix: np.ndarray,
                               roi_names: list = ROI_NAMES,
                               top_k: int = 3) -> list:
    """
    Identify the most highly connected (anomalous) brain regions.
    Used to display "Affected Brain Regions" in the UI.

    Args:
        conn_matrix (np.ndarray): Correlation matrix (num_roi, num_roi)
        roi_names (list):         Names of ROI regions
        top_k (int):              Number of top regions to return

    Returns:
        list: Top-k region names sorted by total connectivity strength
    """
    # Sum absolute correlations for each ROI (exclude self)
    total_connectivity = np.abs(conn_matrix).sum(axis=1)
    np.fill_diagonal(conn_matrix, 0)  # Exclude self

    # Get top-k indices
    top_indices = np.argsort(total_connectivity)[::-1][:top_k]

    return [roi_names[i] for i in top_indices]


# =============================================================
# Self-test
# =============================================================
if __name__ == "__main__":
    print("Testing graph builder pipeline...")

    # Simulate a preprocessed MRI volume
    fake_volume = np.random.randn(1, 64, 64, 64).astype(np.float32)

    # Run full pipeline
    graph, conn_matrix = volume_to_graph(fake_volume)

    print(f"  Node features  : {graph.x.shape}")          # (10, 16)
    print(f"  Edge index     : {graph.edge_index.shape}")  # (2, E)
    print(f"  Edge weights   : {graph.edge_attr.shape}")
    print(f"  Num nodes      : {graph.num_nodes}")
    print(f"  Num edges      : {graph.edge_index.shape[1]}")
    print(f"\n  Connectivity matrix ({conn_matrix.shape}):")
    print(np.round(conn_matrix, 2))

    top_regions = get_top_connected_regions(conn_matrix)
    print(f"\n  Top connected regions: {top_regions}")

    print("\n✅ Graph builder tests passed!")
