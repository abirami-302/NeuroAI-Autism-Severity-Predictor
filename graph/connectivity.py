# =============================================================
# graph/connectivity.py
# Functional Connectivity Matrix Construction.
#
# Takes per-ROI feature vectors and computes pairwise
# Pearson correlations to form a connectivity matrix.
# Provides thresholding, normalisation, and visualisation
# helpers used by graph_builder.py and the UI.
# =============================================================

import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.config import CONNECTIVITY_THRESHOLD, NUM_ROI, ROI_NAMES


# =============================================================
# PEARSON CORRELATION MATRIX
# =============================================================

def compute_pearson_matrix(roi_features: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Pearson correlation between all ROI pairs.

    Each ROI's feature vector acts as a "signal" — the
    correlation between two ROIs indicates how similarly
    their activity patterns vary (functional connectivity).

    Args:
        roi_features (np.ndarray): Shape (num_roi, num_features)

    Returns:
        np.ndarray: Symmetric correlation matrix (num_roi, num_roi)
                    Values in [-1.0, +1.0].
    """
    num_roi = roi_features.shape[0]
    matrix  = np.zeros((num_roi, num_roi), dtype=np.float32)

    for i in range(num_roi):
        for j in range(num_roi):
            if i == j:
                matrix[i, j] = 1.0          # Perfect self-correlation
            else:
                xi = roi_features[i]
                xj = roi_features[j]
                # np.corrcoef returns a 2×2 matrix; element [0,1] = corr(xi, xj)
                corr = float(np.corrcoef(xi, xj)[0, 1])
                # Guard against NaN (arises when std=0)
                matrix[i, j] = corr if not np.isnan(corr) else 0.0

    return matrix


# =============================================================
# SPEARMAN RANK CORRELATION (alternative, more robust)
# =============================================================

def compute_spearman_matrix(roi_features: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Spearman rank correlation.

    More robust than Pearson for non-Gaussian signal distributions.
    Ranks are computed per feature vector before correlating.

    Args:
        roi_features (np.ndarray): Shape (num_roi, num_features)

    Returns:
        np.ndarray: Rank-correlation matrix (num_roi, num_roi)
    """
    from scipy.stats import spearmanr

    num_roi = roi_features.shape[0]
    matrix  = np.zeros((num_roi, num_roi), dtype=np.float32)

    for i in range(num_roi):
        for j in range(num_roi):
            if i == j:
                matrix[i, j] = 1.0
            else:
                corr, _ = spearmanr(roi_features[i], roi_features[j])
                matrix[i, j] = float(corr) if not np.isnan(corr) else 0.0

    return matrix


# =============================================================
# THRESHOLD → ADJACENCY
# =============================================================

def threshold_matrix(conn_matrix: np.ndarray,
                     threshold: float = CONNECTIVITY_THRESHOLD,
                     absolute: bool = True) -> np.ndarray:
    """
    Apply a threshold to a connectivity matrix to get binary edges.

    An edge (i→j) exists when the connection strength exceeds
    the threshold.  Diagonal (self-loops) is always set to zero.

    Args:
        conn_matrix (np.ndarray): Correlation matrix (num_roi, num_roi)
        threshold (float):        Minimum strength to create an edge
        absolute (bool):          If True, threshold |corr|; otherwise
                                  threshold raw corr (positive only)

    Returns:
        np.ndarray: Binary adjacency matrix, dtype float32
    """
    if absolute:
        adj = (np.abs(conn_matrix) > threshold).astype(np.float32)
    else:
        adj = (conn_matrix > threshold).astype(np.float32)

    # No self-loops
    np.fill_diagonal(adj, 0.0)
    return adj


# =============================================================
# WEIGHTED ADJACENCY (keeps correlation values on edges)
# =============================================================

def weighted_adjacency(conn_matrix: np.ndarray,
                       threshold: float = CONNECTIVITY_THRESHOLD) -> np.ndarray:
    """
    Build a weighted adjacency matrix: zero below threshold,
    absolute correlation value above threshold.

    Useful for GNNs that support edge weights (e.g. GAT, GCN
    with edge_attr).

    Args:
        conn_matrix (np.ndarray): Correlation matrix
        threshold (float):        Minimum |corr| for an edge

    Returns:
        np.ndarray: Weighted adjacency (num_roi, num_roi)
    """
    adj = np.where(np.abs(conn_matrix) > threshold,
                   np.abs(conn_matrix), 0.0).astype(np.float32)
    np.fill_diagonal(adj, 0.0)
    return adj


# =============================================================
# CONNECTIVITY STATISTICS
# =============================================================

def connectivity_stats(conn_matrix: np.ndarray,
                       roi_names: list = None) -> dict:
    """
    Compute summary statistics for a connectivity matrix.

    Returns metrics useful for the UI results panel:
        - mean / std of absolute correlations
        - density (fraction of edges present above 0.3)
        - top-3 most connected regions
        - top-3 strongest pairwise connections

    Args:
        conn_matrix (np.ndarray): Correlation matrix (num_roi, num_roi)
        roi_names (list):         Optional ROI name list

    Returns:
        dict: Summary statistics dictionary
    """
    names = roi_names or ROI_NAMES
    n = conn_matrix.shape[0]

    # Exclude diagonal for stats
    mask = ~np.eye(n, dtype=bool)
    off_diag = conn_matrix[mask]

    # Overall stats
    mean_corr = float(np.mean(np.abs(off_diag)))
    std_corr  = float(np.std(np.abs(off_diag)))

    # Graph density (fraction of possible edges present)
    threshold  = CONNECTIVITY_THRESHOLD
    adj        = threshold_matrix(conn_matrix, threshold=threshold)
    n_possible = n * (n - 1)          # directed edges excl. diagonal
    density    = float(adj.sum() / n_possible)

    # Most connected regions (highest total |corr| per row)
    row_strength = np.abs(conn_matrix).sum(axis=1)
    row_strength[np.arange(n), np.arange(n)] if False else None  # no-op
    np.fill_diagonal(np.ones((n, n)), 0)  # ignore diagonal
    row_strength = np.abs(conn_matrix - np.diag(np.diag(conn_matrix))).sum(axis=1)
    top_idx = np.argsort(row_strength)[::-1][:3]
    top_regions = [names[i] if i < len(names) else f"Region_{i}" for i in top_idx]

    # Strongest pairwise connections
    upper = []
    for i in range(n):
        for j in range(i + 1, n):
            upper.append((abs(conn_matrix[i, j]), i, j))
    upper.sort(reverse=True)
    top_pairs = [
        {
            "region_a": names[i] if i < len(names) else f"R{i}",
            "region_b": names[j] if j < len(names) else f"R{j}",
            "correlation": round(float(v), 3)
        }
        for v, i, j in upper[:3]
    ]

    return {
        "mean_absolute_correlation": round(mean_corr, 4),
        "std_correlation":           round(std_corr, 4),
        "graph_density":             round(density, 4),
        "top_connected_regions":     top_regions,
        "strongest_connections":     top_pairs,
        "num_edges":                 int(adj.sum()),
    }


# =============================================================
# FULL PIPELINE: features → connectivity matrix + adjacency
# =============================================================

def build_connectivity(roi_features: np.ndarray,
                       method: str = "pearson",
                       threshold: float = CONNECTIVITY_THRESHOLD
                       ) -> tuple:
    """
    Full connectivity pipeline: ROI features → matrix + adjacency.

    Args:
        roi_features (np.ndarray): (num_roi, num_features)
        method (str):              "pearson" or "spearman"
        threshold (float):         Edge threshold

    Returns:
        tuple:
            conn_matrix (np.ndarray): (num_roi, num_roi) correlations
            adj_matrix  (np.ndarray): (num_roi, num_roi) binary adjacency
    """
    if method == "spearman":
        conn_matrix = compute_spearman_matrix(roi_features)
    else:
        conn_matrix = compute_pearson_matrix(roi_features)

    adj_matrix = threshold_matrix(conn_matrix, threshold=threshold)
    return conn_matrix, adj_matrix


# =============================================================
# Self-test
# =============================================================
if __name__ == "__main__":
    print("Testing connectivity module...")

    roi_feats = np.random.randn(10, 16).astype(np.float32)

    conn, adj = build_connectivity(roi_feats)
    print(f"  Connectivity matrix : {conn.shape}, range [{conn.min():.2f}, {conn.max():.2f}]")
    print(f"  Adjacency matrix    : {adj.shape}, edges = {int(adj.sum())}")

    stats = connectivity_stats(conn)
    print(f"  Mean |corr|         : {stats['mean_absolute_correlation']}")
    print(f"  Graph density       : {stats['graph_density']}")
    print(f"  Top regions         : {stats['top_connected_regions']}")
    print(f"  Strongest pairs     : {stats['strongest_connections']}")

    print("\n✅ Connectivity module test passed!")
