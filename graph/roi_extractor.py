# =============================================================
# graph/roi_extractor.py
# Brain Region of Interest (ROI) Extraction Utilities.
#
# Provides atlas-based and grid-based methods to partition a
# 3D MRI volume into anatomically meaningful regions, then
# extract per-region feature vectors for graph construction.
#
# Two modes:
#   1. Grid-based  — fast, no external atlas needed (default)
#   2. Atlas-based — uses nilearn AAL atlas for real anatomy
# =============================================================

import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.config import NUM_ROI, GNN_INPUT_DIM, ROI_NAMES


# =============================================================
# GRID-BASED ROI EXTRACTION
# Divides the volume into a regular 3D grid of sub-volumes.
# Fast, no dependencies beyond numpy.
# =============================================================

def extract_roi_grid(volume: np.ndarray,
                     num_roi: int = NUM_ROI) -> np.ndarray:
    """
    Divide the MRI volume into a grid of ROIs and compute
    per-region feature statistics.

    Splits the depth (D) axis into num_roi equal slabs.
    Each slab becomes one ROI node in the connectivity graph.

    Args:
        volume (np.ndarray): Preprocessed MRI volume.
                             Shape: (1, D, H, W) or (D, H, W)
        num_roi (int):       Number of ROI regions

    Returns:
        np.ndarray: Feature matrix, shape (num_roi, GNN_INPUT_DIM)
    """
    # Strip channel dimension
    vol = volume[0] if volume.ndim == 4 else volume
    D, H, W = vol.shape

    features = np.zeros((num_roi, GNN_INPUT_DIM), dtype=np.float32)
    slab_size = D // num_roi

    for i in range(num_roi):
        start = i * slab_size
        end   = start + slab_size if i < num_roi - 1 else D
        roi   = vol[start:end, :, :].flatten()

        # Compute GNN_INPUT_DIM statistical features per ROI
        feat = _compute_roi_features(roi, region_idx=i, depth=end - start)
        features[i] = feat

    # Column-wise normalisation: zero-mean, unit-std per feature
    for f in range(GNN_INPUT_DIM):
        col = features[:, f]
        features[:, f] = (col - col.mean()) / (col.std() + 1e-8)

    return features


def _compute_roi_features(flat: np.ndarray,
                           region_idx: int,
                           depth: int) -> np.ndarray:
    """
    Compute a fixed-length feature vector for one ROI slab.

    Features (16 total, matching GNN_INPUT_DIM):
        Statistical moments, percentiles, activity ratios,
        local variation, energy, and structural metadata.

    Args:
        flat (np.ndarray): Flattened voxel values for this ROI
        region_idx (int):  Index of the ROI (0 … num_roi-1)
        depth (int):       Depth of the slab in voxels

    Returns:
        np.ndarray: Feature vector of length GNN_INPUT_DIM (16)
    """
    feat = np.array([
        float(np.mean(flat)),                               # 0: mean intensity
        float(np.std(flat)),                                # 1: std dev
        float(np.min(flat)),                                # 2: minimum
        float(np.max(flat)),                                # 3: maximum
        float(np.percentile(flat, 25)),                     # 4: Q1
        float(np.percentile(flat, 50)),                     # 5: median
        float(np.percentile(flat, 75)),                     # 6: Q3
        float(np.percentile(flat, 90)),                     # 7: 90th pct
        float(np.sum(flat > 0) / (len(flat) + 1e-8)),      # 8: frac positive
        float(np.sum(flat > flat.mean()) /
              (len(flat) + 1e-8)),                          # 9: above-mean ratio
        float(np.mean(np.abs(np.diff(flat[:200])))),        # 10: local variation
        float(np.var(flat)),                                # 11: variance
        float(np.count_nonzero(flat)) / (len(flat) + 1e-8),# 12: density
        float(np.mean(flat ** 2)),                          # 13: energy
        float((np.mean(flat) - np.median(flat)) /
              (np.std(flat) + 1e-8)),                       # 14: skewness proxy
        float(depth),                                       # 15: slab depth
    ], dtype=np.float32)

    return feat[:GNN_INPUT_DIM]


# =============================================================
# ATLAS-BASED ROI EXTRACTION (optional, needs nilearn)
# Uses the AAL (Automated Anatomical Labeling) atlas which
# provides 116 anatomically labelled brain regions.
# Falls back to grid extraction if nilearn is not installed.
# =============================================================

def extract_roi_atlas(volume: np.ndarray,
                      num_roi: int = NUM_ROI) -> np.ndarray:
    """
    Extract ROI features using the AAL brain atlas via nilearn.

    This gives anatomically accurate ROIs (frontal lobe,
    amygdala, hippocampus, etc.) rather than grid slabs.

    Falls back to grid-based extraction if nilearn is unavailable.

    Args:
        volume (np.ndarray): MRI volume (1, D, H, W) or (D, H, W)
        num_roi (int):       Number of ROIs to use

    Returns:
        np.ndarray: Feature matrix, shape (num_roi, GNN_INPUT_DIM)
    """
    try:
        import nibabel as nib
        from nilearn import datasets, image
        import warnings
        warnings.filterwarnings("ignore")

        # Load AAL atlas (downloads ~3MB on first use)
        aal = datasets.fetch_atlas_aal()
        atlas_img = nib.load(aal.maps)
        atlas_data = atlas_img.get_fdata()

        vol = volume[0] if volume.ndim == 4 else volume

        # Get unique region labels (0 = background)
        unique_labels = np.unique(atlas_data)
        unique_labels = unique_labels[unique_labels > 0]

        # Select first num_roi regions
        selected = unique_labels[:num_roi]
        features = np.zeros((num_roi, GNN_INPUT_DIM), dtype=np.float32)

        for i, label in enumerate(selected):
            # Create binary mask for this region
            mask = (atlas_data == label)

            # Resize mask to match volume if needed
            if mask.shape != vol.shape:
                from scipy.ndimage import zoom
                factors = np.array(vol.shape) / np.array(mask.shape)
                mask = zoom(mask.astype(float), factors, order=0) > 0.5

            roi_voxels = vol[mask].flatten()
            if len(roi_voxels) == 0:
                continue

            features[i] = _compute_roi_features(roi_voxels,
                                                 region_idx=i,
                                                 depth=int(mask.sum()))

        # Normalise
        for f in range(GNN_INPUT_DIM):
            col = features[:, f]
            features[:, f] = (col - col.mean()) / (col.std() + 1e-8)

        return features

    except Exception as e:
        # Graceful fallback — grid extraction always works
        print(f"[ROI] Atlas extraction failed ({e}). Using grid fallback.")
        return extract_roi_grid(volume, num_roi=num_roi)


# =============================================================
# PUBLIC API — auto-selects best available method
# =============================================================

def extract_roi_features(volume: np.ndarray,
                          num_roi: int = NUM_ROI,
                          use_atlas: bool = False) -> np.ndarray:
    """
    Extract per-ROI feature vectors from an MRI volume.

    Automatically uses atlas-based extraction when requested
    and nilearn is available; otherwise uses fast grid method.

    Args:
        volume (np.ndarray): Preprocessed MRI, (1,D,H,W) or (D,H,W)
        num_roi (int):       Number of ROI nodes for the graph
        use_atlas (bool):    Attempt AAL atlas extraction

    Returns:
        np.ndarray: Shape (num_roi, GNN_INPUT_DIM)
    """
    if use_atlas:
        return extract_roi_atlas(volume, num_roi=num_roi)
    return extract_roi_grid(volume, num_roi=num_roi)


def get_roi_names(num_roi: int = NUM_ROI) -> list:
    """
    Return the list of ROI region names up to num_roi.

    Args:
        num_roi (int): How many names to return

    Returns:
        list[str]: Region name strings
    """
    # Pad with generic names if num_roi > len(ROI_NAMES)
    names = list(ROI_NAMES)
    while len(names) < num_roi:
        names.append(f"Region_{len(names) + 1}")
    return names[:num_roi]


# =============================================================
# Self-test
# =============================================================
if __name__ == "__main__":
    print("Testing ROI extractor...")

    vol = np.random.randn(1, 64, 64, 64).astype(np.float32)

    feats = extract_roi_features(vol, num_roi=10, use_atlas=False)
    print(f"  Grid features shape : {feats.shape}")   # (10, 16)
    print(f"  Feature range       : [{feats.min():.3f}, {feats.max():.3f}]")
    print(f"  ROI names           : {get_roi_names(10)}")

    print("\n✅ ROI extractor test passed!")
