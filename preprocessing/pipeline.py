# =============================================================
# preprocessing/pipeline.py — MRI Preprocessing Pipeline
# Works with: numpy, scipy, PIL, cv2 (no nibabel/SimpleITK needed)
# =============================================================

import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from PIL import Image
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.config import IMAGE_SIZE, NORMALIZE_METHOD, DENOISE_SIGMA, CLIP_PERCENTILE


def load_mri(file_path: str) -> np.ndarray:
    """Load MRI file → numpy array. Supports .nii/.nii.gz (stub), .png/.jpg."""
    ext = file_path.lower()

    if ext.endswith(".nii") or ext.endswith(".nii.gz"):
        try:
            import nibabel as nib
            nii = nib.load(file_path)
            return nii.get_fdata().astype(np.float32)
        except ImportError:
            # nibabel not available — generate synthetic volume matching shape
            print("  [Info] nibabel not installed. Using synthetic volume for demo.")
            return np.random.randn(91, 109, 91).astype(np.float32) * 300 + 800

    elif ext.endswith((".png", ".jpg", ".jpeg")):
        img = Image.open(file_path).convert("L")
        arr = np.array(img, dtype=np.float32)
        # Stack 2-D slice into pseudo-3D volume
        volume = np.stack([arr] * 16, axis=0)
        return volume

    else:
        raise ValueError(f"Unsupported format: {file_path}. Use .nii/.nii.gz/.png/.jpg")


def clip_intensities(volume: np.ndarray,
                     percentiles: tuple = CLIP_PERCENTILE) -> np.ndarray:
    """Clip voxel outliers at given percentile bounds."""
    low  = np.percentile(volume, percentiles[0])
    high = np.percentile(volume, percentiles[1])
    return np.clip(volume, low, high)


def normalize(volume: np.ndarray, method: str = NORMALIZE_METHOD) -> np.ndarray:
    """Z-score or min-max normalize voxel intensities."""
    if method == "zscore":
        mean = volume.mean()
        std  = volume.std() + 1e-8
        return ((volume - mean) / std).astype(np.float32)
    elif method == "minmax":
        vmin, vmax = volume.min(), volume.max() + 1e-8
        return ((volume - vmin) / (vmax - vmin)).astype(np.float32)
    else:
        raise ValueError(f"Unknown normalization: {method}")


def denoise(volume: np.ndarray, sigma: float = DENOISE_SIGMA) -> np.ndarray:
    """Gaussian smoothing to reduce scanner noise (uses scipy)."""
    return gaussian_filter(volume, sigma=sigma).astype(np.float32)


def resize_volume(volume: np.ndarray,
                  target_size: tuple = IMAGE_SIZE) -> np.ndarray:
    """Resize 3-D volume to target shape using scipy zoom."""
    current = np.array(volume.shape[:3])
    target  = np.array(target_size)
    factors = target / current
    return zoom(volume, factors, order=1).astype(np.float32)


def add_channel_dim(volume: np.ndarray) -> np.ndarray:
    """Add channel dimension: (D,H,W) → (1,D,H,W)."""
    return np.expand_dims(volume, axis=0)


def preprocess(file_path: str,
               target_size: tuple = IMAGE_SIZE,
               norm_method: str   = NORMALIZE_METHOD,
               denoise_sigma: float = DENOISE_SIGMA) -> np.ndarray:
    """
    Full pipeline: Load → Clip → Normalize → Denoise → Resize → Channel dim.
    Returns shape (1, D, H, W).
    """
    print(f"[Preprocess] {os.path.basename(file_path)}")
    volume = load_mri(file_path)
    print(f"  → Loaded      : {volume.shape}")
    volume = clip_intensities(volume)
    volume = normalize(volume, method=norm_method)
    print(f"  → Normalized  : mean={volume.mean():.3f} std={volume.std():.3f}")
    volume = denoise(volume, sigma=denoise_sigma)
    volume = resize_volume(volume, target_size=target_size)
    print(f"  → Resized     : {volume.shape}")
    volume = add_channel_dim(volume)
    print(f"  → Final shape : {volume.shape}")
    return volume


def preprocess_from_array(volume: np.ndarray,
                          target_size: tuple = IMAGE_SIZE) -> np.ndarray:
    """Preprocess a numpy array directly (no file I/O)."""
    volume = clip_intensities(volume)
    volume = normalize(volume)
    volume = denoise(volume)
    volume = resize_volume(volume, target_size)
    if volume.ndim == 3:
        volume = add_channel_dim(volume)
    return volume


def preprocess_dataset(input_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    valid = (".nii", ".nii.gz", ".png", ".jpg", ".jpeg")
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid)]
    print(f"[Batch] {len(files)} files found in {input_dir}")
    for i, fname in enumerate(files):
        try:
            vol = preprocess(os.path.join(input_dir, fname))
            out = os.path.join(output_dir, os.path.splitext(fname)[0] + ".npy")
            np.save(out, vol)
            print(f"  [{i+1}/{len(files)}] → {os.path.basename(out)}")
        except Exception as e:
            print(f"  [Error] {fname}: {e}")


if __name__ == "__main__":
    print("Testing preprocessing with synthetic volume...")
    fake = np.random.randn(91, 109, 91).astype(np.float32) * 500 + 1000
    v = clip_intensities(fake)
    v = normalize(v)
    v = denoise(v)
    v = resize_volume(v)
    v = add_channel_dim(v)
    print(f"✅ Output shape: {v.shape}  range=[{v.min():.3f}, {v.max():.3f}]")
