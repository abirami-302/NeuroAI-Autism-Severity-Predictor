# =============================================================
# utils/helpers.py
# Shared utility functions used across all NeuroAI modules.
#
# Includes: file I/O, timer, reproducibility seed,
#           model parameter counting, safe directory creation,
#           and result serialisation helpers.
# =============================================================

import os
import sys
import time
import json
import random
import hashlib
import numpy as np
import torch
from pathlib import Path
from typing import Any, Optional


# =============================================================
# REPRODUCIBILITY
# =============================================================

def set_seed(seed: int = 42) -> None:
    """
    Set all random seeds for fully reproducible results.

    Sets seeds for: Python random, NumPy, PyTorch CPU, PyTorch CUDA.

    Args:
        seed (int): Random seed value (default 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic CUDA operations (may reduce speed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    print(f"[Seed] Random seed set to {seed}")


# =============================================================
# FILE & DIRECTORY UTILITIES
# =============================================================

def ensure_dir(path: str) -> str:
    """
    Create directory (and parents) if it doesn't exist.

    Args:
        path (str): Directory path to create

    Returns:
        str: The same path (for chaining)
    """
    os.makedirs(path, exist_ok=True)
    return path


def get_file_extension(path: str) -> str:
    """
    Return the lower-cased file extension including the dot.

    Handles double extensions like '.nii.gz' correctly.

    Args:
        path (str): File path

    Returns:
        str: Extension, e.g. ".nii", ".nii.gz", ".png"
    """
    path = path.lower()
    if path.endswith(".nii.gz"):
        return ".nii.gz"
    return os.path.splitext(path)[1]


def is_mri_file(path: str) -> bool:
    """
    Return True if the file has a supported MRI extension.

    Supported: .nii, .nii.gz, .png, .jpg, .jpeg
    """
    ext = get_file_extension(path)
    return ext in {".nii", ".nii.gz", ".png", ".jpg", ".jpeg"}


def list_mri_files(directory: str) -> list:
    """
    Recursively list all MRI files in a directory.

    Args:
        directory (str): Root directory to search

    Returns:
        list[str]: Sorted list of full file paths
    """
    files = []
    for root, _, fnames in os.walk(directory):
        for fname in fnames:
            fpath = os.path.join(root, fname)
            if is_mri_file(fpath):
                files.append(fpath)
    return sorted(files)


def file_md5(path: str) -> str:
    """
    Compute MD5 checksum of a file (useful for cache validation).

    Args:
        path (str): Path to file

    Returns:
        str: Hex MD5 digest
    """
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# =============================================================
# TIMER CONTEXT MANAGER
# =============================================================

class Timer:
    """
    Simple context manager for timing code blocks.

    Usage:
        with Timer("Preprocessing"):
            result = preprocess(volume)
        # Prints: [Timer] Preprocessing: 1.23s
    """
    def __init__(self, name: str = ""):
        self.name    = name
        self.elapsed = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start
        print(f"[Timer] {self.name}: {self.elapsed:.3f}s")


# =============================================================
# MODEL UTILITIES
# =============================================================

def count_parameters(model: torch.nn.Module) -> dict:
    """
    Count total and trainable parameters in a PyTorch model.

    Args:
        model (nn.Module): PyTorch model

    Returns:
        dict: {"total": N, "trainable": M, "frozen": N-M}
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total":     total,
        "trainable": trainable,
        "frozen":    total - trainable,
    }


def model_size_mb(model: torch.nn.Module) -> float:
    """
    Estimate model size in megabytes (parameters only).

    Args:
        model (nn.Module): PyTorch model

    Returns:
        float: Approximate size in MB
    """
    params = count_parameters(model)["total"]
    # Each float32 parameter = 4 bytes
    return round(params * 4 / (1024 ** 2), 2)


def freeze_backbone(model: torch.nn.Module,
                    unfreeze_last_n: int = 2) -> None:
    """
    Freeze all model parameters except the last N modules.
    Useful for fine-tuning a pre-trained model.

    Args:
        model (nn.Module):    Model to partially freeze
        unfreeze_last_n (int): Keep last N child modules trainable
    """
    children = list(model.children())
    n_freeze = max(0, len(children) - unfreeze_last_n)

    for i, child in enumerate(children):
        for param in child.parameters():
            param.requires_grad = (i >= n_freeze)

    trainable = count_parameters(model)["trainable"]
    print(f"[Freeze] {trainable:,} trainable parameters "
          f"(last {unfreeze_last_n} modules unfrozen)")


# =============================================================
# RESULT SERIALISATION
# =============================================================

def save_result_json(result_dict: dict, output_path: str) -> None:
    """
    Save a prediction result dictionary as a JSON file.

    Non-serialisable types (numpy arrays, tensors) are converted
    to Python lists automatically.

    Args:
        result_dict (dict): Prediction result (from predict())
        output_path (str):  Destination .json file path
    """
    ensure_dir(os.path.dirname(output_path) or ".")

    def _convert(obj: Any):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return obj

    serialisable = {
        k: _convert(v) for k, v in result_dict.items()
        if not callable(v)
    }

    with open(output_path, "w") as f:
        json.dump(serialisable, f, indent=2)

    print(f"[Save] Result saved → {output_path}")


def load_result_json(path: str) -> dict:
    """
    Load a saved prediction result from JSON.

    Args:
        path (str): Path to .json file

    Returns:
        dict: Loaded result dictionary
    """
    with open(path, "r") as f:
        return json.load(f)


# =============================================================
# FORMATTING HELPERS
# =============================================================

def format_confidence(confidence: float) -> str:
    """
    Format a confidence float as a display string.

    Examples:
        92.3  → "92.3%"
        100.0 → "100%"
    """
    if confidence == 100.0:
        return "100%"
    return f"{confidence:.1f}%"


def format_seconds(seconds: float) -> str:
    """
    Human-readable duration string.

    Examples:
        0.23  → "230ms"
        1.5   → "1.5s"
        75.0  → "1m 15s"
    """
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}m {s}s"


# =============================================================
# Self-test
# =============================================================
if __name__ == "__main__":
    print("Testing helpers...")

    set_seed(42)

    ensure_dir("/tmp/neuroai_test")
    print(f"  Is MRI (.nii)   : {is_mri_file('brain.nii')}")
    print(f"  Is MRI (.csv)   : {is_mri_file('data.csv')}")
    print(f"  Extension .nii.gz: {get_file_extension('scan.nii.gz')}")

    with Timer("Sleep test"):
        time.sleep(0.05)

    print(f"  format_confidence: {format_confidence(87.3)}")
    print(f"  format_seconds   : {format_seconds(0.23)}")
    print(f"  format_seconds   : {format_seconds(1.5)}")
    print(f"  format_seconds   : {format_seconds(75)}")

    # Model param count test
    m = torch.nn.Linear(128, 3)
    info = count_parameters(m)
    print(f"  Linear(128,3) params: {info}")

    print("\n✅ Helpers tests passed!")
