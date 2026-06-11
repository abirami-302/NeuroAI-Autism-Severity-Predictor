# =============================================================
# inference/gradcam.py
# Grad-CAM (Gradient-weighted Class Activation Mapping)
# for explainable AI heatmap generation.
#
# Grad-CAM highlights WHICH parts of the MRI scan most
# influenced the model's prediction — giving doctors and
# researchers insight into what the model "looked at".
#
# Reference: Selvaraju et al., 2017
#   "Grad-CAM: Visual Explanations from Deep Networks"
# =============================================================

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.config import (
    GRADCAM_TARGET_LAYER, GRADCAM_COLORMAP, GRADCAM_ALPHA,
    CLASS_NAMES, HEATMAP_OUTPUT_DIR, DEVICE
)


# =============================================================
# GRAD-CAM IMPLEMENTATION
# =============================================================

class GradCAM3D:
    """
    Gradient-weighted Class Activation Mapping for 3D CNNs.

    How it works:
    1. Run a forward pass and record activations at target layer
    2. Run a backward pass for the predicted class
    3. Compute gradients of the class score w.r.t. activations
    4. Weight activations by averaged gradients (channel-wise)
    5. Apply ReLU and normalize → heatmap volume

    Args:
        model (nn.Module):    Trained 3D CNN model
        target_layer (str):   Name of the layer to hook (e.g., "features")
    """

    def __init__(self, model, target_layer: str = GRADCAM_TARGET_LAYER):
        self.model        = model
        self.target_layer = target_layer

        # Storage for forward activations and backward gradients
        self.activations  = None
        self.gradients    = None

        # Register hooks on the target layer
        self._register_hooks()

    def _register_hooks(self):
        """
        Register forward and backward hooks on the target layer.
        Forward hook saves activations; backward hook saves gradients.
        """
        # Find the target layer by name
        target = None
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                target = module
                break

        if target is None:
            # Fallback: hook the last Conv3d layer found
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Conv3d):
                    target = module

        if target is None:
            raise ValueError(
                f"Target layer '{self.target_layer}' not found in model."
            )

        # Forward hook: captures feature maps during forward pass
        def forward_hook(module, input, output):
            self.activations = output.detach()

        # Backward hook: captures gradients during backward pass
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target.register_forward_hook(forward_hook)
        target.register_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor,
                 target_class: int = None) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap for the given input.

        Args:
            input_tensor (torch.Tensor): MRI volume (1, 1, D, H, W)
            target_class (int):         Class index to explain.
                                        If None, uses predicted class.

        Returns:
            np.ndarray: 3D heatmap volume (D, H, W), values in [0, 1]
        """
        self.model.eval()
        # Ensure input is on the correct device and tracks gradients
        input_tensor = input_tensor.to(next(self.model.parameters()).device)
        input_tensor = input_tensor.detach().requires_grad_(True)

        # ── Forward pass ─────────────────────────────────────
        output = self.model(input_tensor)          # (1, num_classes)

        if target_class is None:
            # Use the predicted (highest scoring) class
            target_class = output.argmax(dim=1).item()

        # ── Backward pass ────────────────────────────────────
        self.model.zero_grad()

        # Create one-hot score for the target class
        class_score = output[0, target_class]
        class_score.backward()

        # ── Compute Grad-CAM ─────────────────────────────────
        # Gradients shape: (1, C, D', H', W')
        gradients   = self.gradients[0]    # Remove batch dim → (C, D', H', W')
        activations = self.activations[0]  # (C, D', H', W')

        # Global average pool gradients over spatial dimensions
        # This gives the importance weight for each feature map channel
        weights = gradients.mean(dim=(1, 2, 3))   # (C,)

        # Weighted combination of activation maps
        # cam = sum over channels of (weight_c * activation_c)
        cam = torch.zeros(activations.shape[1:], device=DEVICE)  # (D', H', W')

        for c, w in enumerate(weights):
            cam += w * activations[c]

        # ReLU: only keep positive influences
        cam = F.relu(cam)

        # Convert to numpy
        cam = cam.cpu().numpy()

        # ── Normalize to [0, 1] ──────────────────────────────
        cam_min = cam.min()
        cam_max = cam.max() + 1e-8
        cam = (cam - cam_min) / (cam_max - cam_min)

        return cam   # Shape: (D', H', W')


# =============================================================
# HEATMAP OVERLAY UTILITIES
# =============================================================

def resize_heatmap_to_volume(heatmap: np.ndarray,
                              target_shape: tuple) -> np.ndarray:
    """
    Upsample a Grad-CAM heatmap to match the original MRI volume shape.

    Args:
        heatmap (np.ndarray):    Low-res heatmap from Grad-CAM
        target_shape (tuple):    Target (D, H, W) to upsample to

    Returns:
        np.ndarray: Upsampled heatmap matching target_shape
    """
    from scipy.ndimage import zoom

    current = np.array(heatmap.shape)
    target  = np.array(target_shape)
    factors = target / current
    return zoom(heatmap, factors, order=1).astype(np.float32)


def apply_colormap(heatmap_2d: np.ndarray,
                   colormap: str = GRADCAM_COLORMAP) -> np.ndarray:
    """
    Apply a matplotlib colormap to a 2D heatmap slice.

    Args:
        heatmap_2d (np.ndarray): 2D slice values in [0, 1]
        colormap (str):          Matplotlib colormap name (e.g. "jet")

    Returns:
        np.ndarray: RGB image (H, W, 3), values in [0, 255]
    """
    cmap   = cm.get_cmap(colormap)
    colored = cmap(heatmap_2d)[:, :, :3]   # Drop alpha → (H, W, 3)
    return (colored * 255).astype(np.uint8)


def overlay_heatmap_on_slice(mri_slice: np.ndarray,
                              heatmap_slice: np.ndarray,
                              alpha: float = GRADCAM_ALPHA,
                              colormap: str = GRADCAM_COLORMAP
                              ) -> np.ndarray:
    """
    Blend a Grad-CAM heatmap over an MRI slice image.

    Args:
        mri_slice (np.ndarray):     2D MRI slice (H, W), any range
        heatmap_slice (np.ndarray): 2D heatmap values in [0, 1]
        alpha (float):              Heatmap opacity (0=invisible, 1=full)
        colormap (str):             Matplotlib colormap name

    Returns:
        np.ndarray: Blended RGB image (H, W, 3), uint8
    """
    # Normalize MRI slice to [0, 255] grayscale
    mri_norm = mri_slice.astype(np.float32)
    mri_norm = (mri_norm - mri_norm.min()) / (mri_norm.max() - mri_norm.min() + 1e-8)
    mri_rgb  = np.stack([mri_norm * 255] * 3, axis=-1).astype(np.uint8)

    # Colorize heatmap
    heatmap_rgb = apply_colormap(heatmap_slice, colormap=colormap)

    # Blend: overlay = (1-alpha)*mri + alpha*heatmap
    blended = ((1 - alpha) * mri_rgb + alpha * heatmap_rgb).astype(np.uint8)

    return blended


# =============================================================
# VISUALIZATION FUNCTIONS
# =============================================================

def plot_gradcam_slices(volume: np.ndarray,
                        heatmap: np.ndarray,
                        prediction: str,
                        confidence: float,
                        save_path: str = None,
                        num_slices: int = 5) -> plt.Figure:
    """
    Plot multiple MRI slices with Grad-CAM overlays in a grid.

    Shows equally-spaced slices through the depth axis with
    color-coded heatmap indicating most influential regions.

    Args:
        volume (np.ndarray):    MRI volume (1, D, H, W) or (D, H, W)
        heatmap (np.ndarray):   Grad-CAM heatmap (D, H, W)
        prediction (str):       Predicted class label
        confidence (float):     Confidence score (0–100)
        save_path (str):        If given, save figure to this path
        num_slices (int):       Number of depth slices to show

    Returns:
        plt.Figure: Matplotlib figure object
    """
    # Remove channel dim if present
    if volume.ndim == 4:
        volume = volume[0]

    D = volume.shape[0]

    # Resize heatmap to match volume if needed
    if heatmap.shape != volume.shape:
        heatmap = resize_heatmap_to_volume(heatmap, volume.shape)

    # Choose evenly-spaced slices through the volume depth
    slice_indices = np.linspace(D // 4, 3 * D // 4, num_slices, dtype=int)

    # ── Create figure ────────────────────────────────────────
    fig, axes = plt.subplots(2, num_slices, figsize=(4 * num_slices, 8))
    fig.patch.set_facecolor("#1a1a2e")   # Dark background

    # Determine color for prediction label
    label_colors = {"Mild": "#2ECC71", "Moderate": "#F39C12", "Severe": "#E74C3C"}
    label_color  = label_colors.get(prediction, "#FFFFFF")

    # Title
    fig.suptitle(
        f"🧠 NeuroAI Grad-CAM Analysis\n"
        f"Prediction: {prediction}  |  Confidence: {confidence:.1f}%",
        fontsize=14, color="white", y=0.98,
        fontweight="bold"
    )

    for col, idx in enumerate(slice_indices):
        mri_sl    = volume[idx]           # Raw MRI slice
        heat_sl   = heatmap[idx]          # Heatmap slice
        overlay   = overlay_heatmap_on_slice(mri_sl, heat_sl)

        # Top row: raw MRI slices
        axes[0, col].imshow(mri_sl, cmap="gray", aspect="auto")
        axes[0, col].set_title(f"Slice {idx}", color="white", fontsize=9)
        axes[0, col].axis("off")

        # Bottom row: Grad-CAM overlay
        axes[1, col].imshow(overlay, aspect="auto")
        axes[1, col].set_title("Grad-CAM", color=label_color, fontsize=9)
        axes[1, col].axis("off")

    # Row labels
    axes[0, 0].set_ylabel("MRI Slice", color="white", fontsize=10)
    axes[1, 0].set_ylabel("Heatmap\nOverlay", color="white", fontsize=10)

    # Colorbar for heatmap intensity
    sm = plt.cm.ScalarMappable(
        cmap=GRADCAM_COLORMAP,
        norm=plt.Normalize(vmin=0, vmax=1)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[1, :], orientation="horizontal",
                        fraction=0.03, pad=0.08)
    cbar.set_label("Activation Intensity", color="white", fontsize=10)
    cbar.ax.xaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.xaxis.get_ticklabels(), color="white")

    plt.tight_layout()

    # Save to file if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  [Grad-CAM] Saved heatmap → {save_path}")

    return fig


def generate_mock_heatmap(volume_shape: tuple = (64, 64, 64)) -> np.ndarray:
    """
    Generate a realistic-looking mock Grad-CAM heatmap.
    Used when the model cannot compute gradients (e.g., demo mode).

    Creates Gaussian blobs centered on typical autism-related
    brain regions to simulate a realistic heatmap.

    Args:
        volume_shape (tuple): (D, H, W) of the MRI volume

    Returns:
        np.ndarray: Mock heatmap, values in [0, 1]
    """
    D, H, W = volume_shape
    heatmap = np.zeros((D, H, W), dtype=np.float32)

    # Simulate activations in 3 brain regions with Gaussian blobs
    activation_centers = [
        (D // 4,     H // 3,     W // 2),    # Region 1: frontal-ish
        (D // 2,     H // 2,     W // 3),    # Region 2: central-ish
        (3 * D // 4, 2 * H // 3, 2 * W // 3), # Region 3: parietal-ish
    ]
    strengths = [1.0, 0.75, 0.5]

    for (cd, ch, cw), strength in zip(activation_centers, strengths):
        for d in range(D):
            for h in range(H):
                for w in range(W):
                    dist_sq = ((d-cd)/10)**2 + ((h-ch)/15)**2 + ((w-cw)/15)**2
                    heatmap[d, h, w] += strength * np.exp(-dist_sq / 2)

    # Normalize to [0, 1]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap


# =============================================================
# Self-test
# =============================================================
if __name__ == "__main__":
    print("Testing Grad-CAM utilities...")

    # Generate mock heatmap
    heatmap = generate_mock_heatmap((64, 64, 64))
    print(f"  Mock heatmap shape : {heatmap.shape}")
    print(f"  Value range        : [{heatmap.min():.3f}, {heatmap.max():.3f}]")

    # Test overlay
    mri_slice   = np.random.randn(64, 64).astype(np.float32)
    heat_slice  = heatmap[32]
    overlay     = overlay_heatmap_on_slice(mri_slice, heat_slice)
    print(f"  Overlay shape      : {overlay.shape}")   # (64, 64, 3)

    # Test full visualization
    volume = np.random.randn(64, 64, 64).astype(np.float32)
    fig = plot_gradcam_slices(
        volume, heatmap,
        prediction="Moderate",
        confidence=87.5
    )
    print(f"  Figure created     : {type(fig)}")
    plt.close(fig)

    print("\n✅ Grad-CAM tests passed!")
