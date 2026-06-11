# =============================================================
# ui/components/heatmap_viewer.py
# Streamlit Grad-CAM Heatmap Viewer Component.
#
# Renders the interactive heatmap overlay section of the UI,
# including a slice-slider so users can scroll through all
# depth planes of the 3D heatmap interactively.
# =============================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import io
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from inference.gradcam import (
    overlay_heatmap_on_slice,
    plot_gradcam_slices,
    resize_heatmap_to_volume
)
from training.config import GRADCAM_COLORMAP, CLASS_COLORS


def render_heatmap_viewer(result) -> None:
    """
    Render the full Grad-CAM heatmap viewer section.

    Includes:
        1. Static multi-slice overview (5 evenly-spaced slices)
        2. Interactive single-slice slider for exploring all depths
        3. Colorbar legend
        4. Download button for the heatmap image

    Args:
        result (PredictionResult): From inference/predict.py
                                   Uses result.volume, result.heatmap,
                                   result.label, result.confidence
    """
    label      = result.label
    confidence = result.confidence
    color      = CLASS_COLORS.get(label, "#8b949e")

    # Extract 3D volume (remove channel dim if present)
    volume = result.volume[0] if result.volume.ndim == 4 else result.volume
    heatmap = result.heatmap

    # Ensure heatmap matches volume spatial dims
    if heatmap.shape != volume.shape:
        heatmap = resize_heatmap_to_volume(heatmap, volume.shape)

    D = volume.shape[0]

    # ── Section header ─────────────────────────────────────
    st.markdown(
        "##### 🌡️ Gradient-weighted Class Activation Map (Grad-CAM)"
    )
    st.caption(
        "**Warm colors (red/yellow)** = brain regions that most influenced "
        f"the **{label}** prediction. "
        "**Cool colors (blue)** = regions with low influence."
    )

    # ── Static overview: 5-slice grid ─────────────────────
    with st.spinner("Rendering heatmap overview..."):
        fig_overview = plot_gradcam_slices(
            volume, heatmap,
            prediction=label,
            confidence=confidence,
            num_slices=5
        )
        st.pyplot(fig_overview, use_container_width=True)
        plt.close(fig_overview)

    st.divider()

    # ── Interactive single-slice explorer ─────────────────
    st.markdown("**🔍 Interactive Slice Explorer**")
    st.caption("Drag the slider to explore individual depth slices.")

    selected_slice = st.slider(
        "Depth slice (z-axis)",
        min_value=0,
        max_value=D - 1,
        value=D // 2,
        step=1,
        help="Navigate through the 3D MRI volume depth"
    )

    # Render selected slice with heatmap overlay
    mri_sl   = volume[selected_slice]
    heat_sl  = heatmap[selected_slice]
    overlay  = overlay_heatmap_on_slice(mri_sl, heat_sl)

    col_raw, col_heat, col_blend = st.columns(3)

    with col_raw:
        fig, ax = plt.subplots(figsize=(3, 3))
        fig.patch.set_facecolor("#1a1a2e")
        ax.imshow(mri_sl, cmap="gray", aspect="equal")
        ax.set_title("Raw MRI", color="white", fontsize=9)
        ax.axis("off")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_heat:
        fig, ax = plt.subplots(figsize=(3, 3))
        fig.patch.set_facecolor("#1a1a2e")
        ax.imshow(heat_sl, cmap=GRADCAM_COLORMAP,
                  vmin=0, vmax=1, aspect="equal")
        ax.set_title("Activation Map", color=color, fontsize=9)
        ax.axis("off")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_blend:
        fig, ax = plt.subplots(figsize=(3, 3))
        fig.patch.set_facecolor("#1a1a2e")
        ax.imshow(overlay, aspect="equal")
        ax.set_title("Overlay", color="white", fontsize=9)
        ax.axis("off")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ── Colorbar legend ─────────────────────────────────
    st.markdown("")
    _render_colorbar_legend()

    # ── Download button ─────────────────────────────────
    buf = _fig_to_bytes(plot_gradcam_slices(
        volume, heatmap,
        prediction=label,
        confidence=confidence,
        num_slices=5
    ))
    st.download_button(
        label="⬇️ Download Heatmap (PNG)",
        data=buf,
        file_name=f"gradcam_{label.lower()}.png",
        mime="image/png",
        use_container_width=True
    )


def _render_colorbar_legend() -> None:
    """Render a horizontal colorbar legend for the heatmap."""
    fig, ax = plt.subplots(figsize=(5, 0.4))
    fig.patch.set_facecolor("#1a1a2e")

    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient, aspect="auto", cmap=GRADCAM_COLORMAP)
    ax.set_yticks([])
    ax.set_xticks([0, 128, 255])
    ax.set_xticklabels(["Low\nActivation", "Medium", "High\nActivation"],
                       color="white", fontsize=7)
    ax.tick_params(colors="white")

    plt.tight_layout(pad=0.1)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _fig_to_bytes(fig: plt.Figure) -> bytes:
    """Convert matplotlib figure to PNG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150,
                bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    buf.seek(0)
    return buf.read()
