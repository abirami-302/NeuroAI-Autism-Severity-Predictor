# =============================================================
# ui/components/uploader.py
# Streamlit MRI Upload Widget Component.
#
# Renders the file upload section and returns the saved
# temporary file path once a file has been uploaded.
# =============================================================

import streamlit as st
import os
import tempfile
from typing import Optional


def render_uploader() -> Optional[str]:
    """
    Render the MRI file uploader widget in Streamlit.

    Saves the uploaded file to a temporary location on disk
    so the preprocessing pipeline can read it via nibabel
    or PIL (both require a real file path, not a buffer).

    Returns:
        str | None: Temporary file path if a file was uploaded,
                    None otherwise.
    """
    st.markdown(
        '<div class="section-title">📤 Upload Brain MRI Scan</div>',
        unsafe_allow_html=True
    )

    uploaded = st.file_uploader(
        label="Drop your MRI file here",
        type=["nii", "gz", "png", "jpg", "jpeg"],
        help=(
            "Supported formats:\n"
            "• NIfTI 3D volume: .nii, .nii.gz\n"
            "• 2D MRI slice image: .png, .jpg"
        )
    )

    if uploaded is None:
        # Placeholder hint when no file is selected
        st.markdown(
            """
<div style="
    background:#161b22; border:2px dashed #30363d;
    border-radius:10px; padding:1.5rem;
    text-align:center; color:#8b949e;
">
    <div style="font-size:2.5rem; margin-bottom:0.5rem">🫁</div>
    <strong>Drag &amp; drop an MRI file above</strong><br>
    <small>Formats: .nii &nbsp;·&nbsp; .nii.gz &nbsp;·&nbsp; .png &nbsp;·&nbsp; .jpg</small>
</div>
            """,
            unsafe_allow_html=True
        )
        return None

    # Show file info
    file_size_kb = len(uploaded.getvalue()) / 1024
    st.success(
        f"✅ **{uploaded.name}** uploaded "
        f"({file_size_kb:.1f} KB)"
    )

    # Determine suffix for the temp file
    fname  = uploaded.name.lower()
    if fname.endswith(".nii.gz"):
        suffix = ".nii.gz"
    else:
        suffix = os.path.splitext(fname)[1] or ".tmp"

    # Write to a temporary file that persists until the caller deletes it
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getvalue())
    tmp.flush()
    tmp.close()

    return tmp.name


def render_demo_notice() -> None:
    """Render a small info card explaining demo mode."""
    st.info(
        "🎲 **Demo Mode** — Running on a synthetic (randomly generated) "
        "MRI volume. Upload a real MRI file for a meaningful prediction.",
        icon="ℹ️"
    )
