# =============================================================
# ui/components/results.py
# Streamlit Prediction Results Display Component.
#
# Renders the full results panel:
#   - Severity badge + confidence meter
#   - Class probability bar chart
#   - Affected brain regions pills
#   - Connectivity statistics
# =============================================================

import streamlit as st
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from training.config import CLASS_NAMES, CLASS_COLORS
from utils.visualize import plot_probability_bars


# Severity → emoji mapping
SEVERITY_EMOJI = {
    "Mild":     "🟢",
    "Moderate": "🟡",
    "Severe":   "🔴",
    "Unknown":  "⚪",
}


def render_prediction_card(result) -> None:
    """
    Render the main prediction result card.

    Args:
        result (PredictionResult): Output from inference/predict.py
    """
    label      = result.label
    confidence = result.confidence
    label_lower = label.lower()

    color = CLASS_COLORS.get(label, "#8b949e")
    emoji = SEVERITY_EMOJI.get(label, "⚪")

    st.markdown(f"""
<div class="result-card prediction-{label_lower}">
    <div style="margin-bottom:0.8rem">
        <span class="severity-badge badge-{label_lower}">
            {emoji} &nbsp; {label.upper()} AUTISM
        </span>
    </div>

    <div style="display:flex; gap:2rem; margin-bottom:1rem; flex-wrap:wrap;">
        <div>
            <div style="color:#8b949e; font-size:0.82rem">Confidence Score</div>
            <div style="font-size:2.2rem; font-weight:800;
                        color:{color}; line-height:1.1">{confidence:.1f}%</div>
        </div>
        <div>
            <div style="color:#8b949e; font-size:0.82rem">Inference Time</div>
            <div style="font-size:2.2rem; font-weight:800;
                        color:#58a6ff; line-height:1.1">{result.inference_time}s</div>
        </div>
    </div>

    <!-- Confidence progress bar -->
    <div style="background:#21262d; border-radius:8px;
                height:10px; margin-bottom:1.2rem; overflow:hidden;">
        <div style="width:{confidence}%; height:100%; border-radius:8px;
                    background:linear-gradient(90deg,{color},{color}aa);
                    transition:width 1s ease;"></div>
    </div>

    <!-- Affected regions -->
    <div>
        <div style="color:#8b949e; font-size:0.82rem;
                    margin-bottom:0.5rem;">🧬 Affected Brain Regions</div>
        {"".join([
            f'<span class="region-pill">📍 {r}</span>'
            for r in result.affected_regions
        ])}
    </div>
</div>
    """, unsafe_allow_html=True)


def render_probability_chart(result) -> None:
    """
    Render the class probability horizontal bar chart.

    Args:
        result (PredictionResult): Prediction result object
    """
    st.markdown("##### Class Probability Distribution")

    fig = plot_probability_bars(result.probabilities, result.label)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Text breakdown below the chart
    st.markdown("**Probability breakdown:**")
    cols = st.columns(len(CLASS_NAMES))
    for col, cls in zip(cols, CLASS_NAMES):
        prob  = result.probabilities.get(cls, 0.0)
        color = CLASS_COLORS.get(cls, "#8b949e")
        col.markdown(
            f"<div style='text-align:center'>"
            f"<div style='font-size:1.4rem; font-weight:700; color:{color}'>"
            f"{prob:.1f}%</div>"
            f"<div style='color:#8b949e; font-size:0.8rem'>{cls}</div>"
            f"</div>",
            unsafe_allow_html=True
        )


def render_connectivity_stats(conn_matrix) -> None:
    """
    Render summary statistics for the connectivity matrix.

    Args:
        conn_matrix (np.ndarray): (num_roi, num_roi) correlation matrix
    """
    import numpy as np
    from graph.connectivity import connectivity_stats

    stats = connectivity_stats(conn_matrix)

    st.markdown("##### Connectivity Summary")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Mean |Correlation|",
                  f"{stats['mean_absolute_correlation']:.3f}")
    with c2:
        st.metric("Graph Density",
                  f"{stats['graph_density']:.1%}")
    with c3:
        st.metric("Active Edges",
                  str(stats["num_edges"]))

    st.markdown("**Strongest pairwise connections:**")
    for pair in stats["strongest_connections"]:
        st.markdown(
            f"- **{pair['region_a']}** ↔ **{pair['region_b']}**: "
            f"`{pair['correlation']:.3f}`"
        )


def render_error_card(error_msg: str) -> None:
    """
    Render an error card when prediction fails.

    Args:
        error_msg (str): Error message string
    """
    st.markdown(f"""
<div class="result-card" style="border-left:5px solid #E74C3C">
    <div style="font-size:1.5rem; margin-bottom:0.5rem">❌ Prediction Failed</div>
    <div style="color:#8b949e; font-size:0.9rem; font-family:monospace;
                background:#0d1117; padding:0.8rem; border-radius:6px;">
        {error_msg}
    </div>
    <div style="margin-top:1rem; color:#8b949e; font-size:0.85rem">
        💡 Try the <strong>Run Demo</strong> button to test with synthetic data,
        or check that your MRI file is in a supported format (.nii, .nii.gz, .png).
    </div>
</div>
    """, unsafe_allow_html=True)
