# =============================================================
# utils/visualize.py
# Visualization utilities for NeuroAI.
# Provides reusable plotting functions for:
#   - Functional connectivity matrices
#   - Prediction probability bar charts
#   - Brain region analysis plots
#   - MRI slice viewers
# =============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import io
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.config import CLASS_NAMES, CLASS_COLORS, ROI_NAMES


# =============================================================
# 1. CONNECTIVITY MATRIX HEATMAP
# =============================================================

def plot_connectivity_matrix(conn_matrix: np.ndarray,
                              roi_names: list = ROI_NAMES,
                              title: str = "Brain Functional Connectivity",
                              save_path: str = None) -> plt.Figure:
    """
    Visualize the functional connectivity matrix as a heatmap.

    Each cell (i, j) shows the Pearson correlation between
    brain region i and brain region j.

    Args:
        conn_matrix (np.ndarray): (num_roi, num_roi) correlation matrix
        roi_names (list):         Labels for each ROI
        title (str):              Plot title
        save_path (str):          If given, save figure to this path

    Returns:
        plt.Figure
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    # Seaborn heatmap with diverging colormap (blue=negative, red=positive)
    sns.heatmap(
        conn_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1, vmax=1,
        xticklabels=roi_names,
        yticklabels=roi_names,
        linewidths=0.5,
        linecolor="#2d2d2d",
        ax=ax,
        annot_kws={"size": 7}
    )

    ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=15)
    ax.tick_params(colors="white", labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", color="white")
    plt.setp(ax.get_yticklabels(), rotation=0, color="white")

    # Colorbar styling
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors="white")
    cbar.set_label("Pearson Correlation", color="white")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())

    return fig


# =============================================================
# 2. PREDICTION PROBABILITY BAR CHART
# =============================================================

def plot_probability_bars(probabilities: dict,
                           predicted_label: str,
                           save_path: str = None) -> plt.Figure:
    """
    Horizontal bar chart showing class probabilities.

    The predicted class bar is highlighted in its severity color.

    Args:
        probabilities (dict): {"Mild": 12.3, "Moderate": 72.1, "Severe": 15.6}
        predicted_label (str): The predicted class name
        save_path (str):       Optional save path

    Returns:
        plt.Figure
    """
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#0d1117")

    labels = list(probabilities.keys())
    values = list(probabilities.values())

    # Color: highlight predicted class, mute others
    colors = []
    for label in labels:
        if label == predicted_label:
            colors.append(CLASS_COLORS.get(label, "#4ECDC4"))
        else:
            colors.append("#3d3d5c")

    bars = ax.barh(labels, values, color=colors, height=0.5, edgecolor="none")

    # Add value labels inside bars
    for bar, val in zip(bars, values):
        x_pos = min(val - 3, val * 0.85)
        ax.text(
            max(x_pos, 2), bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center", ha="left", color="white",
            fontsize=11, fontweight="bold"
        )

    ax.set_xlim(0, 110)
    ax.set_xlabel("Confidence (%)", color="white", fontsize=10)
    ax.set_title("Prediction Probabilities", color="white",
                 fontsize=12, fontweight="bold")
    ax.tick_params(colors="white", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#555")
    ax.spines["left"].set_color("#555")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())

    return fig


# =============================================================
# 3. PLOTLY INTERACTIVE CONNECTIVITY GRAPH
# =============================================================

def plot_connectivity_network(conn_matrix: np.ndarray,
                               roi_names: list = ROI_NAMES,
                               threshold: float = 0.3) -> go.Figure:
    """
    Interactive 3D network graph of brain connectivity using Plotly.

    Nodes = Brain ROI regions positioned in a circle.
    Edges = Functional connections above threshold.
    Edge color/width = Correlation strength.

    Args:
        conn_matrix (np.ndarray): Correlation matrix
        roi_names (list):         ROI labels
        threshold (float):        Only show edges above this value

    Returns:
        plotly.graph_objects.Figure
    """
    n = len(roi_names)

    # Position nodes in a circle
    angles  = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x_nodes = np.cos(angles)
    y_nodes = np.sin(angles)

    # ── Build edge traces ────────────────────────────────────
    edge_traces = []
    for i in range(n):
        for j in range(i + 1, n):
            corr = conn_matrix[i, j]
            if abs(corr) > threshold:
                # Color: blue for negative, red for positive correlation
                color = f"rgba(231,76,60,{abs(corr):.2f})" if corr > 0 \
                    else f"rgba(52,152,219,{abs(corr):.2f})"
                width = abs(corr) * 4

                edge_traces.append(go.Scatter(
                    x=[x_nodes[i], x_nodes[j], None],
                    y=[y_nodes[i], y_nodes[j], None],
                    mode="lines",
                    line=dict(width=width, color=color),
                    hoverinfo="none",
                    showlegend=False
                ))

    # ── Node trace ───────────────────────────────────────────
    # Node size = total connectivity strength
    node_sizes = (np.abs(conn_matrix).sum(axis=1) /
                  np.abs(conn_matrix).sum(axis=1).max() * 20 + 10)

    node_trace = go.Scatter(
        x=x_nodes, y=y_nodes,
        mode="markers+text",
        text=roi_names,
        textposition="top center",
        marker=dict(
            size=node_sizes,
            color=list(range(n)),
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Region Index"),
            line=dict(width=1, color="white")
        ),
        hovertemplate="<b>%{text}</b><extra></extra>",
        textfont=dict(color="white", size=9)
    )

    fig = go.Figure(data=edge_traces + [node_trace])

    fig.update_layout(
        title="Brain Functional Connectivity Network",
        titlefont_color="white",
        showlegend=False,
        hovermode="closest",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#0d1117",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500
    )

    return fig


# =============================================================
# 4. MRI SLICE GRID VIEWER
# =============================================================

def plot_mri_slices(volume: np.ndarray,
                    num_slices: int = 9,
                    title: str = "MRI Brain Slices",
                    save_path: str = None) -> plt.Figure:
    """
    Display a grid of evenly-spaced MRI depth slices.

    Args:
        volume (np.ndarray): (1, D, H, W) or (D, H, W)
        num_slices (int):    Number of slices to show (perfect square preferred)
        title (str):         Figure title
        save_path (str):     Optional save path

    Returns:
        plt.Figure
    """
    if volume.ndim == 4:
        volume = volume[0]

    D = volume.shape[0]
    slice_idx = np.linspace(0, D - 1, num_slices, dtype=int)

    # Make grid as square as possible
    cols = int(np.ceil(np.sqrt(num_slices)))
    rows = int(np.ceil(num_slices / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    fig.patch.set_facecolor("#1a1a2e")

    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, idx in enumerate(slice_idx):
        axes[i].imshow(volume[idx], cmap="gray", aspect="equal")
        axes[i].set_title(f"z={idx}", color="#aaaaaa", fontsize=8)
        axes[i].axis("off")

    # Turn off unused axes
    for j in range(len(slice_idx), len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, color="white", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())

    return fig


# =============================================================
# 5. AFFECTED REGIONS RADAR CHART (Plotly)
# =============================================================

def plot_region_radar(conn_matrix: np.ndarray,
                      roi_names: list = ROI_NAMES) -> go.Figure:
    """
    Radar / spider chart showing connectivity strength per brain region.

    Args:
        conn_matrix (np.ndarray): Correlation matrix
        roi_names (list):         ROI region names

    Returns:
        plotly.graph_objects.Figure
    """
    # Total connectivity per region (excluding self)
    connectivity = np.abs(conn_matrix).sum(axis=1)
    connectivity_norm = connectivity / (connectivity.max() + 1e-8)

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=list(connectivity_norm) + [connectivity_norm[0]],
        theta=roi_names + [roi_names[0]],
        fill="toself",
        fillcolor="rgba(52,152,219,0.2)",
        line=dict(color="#3498DB", width=2),
        name="Connectivity"
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1],
                            gridcolor="#444", tickfont=dict(color="white")),
            angularaxis=dict(tickfont=dict(color="white", size=10),
                             gridcolor="#444"),
            bgcolor="#0d1117"
        ),
        showlegend=False,
        paper_bgcolor="#1a1a2e",
        title=dict(text="Region Connectivity Strength", font=dict(color="white")),
        height=400
    )

    return fig


# =============================================================
# Utility: Convert matplotlib figure → bytes for Streamlit
# =============================================================

def fig_to_bytes(fig: plt.Figure) -> bytes:
    """
    Convert a matplotlib figure to PNG bytes for Streamlit display.

    Args:
        fig (plt.Figure): Matplotlib figure

    Returns:
        bytes: PNG image bytes
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()
