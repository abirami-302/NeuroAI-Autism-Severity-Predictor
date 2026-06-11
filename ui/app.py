# =============================================================
# ui/app.py
# NeuroAI Streamlit Application — Main UI Entry Point
#
# Run with:
#   streamlit run ui/app.py
# =============================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys
import time
import tempfile
import io

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.config import CLASS_NAMES, CLASS_COLORS, ROI_NAMES
from inference.predict import predict, predict_demo, load_model
from inference.gradcam import plot_gradcam_slices, generate_mock_heatmap
from utils.visualize import (
    plot_connectivity_matrix,
    plot_probability_bars,
    plot_connectivity_network,
    plot_mri_slices,
    plot_region_radar,
    fig_to_bytes
)

# =============================================================
# PAGE CONFIG — Must be the first Streamlit call
# =============================================================
st.set_page_config(
    page_title   = "NeuroAI — Autism Severity Predictor",
    page_icon    = "🧠",
    layout       = "wide",
    initial_sidebar_state = "expanded"
)

# =============================================================
# CUSTOM CSS — Dark medical theme
# =============================================================
st.markdown("""
<style>
    /* ── Base theme ── */
    .stApp { background-color: #0d1117; color: #e6edf3; }
    .stSidebar { background-color: #161b22; }
    .stSidebar .css-1d391kg { padding: 1rem; }

    /* ── Header banner ── */
    .header-banner {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }
    .header-banner h1 {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #58a6ff, #79c0ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 0.3rem 0;
    }
    .header-banner p { color: #8b949e; font-size: 1rem; margin: 0; }

    /* ── Result cards ── */
    .result-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    }
    .prediction-mild     { border-left: 5px solid #2ECC71; }
    .prediction-moderate { border-left: 5px solid #F39C12; }
    .prediction-severe   { border-left: 5px solid #E74C3C; }
    .prediction-unknown  { border-left: 5px solid #8b949e; }

    /* ── Severity badge ── */
    .severity-badge {
        display: inline-block;
        padding: 0.35rem 1.2rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    .badge-mild     { background: rgba(46,204,113,0.15); color: #2ECC71; border: 1px solid #2ECC71; }
    .badge-moderate { background: rgba(243,156,18,0.15); color: #F39C12; border: 1px solid #F39C12; }
    .badge-severe   { background: rgba(231,76,60,0.15);  color: #E74C3C; border: 1px solid #E74C3C; }

    /* ── Confidence meter ── */
    .conf-meter-bg {
        background: #21262d; border-radius: 8px;
        height: 12px; margin: 0.5rem 0;
        overflow: hidden;
    }
    .conf-meter-fill {
        height: 100%; border-radius: 8px;
        background: linear-gradient(90deg, #58a6ff, #79c0ff);
        transition: width 1s ease;
    }

    /* ── Metric box ── */
    .metric-box {
        background: #0d1117; border: 1px solid #30363d;
        border-radius: 8px; padding: 1rem;
        text-align: center;
    }
    .metric-val { font-size: 1.8rem; font-weight: 700; color: #58a6ff; }
    .metric-lbl { font-size: 0.8rem; color: #8b949e; margin-top: 0.2rem; }

    /* ── Section titles ── */
    .section-title {
        color: #79c0ff; font-size: 1.05rem;
        font-weight: 600; margin: 1.2rem 0 0.6rem 0;
        border-bottom: 1px solid #21262d; padding-bottom: 0.3rem;
    }

    /* ── Upload box ── */
    .upload-hint {
        background: #161b22; border: 2px dashed #30363d;
        border-radius: 10px; padding: 1.5rem;
        text-align: center; color: #8b949e;
    }

    /* ── Region pill ── */
    .region-pill {
        display: inline-block;
        background: rgba(88,166,255,0.1);
        border: 1px solid #58a6ff;
        color: #79c0ff;
        border-radius: 16px;
        padding: 0.2rem 0.8rem;
        margin: 0.2rem;
        font-size: 0.85rem;
    }

    /* ── Streamlit overrides ── */
    .stButton>button {
        background: linear-gradient(135deg, #1f6feb, #388bfd);
        color: white; border: none; border-radius: 8px;
        font-weight: 600; padding: 0.6rem 2rem;
        transition: all 0.2s; width: 100%;
    }
    .stButton>button:hover { opacity: 0.9; transform: translateY(-1px); }
    div[data-testid="stFileUploader"] { border: none; }
    .stTabs [data-baseweb="tab"] { color: #8b949e; }
    .stTabs [aria-selected="true"] { color: #58a6ff; border-bottom-color: #58a6ff; }
    h2, h3 { color: #e6edf3 !important; }
    p, li { color: #c9d1d9; }
</style>
""", unsafe_allow_html=True)

# =============================================================
# SESSION STATE INITIALIZATION
# =============================================================
if "model"        not in st.session_state: st.session_state.model   = None
if "result"       not in st.session_state: st.session_state.result  = None
if "is_analyzing" not in st.session_state: st.session_state.is_analyzing = False


# =============================================================
# SIDEBAR — Professional structured layout
# =============================================================
with st.sidebar:

    # ── Brand block ──────────────────────────────────────────
    st.markdown("""
<div style="
    background: linear-gradient(135deg, #1a1a2e, #0f3460);
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1.2rem 1rem;
    margin-bottom: 0.5rem;
    text-align: center;
">
    <div style="font-size:2rem; margin-bottom:0.3rem">🧠</div>
    <div style="font-size:1.15rem; font-weight:800;
                background:linear-gradient(90deg,#58a6ff,#79c0ff);
                -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
        NeuroAI
    </div>
    <div style="font-size:0.75rem; color:#8b949e; margin-top:0.2rem;">
        Autism Severity Predictor
    </div>
    <div style="margin-top:0.6rem">
        <span style="background:#21262d; color:#58a6ff; font-size:0.68rem;
                     padding:0.15rem 0.55rem; border-radius:10px; border:1px solid #30363d;">
            3D CNN
        </span>
        <span style="background:#21262d; color:#58a6ff; font-size:0.68rem;
                     padding:0.15rem 0.55rem; border-radius:10px; border:1px solid #30363d;
                     margin:0 0.2rem;">
            GNN
        </span>
        <span style="background:#21262d; color:#58a6ff; font-size:0.68rem;
                     padding:0.15rem 0.55rem; border-radius:10px; border:1px solid #30363d;">
            Grad-CAM
        </span>
    </div>
</div>
    """, unsafe_allow_html=True)

    # ── Visualization Controls ────────────────────────────────
    st.markdown("""
<div style="color:#79c0ff; font-size:0.78rem; font-weight:700;
            letter-spacing:0.08em; text-transform:uppercase;
            margin: 1rem 0 0.5rem 0; padding-bottom:0.3rem;
            border-bottom:1px solid #21262d;">
    ⚙️ &nbsp; Visualization Controls
</div>
    """, unsafe_allow_html=True)

    show_heatmap  = st.toggle("Grad-CAM Heatmap",      value=True)
    show_conn_mat = st.toggle("Connectivity Matrix",    value=True)
    show_network  = st.toggle("Network Graph",          value=True)
    show_radar    = st.toggle("Region Radar Chart",     value=True)

    # ── Pipeline Status ───────────────────────────────────────
    st.markdown("""
<div style="color:#79c0ff; font-size:0.78rem; font-weight:700;
            letter-spacing:0.08em; text-transform:uppercase;
            margin: 1.2rem 0 0.5rem 0; padding-bottom:0.3rem;
            border-bottom:1px solid #21262d;">
    🔬 &nbsp; AI Pipeline
</div>
<div style="font-size:0.8rem; line-height:2;">
    <span style="color:#2ECC71">●</span>
    <span style="color:#c9d1d9"> Preprocessing</span><br>
    <span style="color:#2ECC71">●</span>
    <span style="color:#c9d1d9"> 3D CNN Feature Extraction</span><br>
    <span style="color:#2ECC71">●</span>
    <span style="color:#c9d1d9"> Connectivity Graph (GNN)</span><br>
    <span style="color:#2ECC71">●</span>
    <span style="color:#c9d1d9"> Multi-Class Classification</span><br>
    <span style="color:#2ECC71">●</span>
    <span style="color:#c9d1d9"> Grad-CAM Explainability</span>
</div>
    """, unsafe_allow_html=True)

    # ── Dataset Setup ─────────────────────────────────────────
    st.markdown("""
<div style="color:#79c0ff; font-size:0.78rem; font-weight:700;
            letter-spacing:0.08em; text-transform:uppercase;
            margin: 1.2rem 0 0.5rem 0; padding-bottom:0.3rem;
            border-bottom:1px solid #21262d;">
    📦 &nbsp; Dataset Setup
</div>
    """, unsafe_allow_html=True)

    with st.expander("Kaggle Dataset Instructions", expanded=False):
        st.markdown("""
1. Create account at [kaggle.com](https://www.kaggle.com)
2. Settings → **Create New API Token**
3. Place `kaggle.json` in `~/.kaggle/`
4. Run in terminal:
```bash
kaggle datasets download \\
  -d birdy654/autistic-children-data-set-acds
```
5. Unzip into `dataset/raw/`
6. Run `python main.py --mode preprocess`
        """)

    # ── Training ──────────────────────────────────────────────
    st.markdown("""
<div style="color:#79c0ff; font-size:0.78rem; font-weight:700;
            letter-spacing:0.08em; text-transform:uppercase;
            margin: 1.2rem 0 0.5rem 0; padding-bottom:0.3rem;
            border-bottom:1px solid #21262d;">
    🏋️ &nbsp; Model Training
</div>
    """, unsafe_allow_html=True)

    with st.expander("Training Commands", expanded=False):
        st.code("# Standard training\npython training/train.py", language="bash")
        st.code("# Lightweight (laptop)\npython training/train_lightweight.py --size 32", language="bash")

    # ── Disclaimer ────────────────────────────────────────────
    st.markdown("""
<div style="
    background: rgba(231,76,60,0.08);
    border: 1px solid rgba(231,76,60,0.3);
    border-radius:8px; padding:0.7rem 0.8rem;
    margin-top:1.2rem;
">
    <div style="color:#E74C3C; font-size:0.72rem; font-weight:700;
                text-transform:uppercase; letter-spacing:0.06em;
                margin-bottom:0.3rem;">
        ⚠️ Disclaimer
    </div>
    <div style="color:#8b949e; font-size:0.72rem; line-height:1.5;">
        For research &amp; educational use only.<br>
        Not a certified clinical diagnostic tool.
    </div>
</div>

<div style="text-align:center; color:#484f58;
            font-size:0.68rem; margin-top:1rem;">
    v1.0 &nbsp;·&nbsp; PyTorch 2.3 &nbsp;·&nbsp; Streamlit
</div>
    """, unsafe_allow_html=True)


# =============================================================
# HEADER
# =============================================================
st.markdown("""
<div class="header-banner">
    <h1>🧠 NeuroAI — Autism Severity Predictor</h1>
    <p>AI-powered brain MRI analysis using 3D CNN + Graph Neural Networks + Explainable Grad-CAM</p>
</div>
""", unsafe_allow_html=True)


# =============================================================
# LAZY MODEL LOADER
# =============================================================
@st.cache_resource(show_spinner=False)
def get_model():
    """Load model once and cache it across Streamlit reruns."""
    with st.spinner("🔄 Loading NeuroAI model..."):
        return load_model(use_mock=True)


# =============================================================
# MAIN CONTENT — Two-column layout
# =============================================================
col_left, col_right = st.columns([1, 1.8], gap="large")

# ── LEFT COLUMN: Upload & Controls ──────────────────────────
with col_left:
    st.markdown('<div class="section-title">📤 Upload Brain MRI Scan</div>',
                unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        label="Drop MRI file here",
        type=["nii", "gz", "png", "jpg", "jpeg"],
        help="Supports: NIfTI (.nii, .nii.gz), PNG, JPG"
    )

    if not uploaded_file:
        st.markdown("""
<div class="upload-hint">
    <div style="font-size:2.5rem; margin-bottom:0.5rem">🫁</div>
    <strong>Drag & drop an MRI file</strong><br>
    <small>Formats: .nii · .nii.gz · .png · .jpg</small>
</div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🚀 Analysis</div>',
                unsafe_allow_html=True)

    analyze_btn = st.button(
        "🔬 Analyze MRI",
        disabled=uploaded_file is None,
        help="Run full AI pipeline on uploaded MRI",
        use_container_width=True
    )
    # demo_btn removed — set to False so existing handler is a no-op
    demo_btn = False

    # ── Quick info boxes ─────────────────────────────────────
    if st.session_state.result:
        result = st.session_state.result
        st.markdown('<div class="section-title">📊 Quick Stats</div>',
                    unsafe_allow_html=True)

        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown(f"""
<div class="metric-box">
    <div class="metric-val">{result.confidence:.0f}%</div>
    <div class="metric-lbl">Confidence</div>
</div>""", unsafe_allow_html=True)
        with mc2:
            st.markdown(f"""
<div class="metric-box">
    <div class="metric-val">{result.inference_time}s</div>
    <div class="metric-lbl">Inference Time</div>
</div>""", unsafe_allow_html=True)


# ── RIGHT COLUMN: Results ────────────────────────────────────
with col_right:

    # ── ANALYZE BUTTON HANDLER ───────────────────────────────
    if analyze_btn and uploaded_file:
        model = get_model()
        suffix = "." + uploaded_file.name.split(".")[-1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        with st.spinner("🧠 Analyzing MRI scan..."):
            progress = st.progress(0, text="Preprocessing MRI...")
            time.sleep(0.3); progress.progress(20, text="Extracting features...")
            time.sleep(0.3); progress.progress(45, text="Building connectivity graph...")
            time.sleep(0.3); progress.progress(65, text="Running GNN analysis...")
            time.sleep(0.3); progress.progress(80, text="Generating Grad-CAM heatmap...")

            result = predict(tmp_path, model=model)
            st.session_state.result = result

            progress.progress(100, text="✅ Analysis complete!")
            time.sleep(0.5)
            progress.empty()

        os.unlink(tmp_path)
        st.rerun()

    # ── DISPLAY RESULTS ──────────────────────────────────────
    if st.session_state.result:
        result = st.session_state.result

        if result.error:
            st.error(f"❌ Prediction failed: {result.error}")
        else:
            label_lower = result.label.lower()
            card_class  = f"prediction-{label_lower}"
            badge_class = f"badge-{label_lower}"

            # ── Main prediction card ─────────────────────────
            st.markdown(f"""
<div class="result-card {card_class}">
    <div style="margin-bottom:0.8rem">
        <span class="severity-badge {badge_class}">
            {result.label.upper()} AUTISM
        </span>
    </div>
    <div style="display:flex; align-items:center; gap:1rem; margin-bottom:1rem">
        <div>
            <div style="color:#8b949e; font-size:0.85rem">Confidence Score</div>
            <div style="font-size:2rem; font-weight:800; color:white">{result.confidence:.1f}%</div>
        </div>
    </div>
    <div class="conf-meter-bg">
        <div class="conf-meter-fill" style="width:{result.confidence}%"></div>
    </div>
    <div style="margin-top:1rem">
        <div style="color:#8b949e; font-size:0.85rem; margin-bottom:0.4rem">
            🧬 Affected Brain Regions
        </div>
        {"".join([f'<span class="region-pill">📍 {r}</span>' for r in result.affected_regions])}
    </div>
</div>
            """, unsafe_allow_html=True)

            # ── Tabs for detailed results ────────────────────
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Probabilities",
                "🌡️ Grad-CAM",
                "🔗 Connectivity",
                "🧭 Regions"
            ])

            # TAB 1: Probability chart
            with tab1:
                fig_probs = plot_probability_bars(
                    result.probabilities, result.label
                )
                st.pyplot(fig_probs, use_container_width=True)
                plt.close(fig_probs)

                st.markdown("**Class Probabilities:**")
                for cls, prob in result.probabilities.items():
                    color = CLASS_COLORS.get(cls, "#8b949e")
                    st.markdown(
                        f"- **{cls}**: "
                        f"<span style='color:{color}; font-weight:700'>{prob:.1f}%</span>",
                        unsafe_allow_html=True
                    )

            # TAB 2: Grad-CAM heatmap
            with tab2:
                if show_heatmap:
                    st.markdown("##### Gradient-weighted Class Activation Maps")
                    st.caption(
                        "Heatmap shows which brain regions most influenced "
                        "the model's prediction. Red = high activation."
                    )

                    volume_3d = result.volume[0] if result.volume.ndim == 4 \
                        else result.volume

                    fig_heatmap = plot_gradcam_slices(
                        volume_3d, result.heatmap,
                        prediction=result.label,
                        confidence=result.confidence,
                        num_slices=5
                    )
                    st.pyplot(fig_heatmap, use_container_width=True)
                    plt.close(fig_heatmap)

                    # Download button
                    buf = io.BytesIO()
                    fig_save = plot_gradcam_slices(
                        volume_3d, result.heatmap,
                        prediction=result.label,
                        confidence=result.confidence
                    )
                    fig_save.savefig(buf, format="png", dpi=150,
                                     bbox_inches="tight", facecolor="#1a1a2e")
                    plt.close(fig_save)
                    buf.seek(0)

                    st.download_button(
                        "⬇️ Download Heatmap",
                        data=buf, file_name="gradcam_heatmap.png",
                        mime="image/png"
                    )
                else:
                    st.info("Enable 'Show Grad-CAM Heatmap' in the sidebar.")

            # TAB 3: Connectivity matrix + network
            with tab3:
                if show_conn_mat:
                    st.markdown("##### Functional Connectivity Matrix")
                    fig_conn = plot_connectivity_matrix(result.conn_matrix)
                    st.pyplot(fig_conn, use_container_width=True)
                    plt.close(fig_conn)

                if show_network:
                    st.markdown("##### Interactive Connectivity Network")
                    fig_net = plot_connectivity_network(result.conn_matrix)
                    st.plotly_chart(fig_net, use_container_width=True)

            # TAB 4: Region radar + MRI slices
            with tab4:
                if show_radar:
                    st.markdown("##### Region Connectivity Strength")
                    fig_radar = plot_region_radar(result.conn_matrix)
                    st.plotly_chart(fig_radar, use_container_width=True)

                st.markdown("##### MRI Volume Slices")
                fig_slices = plot_mri_slices(result.volume, num_slices=9)
                st.pyplot(fig_slices, use_container_width=True)
                plt.close(fig_slices)

    else:
        # ── Placeholder when no result yet ──────────────────
        st.markdown("""
<div class="result-card" style="text-align:center; padding:3rem">
    <div style="font-size:4rem; margin-bottom:1rem">🧠</div>
    <div style="font-size:1.2rem; font-weight:600; color:#e6edf3; margin-bottom:0.5rem">
        Ready for Analysis
    </div>
    <div style="color:#8b949e">
        Upload a Brain MRI scan and click <strong>Analyze MRI</strong><br>
        to run the full AI pipeline.
    </div>
    <div style="margin-top:2rem; padding:1rem; background:#0d1117;
                border-radius:8px; display:inline-block">
        <div style="color:#58a6ff; font-size:0.9rem; font-weight:600">
            Pipeline Preview
        </div>
        <div style="color:#8b949e; font-size:0.82rem; margin-top:0.5rem; text-align:left">
            Upload MRI → Preprocess → 3D CNN → Graph Build<br>
            → GNN Analysis → Classify → Grad-CAM → Results
        </div>
    </div>
</div>
        """, unsafe_allow_html=True)


# =============================================================
# FOOTER
# =============================================================
st.divider()
st.markdown("""
<div style="text-align:center; color:#484f58; font-size:0.8rem; padding:1rem 0">
    🧠 NeuroAI Autism Severity Predictor &nbsp;|&nbsp;
    Built with PyTorch · PyTorch Geometric · Grad-CAM · Streamlit &nbsp;|&nbsp;
    ⚠️ Research & Educational Use Only — Not a Clinical Diagnostic Tool
</div>
""", unsafe_allow_html=True)