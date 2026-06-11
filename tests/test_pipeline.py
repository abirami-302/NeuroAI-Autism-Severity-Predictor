# =============================================================
# tests/test_pipeline.py
# Unit tests for all NeuroAI modules.
# Run with: pytest tests/ -v
# =============================================================

import pytest
import numpy as np
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================
# TEST: PREPROCESSING
# =============================================================

class TestPreprocessing:

    def test_normalize_zscore(self):
        from preprocessing.pipeline import normalize
        vol = np.random.randn(64, 64, 64).astype(np.float32) * 500 + 1000
        out = normalize(vol, method="zscore")
        assert abs(out.mean()) < 0.1, "Z-score mean should be ~0"
        assert abs(out.std() - 1.0) < 0.1, "Z-score std should be ~1"

    def test_normalize_minmax(self):
        from preprocessing.pipeline import normalize
        vol = np.random.randn(64, 64, 64).astype(np.float32) * 300
        out = normalize(vol, method="minmax")
        assert out.min() >= 0.0
        assert out.max() <= 1.0 + 1e-5

    def test_resize_volume(self):
        from preprocessing.pipeline import resize_volume
        vol = np.random.randn(91, 109, 91).astype(np.float32)
        out = resize_volume(vol, target_size=(64, 64, 64))
        assert out.shape == (64, 64, 64)

    def test_add_channel_dim(self):
        from preprocessing.pipeline import add_channel_dim
        vol = np.random.randn(64, 64, 64).astype(np.float32)
        out = add_channel_dim(vol)
        assert out.shape == (1, 64, 64, 64)

    def test_clip_intensities(self):
        from preprocessing.pipeline import clip_intensities
        vol = np.random.randn(64, 64, 64).astype(np.float32) * 1000
        out = clip_intensities(vol, percentiles=(1, 99))
        p1  = np.percentile(vol, 1)
        p99 = np.percentile(vol, 99)
        assert out.min() >= p1 - 1e-3
        assert out.max() <= p99 + 1e-3


# =============================================================
# TEST: GRAPH BUILDER
# =============================================================

class TestGraphBuilder:

    def test_roi_signal_extraction(self):
        from graph.graph_builder import extract_roi_signals
        vol = np.random.randn(1, 64, 64, 64).astype(np.float32)
        signals = extract_roi_signals(vol, num_roi=10)
        assert signals.shape == (10, 16), f"Expected (10,16), got {signals.shape}"

    def test_connectivity_matrix(self):
        from graph.graph_builder import build_connectivity_matrix
        signals = np.random.randn(10, 16).astype(np.float32)
        conn = build_connectivity_matrix(signals)
        assert conn.shape == (10, 10)
        # Diagonal should be 1 (self-correlation)
        assert all(abs(conn[i, i] - 1.0) < 1e-4 for i in range(10))

    def test_adjacency_matrix(self):
        from graph.graph_builder import build_adjacency_matrix
        conn = np.random.uniform(-1, 1, (10, 10)).astype(np.float32)
        np.fill_diagonal(conn, 1.0)
        adj = build_adjacency_matrix(conn, threshold=0.3)
        # Diagonal should be 0 (no self-loops)
        assert all(adj[i, i] == 0 for i in range(10))
        # Values should be 0 or 1
        assert set(np.unique(adj)).issubset({0.0, 1.0})

    def test_volume_to_graph(self):
        from graph.graph_builder import volume_to_graph
        vol = np.random.randn(1, 64, 64, 64).astype(np.float32)
        graph, conn = volume_to_graph(vol)
        assert graph.num_nodes == 10
        assert graph.x.shape == (10, 16)
        assert graph.edge_index.shape[0] == 2


# =============================================================
# TEST: MODELS
# =============================================================

class TestCNN3D:

    def test_cnn3d_output_shape(self):
        from models.cnn3d import CNN3D
        model = CNN3D()
        model.eval()
        x = torch.randn(2, 1, 64, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 128), f"Expected (2,128), got {out.shape}"

    def test_cnn3d_classifier_shape(self):
        from models.cnn3d import CNN3DClassifier
        model = CNN3DClassifier(num_classes=3)
        model.eval()
        x = torch.randn(2, 1, 64, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 3), f"Expected (2,3), got {out.shape}"

    def test_cnn3d_no_nan(self):
        from models.cnn3d import CNN3DClassifier
        model = CNN3DClassifier()
        model.eval()
        x = torch.randn(1, 1, 64, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert not torch.isnan(out).any(), "Output contains NaN values"


class TestGNN:

    def test_gnn_output_shape(self):
        from models.gnn import BrainGNN, create_dummy_graph
        model = BrainGNN()
        model.eval()
        graph = create_dummy_graph(batch_size=2)
        with torch.no_grad():
            out = model(graph.x, graph.edge_index, graph.batch)
        assert out.shape == (2, 32), f"Expected (2,32), got {out.shape}"

    def test_combined_classifier(self):
        from models.gnn import NeuroAIClassifier, create_dummy_graph
        from training.config import CNN_FEATURE_DIM
        model = NeuroAIClassifier()
        model.eval()
        graph = create_dummy_graph(batch_size=2)
        mri_feats = torch.randn(2, CNN_FEATURE_DIM)
        with torch.no_grad():
            out = model(mri_feats, graph)
        assert out.shape == (2, 3)


# =============================================================
# TEST: GRAD-CAM
# =============================================================

class TestGradCAM:

    def test_mock_heatmap_shape(self):
        from inference.gradcam import generate_mock_heatmap
        heatmap = generate_mock_heatmap((64, 64, 64))
        assert heatmap.shape == (64, 64, 64)
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0 + 1e-5

    def test_overlay_shape(self):
        from inference.gradcam import overlay_heatmap_on_slice
        mri_slice  = np.random.randn(64, 64).astype(np.float32)
        heat_slice = np.random.uniform(0, 1, (64, 64)).astype(np.float32)
        overlay = overlay_heatmap_on_slice(mri_slice, heat_slice)
        assert overlay.shape == (64, 64, 3)
        assert overlay.dtype == np.uint8

    def test_plot_gradcam_creates_figure(self):
        import matplotlib.pyplot as plt
        from inference.gradcam import plot_gradcam_slices, generate_mock_heatmap
        volume  = np.random.randn(64, 64, 64).astype(np.float32)
        heatmap = generate_mock_heatmap((64, 64, 64))
        fig = plot_gradcam_slices(volume, heatmap, "Mild", 85.0, num_slices=3)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# =============================================================
# TEST: PREDICTION PIPELINE
# =============================================================

class TestPrediction:

    def test_predict_demo_returns_result(self):
        from inference.predict import predict_demo, PredictionResult
        result = predict_demo()
        assert isinstance(result, PredictionResult)
        assert result.label in ["Mild", "Moderate", "Severe"]
        assert 0 <= result.confidence <= 100
        assert result.heatmap.shape == (64, 64, 64)
        assert result.error is None

    def test_probabilities_sum_to_100(self):
        from inference.predict import predict_demo
        result = predict_demo()
        total = sum(result.probabilities.values())
        assert abs(total - 100.0) < 1.0, f"Probs sum to {total}, expected ~100"

    def test_affected_regions_are_valid(self):
        from inference.predict import predict_demo
        from training.config import ROI_NAMES
        result = predict_demo()
        for region in result.affected_regions:
            assert region in ROI_NAMES, f"Unknown region: {region}"


# =============================================================
# Run tests directly
# =============================================================
if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
