from pathlib import Path
import sys
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.spatial_tft import SpatialTFT


def test_spatial_tft_forward():
    b, t, h, n = 1, 12, 4, 5
    model = SpatialTFT(context_length=t, horizon_length=h, graph_input_dim=3,
        known_feature_names=[f"k{i}" for i in range(7)], static_feature_names=["s0", "s1"],
        hidden_dim=16, gnn_hidden_dim=8, num_heads=4, dropout=0.0)
    observed = torch.randn(b, t, n, 1)
    graph = torch.randn(b, t, n, 3)
    known = torch.randn(b, t+h, n, 7)
    static = torch.randn(n, 2)
    adj = torch.eye(n)
    out = model(observed, graph, known, static, adj)
    assert out.shape == (b, n, h, 3)
    assert torch.isfinite(out).all()
