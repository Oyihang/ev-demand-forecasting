from pathlib import Path
import sys
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.spatial_tft import SpatialTFT


def test_spatial_tft_forward():
    b, t, h, n = 1, 12, 4, 5
    model = SpatialTFT(context_length=t, horizon_length=h, graph_input_dim=5,
        known_feature_names=["day_of_week", "hour", "is_weekend"],
        known_input_cardinalities=[7, 24, 2],
        static_feature_names=["charge_count", "area", "perimeter"],
        hidden_dim=16, gnn_hidden_dim=8, num_heads=4, dropout=0.0)
    observed = torch.randn(b, t, n, 1)
    graph = torch.randn(b, t, n, 5)
    known_calendar = torch.stack([
        torch.randint(0, 7, (b, t+h, n)),
        torch.randint(0, 24, (b, t+h, n)),
        torch.randint(0, 2, (b, t+h, n)),
    ], dim=-1)
    known = known_calendar
    static = torch.randn(n, 3)
    adj = torch.eye(n)
    tft_inputs = {}
    gnn_outputs = {}

    def capture_tft_inputs(module, args, kwargs):
        tft_inputs["observed"] = kwargs["observed_inputs"].detach()

    def capture_gnn_outputs(module, args, output):
        gnn_outputs["sequence"] = output.detach()

    hook = model.tft.register_forward_pre_hook(capture_tft_inputs, with_kwargs=True)
    gnn_hook = model.spatial_encoder.register_forward_hook(capture_gnn_outputs)
    out = model(observed, graph, known, static, adj)
    hook.remove()
    gnn_hook.remove()
    assert out.shape == (b, n, h, 3)
    assert torch.isfinite(out).all()
    assert tft_inputs["observed"].shape == (b * n, t, 1 + model.spatial_encoder.output_dim)
    assert gnn_outputs["sequence"].shape == (b, t, n, model.spatial_encoder.output_dim)
    assert model.tft.embeddings.known_inputs_categorical_cardinalities == [7, 24, 2]
