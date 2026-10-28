from pathlib import Path
import sys
import torch
import yaml
from torch.utils.data import DataLoader, Subset
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.dataset import load_urbanev, WindowDataset
from src.models.spatial_tft import SpatialTFT
from src.training.train import make_model, run_epoch


def test_spatial_tft_forward():
    b, t, h, n = 1, 12, 4, 5
    model = SpatialTFT(context_length=t, horizon_length=h, graph_input_dim=5,
        known_feature_names=["T", "P0", "P", "U", "Td", "day_of_week", "hour", "is_weekend", "nRAIN", "is_china_holiday"],
        known_input_cardinalities=[7, 24, 2, 4, 2],
        static_feature_names=["charge_count", "area", "perimeter"],
        hidden_dim=16, gnn_hidden_dim=8, num_heads=4, dropout=0.0)
    observed = torch.randn(b, t, n, 1)
    graph = torch.randn(b, t, n, 5)
    known_calendar = torch.stack([
        torch.randint(0, 7, (b, t+h, n)),
        torch.randint(0, 24, (b, t+h, n)),
        torch.randint(0, 2, (b, t+h, n)),
        torch.randint(0, 4, (b, t+h, n)),
        torch.randint(0, 2, (b, t+h, n)),
    ], dim=-1)
    known = torch.cat([torch.randn(b, t+h, n, 5), known_calendar], dim=-1)
    static = torch.randn(n, 3)
    adj = torch.eye(n)
    tft_inputs = {}
    gnn_outputs = {}

    def capture_tft_inputs(module, args, kwargs):
        tft_inputs["observed"] = kwargs["observed_inputs"].detach()
        tft_inputs["known"] = kwargs["known_inputs"].detach()

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
    torch.testing.assert_close(tft_inputs["known"], known.permute(0, 2, 1, 3).reshape(b * n, t + h, 10))
    assert model.tft.embeddings.known_inputs_categorical_cardinalities == [7, 24, 2, 4, 2]


def test_spatial_tft_lstm_initial_states_are_contiguous():
    b, t, h, n = 2, 12, 4, 5
    model = SpatialTFT(context_length=t, horizon_length=h, graph_input_dim=5,
        known_feature_names=["day_of_week", "hour", "is_weekend"],
        known_input_cardinalities=[7, 24, 2],
        static_feature_names=["charge_count", "area", "perimeter"],
        hidden_dim=16, gnn_hidden_dim=8, num_heads=4, dropout=0.0, lstm_layers=2)
    observed = torch.randn(b, t, n, 1)
    graph = torch.randn(b, t, n, 5)
    known = torch.stack([
        torch.randint(0, 7, (b, t+h, n)),
        torch.randint(0, 24, (b, t+h, n)),
        torch.randint(0, 2, (b, t+h, n)),
    ], dim=-1)
    static = torch.randn(n, 3)
    adjacency = torch.eye(n)
    initial_states = {}

    def capture_initial_states(*hook_args):
        initial_states["hidden"], initial_states["cell"] = hook_args[1][1]

    hook = model.tft.encoder_lstm.register_forward_pre_hook(capture_initial_states)
    model(observed, graph, known, static, adjacency)
    hook.remove()

    assert initial_states["hidden"].is_contiguous()
    assert initial_states["cell"].is_contiguous()


def test_weather_known_features_train_with_and_without_spatial():
    cfg = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text())
    cfg["data"].update(max_zones=2, context_length=4, horizon_length=2)
    cfg["model"].update(hidden_dim=8, gnn_hidden_dim=4, num_heads=2, lstm_layers=1, dropout=0.0)
    data = load_urbanev(cfg, ROOT)
    windows = WindowDataset(data, "train", context_length=4, horizon_length=2)
    loader = DataLoader(Subset(windows, [0]), batch_size=1)
    for use_spatial in (True, False):
        model = make_model(data, cfg, use_spatial=use_spatial)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        loss, _, _, _ = run_epoch(model, loader, data, torch.device("cpu"), optimizer)
        assert torch.isfinite(torch.tensor(loss))
