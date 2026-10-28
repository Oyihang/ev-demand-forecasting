from pathlib import Path
import sys
import yaml
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.dataset import load_urbanev, WindowDataset


def test_data_shapes_and_graph():
    cfg = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text())
    cfg["data"]["max_zones"] = 8
    data = load_urbanev(cfg, ROOT)
    assert data.target.shape[1] == 8
    assert data.graph_features.shape[:2] == data.target.shape
    assert data.feature_names["graph"] == [
        "duration", "electricity_price", "service_price", "occupancy", "volume"
    ]
    assert data.graph_features.shape[-1] == 5
    assert data.feature_names["static"] == ["charge_count", "area", "perimeter"]
    assert data.static.shape == (8, 3)
    assert data.feature_names["known"] == ["day_of_week", "hour", "is_weekend"]
    assert data.known.shape == (len(data.times), 3)
    assert data.known.dtype == np.int64
    assert np.all((0 <= data.known[:, 0]) & (data.known[:, 0] < 7))
    assert np.all((0 <= data.known[:, 1]) & (data.known[:, 1] < 24))
    assert np.all((0 <= data.known[:, 2]) & (data.known[:, 2] < 2))
    assert data.adjacency.shape == (8, 8)
    assert np.isfinite(data.target).all()
    assert np.allclose(data.adjacency, data.adjacency.T, atol=1e-5)
    ds = WindowDataset(data, "train", 168, 24, stride=12)
    sample = ds[0]
    assert sample["observed"].shape == (168, 8, 1)
    assert sample["graph_features"].shape == (168, 8, 5)
    assert sample["known"].shape == (192, 8, 3)
    assert sample["known"].dtype == torch.int64
    assert sample["target"].shape == (8, 24)


def test_forecast_targets_stay_inside_split():
    cfg = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text())
    cfg["data"]["max_zones"] = 4
    data = load_urbanev(cfg, ROOT)
    context, horizon = cfg["data"]["context_length"], cfg["data"]["horizon_length"]
    for split, lo, hi in [("train", context, data.split_train_end),
                          ("val", data.split_train_end, data.split_val_end),
                          ("test", data.split_val_end, len(data.times))]:
        ds = WindowDataset(data, split, context, horizon, stride=6)
        assert all(start >= lo and start + horizon <= hi for start in ds.starts)
