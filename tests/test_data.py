from pathlib import Path
import sys
import yaml
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.dataset import load_urbanev, WindowDataset


def test_data_shapes_and_graph():
    cfg = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text())
    cfg["data"]["max_zones"] = 8
    data = load_urbanev(cfg, ROOT)
    assert data.target.shape[1] == 8
    assert data.graph_features.shape[:2] == data.target.shape
    assert data.adjacency.shape == (8, 8)
    assert np.isfinite(data.target).all()
    assert np.allclose(data.adjacency, data.adjacency.T, atol=1e-5)
    ds = WindowDataset(data, "train", 168, 24, stride=12)
    sample = ds[0]
    assert sample["observed"].shape == (168, 8, 1)
    assert sample["known"].shape == (192, 8, 7)
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
