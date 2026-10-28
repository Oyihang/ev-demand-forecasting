from pathlib import Path
import sys
import yaml
import numpy as np
import pandas as pd
import pytest
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.dataset import _china_holiday_flags, load_urbanev, WindowDataset


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
    assert data.feature_names["known"] == [
        "T", "P0", "P", "U", "Td", "day_of_week", "hour", "is_weekend", "nRAIN", "is_china_holiday"
    ]
    assert data.known.shape == (len(data.times), 10)
    assert data.known.dtype == np.float32
    assert np.isfinite(data.known).all()
    assert np.allclose(data.known[:data.split_train_end, :5].mean(axis=0), 0, atol=1e-4)
    assert np.allclose(data.known[:data.split_train_end, :5].std(axis=0), 1, atol=1e-4)
    for column, cardinality in enumerate([7, 24, 2, 4, 2], start=5):
        assert np.all((0 <= data.known[:, column]) & (data.known[:, column] < cardinality))
        assert np.all(data.known[:, column] == np.floor(data.known[:, column]))
    assert data.adjacency.shape == (8, 8)
    assert np.isfinite(data.target).all()
    assert np.allclose(data.adjacency, data.adjacency.T, atol=1e-5)
    ds = WindowDataset(data, "train", 168, 24, stride=12)
    sample = ds[0]
    assert sample["observed"].shape == (168, 8, 1)
    assert sample["graph_features"].shape == (168, 8, 5)
    assert sample["known"].shape == (192, 8, 10)
    assert sample["known"].dtype == torch.float32
    start = ds.starts[0]
    np.testing.assert_array_equal(sample["known"][:, 0].numpy(), data.known[start - 168:start + 24])
    np.testing.assert_array_equal(sample["known"][:, 0].numpy(), sample["known"][:, 7].numpy())
    assert np.any(sample["known"][:, :, :5].numpy() % 1 != 0)
    assert sample["target"].shape == (8, 24)


@pytest.mark.parametrize(("day", "expected"), [
    ("2022-09-09", 0),  # Friday before Mid-Autumn
    ("2022-09-10", 1),  # Mid-Autumn weekend
    ("2022-09-12", 1),  # Mid-Autumn observed day off
    ("2022-09-17", 0),  # ordinary Saturday
    ("2022-10-01", 1),
    ("2022-10-07", 1),
    ("2022-10-08", 0),  # compensatory working Saturday
    ("2022-12-31", 1),  # New Year holiday starts in the previous year
    ("2023-01-02", 1),
    ("2023-01-21", 1),
    ("2023-01-27", 1),
    ("2023-01-28", 0),  # compensatory working Saturday
    ("2023-01-29", 0),  # compensatory working Sunday
    ("2023-02-05", 0),  # ordinary Sunday
])
def test_china_holiday_dates(day, expected):
    times = pd.date_range(day, periods=24, freq="h")
    np.testing.assert_array_equal(_china_holiday_flags(times), np.full(24, expected))


def test_china_holiday_unsupported_year():
    with pytest.raises(ValueError, match="Chinese holiday calendar does not cover 2027-01-01"):
        _china_holiday_flags(pd.DatetimeIndex(["2027-01-01"]))


def test_china_holiday_reaches_future_known_window():
    cfg = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text())
    cfg["data"]["max_zones"] = 2
    data = load_urbanev(cfg, ROOT)
    start = data.times.get_loc(pd.Timestamp("2022-12-30 12:00"))
    windows = WindowDataset(data, "train", context_length=168, horizon_length=24)
    sample = windows[windows.starts.index(start)]
    future_holidays = sample["known"][168:, :, -1].numpy()
    np.testing.assert_array_equal(future_holidays[:12], np.zeros((12, 2)))
    np.testing.assert_array_equal(future_holidays[12:], np.ones((12, 2)))


def test_weather_alignment_scaling_and_future_window():
    cfg = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text())
    cfg["data"]["max_zones"] = 2
    data = load_urbanev(cfg, ROOT)
    path = ROOT / cfg["data"]["root"] / cfg["data"]["weather_file"]
    weather = pd.read_csv(path, parse_dates=["time"]).set_index("time").reindex(data.times)
    raw = weather[["T", "P0", "P", "U", "Td"]].to_numpy(dtype=np.float32)
    mean = raw[:data.split_train_end].mean(axis=0)
    std = raw[:data.split_train_end].std(axis=0)
    expected = (raw - mean) / np.where(std < 1e-6, 1.0, std)
    np.testing.assert_allclose(data.known[:, :5], expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(data.known[:, 8], weather["nRAIN"].to_numpy())
    ds = WindowDataset(data, "test", 168, 24)
    start = ds.starts[0]
    np.testing.assert_allclose(ds[0]["known"][168:, 0, :5].numpy(), expected[start:start + 24])


def test_weather_rejects_missing_or_invalid_values(tmp_path):
    cfg = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text())
    cfg["data"]["max_zones"] = 2
    path = ROOT / cfg["data"]["root"] / cfg["data"]["weather_file"]
    weather = pd.read_csv(path)
    weather.loc[0, "nRAIN"] = 4
    invalid_path = tmp_path / "weather.csv"
    weather.to_csv(invalid_path, index=False)
    cfg["data"]["weather_file"] = str(invalid_path)
    with pytest.raises(ValueError, match="nRAIN"):
        load_urbanev(cfg, ROOT)
    weather = weather.iloc[1:]
    weather.to_csv(invalid_path, index=False)
    with pytest.raises(ValueError, match="missing or invalid values"):
        load_urbanev(cfg, ROOT)


def test_weather_scaling_uses_only_training_period(tmp_path):
    cfg = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text())
    cfg["data"]["max_zones"] = 2
    original = load_urbanev(cfg, ROOT)
    path = ROOT / cfg["data"]["root"] / cfg["data"]["weather_file"]
    weather = pd.read_csv(path)
    weather.loc[original.split_train_end:, "T"] += 100
    updated_path = tmp_path / "weather.csv"
    weather.to_csv(updated_path, index=False)
    cfg["data"]["weather_file"] = str(updated_path)
    updated = load_urbanev(cfg, ROOT)
    np.testing.assert_array_equal(
        updated.known[:original.split_train_end], original.known[:original.split_train_end]
    )
    assert updated.known[original.split_train_end, 0] > original.known[original.split_train_end, 0]


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
