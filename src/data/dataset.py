"""UrbanEV hourly data loader, chronological splits, and window dataset."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import torch
from chinese_calendar import get_holiday_detail
from torch.utils.data import Dataset


@dataclass
class PreparedData:
    times: pd.DatetimeIndex
    zone_ids: List[str]
    target: np.ndarray            # (T,N), log1p transformed and train-normalized
    graph_features: np.ndarray   # (T,N,F), normalized dynamic signals
    known: np.ndarray            # (T,K), weather and calendar features known ahead
    static: np.ndarray           # (N,S), static zone features
    adjacency: np.ndarray        # (N,N), normalized with self-loops
    target_mean: np.ndarray      # (N,)
    target_std: np.ndarray       # (N,)
    split_train_end: int
    split_val_end: int
    feature_names: Dict[str, List[str]]


def _read_wide(path: Path, time_col: bool = True) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required dataset file not found: {path}")
    df = pd.read_csv(path)
    if time_col:
        if "time" not in df.columns:
            raise ValueError(f"Expected a 'time' column in {path}")
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"]).drop_duplicates("time").sort_values("time").set_index("time")
    return df


def _normalize_adjacency(adj: np.ndarray) -> np.ndarray:
    adj = np.nan_to_num(adj.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    # Dataset adjacency may be binary or weighted. Keep nonnegative edge weights.
    adj = np.maximum(adj, 0)
    adj = (adj + adj.T) / 2.0
    adj = adj + np.eye(adj.shape[0], dtype=np.float32)
    degree = adj.sum(axis=1).clip(min=1e-8)
    inv_sqrt = 1.0 / np.sqrt(degree)
    return (inv_sqrt[:, None] * adj * inv_sqrt[None, :]).astype(np.float32)


def _china_holiday_flags(times: pd.DatetimeIndex) -> np.ndarray:
    dates = times.date
    flags = {}
    for day in np.unique(dates):
        try:
            is_holiday, holiday_name = get_holiday_detail(day)
        except NotImplementedError as exc:
            raise ValueError(f"Chinese holiday calendar does not cover {day}: {exc}") from exc
        flags[day] = int(is_holiday and holiday_name is not None)
    return np.array([flags[day] for day in dates], dtype=np.int64)


def load_urbanev(config: dict, root_dir: str | Path = ".") -> PreparedData:
    root_dir = Path(root_dir)
    dcfg = config["data"]
    data_root = root_dir / dcfg["root"]
    freq_dir = data_root / dcfg.get("frequency_dir", "charge_1hour")

    target_df = _read_wide(freq_dir / dcfg.get("target_file", "volume.csv"))
    target_df.columns = target_df.columns.astype(str)
    zone_ids = list(target_df.columns)
    times = target_df.index

    # Align all dynamic files strictly by timestamps and zone IDs.
    dynamic_files = {
        "occupancy": dcfg.get("occupancy_file", "occupancy.csv"),
        "duration": dcfg.get("duration_file", "duration.csv"),
        "electricity_price": dcfg.get("electricity_price_file", "e_price.csv"),
        "service_price": dcfg.get("service_price_file", "s_price.csv"),
    }
    dynamic_frames = {}
    for name, filename in dynamic_files.items():
        frame = _read_wide(freq_dir / filename)
        frame.columns = frame.columns.astype(str)
        dynamic_frames[name] = frame.reindex(index=times, columns=zone_ids)

    # Align graph matrices to target zone order; their column names are zone IDs.
    adjacency_df = _read_wide(data_root / dcfg.get("adjacency_file", "adj.csv"), time_col=False)
    adjacency_df.columns = adjacency_df.columns.astype(str)
    # The published matrix is header-only (row IDs are omitted); row order matches column order.
    if len(adjacency_df) == len(adjacency_df.columns):
        adjacency_df.index = adjacency_df.columns.copy()
    else:
        adjacency_df.index = adjacency_df.index.astype(str)
    adjacency_df = adjacency_df.reindex(index=zone_ids, columns=zone_ids).fillna(0)
    raw_adjacency = adjacency_df.to_numpy(dtype=np.float32)
    graph_type = dcfg.get("graph_type", "adjacency")
    if graph_type == "distance_weighted":
        distance_path = data_root / dcfg.get("distance_file", "distance.csv")
        distance_df = _read_wide(distance_path, time_col=False)
        distance_df.columns = distance_df.columns.astype(str)
        if len(distance_df) == len(distance_df.columns):
            distance_df.index = distance_df.columns.copy()
        else:
            distance_df.index = distance_df.index.astype(str)
        distance_df = distance_df.reindex(index=zone_ids, columns=zone_ids).fillna(np.inf)
        distances = distance_df.to_numpy(dtype=np.float32)
        k = min(int(dcfg.get("distance_k", 8)), max(1, len(zone_ids) - 1))
        finite_positive = distances[np.isfinite(distances) & (distances > 0)]
        sigma = float(np.median(finite_positive)) if finite_positive.size else 1.0
        weights = np.exp(-distances / max(sigma, 1e-6)).astype(np.float32)
        np.fill_diagonal(weights, 0.0)
        # Keep only each node's k nearest neighbors, then symmetrize.
        knn = np.zeros_like(weights)
        for row in range(len(zone_ids)):
            nearest = np.argsort(distances[row])
            nearest = [j for j in nearest if j != row and np.isfinite(distances[row, j])][:k]
            knn[row, nearest] = weights[row, nearest]
        raw_adjacency = np.maximum(knn, knn.T)
    elif graph_type != "adjacency":
        raise ValueError("data.graph_type must be 'adjacency' or 'distance_weighted'")
    adjacency = _normalize_adjacency(raw_adjacency)

    # Define chronological split boundaries before fitting any time-dependent scalers.
    total_t = len(times)
    train_end = int(total_t * float(dcfg.get("train_ratio", 0.70)))
    val_end = train_end + int(total_t * float(dcfg.get("val_ratio", 0.15)))
    if not (0 < train_end < val_end < total_t):
        raise ValueError(
            f"Invalid chronological split: train_end={train_end}, val_end={val_end}, total={total_t}. "
            "Check train_ratio and val_ratio."
        )

    # Keep only the requested time-invariant zone attributes.
    info_path = data_root / dcfg.get("zone_info_file", "zone-information.csv")
    info = pd.read_csv(info_path)
    if "TAZID" not in info.columns:
        raise ValueError("zone-information.csv must contain TAZID")
    info["TAZID"] = info["TAZID"].astype(str)
    info = info.drop_duplicates("TAZID").set_index("TAZID").reindex(zone_ids)
    static_cols = ["charge_count", "area", "perimeter"]
    missing_static = [name for name in static_cols if name not in info.columns]
    if missing_static:
        raise ValueError(f"zone-information.csv is missing static features: {missing_static}")
    static_raw = info[static_cols].apply(pd.to_numeric, errors="coerce")
    static_raw = static_raw.replace([np.inf, -np.inf], np.nan)
    static_raw = static_raw.fillna(static_raw.median()).fillna(0.0)
    static_names = list(static_raw.columns)
    static_values = static_raw.to_numpy(dtype=np.float32)
    s_mean = static_values.mean(axis=0, keepdims=True)
    s_std = static_values.std(axis=0, keepdims=True)
    static_values = (static_values - s_mean) / np.where(s_std < 1e-6, 1.0, s_std)

    target_raw = target_df.to_numpy(dtype=np.float32)
    target_raw = np.nan_to_num(target_raw, nan=0.0, posinf=0.0, neginf=0.0).clip(min=0)
    target_log = np.log1p(target_raw)
    target_mean = target_log[:train_end].mean(axis=0)
    target_std = target_log[:train_end].std(axis=0)
    target_std = np.where(target_std < 1e-6, 1.0, target_std)
    target_scaled = ((target_log - target_mean) / target_std).astype(np.float32)

    graph_arrays = []
    graph_names = ["duration", "electricity_price", "service_price", "occupancy", "volume"]
    for name in graph_names[:-1]:
        arr = dynamic_frames[name].to_numpy(dtype=np.float32)
        # Impute with forward-fill and then zero; no future values are used to fill earlier rows.
        frame = pd.DataFrame(arr).ffill().fillna(0.0)
        arr = frame.to_numpy(dtype=np.float32)
        # Train-only feature scaling per feature across time and zones.
        mean = arr[:train_end].mean()
        std = arr[:train_end].std()
        graph_arrays.append(((arr - mean) / (std if std > 1e-6 else 1.0)).astype(np.float32))
    graph_arrays.append(target_scaled)

    # Weather observations are treated as perfect forecasts for future timesteps.
    weather_path = data_root / dcfg.get("weather_file", "weather_central.csv")
    weather = pd.read_csv(weather_path)
    weather_cols = ["T", "P0", "P", "U", "Td", "nRAIN"]
    missing_cols = [name for name in ["time"] + weather_cols if name not in weather.columns]
    if missing_cols:
        raise ValueError(f"{weather_path} is missing weather columns: {missing_cols}")
    weather["time"] = pd.to_datetime(weather["time"], errors="coerce")
    if weather["time"].isna().any() or weather["time"].duplicated().any():
        raise ValueError(f"{weather_path} contains invalid or duplicate timestamps")
    weather = weather.set_index("time").reindex(times)
    weather_values = weather[weather_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    if not np.isfinite(weather_values).all():
        raise ValueError(f"{weather_path} contains missing or invalid values for target timestamps")
    rain = weather_values[:, -1]
    if not np.all((rain >= 0) & (rain <= 3) & (rain == np.floor(rain))):
        raise ValueError(f"{weather_path} nRAIN must contain integer categories from 0 to 3")
    weather_continuous = weather_values[:, :-1]
    weather_mean = weather_continuous[:train_end].mean(axis=0)
    weather_std = weather_continuous[:train_end].std(axis=0)
    weather_scaled = (weather_continuous - weather_mean) / np.where(weather_std < 1e-6, 1.0, weather_std)

    # Continuous inputs precede categorical inputs for the TFT embeddings.
    idx = pd.DatetimeIndex(times)
    dow = idx.dayofweek.to_numpy(dtype=np.int64)
    known = np.column_stack([
        weather_scaled,
        dow,
        idx.hour.to_numpy(dtype=np.int64),
        (dow >= 5).astype(np.int64),
        rain,
        _china_holiday_flags(idx),
    ]).astype(np.float32)
    known_names = [
        "T", "P0", "P", "U", "Td", "day_of_week", "hour", "is_weekend", "nRAIN", "is_china_holiday"
    ]
    graph_features = np.stack(graph_arrays, axis=-1).astype(np.float32)

    max_zones = dcfg.get("max_zones")
    if max_zones is not None:
        n = min(int(max_zones), len(zone_ids))
        zone_ids = zone_ids[:n]
        target_scaled = target_scaled[:, :n]
        graph_features = graph_features[:, :n, :]
        static_values = static_values[:n]
        adjacency = _normalize_adjacency(raw_adjacency[:n, :n])

    return PreparedData(
        times=pd.DatetimeIndex(times), zone_ids=zone_ids, target=target_scaled,
        graph_features=graph_features, known=known, static=static_values,
        adjacency=adjacency, target_mean=target_mean[:len(zone_ids)].astype(np.float32),
        target_std=target_std[:len(zone_ids)].astype(np.float32),
        split_train_end=train_end, split_val_end=val_end,
        feature_names={"known": known_names, "graph": graph_names, "static": static_names},
    )


class WindowDataset(Dataset):
    """Returns windows as (T,N,F), retaining all zones for graph message passing."""
    def __init__(self, data: PreparedData, split: str, context_length: int, horizon_length: int, stride: int = 1):
        self.data = data
        self.context = int(context_length)
        self.horizon = int(horizon_length)
        self.stride = max(1, int(stride))
        if split == "train":
            self.start_min, self.start_max = self.context, data.split_train_end - self.horizon
        elif split == "val":
            self.start_min, self.start_max = data.split_train_end, data.split_val_end - self.horizon
        elif split == "test":
            self.start_min, self.start_max = data.split_val_end, len(data.times) - self.horizon
        else:
            raise ValueError("split must be train, val, or test")
        # The forecast target is always wholly inside its split. Validation/test history may
        # use prior observations, as would happen in real rolling forecasting.
        self.starts = list(range(self.start_min, self.start_max + 1, self.stride))
        if not self.starts:
            raise ValueError(f"No windows for split={split}; check dataset length and split ratios")

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        start = self.starts[idx]
        c, h = self.context, self.horizon
        history_start = start - c
        obs = self.data.target[history_start:start, :, None]
        graph = self.data.graph_features[history_start:start]
        known = self.data.known[history_start:start + h]
        known = np.repeat(known[:, None, :], len(self.data.zone_ids), axis=1)
        target = self.data.target[start:start + h]
        return {
            "observed": torch.from_numpy(obs.copy()),
            "graph_features": torch.from_numpy(graph.copy()),
            "known": torch.from_numpy(known.copy()),
            "target": torch.from_numpy(target.T.copy()),  # N,H
        }
