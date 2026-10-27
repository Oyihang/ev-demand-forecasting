"""Evaluate persistence and 24-hour seasonal-naive baselines on the same test windows."""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.dataset import load_urbanev, WindowDataset
from src.training.metrics import regression_metrics

cfg = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text())
data = load_urbanev(cfg, ROOT)
dcfg = cfg["data"]
ds = WindowDataset(data, "test", dcfg["context_length"], dcfg["horizon_length"], stride=int(dcfg.get("test_stride", 1)))
y_true, persistence, seasonal = [], [], []
for start in ds.starts:
    # Predictions are made at 'start'; use the most recent observed target and the
    # same hour one day earlier, both of which are available at forecast time.
    y_true.append(data.target[start:start + ds.horizon].T)
    persistence.append(np.repeat(data.target[start - 1:start, :].T, ds.horizon, axis=1))
    seasonal.append(data.target[start - 24:start - 24 + ds.horizon].T if start >= 24 else np.repeat(data.target[start - 1:start, :].T, ds.horizon, axis=1))
y_true = np.stack(y_true)
persistence = np.stack(persistence)
seasonal = np.stack(seasonal)
def inverse(x):
    return np.expm1(x * data.target_std[None, :, None] + data.target_mean[None, :, None]).clip(min=0)
true_raw = inverse(y_true)
outdir = ROOT / cfg["outputs"]["directory"] / "metrics"
outdir.mkdir(parents=True, exist_ok=True)
results = {}
for name, pred in [("persistence", persistence), ("seasonal_naive_24h", seasonal)]:
    m = regression_metrics(true_raw, inverse(pred))
    m["n_test_windows"] = len(ds)
    m["baseline"] = name
    results[name] = m
    (outdir / f"{name}_test_metrics.json").write_text(json.dumps(m, indent=2))
print(json.dumps(results, indent=2))
