"""Inspect the downloaded UrbanEV files and verify zone/time alignment."""
from pathlib import Path
import sys, json
import pandas as pd
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import yaml
from src.data.dataset import load_urbanev

cfg = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text())
data = load_urbanev(cfg, ROOT)
report = {
    "rows": len(data.times), "zones": len(data.zone_ids),
    "start_time": str(data.times.min()), "end_time": str(data.times.max()),
    "target_shape": list(data.target.shape), "graph_features_shape": list(data.graph_features.shape),
    "known_features_shape": list(data.known.shape), "static_shape": list(data.static.shape),
    "adjacency_shape": list(data.adjacency.shape), "adjacency_nonzero_ratio": float((data.adjacency > 0).mean()),
    "feature_names": data.feature_names,
    "train_end_index": data.split_train_end, "validation_end_index": data.split_val_end,
    "target_finite": bool(np.isfinite(data.target).all()),
    "graph_features_finite": bool(np.isfinite(data.graph_features).all()),
    "static_finite": bool(np.isfinite(data.static).all()),
    "adjacency_finite": bool(np.isfinite(data.adjacency).all()),
    "train_ratio_actual": data.split_train_end / len(data.times),
    "val_ratio_actual": (data.split_val_end - data.split_train_end) / len(data.times),
    "test_ratio_actual": (len(data.times) - data.split_val_end) / len(data.times),
    "target_min_log_scaled": float(np.min(data.target)),
    "target_max_log_scaled": float(np.max(data.target)),
}
out = ROOT / "results/metrics/data_audit.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
print(f"Saved: {out}")
