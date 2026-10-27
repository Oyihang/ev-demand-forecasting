from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.data.dataset import load_urbanev, WindowDataset
from src.models.spatial_tft import SpatialTFT
from src.training.metrics import pinball_loss, regression_metrics, pinball_numpy, interval_metrics


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def get_device(name):
    if name != "auto": return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_model(data, cfg, use_spatial=True):
    m = cfg["model"]
    return SpatialTFT(
        context_length=cfg["data"]["context_length"],
        horizon_length=cfg["data"]["horizon_length"],
        graph_input_dim=data.graph_features.shape[-1],
        known_feature_names=data.feature_names["known"],
        static_feature_names=data.feature_names["static"],
        hidden_dim=int(m["hidden_dim"]), gnn_hidden_dim=int(m["gnn_hidden_dim"]),
        num_heads=int(m["num_heads"]), dropout=float(m["dropout"]),
        lstm_layers=int(m["lstm_layers"]), quantiles=tuple(m["quantiles"]), use_spatial=use_spatial,
    )


def run_epoch(model, loader, data, device, optimizer=None, quantiles=(0.1,0.5,0.9), grad_clip=1.0):
    training = optimizer is not None
    model.train(training)
    loss_sum, window_count = 0.0, 0
    preds, targets, quantile_preds = [], [], []
    adjacency = torch.from_numpy(data.adjacency).to(device)
    static = torch.from_numpy(data.static).to(device)
    for batch in loader:
        observed = batch["observed"].to(device)
        graph = batch["graph_features"].to(device)
        known = batch["known"].to(device)
        target = batch["target"].to(device)
        if training: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(observed, graph, known, static, adjacency)
            loss = pinball_loss(output, target, quantiles)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        batch_windows = int(target.shape[0])
        loss_sum += float(loss.detach().cpu()) * batch_windows
        window_count += batch_windows
        mid = min(range(len(quantiles)), key=lambda i: abs(quantiles[i] - 0.5))
        preds.append(output[..., mid].detach().cpu().numpy())
        quantile_preds.append(output.detach().cpu().numpy())
        targets.append(target.detach().cpu().numpy())
    return (loss_sum / max(window_count, 1), np.concatenate(targets, axis=0),
            np.concatenate(preds, axis=0), np.concatenate(quantile_preds, axis=0))


def inverse_target(arr, mean, std):
    # arr shape (..., N, H) or (...,N); zone dimension is the penultimate/last axis handled below
    return np.expm1(arr * std.reshape(1, -1, 1) + mean.reshape(1, -1, 1)).clip(min=0)


def main():
    parser = argparse.ArgumentParser(description="Train Spatial-GNN + TFT on UrbanEV hourly data")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-zones", type=int, default=None, help="quick test using first N zones")
    parser.add_argument("--smoke-test", action="store_true", help="one train/validation step only")
    parser.add_argument("--no-spatial", action="store_true", help="ablation: disable graph-derived spatial context")
    parser.add_argument("--run-name", default=None, help="name prefix for checkpoints and metrics")
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text())
    if args.max_zones is not None: cfg["data"]["max_zones"] = args.max_zones
    if args.smoke_test:
        cfg["training"]["epochs"] = 1
        cfg["training"]["batch_size"] = 1
        cfg["training"]["patience"] = 1
    if args.epochs is not None: cfg["training"]["epochs"] = args.epochs
    seed_everything(int(cfg["training"]["seed"]))
    device = get_device(cfg["training"].get("device", "auto"))
    data = load_urbanev(cfg, ROOT)
    print(f"Loaded {len(data.times)} hourly rows, {len(data.zone_ids)} zones, device={device}")
    print(f"Time range: {data.times.min()} -> {data.times.max()}")
    print(f"Split endpoints: train={data.split_train_end}, val={data.split_val_end}, total={len(data.times)}")

    dcfg, tcfg = cfg["data"], cfg["training"]
    datasets = {s: WindowDataset(data, s, dcfg["context_length"], dcfg["horizon_length"], stride=int(dcfg.get(s + "_stride", 1))) for s in ["train", "val", "test"]}
    loaders = {}
    for s, ds in datasets.items():
        active_ds = Subset(ds, list(range(min(len(ds), 1)))) if args.smoke_test else ds
        loaders[s] = DataLoader(active_ds, batch_size=int(tcfg["batch_size"]), shuffle=(s == "train"),
                                num_workers=int(tcfg.get("num_workers", 0)), drop_last=False)
    run_name = args.run_name or ("tft_no_spatial" if args.no_spatial else "st_tft")
    model = make_model(data, cfg, use_spatial=not args.no_spatial).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tcfg["learning_rate"]),
                                  weight_decay=float(tcfg.get("weight_decay", 0.0)))
    quantiles = tuple(cfg["model"]["quantiles"])
    outdir = ROOT / cfg["outputs"]["directory"]
    (outdir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (outdir / "metrics").mkdir(parents=True, exist_ok=True)
    best_val, patience, history = float("inf"), 0, []
    best_path = outdir / "checkpoints" / f"{run_name}_best.pt"
    max_epochs = int(tcfg["epochs"])
    for epoch in range(1, max_epochs + 1):
        train_loss, _, _, _ = run_epoch(model, loaders["train"], data, device, optimizer, quantiles, float(tcfg["gradient_clip_norm"]))
        with torch.no_grad():
            val_loss, val_y, val_pred, _ = run_epoch(model, loaders["val"], data, device, None, quantiles)
        history.append({"epoch": epoch, "train_pinball": train_loss, "val_pinball": val_loss})
        print(f"Epoch {epoch:03d}/{max_epochs} | train pinball={train_loss:.5f} | val pinball={val_loss:.5f}")
        if val_loss < best_val:
            best_val, patience = val_loss, 0
            torch.save({"model_state_dict": model.state_dict(), "config": cfg,
                        "zone_ids": data.zone_ids, "target_mean": data.target_mean,
                        "target_std": data.target_std, "feature_names": data.feature_names,
                        "best_val_pinball": best_val}, best_path)
        else:
            patience += 1
            if patience >= int(tcfg["patience"]):
                print("Early stopping triggered.")
                break
        if args.smoke_test: break

    pd.DataFrame(history).to_csv(outdir / "metrics" / f"{run_name}_training_history.csv", index=False)
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    with torch.no_grad():
        test_loss, test_y, test_pred, test_pred_q = run_epoch(model, loaders["test"], data, device, None, quantiles)
    # Arrays: y / median=(windows,N,H), all quantiles=(windows,N,H,Q).
    scale = data.target_std[None, :, None]
    mean = data.target_mean[None, :, None]
    true_raw = np.expm1(test_y * scale + mean).clip(min=0)
    pred_raw = np.expm1(test_pred * scale + mean).clip(min=0)
    q_scale = data.target_std[None, :, None, None]
    q_mean = data.target_mean[None, :, None, None]
    pred_q_raw = np.expm1(test_pred_q * q_scale + q_mean).clip(min=0)
    metrics = regression_metrics(true_raw, pred_raw)
    metrics["test_pinball_normalized"] = test_loss
    metrics["best_val_pinball_normalized"] = best_val
    metrics["n_test_windows"] = int(len(test_y))
    metrics["n_zones"] = int(len(data.zone_ids))
    metrics["horizon_length"] = int(dcfg["horizon_length"])
    metrics["quantile_pinball_original_units"] = {}
    for qi, q in enumerate(quantiles):
        metrics["quantile_pinball_original_units"][str(q)] = pinball_numpy(true_raw, pred_q_raw[..., qi], float(q))
    if 0.1 in quantiles and 0.9 in quantiles:
        lo_i, hi_i = quantiles.index(0.1), quantiles.index(0.9)
        metrics["interval_80pct"] = interval_metrics(true_raw, pred_q_raw[..., lo_i], pred_q_raw[..., hi_i])
        metrics["quantile_crossing_pct"] = float(np.mean(
            (pred_q_raw[..., lo_i] > pred_q_raw[..., quantiles.index(0.5)]) |
            (pred_q_raw[..., quantiles.index(0.5)] > pred_q_raw[..., hi_i])
        ) * 100.0)

    metrics_dir = outdir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / f"{run_name}_test_metrics.json").write_text(json.dumps(metrics, indent=2))

    # Horizon-level metrics: each row evaluates one forecast lead time across all windows/zones.
    horizon_rows = []
    for hi in range(true_raw.shape[-1]):
        row = {"horizon_hour": hi + 1, **regression_metrics(true_raw[..., hi], pred_raw[..., hi])}
        if 0.1 in quantiles and 0.9 in quantiles:
            row.update(interval_metrics(true_raw[..., hi], pred_q_raw[..., hi, quantiles.index(0.1)], pred_q_raw[..., hi, quantiles.index(0.9)]))
        horizon_rows.append(row)
    pd.DataFrame(horizon_rows).to_csv(metrics_dir / f"{run_name}_horizon_metrics.csv", index=False)

    # Zone-level metrics aggregate all forecast windows and lead times for each zone.
    zone_rows = []
    for zi, zone_id in enumerate(data.zone_ids):
        zone_rows.append({"zone_id": zone_id, **regression_metrics(true_raw[:, zi, :], pred_raw[:, zi, :])})
    pd.DataFrame(zone_rows).to_csv(metrics_dir / f"{run_name}_zone_metrics.csv", index=False)

    # Save one window in a readable, long-form CSV with timestamps and optional interval bounds.
    first_start = data.split_val_end
    rows = []
    for zi, zone_id in enumerate(data.zone_ids):
        for hi in range(true_raw.shape[-1]):
            item = {"zone_id": zone_id, "forecast_hour": hi + 1,
                    "target_time": str(data.times[first_start + hi]),
                    "actual_kwh": float(true_raw[0, zi, hi]),
                    "predicted_p50_kwh": float(pred_raw[0, zi, hi])}
            if 0.1 in quantiles and 0.9 in quantiles:
                item["predicted_p10_kwh"] = float(pred_q_raw[0, zi, hi, quantiles.index(0.1)])
                item["predicted_p90_kwh"] = float(pred_q_raw[0, zi, hi, quantiles.index(0.9)])
            rows.append(item)
    pd.DataFrame(rows).to_csv(metrics_dir / f"{run_name}_sample_forecast.csv", index=False)
    print("Test metrics:", json.dumps(metrics, indent=2))
    print(f"Best checkpoint: {best_path}")

if __name__ == "__main__": main()
