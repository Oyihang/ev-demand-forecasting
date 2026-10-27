from __future__ import annotations
import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Point-forecast metrics on original target units; accepts any aligned shape."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[valid], y_pred[valid]
    if not len(y_true):
        return {"MAE": float("nan"), "RMSE": float("nan"), "WAPE_pct": float("nan"), "MAPE_nonzero_pct": float("nan")}
    error = y_true - y_pred
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))
    denom = np.sum(np.abs(y_true))
    wape = np.sum(np.abs(error)) / denom * 100 if denom > 1e-12 else np.nan
    nz = np.abs(y_true) > 1e-6
    mape = np.mean(np.abs(error[nz] / y_true[nz])) * 100 if nz.any() else np.nan
    return {"MAE": float(mae), "RMSE": float(rmse), "WAPE_pct": float(wape), "MAPE_nonzero_pct": float(mape)}


def pinball_loss(predictions, target, quantiles):
    """Torch pinball loss; predictions (..., Q), target (...)."""
    import torch
    q = torch.as_tensor(quantiles, device=predictions.device, dtype=predictions.dtype)
    error = target.unsqueeze(-1) - predictions
    return torch.maximum(q * error, (q - 1.0) * error).mean()


def pinball_numpy(y_true: np.ndarray, y_pred_q: np.ndarray, quantile: float) -> float:
    """Pinball loss for aligned original-scale arrays, where y_pred_q has same shape as y_true."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred_q = np.asarray(y_pred_q, dtype=np.float64)
    valid = np.isfinite(y_true) & np.isfinite(y_pred_q)
    if not valid.any():
        return float("nan")
    error = y_true[valid] - y_pred_q[valid]
    return float(np.maximum(quantile * error, (quantile - 1.0) * error).mean())


def interval_metrics(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict:
    """Coverage and normalized width of a central prediction interval."""
    y_true = np.asarray(y_true, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    valid = np.isfinite(y_true) & np.isfinite(lower) & np.isfinite(upper)
    if not valid.any():
        return {"PICP_pct": float("nan"), "PINAW_pct": float("nan"), "mean_interval_width": float("nan")}
    y, lo, hi = y_true[valid], lower[valid], upper[valid]
    width = np.maximum(hi - lo, 0.0)
    scale = float(np.max(y) - np.min(y))
    return {
        "PICP_pct": float(np.mean((y >= lo) & (y <= hi)) * 100.0),
        "PINAW_pct": float(np.mean(width) / scale * 100.0) if scale > 1e-12 else float("nan"),
        "mean_interval_width": float(np.mean(width)),
    }
