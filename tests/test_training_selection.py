import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.training import train


def test_default_config_selects_validation_pinball():
    cfg = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text())
    assert cfg["training"]["selection_metric"] == "val_pinball"


def test_validation_wape_original_units():
    actual = np.log1p(np.array([[[1.0, 9.0]]]))
    predicted = np.log1p(np.array([[[2.0, 7.0]]]))
    score = train.validation_wape_pct(actual, predicted, np.array([0.0]), np.array([1.0]))
    assert score == pytest.approx(30.0)


def test_validation_wape_rejects_undefined_or_nonfinite():
    with pytest.raises(ValueError, match="undefined"):
        train.validation_wape_pct(np.zeros((1, 1, 1)), np.zeros((1, 1, 1)), np.zeros(1), np.ones(1))
    with pytest.raises(ValueError, match="finite"):
        train.validation_wape_pct(np.ones((1, 1, 1)), np.full((1, 1, 1), np.inf), np.zeros(1), np.ones(1))


@pytest.mark.parametrize(("selection_metric", "expected_epoch"), [
    ("val_pinball", 1),
    ("val_wape_pct", 2),
])
def test_checkpoint_uses_selected_validation_metric(tmp_path, monkeypatch, selection_metric, expected_epoch):
    cfg = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text())
    cfg["data"].update(max_zones=2, context_length=4, horizon_length=2)
    cfg["model"].update(hidden_dim=8, gnn_hidden_dim=4, num_heads=2, lstm_layers=1)
    cfg["training"].update(epochs=2, batch_size=1, device="cpu", selection_metric=selection_metric)
    cfg["outputs"]["directory"] = str(tmp_path / "output")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(config_path), "--run-name", "selection"])
    validation_calls = 0

    def fake_run_epoch(model, loader, data, device, optimizer=None, quantiles=(0.1, 0.5, 0.9), grad_clip=1.0):
        nonlocal validation_calls
        target = np.zeros((1, 2, 2), dtype=np.float32)
        if optimizer is not None:
            loss, prediction = 0.3, target
        else:
            validation_calls += 1
            loss = 0.1 if validation_calls == 1 else 0.2
            prediction = target + 0.5 if validation_calls == 1 else target
        quantile_predictions = np.repeat(prediction[..., None], 3, axis=-1)
        return loss, target, prediction, quantile_predictions

    monkeypatch.setattr(train, "run_epoch", fake_run_epoch)
    train.main()
    checkpoint = torch.load(tmp_path / "output" / "checkpoints" / "selection_best.pt",
                            map_location="cpu", weights_only=False)
    metrics_path = tmp_path / "output" / "metrics" / "selection_test_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    assert checkpoint["selection_metric"] == selection_metric
    assert checkpoint["best_val_pinball"] == pytest.approx(0.1 if expected_epoch == 1 else 0.2)
    assert metrics["best_val_pinball_normalized"] == pytest.approx(checkpoint["best_val_pinball"])
    assert metrics["best_val_score"] == pytest.approx(checkpoint["best_val_score"])
    if selection_metric == "val_wape_pct":
        assert metrics["best_val_wape_pct"] == pytest.approx(0.0)
    else:
        assert "best_val_wape_pct" not in metrics
