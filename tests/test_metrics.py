import numpy as np
from src.training.metrics import regression_metrics, interval_metrics, pinball_numpy

def test_regression_metrics_simple():
    m = regression_metrics(np.array([1., 2., 3.]), np.array([1., 2., 4.]))
    assert abs(m["MAE"] - 1/3) < 1e-8
    assert m["RMSE"] > m["MAE"]

def test_interval_metrics_and_pinball():
    y = np.array([1., 2., 3.])
    lo = np.array([0., 1., 2.])
    hi = np.array([2., 3., 4.])
    m = interval_metrics(y, lo, hi)
    assert m["PICP_pct"] == 100.0
    assert m["mean_interval_width"] == 2.0
    assert pinball_numpy(y, y, 0.5) == 0.0
