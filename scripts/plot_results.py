"""Plot the first test forecast window and training/validation loss history."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
results = ROOT / "results"
figdir = results / "figures"
figdir.mkdir(parents=True, exist_ok=True)
forecast = results / "metrics/st_tft_sample_forecast.csv"
# forecast = results / "metrics/tft_no_spatial_sample_forecast.csv"
history = results / "metrics/st_tft_training_history.csv"
if forecast.exists():
    df = pd.read_csv(forecast)
    if {"zone_id", "forecast_hour", "actual_kwh", "predicted_p50_kwh"}.issubset(df.columns):
        zone = df.zone_id.iloc[100]
        sample = df[df.zone_id == zone].sort_values("forecast_hour")
        plt.figure(figsize=(10, 4))
        plt.plot(sample.forecast_hour, sample.actual_kwh, label="Actual")
        plt.plot(sample.forecast_hour, sample.predicted_p50_kwh, label="Predicted P50")
        if "predicted_p10_kwh" in sample and "predicted_p90_kwh" in sample:
            plt.fill_between(sample.forecast_hour, sample.predicted_p10_kwh, sample.predicted_p90_kwh, alpha=0.2, label="P10-P90 interval")
        plt.title(f"ST-TFT: test forecast for zone {zone}")
        plt.xlabel("Forecast hour")
        plt.ylabel("Energy (original units)")
        plt.legend(); plt.tight_layout()
        plt.savefig(figdir / "sample_forecast.png", dpi=160); plt.close()
if history.exists():
    df = pd.read_csv(history)
    plt.figure(figsize=(8, 4))
    plt.plot(df.epoch, df.train_pinball, label="Train pinball")
    plt.plot(df.epoch, df.val_pinball, label="Validation pinball")
    plt.xlabel("Epoch"); plt.ylabel("Pinball loss"); plt.title("Training history")
    plt.legend(); plt.tight_layout()
    plt.savefig(figdir / "training_history.png", dpi=160); plt.close()
print(f"Plots saved under {figdir}")
