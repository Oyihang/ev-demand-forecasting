# ST-TFT: Spatial-Temporal Fusion Transformer for EV Charging Demand Forecasting

A reproducible research project that combines **graph convolution (spatial dependency)** and a **Temporal Fusion Transformer (temporal dependency)** to forecast hourly EV charging energy demand across urban traffic analysis zones.

> The raw UrbanEV data is **not included** in this repository. Obtain it from the upstream source and place it in the expected directory before running data-dependent scripts. The model predicts `volume.csv` (energy demand) and uses historical occupancy/duration plus zone adjacency as spatial signals. The scripts do not claim any performance result until you run the experiments.

## Research questions

1. Does adding graph-based spatial information improve over a temporal-only TFT baseline?
2. How does performance change across 1-hour to 24-hour horizons and across zones?
3. Which historical variables and calendar patterns are associated with predictions?
4. Are results robust to changing the spatial graph (adjacency vs distance-weighted graph)?

## Repository layout

```text
configs/experiment.yaml      Experiment configuration
 data/raw/                   Local dataset location (raw files are not committed)
 data/README.md              Dataset source, placement, and licensing notes
src/data/dataset.py           Time/zone alignment, train-only scaling, windows
src/models/tft.py             TFT implementation supplied with the project
src/models/spatial_tft.py      Graph convolution + TFT fusion model
src/training/train.py         Training, early stopping, test evaluation
scripts/audit_data.py         Dataset integrity report
scripts/plot_results.py       Plot generated metrics and forecast samples
tests/                        Data/model smoke tests
report/report.md              Research report template
results/                      Generated metrics, plots, checkpoints (after runs)
```

## Environment

Python 3.10+ is recommended. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install a PyTorch build appropriate for your machine if the default package index does not provide the desired CUDA build. CPU training is supported, but the full 275-zone model can be slow; use a small zone subset for smoke tests first.

## 1. Obtain the data

Download the cleaned, aggregated UrbanEV dataset from the
[upstream UrbanEV repository](https://github.com/IntelligentSystemsLab/UrbanEV)
and review its license and terms. Place the dataset files at
`data/raw/20220901-20230228_zone-cleaned-aggregated/`, including the
`charge_1hour/` directory, `adj.csv`, `distance.csv`, and
`zone-information.csv`. Expected filenames are listed in
`configs/experiment.yaml`; more details are in [data/README.md](data/README.md).

Raw dataset files are intentionally excluded from Git and are not redistributed
by this project.

## 2. Audit the data

```bash
python scripts/audit_data.py
```

The script verifies timestamp/zone alignment and writes `results/metrics/data_audit.json`. Expected core files:

- `data/raw/20220901-20230228_zone-cleaned-aggregated/charge_1hour/volume.csv`
- `.../occupancy.csv`, `duration.csv`, `e_price.csv`, `s_price.csv`
- `.../adj.csv`, `distance.csv`, `zone-information.csv`

The hourly matrices use one row per timestamp, a `time` column, and one column per zone. The adjacency matrix is header-only and its row order matches its column order in the provided dataset.

## 3. Run a fast smoke test

```bash
python -m src.training.train --smoke-test --max-zones 8
```

This trains one epoch on eight zones to verify loading, tensor dimensions, forward/backward passes, checkpoint writing, and metrics output. It is not a meaningful benchmark.

## 4. Train the full model

```bash
python -m src.training.train
```

To keep earlier results and checkpoints untouched, choose a new run name:

```bash
python -m src.training.train --run-name st_tft_weather_holiday_pinball
```

Compare temporal-only TFT with the spatial model (two separate runs):

```bash
python scripts/run_ablation.py --epochs 20
python scripts/build_report.py
```

The ablation writes `tft_no_spatial_test_metrics.json` and `st_tft_test_metrics.json`. The no-spatial setting disables graph-derived context; it is the controlled temporal-only baseline.

Optional overrides:

```bash
python -m src.training.train --epochs 30
python -m src.training.train --max-zones 40 --epochs 2
```

Main settings live in `configs/experiment.yaml`: 168-hour context, 24-hour horizon, chronological 70%/15%/15% split, stride-based sampling, hidden dimensions, quantiles and early stopping. The baseline uses at most 30 epochs and patience 7; these are starting values, not verified optimal settings. Tune settings only after establishing a reproducible baseline.

## Model and tensor conventions

- `observed`: `(batch, context, zones, 1)` historical target (`log1p` volume, standardized per zone using training-period statistics only).
- `graph_features`: `(batch, context, zones, 5)` historical duration, electricity price, service price, occupancy, and volume features.
- `known`: `(batch, context+horizon, zones, 10)` with continuous weather `T`, `P0`, `P`, `U`, `Td`, followed by categorical day-of-week, hour, weekend, `nRAIN`, and `is_china_holiday` (cardinalities `[7, 24, 2, 4, 2]`). Continuous weather is standardized using training-period statistics only. Central weather is shared by all zones.
- `static`: `(zones, 3)` charge count, area, and perimeter.
- `adjacency`: `(zones, zones)` symmetric, self-looped, degree-normalized graph.
- Output: `(batch, zones, horizon, 3)` quantiles (0.1, 0.5, 0.9); the median is used for point metrics.

Spatial graph convolution is applied independently at every historical timestamp. The full sequence of spatial embeddings is passed to TFT together with the historical target; it is not averaged over time. TFT handles the temporal sequence, while the three static zone attributes condition its static encoder.

## Data leakage and evaluation

- Splits are chronological; the forecast target is wholly inside its assigned split.
- Scaling statistics are fit on the training period only.
- Future inputs include calendar features and weather under a **perfect-weather-forecast assumption**: actual future weather from `weather_central.csv` is treated as if it were known at forecast time. This is an optimistic oracle experiment, not a realistic forecast backtest. For deployment or realistic evaluation, replace future observations with forecasts issued by each forecast origin. Future observed volume, occupancy and duration are never fed into the decoder.
- Weather changes the model inputs, so older checkpoints must be retrained and existing metrics do not reflect the weather-enabled configuration.
- `is_china_holiday` is 1 on mainland China's official holiday and scheduled holiday-break dates, including weekends within those breaks. Ordinary weekends and compensatory working days are 0; `is_weekend` remains a separate calendar-day indicator. It is calculated for both past and future timestamps using `chinese-calendar==1.11.0`, whose published calendar covers 2004–2026. Dates outside that range raise an error; update the calendar dependency after future official schedules are published. Future holiday inputs assume the relevant schedule is already announced at forecast time.
- Adding `is_china_holiday` changes the model inputs again: previously trained checkpoints and metrics do not include it and require retraining for a fair comparison.
- Metrics: MAE, RMSE, WAPE, and MAPE over nonzero targets. MAPE is reported with care because near-zero loads can make it unstable.
- Training minimizes quantile Pinball Loss. The default `training.selection_metric: val_pinball` selects the checkpoint and applies early stopping using validation Pinball Loss. Set it to `val_wape_pct` to select by validation WAPE on the median forecast in original units; validation WAPE requires a 0.5 quantile.
- Training history records validation WAPE when it is the selection metric; `best_val_pinball_normalized` in the test metrics is the validation pinball loss of the selected checkpoint, not necessarily the lowest pinball loss across epochs.
- Use the same split, windows, target transform and metrics for every baseline.

## Outputs

Training writes run-specific artifacts (the default run name is `st_tft`):

- `results/checkpoints/st_tft_best.pt` — checkpoint selected by the configured validation metric
- `results/metrics/st_tft_training_history.csv` — train/validation loss by epoch
- `results/metrics/st_tft_test_metrics.json` — aggregate test metrics and interval coverage
- `results/metrics/st_tft_horizon_metrics.csv` — metrics for each lead time (1..24 hours)
- `results/metrics/st_tft_zone_metrics.csv` — metrics per zone
- `results/metrics/st_tft_sample_forecast.csv` — first test window with timestamps, P10/P50/P90

Run simple baselines on the exact same chronological test split:

```bash
python scripts/evaluate_naive_baselines.py
```

The baseline outputs are `persistence_test_metrics.json` and `seasonal_naive_24h_test_metrics.json`. In one run on this fixed split, the 24-hour seasonal-naive baseline achieved WAPE around 22.70%; the earlier ST-TFT WAPE of 28.70% was produced before the data-feature fix and is not a fair final comparison. Retrain ST-TFT with this version before drawing conclusions. The test period is for final evaluation only; use validation metrics for model selection.

Then generate plots:

```bash
python scripts/plot_results.py
python scripts/build_report.py
```

Run tests:

```bash
pytest -q
```

## Important limitations and next experiments

1. This initial model is a research baseline, not a production deployment model.
2. The graph encoder performs spatial message passing per timestamp; temporal dependencies are learned by TFT. Compare this with jointly temporal-spatial graph architectures such as Graph WaveNet, GAT, or PAG.
3. Electricity and service prices are historical GNN inputs. Do not pass future observed prices into the forecast unless their values are genuinely known at prediction time.
4. Report metrics by forecast horizon and zone, not just a single aggregate number.
5. Compare temporal-only TFT, the current adjacency graph, and distance-weighted graph before claiming the spatial module improves accuracy. Also compare against persistence and 24-hour seasonal-naive baselines.
6. The six-month period does not provide a full annual seasonal cycle.

## Data provenance

UrbanEV / IntelligentSystemsLab: https://github.com/IntelligentSystemsLab/UrbanEV

Check the upstream dataset and repository licenses/terms before redistribution. Do not publish credentials, private data, or large raw files if the upstream terms prohibit it.
