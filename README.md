# Hong Kong AQI Short-Term Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project predicts short-term air-quality trends in Hong Kong with Python.
The current pipeline collects hourly pollutant and weather data, preprocesses
the time series, engineers features, trains multiple models, and generates a
final forecast.

Core topic:

- Domain: Hong Kong air quality forecasting
- Target horizon: next 24 to 72 hours
- Data type: hourly multivariate time series
- Main sources: Open-Meteo Air Quality API and Weather Archive API

## Current Status

Data collection and ETL have been stabilized.

Verified status from the latest live run:

- `data/collection_metadata.json`
  - `source_mode = "live_api"`
  - `fallback_to_sample = false`
  - `weather_synthetic_fill.used = false`
- `data/preprocessing_report.json`
  - records = 17544
  - missing values = 0
  - duplicate timestamps = 0
  - default outlier mode = `flag_only`

This means the current raw and processed datasets were collected from live APIs,
not generated from sample mode.

## Project Structure

```text
hongkong_aqi_prediction/
|-- README.md
|-- requirements.txt
|-- data/
|   |-- raw_aqi_data.csv
|   |-- processed_aqi_data.csv
|   |-- featured_aqi_data.csv
|   |-- collection_metadata.json
|   |-- preprocessing_report.json
|   `-- data_dictionary.csv
|-- src/
|   |-- 01_data_collection.py
|   |-- 02_preprocessing.py
|   |-- 03_feature_engineering.py
|   |-- 04_models.py
|   |-- 04b_lstm_model.py
|   |-- 05_evaluation.py
|   |-- 06_visualization.py
|   |-- 07_forecast.py
|   |-- 08_model_optimization.py
|   `-- 09_final_evaluation.py
|-- report/
|   |-- data_collection_etl.md
|   |-- team_handoff.md
|   |-- evaluation_results.csv
|   |-- aqi_forecast.csv
|   `-- figures/
`-- notebooks/
    `-- analysis.ipynb
```

## Environment Setup

Use one Python environment only.

Recommended checks:

```bash
python -c "import sys; print(sys.executable)"
python -m pip --version
```

Install dependencies into the same interpreter that will run the scripts:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you use a project-local virtual environment on Windows PowerShell:

```bash
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Recommended Workflow

Run the project in this order:

```bash
python src/01_data_collection.py
python src/02_preprocessing.py
python src/03_feature_engineering.py
python src/04_models.py
python src/09_final_evaluation.py
python src/10_report_visualization.py
python src/07_forecast.py
```

## Data Collection

Live API collection is the default behavior:

```bash
python src/01_data_collection.py
```

Sample mode is only for offline debugging:

```bash
python src/01_data_collection.py --mode sample --output-dir tmp/etl_check_v2
```

Important notes:

- Sample data is never used unless explicitly requested, or unless
  `--allow-sample-fallback` is intentionally passed.
- Every run writes `data/collection_metadata.json`.
- The metadata file records API success, chunk-level results, date range,
  fallback status, and whether synthetic weather filling was needed.

Quick verification:

```bash
python -c "import json; d=json.load(open('data/collection_metadata.json', encoding='utf-8')); print(d['source_mode'], d['fallback_to_sample'], d['weather_synthetic_fill']['used'])"
```

Expected live output:

- `live_api False False`

## Preprocessing

Run preprocessing with the default conservative setting:

```bash
python src/02_preprocessing.py
```

Default behavior:

- interpolate missing values if needed
- replace infinity values if needed
- enforce physical bounds
- flag outliers for review without clipping pollutant spikes

Why this changed:

- AQI and pollutant spikes can be real pollution events
- aggressive IQR clipping was suppressing meaningful extremes
- the project now preserves those events by default and documents them in
  `data/preprocessing_report.json`

Optional comparison experiments:

```bash
python src/02_preprocessing.py --outlier-mode clip
python src/02_preprocessing.py --outlier-mode remove
```

## Audit Files

These files are useful for both verification and report writing:

- `data/collection_metadata.json`
- `data/preprocessing_report.json`
- `data/data_dictionary.csv`
- `report/data_collection_etl.md`
- `report/team_handoff.md`

## Modeling Snapshot

The latest saved evaluation results currently rank the models roughly as:

1. Ensemble (Weighted)
2. Ridge
3. Lasso
4. XGBoost
5. Random Forest
6. Gradient Boosting

The exact metrics in use are stored in:

- `report/evaluation_results.csv`

For report-ready model-effect figures, run:

```bash
python src/10_report_visualization.py
```

This writes:

- `report/model_predictions_test.csv`
- `report/best_model_feature_importance.csv`
- `report/figures/best_model_prediction_overview.png`
- `report/figures/best_model_actual_vs_predicted.png`
- `report/figures/best_model_residual_diagnostics.png`
- `report/figures/best_model_feature_importance.png`
- `report/figures/ensemble_weights.png`

## Known Gaps

The project is now much stronger on data provenance and ETL, but several items
still need work:

- `aqi` is currently a PM2.5-based AQI proxy, not a full multi-pollutant AQI
- the collection location is a single Hong Kong coordinate, not a multi-station network
- `07_forecast.py` still needs closer alignment with the actual best model
- LSTM and some advanced models still need fuller integration and evaluation
- `03_feature_engineering.py` still has one future-compatibility warning and several performance warnings
- final presentation and report still need polishing

## Team Notes

For teammate-facing progress and next actions, see:

- `report/team_handoff.md`

For ETL/report-writing support, see:

- `report/data_collection_etl.md`
