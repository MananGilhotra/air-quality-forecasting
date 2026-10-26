# Air Quality Forecasting

Forecasting Beijing PM2.5 6 hours ahead with XGBoost, compared against persistence and seasonal naive baselines using a rolling chronological backtest. Forecasts are mapped to US EPA (2024) AQI categories.

Dataset: [UCI Beijing PM2.5](https://archive.ics.uci.edu/dataset/381/beijing), hourly, 2010 to 2014 (`data/PRSA_data_2010.1.1-2014.12.31.csv`).

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Run every command from inside this folder.

## Team progress

| Member | Files | Status |
| --- | --- | --- |
| Member 1: Manan Gilhotra | requirements.txt, config.py, data_loader.py | Done |
| Member 2: Harshita | features.py | Done |
| Member 3 | backtest.py, baselines.py | To do |
| Member 4 | xgboost_model.py | To do |
| Member 5 | aqi.py, plots.py, main.py, app.py | To do |

## Member 1: data cleaning

```bash
python data_loader.py
```

This creates `data/clean_data.csv`: one row per hour, a `pm25_missing` flag, gaps filled with the last known value, and wind direction one-hot encoded.

## Member 2: feature engineering

```bash
python features.py
```

This reads `data/clean_data.csv` and creates `data/features.csv`: lag features (1, 2, 3, 6, 12 and 24 hours), rolling means (6 and 24 hours), calendar features (hour, day of week, month, weekend, sin/cos), weather features (temperature–dew point gap, 3-hour pressure change) and the target, PM2.5 6 hours ahead.
