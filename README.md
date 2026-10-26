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
| Member 3: satyam-coder07 | backtest.py, baselines.py | Done |
| Member 4: Yashi Agarwal | xgboost_model.py | Done |
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

## Member 3: rolling backtest and baselines

```bash
python baselines.py
```

backtest.py splits the data by time into 4 folds. Each fold trains on everything before its test period and tests on the next 6 months (Jan–Jun 2013, Jul–Dec 2013, Jan–Jun 2014, Jul–Dec 2014). It also scores forecasts with MAE, RMSE and R squared, only on hours where the real PM2.5 was measured. baselines.py scores two simple forecasts every model must beat: persistence (PM2.5 in 6 hours = PM2.5 now) and seasonal naive (PM2.5 at the same hour yesterday).

| Fold | Persistence MAE | Persistence RMSE | Persistence R² | Seasonal naive MAE | Seasonal naive RMSE | Seasonal naive R² |
| --- | --- | --- | --- | --- | --- | --- |
| 1 (Jan–Jun 2013) | 46.77 | 78.23 | 0.506 | 88.17 | 125.13 | -0.263 |
| 2 (Jul–Dec 2013) | 38.46 | 59.75 | 0.440 | 63.80 | 92.57 | -0.344 |
| 3 (Jan–Jun 2014) | 40.21 | 64.21 | 0.568 | 69.09 | 101.36 | -0.077 |
| 4 (Jul–Dec 2014) | 37.49 | 59.06 | 0.558 | 66.30 | 97.68 | -0.209 |

## Member 4: XGBoost model

```bash
python xgboost_model.py
```

XGBoost builds 300 small decision trees, each one correcting the errors of the trees before it. It uses all 31 features from features.py, trains only on hours where the future PM2.5 was really measured, and is tested on the same 4 folds as the baselines.

| Fold | XGBoost MAE | XGBoost RMSE | XGBoost R² | Persistence MAE | MAE improvement |
| --- | --- | --- | --- | --- | --- |
| 1 (Jan–Jun 2013) | 42.96 | 70.61 | 0.598 | 46.77 | 8% |
| 2 (Jul–Dec 2013) | 32.24 | 48.26 | 0.635 | 38.46 | 16% |
| 3 (Jan–Jun 2014) | 36.11 | 56.36 | 0.667 | 40.21 | 10% |
| 4 (Jul–Dec 2014) | 31.72 | 47.96 | 0.708 | 37.49 | 15% |
| Average | 35.76 | 55.80 | 0.652 | 40.73 | 12% |

XGBoost beats both baselines in every fold. The most important features in the last fold were current PM2.5, northwest wind, season (month_cos), rain hours and time of day.


