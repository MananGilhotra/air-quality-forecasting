# baselines.py
# Two simple forecasts that every real model must beat.

import pandas as pd
import config
from backtest import make_folds, compute_metrics, observed_only


def persistence_forecast(test_df):
    """'Tomorrow looks like today': PM2.5 in 6 hours = PM2.5 now."""
    return test_df["pm25"].values


def seasonal_naive_forecast(test_df):
    """PM2.5 in 6 hours = PM2.5 at that same clock hour yesterday."""
    return test_df["pm25_same_hour_yesterday"].values


if __name__ == "__main__":
    df = pd.read_csv(config.FEATURES_FILE, index_col="datetime", parse_dates=True)
    for fold, train, test in make_folds(df):
        test = observed_only(test)
        print(f"Fold {fold}")
        print("  Persistence:   ", compute_metrics(test["target"], persistence_forecast(test)))
        print("  Seasonal naive:", compute_metrics(test["target"], seasonal_naive_forecast(test)))
