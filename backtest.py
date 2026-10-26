# backtest.py
# Rolling chronological backtest: split data by time (never shuffle) and score predictions.

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import config


def make_folds(df):
    """Return a list of (fold_number, train_df, test_df), split by date."""
    folds = []
    for i, (start, end) in enumerate(config.TEST_PERIODS, start=1):
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)

        train = df[df.index < start].copy()
        test = df[(df.index >= start) & (df.index <= end)].copy()

        # The last HORIZON training rows have targets INSIDE the test period.
        # We keep them as history but do not learn from them (no leakage).
        train.iloc[-config.HORIZON:, train.columns.get_loc("target_observed")] = 0

        folds.append((i, train, test))
    return folds


def compute_metrics(y_true, y_pred):
    """MAE, RMSE and R squared."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": round(float(mae), 2), "RMSE": round(float(rmse), 2), "R2": round(float(r2), 3)}


def observed_only(df):
    """Keep only rows where the future PM2.5 was really measured."""
    return df[df["target_observed"] == 1]
