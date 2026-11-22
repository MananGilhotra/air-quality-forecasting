# sequences.py
# Phase 2, step 2: turn the hourly table into SEQUENCES for the neural networks.
# XGBoost sees one row of hand-made lag features. The networks instead see the raw
# last 48 hours, hour by hour, and learn the time patterns themselves.

import numpy as np
import pandas as pd
import config
from dl_data import split_train_validation, fit_scaler, apply_scaler


def get_sequence_columns(df):
    """Raw hourly inputs only: PM2.5, missing flag, weather, wind and time of day/year."""
    columns = ["pm25", "pm25_missing"] + config.WEATHER_COLS
    columns += ["hour_sin", "hour_cos", "month_sin", "month_cos"]
    columns += [c for c in df.columns if c.startswith("wind_")]
    return columns


def make_windows(values, seq_len):
    """Cut a table into overlapping windows of seq_len hours.
    Window number k covers rows k .. k+seq_len-1, so it ENDS at row k+seq_len-1."""
    windows = []
    for end in range(seq_len, len(values) + 1):
        windows.append(values[end - seq_len:end])
    return np.array(windows, dtype=np.float32)


def to_pm25(model_output, current_pm25, change_std):
    """The networks predict the scaled CHANGE in PM2.5 over the next 6 hours.
    Convert that back to a PM2.5 forecast: current value + predicted change."""
    forecast = current_pm25 + model_output * change_std
    return np.clip(forecast, 0, None)      # PM2.5 can't be negative


def prepare_fold(train_df, test_df):
    """Build everything a network needs for one backtest fold."""
    columns = get_sequence_columns(train_df)
    seq_len = config.SEQ_LEN

    # 1. Time-ordered split and scaling fitted on the training rows only
    fit_df, val_df = split_train_validation(train_df)
    mean, std = fit_scaler(fit_df, columns)

    # 2. One continuous table (training then test), scaled and cut into windows
    full = pd.concat([train_df, test_df])
    windows = make_windows(apply_scaler(full, columns, mean, std), seq_len)
    current_pm25 = full["pm25"].values
    future_pm25 = full["target"].values

    def collect(part_df, only_measured):
        """Windows, current PM2.5 and future PM2.5 for the rows of one part."""
        rows = full.index.get_indexer(part_df.index)       # row numbers in the full table
        keep = rows >= seq_len - 1                         # need 48 hours of history
        if only_measured:
            keep = keep & (part_df["target_observed"].values == 1)
        rows = rows[keep]
        return windows[rows - (seq_len - 1)], current_pm25[rows], future_pm25[rows]

    X_fit, fit_current, fit_future = collect(fit_df, only_measured=True)
    X_val, val_current, val_future = collect(val_df, only_measured=True)
    X_test, test_current, _ = collect(test_df, only_measured=False)

    # 3. The learning target: the CHANGE in PM2.5 over the next 6 hours, scaled
    fit_change = fit_future - fit_current
    change_std = fit_change.std()

    return {
        "n_features": len(columns),
        "X_fit": X_fit, "y_fit": (fit_change / change_std).astype(np.float32), "fit_current": fit_current,
        "X_val": X_val, "val_current": val_current, "val_actual": val_future,
        "X_test": X_test, "test_current": test_current,
        "change_std": change_std,
    }


if __name__ == "__main__":
    from backtest import make_folds

    df = pd.read_csv(config.FEATURES_FILE, index_col="datetime", parse_dates=True)
    print("Sequence columns:", get_sequence_columns(df))
    for fold, train, test in make_folds(df):
        data = prepare_fold(train, test)
        print(f"Fold {fold}: X_fit {data['X_fit'].shape}  X_val {data['X_val'].shape}  "
              f"X_test {data['X_test'].shape}  change_std {data['change_std']:.2f}")
