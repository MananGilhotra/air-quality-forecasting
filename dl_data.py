# dl_data.py
# Phase 2, step 1: get the data ready for neural networks.
# - a time-ordered validation split (to decide when to stop training)
# - scaling that is fitted on TRAINING dates only (no information from the future)

import random
import numpy as np
import pandas as pd
import torch
import config


def set_seed(seed):
    """Make the random numbers repeatable, so two runs give the same model."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_train_validation(train_df):
    """Split a training period by time: first 90% to learn from, last 10% to validate on."""
    n_val = int(len(train_df) * config.VAL_FRACTION)
    fit_df = train_df.iloc[:-n_val].copy()
    val_df = train_df.iloc[-n_val:].copy()

    # The last HORIZON rows of fit_df have targets inside the validation period.
    # We keep them as history but do not learn from them (no leakage).
    fit_df.iloc[-config.HORIZON:, fit_df.columns.get_loc("target_observed")] = 0
    return fit_df, val_df


def fit_scaler(fit_df, columns):
    """Learn the mean and spread of each column from the TRAINING rows only."""
    mean = fit_df[columns].mean()
    std = fit_df[columns].std().replace(0, 1)   # avoid dividing by zero
    return mean, std


def apply_scaler(df, columns, mean, std):
    """Scale any rows (training, validation or test) with the training mean and spread."""
    scaled = (df[columns] - mean) / std
    return scaled.values.astype(np.float32)


if __name__ == "__main__":
    from backtest import make_folds

    df = pd.read_csv(config.FEATURES_FILE, index_col="datetime", parse_dates=True)
    fold, train, test = make_folds(df)[0]
    fit_df, val_df = split_train_validation(train)

    print("Fold 1")
    print("  learn from :", fit_df.index.min(), "to", fit_df.index.max(), f"({len(fit_df)} hours)")
    print("  validate on:", val_df.index.min(), "to", val_df.index.max(), f"({len(val_df)} hours)")
    print("  test on    :", test.index.min(), "to", test.index.max(), f"({len(test)} hours)")

    columns = ["pm25", "TEMP", "PRES"]
    mean, std = fit_scaler(fit_df, columns)
    scaled_fit = apply_scaler(fit_df, columns, mean, std)
    scaled_test = apply_scaler(test, columns, mean, std)
    print("  scaled training mean (should be about 0):", scaled_fit.mean(axis=0).round(3))
    print("  scaled training std  (should be about 1):", scaled_fit.std(axis=0).round(3))
    print("  scaled test mean (not 0, because the scaler never saw test data):", scaled_test.mean(axis=0).round(3))
