# features.py
# Step 2: turn the clean hourly data into features + the target we want to predict.

import numpy as np
import pandas as pd
import config


def add_lag_features(df):
    """PM2.5 from 1, 2, 3, 6, 12 and 24 hours ago."""
    for lag in config.LAGS:
        df[f"pm25_lag_{lag}"] = df["pm25"].shift(lag)
    return df


def add_rolling_features(df):
    """Average PM2.5 over the last 6 and 24 hours (only past and current hours)."""
    for window in config.ROLLING_WINDOWS:
        df[f"pm25_roll_mean_{window}"] = df["pm25"].rolling(window).mean()
    return df


def add_calendar_features(df):
    """Hour, day of week and month, plus sin/cos versions so 23:00 sits next to 00:00."""
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.day_of_week
    df["month"] = df.index.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_weather_features(df):
    """Extra weather covariates built from the existing columns."""
    # Gap between temperature and dew point: small gap = humid air
    df["temp_dewp_gap"] = df["TEMP"] - df["DEWP"]
    # How much the air pressure changed in the last 3 hours
    df["pres_change_3h"] = df["PRES"].diff(3)
    return df


def add_target(df):
    """Target = PM2.5 HORIZON hours in the future."""
    h = config.HORIZON
    df["target"] = df["pm25"].shift(-h)

    # Was the future PM2.5 value really measured, or filled by us?
    # We only train and score on hours where it was really measured.
    df["target_observed"] = 1 - df["pm25_missing"].shift(-h)

    # Value at the same clock hour one day before the target hour.
    # Used by the seasonal naive baseline (and also useful as a feature).
    df["pm25_same_hour_yesterday"] = df["pm25"].shift(24 - h)
    return df


def build_features(clean_df):
    df = clean_df.copy()
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_calendar_features(df)
    df = add_weather_features(df)
    df = add_target(df)

    # Lags leave empty rows at the start, the target leaves empty rows at the end
    df = df.dropna()
    return df


def get_feature_columns(df):
    """Every column except the answer columns is a feature."""
    return [c for c in df.columns if c not in ["target", "target_observed"]]


if __name__ == "__main__":
    clean = pd.read_csv(config.CLEAN_FILE, index_col="datetime", parse_dates=True)
    features = build_features(clean)
    features.to_csv(config.FEATURES_FILE)
    print("Feature table shape:", features.shape)
    print("Feature columns:", get_feature_columns(features))
    print("Saved to", config.FEATURES_FILE)
