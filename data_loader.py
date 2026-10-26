# data_loader.py
# Step 1: load the raw CSV, put it on a regular hourly clock and handle missing values.

import pandas as pd
import config


def load_raw_data(path):
    """Read the CSV and build one proper datetime column."""
    df = pd.read_csv(path)

    # Combine year, month, day, hour into one datetime
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])

    # Sort by time FIRST (lags only make sense on sorted data)
    df = df.sort_values("datetime").set_index("datetime")

    # "pm2.5" has a dot in its name, which is awkward in code, so rename it
    df = df.rename(columns={"pm2.5": "pm25"})

    # These columns are no longer needed
    df = df.drop(columns=["No", "year", "month", "day", "hour"])
    return df


def clean_data(df):
    """Make the series regular (one row per hour) and fill the gaps."""
    # 1. Regular interval: if any hour is missing, add an empty row for it
    df = df.asfreq("h")

    # 2. Missingness flag: remember where PM2.5 was missing BEFORE filling
    df["pm25_missing"] = df["pm25"].isna().astype(int)

    # 3. Fill gaps with the last known value (forward fill).
    #    We never fill with future values, so no information leaks from the future.
    df["pm25"] = df["pm25"].ffill()
    df[config.WEATHER_COLS] = df[config.WEATHER_COLS].ffill()
    df["cbwd"] = df["cbwd"].ffill()

    # The very first rows have nothing before them to copy, so drop them
    df = df.dropna()

    # 4. Wind direction is text (NE, NW, SE, cv), so turn it into 0/1 columns
    wind = pd.get_dummies(df["cbwd"], prefix="wind").astype(int)
    df = pd.concat([df.drop(columns="cbwd"), wind], axis=1)
    return df


if __name__ == "__main__":
    raw = load_raw_data(config.RAW_FILE)
    print("Raw data shape:", raw.shape)
    print("Missing PM2.5 values:", raw["pm25"].isna().sum())

    clean = clean_data(raw)
    clean.to_csv(config.CLEAN_FILE)
    print("Clean data shape:", clean.shape)
    print("Saved to", config.CLEAN_FILE)
    print(clean.head())
