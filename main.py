# main.py
# Runs the whole project: data -> features -> rolling backtest -> all models -> results.

import os
import pandas as pd
import config
from data_loader import load_raw_data, clean_data
from features import build_features
from backtest import make_folds, compute_metrics, observed_only
from baselines import persistence_forecast, seasonal_naive_forecast
from xgboost_model import train_and_predict_xgboost
from aqi import category_accuracy
from plots import MODELS, plot_metrics, plot_forecast, plot_aqi


def main():
    os.makedirs(config.RESULTS_FOLDER, exist_ok=True)

    print("Step 1: loading and cleaning data")
    clean = clean_data(load_raw_data(config.RAW_FILE))

    print("Step 2: building features")
    df = build_features(clean)

    print("Step 3: rolling backtest")
    metric_rows = []
    all_preds = []
    for fold, train, test in make_folds(df):
        print(f"\nFold {fold}: train up to {train.index.max()}, test {test.index.min()} to {test.index.max()}")

        preds = pd.DataFrame(index=test.index)
        preds["forecast_for"] = test.index + pd.Timedelta(hours=config.HORIZON)
        preds["actual"] = test["target"]
        preds["target_observed"] = test["target_observed"]
        preds["fold"] = fold

        preds["Persistence"] = persistence_forecast(test)
        preds["Seasonal naive"] = seasonal_naive_forecast(test)
        xgb_preds, _ = train_and_predict_xgboost(train, test)
        preds["XGBoost"] = xgb_preds

        scored = observed_only(preds)
        for model in MODELS:
            scores = compute_metrics(scored["actual"], scored[model])
            scores["AQI category accuracy"] = category_accuracy(scored["actual"], scored[model])
            metric_rows.append({"fold": fold, "model": model, **scores})
            print(f"  {model:15s} {scores}")
        all_preds.append(preds)

    print("\nStep 4: saving results")
    per_fold = pd.DataFrame(metric_rows)
    summary = per_fold.groupby("model")[["MAE", "RMSE", "R2", "AQI category accuracy"]].mean().round(3)
    summary = summary.loc[MODELS]           # keep a fixed model order
    predictions = pd.concat(all_preds)
    predictions.index.name = "forecast_made_at"

    per_fold.to_csv(f"{config.RESULTS_FOLDER}/metrics_per_fold.csv", index=False)
    summary.to_csv(f"{config.RESULTS_FOLDER}/metrics_summary.csv")
    predictions.to_csv(f"{config.RESULTS_FOLDER}/predictions.csv")

    plot_metrics(summary)
    plot_forecast(predictions)
    plot_aqi(predictions)

    print(f"\nAverage over {len(config.TEST_PERIODS)} folds, horizon = {config.HORIZON} hours:")
    print(summary)
    print(f"\nAll files saved in the '{config.RESULTS_FOLDER}' folder.")


if __name__ == "__main__":
    main()
