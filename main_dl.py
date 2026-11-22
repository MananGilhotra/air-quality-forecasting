# main_dl.py
# Runs Phase 2 end to end:
# Phase 1 results -> LSTM backtest -> temporal CNN backtest -> hybrid -> comparison of all models.

import os
import subprocess
import sys
import pandas as pd
import config
from data_loader import load_raw_data, clean_data
from features import build_features
from backtest import compute_metrics, observed_only
from dl_training import run_dl_backtest
from hybrid import hybrid_forecast
from aqi import category_accuracy, within_one_category_accuracy
from plots_dl import ALL_MODELS, plot_all_models, plot_hybrid_forecast


def main():
    os.makedirs(config.RESULTS_FOLDER, exist_ok=True)

    print("Step 1: Phase 1 models (baselines and XGBoost)")
    # main.py runs in its own process. XGBoost and PyTorch each bring their own
    # multi-threading library, and on some computers they freeze each other
    # when both are used inside one program.
    subprocess.run([sys.executable, "main.py"], check=True, stdout=subprocess.DEVNULL)
    preds = pd.read_csv(f"{config.RESULTS_FOLDER}/predictions.csv",
                        index_col="forecast_made_at", parse_dates=["forecast_made_at", "forecast_for"])

    print("Step 2: building features")
    df = build_features(clean_data(load_raw_data(config.RAW_FILE)))

    print("Step 3: LSTM backtest")
    preds["LSTM"] = run_dl_backtest("LSTM", df)["LSTM"]

    print("Step 4: temporal CNN backtest")
    preds["TCN"] = run_dl_backtest("TCN", df)["TCN"]

    print("Step 5: hybrid ensemble of", config.HYBRID_MODELS)
    preds["Hybrid"] = hybrid_forecast(preds)

    print("Step 6: scoring every model on the same hours")
    scored = observed_only(preds)
    rows = []
    for fold, part in scored.groupby("fold"):
        for model in ALL_MODELS:
            scores = compute_metrics(part["actual"], part[model])
            scores["AQI category accuracy"] = category_accuracy(part["actual"], part[model])
            scores["AQI within one category"] = within_one_category_accuracy(part["actual"], part[model])
            rows.append({"fold": fold, "model": model, **scores})
    per_fold = pd.DataFrame(rows)
    summary = per_fold.groupby("model")[["MAE", "RMSE", "R2", "AQI category accuracy", "AQI within one category"]].mean().round(3)
    summary = summary.loc[ALL_MODELS]

    per_fold.to_csv(f"{config.RESULTS_FOLDER}/dl_metrics_per_fold.csv", index=False)
    summary.to_csv(f"{config.RESULTS_FOLDER}/dl_metrics_summary.csv")
    preds.to_csv(f"{config.RESULTS_FOLDER}/dl_predictions.csv")
    plot_all_models(summary, per_fold, f"{config.RESULTS_FOLDER}/dl_model_comparison.png")
    plot_hybrid_forecast(preds, f"{config.RESULTS_FOLDER}/dl_forecast_vs_actual.png")

    print(f"\nAverage over {len(config.TEST_PERIODS)} folds, horizon = {config.HORIZON} hours:")
    print(summary.to_string())
    best = summary["MAE"].idxmin()
    gain = 1 - summary.loc[best, "MAE"] / summary.loc["Persistence", "MAE"]
    print(f"\nBest model: {best} (MAE {summary.loc[best, 'MAE']:.2f}, {gain:.0%} lower than persistence)")
    print(f"{best} AQI category: exactly right {summary.loc[best, 'AQI category accuracy']:.1%} of hours, "
          f"right or one category off {summary.loc[best, 'AQI within one category']:.1%} of hours")
    print(f"All files saved in the '{config.RESULTS_FOLDER}' folder.")


if __name__ == "__main__":
    main()
