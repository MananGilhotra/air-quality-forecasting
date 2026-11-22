# plots_dl.py
# Phase 2 charts: all six models side by side, and the hybrid forecast against reality.

import matplotlib
matplotlib.use("Agg")            # save charts to files without opening windows
import matplotlib.pyplot as plt
import numpy as np
import config

ALL_MODELS = ["Persistence", "Seasonal naive", "XGBoost", "LSTM", "TCN", "Hybrid"]
COLORS = {"Persistence": "tab:gray", "Seasonal naive": "tab:olive", "XGBoost": "tab:blue",
          "LSTM": "tab:green", "TCN": "tab:purple", "Hybrid": "tab:red"}


def plot_all_models(summary, per_fold, path):
    """Left: average MAE and RMSE per model. Right: MAE in each test period."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))

    x = np.arange(len(ALL_MODELS))
    axes[0].bar(x - 0.2, summary.loc[ALL_MODELS, "MAE"], width=0.4, label="MAE", color="tab:blue")
    axes[0].bar(x + 0.2, summary.loc[ALL_MODELS, "RMSE"], width=0.4, label="RMSE", color="tab:orange")
    axes[0].set_xticks(x, ALL_MODELS, rotation=15)
    axes[0].set_ylabel("µg/m³ (lower is better)")
    axes[0].set_title(f"Average error over all folds ({config.HORIZON}-hour-ahead PM2.5)")
    axes[0].legend()

    for model in ["Persistence", "XGBoost", "LSTM", "TCN", "Hybrid"]:
        part = per_fold[per_fold["model"] == model]
        axes[1].plot(part["fold"], part["MAE"], marker="o", label=model, color=COLORS[model])
    axes[1].set_xticks([1, 2, 3, 4], ["Jan–Jun 13", "Jul–Dec 13", "Jan–Jun 14", "Jul–Dec 14"])
    axes[1].set_ylabel("MAE (µg/m³)")
    axes[1].set_title("MAE in each test period")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_hybrid_forecast(preds, path, days=7):
    """Actual PM2.5 against XGBoost, LSTM and the hybrid for the last few days."""
    last = preds.tail(days * 24)
    plt.figure(figsize=(12, 5))
    plt.plot(last["forecast_for"], last["actual"], color="black", linewidth=2, label="Actual")
    for model in ["XGBoost", "LSTM", "Hybrid"]:
        plt.plot(last["forecast_for"], last[model], color=COLORS[model], alpha=0.85, label=model)
    plt.title(f"{config.HORIZON}-hour-ahead PM2.5 forecast, last {days} days")
    plt.ylabel("PM2.5 (µg/m³)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
