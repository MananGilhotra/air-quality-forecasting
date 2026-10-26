# plots.py
# Charts for the report: model errors, forecast vs actual, and the AQI view.

import matplotlib
matplotlib.use("Agg")            # save charts to files without opening windows
import matplotlib.pyplot as plt
import config
from aqi import AQI_CATEGORIES

MODELS = ["Persistence", "Seasonal naive", "XGBoost"]


def plot_metrics(summary):
    """Bar chart of average MAE and RMSE for each model (lower is better)."""
    ax = summary[["MAE", "RMSE"]].plot(kind="bar", figsize=(8, 5), rot=0)
    ax.set_title(f"Average error over all folds ({config.HORIZON}-hour-ahead PM2.5)")
    ax.set_ylabel("µg/m³")
    plt.tight_layout()
    plt.savefig(f"{config.RESULTS_FOLDER}/model_errors.png")
    plt.close()


def plot_forecast(preds, days=7):
    """Actual PM2.5 vs each model for the last few days of the last fold."""
    last = preds.tail(days * 24)
    plt.figure(figsize=(12, 5))
    plt.plot(last["forecast_for"], last["actual"], label="Actual", color="black", linewidth=2)
    for model in MODELS:
        plt.plot(last["forecast_for"], last[model], label=model, alpha=0.8)
    plt.title(f"{config.HORIZON}-hour-ahead PM2.5 forecast, last {days} days")
    plt.ylabel("PM2.5 (µg/m³)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{config.RESULTS_FOLDER}/forecast_vs_actual.png")
    plt.close()


def plot_aqi(preds, model="XGBoost", days=14):
    """Forecast drawn over coloured AQI category bands."""
    last = preds.tail(days * 24)
    plt.figure(figsize=(12, 5))

    lower = 0
    top = max(last["actual"].max(), last[model].max()) * 1.1
    for upper, name, colour in AQI_CATEGORIES:
        if lower < top:
            plt.axhspan(lower, min(upper, top), color=colour, alpha=0.15, label=name)
        lower = upper

    plt.plot(last["forecast_for"], last["actual"], color="black", label="Actual")
    plt.plot(last["forecast_for"], last[model], color="blue", label=f"{model} forecast")
    plt.ylim(0, top)
    plt.title(f"{model} forecast on US EPA AQI categories, last {days} days")
    plt.ylabel("PM2.5 (µg/m³)")
    plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{config.RESULTS_FOLDER}/aqi_forecast.png")
    plt.close()
