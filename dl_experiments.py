# dl_experiments.py
# Phase 2 improvement experiments: what we tried to make the networks more accurate.
# Each experiment is trained ONCE (one seed) on the same 4 folds, so they can be compared fairly.
# Results go to data/GeneratedOutput/05.DeepLearning/experiments/ (one CSV per experiment).
#
#   python dl_experiments.py              runs every experiment (about 30 to 60 minutes)
#   python dl_experiments.py GRU LSTM     runs only the experiments you name
# An experiment whose CSV already exists is skipped, so the script can be stopped and restarted.

import os
import sys
import numpy as np
import pandas as pd
import torch.nn as nn
import config
from backtest import make_folds, compute_metrics, observed_only
from dl_data import set_seed
from sequences import prepare_fold
from dl_training import train_one_model, predict
from lstm_model import LSTMForecaster
from tcn_model import TCNForecaster
from dl_experiment_models import (GRUForecaster, BiLSTMForecaster, CNNLSTMForecaster,
                                  AttentionLSTMForecaster, TransformerForecaster)

OUT = "data/GeneratedOutput/05.DeepLearning/experiments"

# name: (network class, what the network learns, loss function)
EXPERIMENTS = {
    # Step by step towards our final set-up (all with the LSTM)
    "LSTM, level target, MSE loss":  (LSTMForecaster, "level", nn.MSELoss),
    "LSTM, change target, MSE loss": (LSTMForecaster, "change", nn.MSELoss),
    "LSTM":                          (LSTMForecaster, "change", nn.L1Loss),   # our final set-up
    # Other network designs, all with the final set-up (change target, L1 loss)
    "GRU":            (GRUForecaster, "change", nn.L1Loss),
    "TCN":            (TCNForecaster, "change", nn.L1Loss),
    "Bi-LSTM":        (BiLSTMForecaster, "change", nn.L1Loss),
    "CNN-LSTM":       (CNNLSTMForecaster, "change", nn.L1Loss),
    "Attention LSTM": (AttentionLSTMForecaster, "change", nn.L1Loss),
    "Transformer":    (TransformerForecaster, "change", nn.L1Loss),
}


def as_level_target(data):
    """Change the learning target from 'change in PM2.5' to 'PM2.5 level'.
    The level is scaled with the mean and spread of the training rows, and
    predict() then turns the output back with: mean + output * spread."""
    future = data["y_fit"] * data["change_std"] + data["fit_current"]
    mean, std = future.mean(), future.std()
    level = dict(data)
    level["y_fit"] = ((future - mean) / std).astype(np.float32)
    level["val_current"] = np.full(len(data["val_current"]), mean)
    level["test_current"] = np.full(len(data["test_current"]), mean)
    level["change_std"] = std
    return level


def run_experiment(name, df):
    network, target, loss = EXPERIMENTS[name]
    all_preds = []
    for fold, train, test in make_folds(df):
        data = prepare_fold(train, test)
        if target == "level":
            data = as_level_target(data)
        set_seed(config.RANDOM_SEED)
        model = network(data["n_features"])
        history = train_one_model(model, data, loss_fn=loss())
        preds = pd.DataFrame(index=test.index)
        preds["fold"] = fold
        preds["target"] = test["target"]
        preds["target_observed"] = test["target_observed"]
        preds["forecast"] = predict(model, data["X_test"], data["test_current"], data["change_std"])
        scored = observed_only(preds)
        print(f"  {name} | fold {fold}: {len(history)} epochs,",
              compute_metrics(scored["target"], scored["forecast"]), flush=True)
        all_preds.append(preds)
    return pd.concat(all_preds)


def file_name(name):
    return f"{OUT}/{name.replace(', ', '_').replace(' ', '_')}.csv"


def load_results():
    """Average MAE, RMSE and R2 over the 4 folds for every finished experiment."""
    rows = []
    for name in EXPERIMENTS:
        if not os.path.exists(file_name(name)):
            continue
        preds = observed_only(pd.read_csv(file_name(name), index_col=0, parse_dates=True))
        scores = [compute_metrics(p["target"], p["forecast"]) for _, p in preds.groupby("fold")]
        rows.append({"experiment": name, **pd.DataFrame(scores).mean().round(3).to_dict()})
    return pd.DataFrame(rows).set_index("experiment")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(config.FEATURES_FILE, index_col="datetime", parse_dates=True)
    names = sys.argv[1:] or list(EXPERIMENTS)
    for name in names:
        if os.path.exists(file_name(name)):
            print("Already done:", name)
            continue
        print("Running:", name, flush=True)
        run_experiment(name, df).to_csv(file_name(name))
    print("\nAverage over 4 folds (one training run each):")
    print(load_results().to_string())
