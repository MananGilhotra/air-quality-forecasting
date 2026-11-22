# dl_training.py
# Phase 2, step 4: train the networks and run them through the SAME rolling
# chronological backtest as the baselines and XGBoost.

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import config
from backtest import make_folds, compute_metrics, observed_only
from dl_data import set_seed
from sequences import prepare_fold, to_pm25
from lstm_model import LSTMForecaster
from tcn_model import TCNForecaster

MODEL_CLASSES = {"LSTM": LSTMForecaster, "TCN": TCNForecaster}


def predict(model, X, current_pm25, change_std):
    """Run the network and convert its output back to PM2.5 in µg/m³."""
    model.eval()
    with torch.no_grad():
        output = model(torch.tensor(X)).numpy()
    return to_pm25(output, current_pm25, change_std)


def train_one_model(model, data, loss_fn=None):
    """Train with early stopping. Returns the validation MAE after every epoch."""
    X_fit = torch.tensor(data["X_fit"])
    y_fit = torch.tensor(data["y_fit"])
    if loss_fn is None:
        loss_fn = nn.L1Loss()              # mean absolute error, the metric we report
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    best_mae = float("inf")
    best_weights = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(config.MAX_EPOCHS):
        # One pass over the training windows, in random order
        # (shuffling whole windows is fine: the time order INSIDE each window is kept)
        model.train()
        order = torch.randperm(len(X_fit))
        for start in range(0, len(X_fit), config.BATCH_SIZE):
            batch = order[start:start + config.BATCH_SIZE]
            optimizer.zero_grad()
            loss = loss_fn(model(X_fit[batch]), y_fit[batch])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # keeps training stable
            optimizer.step()

        # Check on the validation period (later in time than the training rows)
        val_pred = predict(model, data["X_val"], data["val_current"], data["change_std"])
        val_mae = float(np.abs(val_pred - data["val_actual"]).mean())
        history.append(val_mae)

        if val_mae < best_mae:
            best_mae = val_mae
            best_weights = {name: w.clone() for name, w in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.PATIENCE:
                break                      # early stopping

    model.load_state_dict(best_weights)    # go back to the best epoch
    return history


def train_and_predict_dl(model_name, train_df, test_df):
    """Train one kind of network on a fold and forecast every test hour."""
    data = prepare_fold(train_df, test_df)
    forecasts = []
    for i in range(config.N_SEEDS):
        set_seed(config.RANDOM_SEED + i)
        model = MODEL_CLASSES[model_name](data["n_features"])
        history = train_one_model(model, data)
        print(f"    {model_name} run {i + 1}/{config.N_SEEDS}: {len(history)} epochs, "
              f"best validation MAE {min(history):.2f}")
        forecasts.append(predict(model, data["X_test"], data["test_current"], data["change_std"]))
    # Average the runs: each run starts from different random weights,
    # and their average is more accurate than any single run
    return np.mean(forecasts, axis=0)


def run_dl_backtest(model_name, df):
    """Forecasts of one network for every test hour of every fold."""
    all_preds = []
    for fold, train, test in make_folds(df):
        print(f"  Fold {fold}: training {model_name}")
        preds = pd.DataFrame(index=test.index)
        preds["fold"] = fold
        preds["target"] = test["target"]
        preds["target_observed"] = test["target_observed"]
        preds[model_name] = train_and_predict_dl(model_name, train, test)
        scored = observed_only(preds)
        print(f"  Fold {fold} {model_name}:", compute_metrics(scored["target"], scored[model_name]))
        all_preds.append(preds)
    return pd.concat(all_preds)


if __name__ == "__main__":
    df = pd.read_csv(config.FEATURES_FILE, index_col="datetime", parse_dates=True)
    run_dl_backtest("LSTM", df)
