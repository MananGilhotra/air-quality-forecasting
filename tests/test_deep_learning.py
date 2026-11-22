# test_deep_learning.py
# Checks for Phase 2: validation split, scaling, sequences, networks, training and the hybrid.
# Run from the project folder:  pytest tests/

import numpy as np
import pandas as pd
import pytest
import torch
import config
from data_loader import load_raw_data, clean_data
from features import build_features
from backtest import make_folds
from dl_data import set_seed, split_train_validation, fit_scaler, apply_scaler
from sequences import get_sequence_columns, make_windows, to_pm25, prepare_fold
from lstm_model import LSTMForecaster
from tcn_model import TCNForecaster
from dl_training import train_one_model, predict
from hybrid import hybrid_forecast
from aqi import within_one_category_accuracy


@pytest.fixture(scope="module")
def first_fold():
    features = build_features(clean_data(load_raw_data(config.RAW_FILE)))
    fold, train, test = make_folds(features)[0]
    return train, test


@pytest.fixture(scope="module")
def data(first_fold):
    train, test = first_fold
    return prepare_fold(train, test)


def test_validation_comes_after_training_rows(first_fold):
    train, _ = first_fold
    fit_df, val_df = split_train_validation(train)
    assert fit_df.index.max() < val_df.index.min()
    assert len(fit_df) + len(val_df) == len(train)
    # the last HORIZON training rows must not be used as targets
    assert (fit_df["target_observed"].iloc[-config.HORIZON:] == 0).all()


def test_scaler_is_fitted_on_training_rows_only(first_fold):
    train, test = first_fold
    fit_df, _ = split_train_validation(train)
    columns = ["pm25", "TEMP"]
    mean, std = fit_scaler(fit_df, columns)
    assert np.allclose(mean.values, fit_df[columns].mean().values)
    scaled_fit = apply_scaler(fit_df, columns, mean, std)
    assert np.allclose(scaled_fit.mean(axis=0), 0, atol=1e-3)
    # test rows are scaled with the training mean, so their mean is not forced to 0
    scaled_test = apply_scaler(test, columns, mean, std)
    assert not np.allclose(scaled_test.mean(axis=0), 0, atol=1e-3)


def test_windows_end_at_the_right_hour():
    values = np.arange(20, dtype=np.float32).reshape(10, 2)
    windows = make_windows(values, seq_len=4)
    assert windows.shape == (7, 4, 2)
    assert np.array_equal(windows[0], values[0:4])
    assert np.array_equal(windows[-1], values[6:10])


def test_fold_data_shapes(first_fold, data):
    train, test = first_fold
    n_features = len(get_sequence_columns(train))
    assert n_features == 16
    assert data["X_fit"].shape[1:] == (config.SEQ_LEN, n_features)
    assert len(data["X_fit"]) == len(data["y_fit"])
    assert len(data["X_val"]) == len(data["val_actual"])
    assert len(data["X_test"]) == len(test)          # one forecast for every test hour


def test_last_step_of_a_test_window_is_the_forecast_hour(first_fold, data):
    train, test = first_fold
    # PM2.5 is the first sequence column; undo the scaling of the last step
    fit_df, _ = split_train_validation(train)
    mean, std = fit_scaler(fit_df, ["pm25"])
    last_step_pm25 = data["X_test"][:, -1, 0] * std["pm25"] + mean["pm25"]
    assert np.allclose(last_step_pm25, test["pm25"].values, atol=1e-2)
    assert np.allclose(data["test_current"], test["pm25"].values)


def test_to_pm25_adds_the_change_and_is_never_negative():
    forecast = to_pm25(np.array([1.0, -5.0]), np.array([100.0, 20.0]), change_std=10.0)
    assert np.allclose(forecast, [110.0, 0.0])


@pytest.mark.parametrize("model_class", [LSTMForecaster, TCNForecaster])
def test_networks_return_one_number_per_sample(model_class):
    model = model_class(n_features=16)
    output = model(torch.randn(5, config.SEQ_LEN, 16))
    assert output.shape == (5,)


def test_tcn_only_uses_the_last_31_hours():
    set_seed(0)
    model = TCNForecaster(n_features=16).eval()
    x = torch.randn(1, config.SEQ_LEN, 16)
    changed = x.clone()
    changed[:, :10, :] += 5.0          # change only the oldest 10 hours (outside the 31-hour view)
    with torch.no_grad():
        assert torch.allclose(model(x), model(changed), atol=1e-5)


def test_training_improves_on_persistence(data, monkeypatch):
    # A short training run on part of the data must already beat "no change" on validation
    monkeypatch.setattr(config, "MAX_EPOCHS", 2)
    small = dict(data, X_fit=data["X_fit"][-6000:], y_fit=data["y_fit"][-6000:])
    set_seed(config.RANDOM_SEED)
    model = LSTMForecaster(small["n_features"])
    history = train_one_model(model, small)
    persistence_mae = np.abs(small["val_current"] - small["val_actual"]).mean()
    assert len(history) == 2
    assert min(history) < persistence_mae
    forecast = predict(model, small["X_val"], small["val_current"], small["change_std"])
    assert (forecast >= 0).all()


def test_hybrid_is_the_average_of_its_models():
    preds = pd.DataFrame({"XGBoost": [100.0, 40.0], "LSTM": [90.0, 50.0], "TCN": [80.0, 60.0]})
    assert np.allclose(hybrid_forecast(preds), [90.0, 50.0])


def test_within_one_category_accuracy():
    # true value 40 is "Unhealthy for Sensitive Groups"
    actual = [40, 40, 40, 40]
    predicted = [45, 30, 60, 130]   # same, one below, one above, three above
    assert within_one_category_accuracy(actual, predicted) == 0.75
