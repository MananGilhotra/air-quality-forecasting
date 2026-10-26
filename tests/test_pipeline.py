# test_pipeline.py
# Checks that cleaning, features, the backtest and the models behave as described.
# Run from the project folder:  pytest tests/

import numpy as np
import pandas as pd
import pytest
import config
from data_loader import load_raw_data, clean_data
from features import build_features, get_feature_columns
from backtest import make_folds, compute_metrics, observed_only
from baselines import persistence_forecast
from xgboost_model import train_and_predict_xgboost


@pytest.fixture(scope="module")
def clean():
    return clean_data(load_raw_data(config.RAW_FILE))


@pytest.fixture(scope="module")
def features(clean):
    return build_features(clean)


def test_clean_data_is_regular_hourly(clean):
    steps = clean.index.to_series().diff().dropna()
    assert (steps == pd.Timedelta(hours=1)).all()


def test_clean_data_has_no_empty_values(clean):
    assert clean.isna().sum().sum() == 0


def test_missing_flag_is_kept(clean):
    assert clean["pm25_missing"].sum() == 2043


def test_feature_table_shape(features):
    assert features.shape == (43770, 33)
    assert len(get_feature_columns(features)) == 31


def test_target_is_6_hours_ahead(clean, features):
    expected = clean["pm25"].shift(-config.HORIZON).loc[features.index]
    assert (features["target"] == expected).all()


def test_lags_only_look_backwards(clean, features):
    expected = clean["pm25"].shift(6).loc[features.index]
    assert (features["pm25_lag_6"] == expected).all()


def test_answer_columns_are_not_features(features):
    columns = get_feature_columns(features)
    assert "target" not in columns
    assert "target_observed" not in columns


def test_folds_are_in_time_order_with_leakage_guard(features):
    for fold, train, test in make_folds(features):
        assert train.index.max() < test.index.min()
        assert (train["target_observed"].iloc[-config.HORIZON:] == 0).all()


def test_metrics_for_a_perfect_forecast():
    y = np.array([10.0, 20.0, 30.0])
    scores = compute_metrics(y, y)
    assert scores["MAE"] == 0
    assert scores["RMSE"] == 0
    assert scores["R2"] == 1


def test_xgboost_beats_persistence_on_first_fold(features):
    fold, train, test = make_folds(features)[0]
    preds, _ = train_and_predict_xgboost(train, test)
    scored = observed_only(test.assign(xgb=preds, persistence=persistence_forecast(test)))
    xgb_mae = compute_metrics(scored["target"], scored["xgb"])["MAE"]
    persistence_mae = compute_metrics(scored["target"], scored["persistence"])["MAE"]
    assert xgb_mae < persistence_mae
