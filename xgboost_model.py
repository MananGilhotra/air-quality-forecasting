# xgboost_model.py
# The AML baseline: XGBoost regression on lag, rolling, calendar and weather features.

import pandas as pd
from xgboost import XGBRegressor
import config
from backtest import make_folds, compute_metrics, observed_only
from features import get_feature_columns


def train_and_predict_xgboost(train_df, test_df):
    feature_cols = get_feature_columns(train_df)
    train_df = observed_only(train_df)

    model = XGBRegressor(
        n_estimators=300,      # number of trees
        learning_rate=0.05,    # how big each correction step is
        max_depth=6,           # how deep each tree can grow
        subsample=0.8,         # each tree sees 80% of rows
        colsample_bytree=0.8,  # each tree sees 80% of features
        random_state=config.RANDOM_SEED,
    )
    model.fit(train_df[feature_cols], train_df["target"])

    predictions = model.predict(test_df[feature_cols])
    return predictions, model


if __name__ == "__main__":
    df = pd.read_csv(config.FEATURES_FILE, index_col="datetime", parse_dates=True)
    for fold, train, test in make_folds(df):
        preds, model = train_and_predict_xgboost(train, test)
        test = test.assign(pred=preds)
        test = observed_only(test)
        print(f"Fold {fold} XGBoost:", compute_metrics(test["target"], test["pred"]))

    # Which features mattered most in the last fold?
    importance = pd.Series(model.feature_importances_, index=get_feature_columns(train))
    print("\nTop 10 features:")
    print(importance.sort_values(ascending=False).head(10))
