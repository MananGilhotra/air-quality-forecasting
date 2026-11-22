# hybrid.py
# Phase 2, step 5a: the hybrid ensemble.
# XGBoost (hand-made lag features) and the neural networks (raw 48-hour sequences)
# make different mistakes. Averaging their forecasts cancels part of those mistakes.

import pandas as pd
import config


def hybrid_forecast(predictions, models=config.HYBRID_MODELS):
    """Simple average of the chosen models' forecasts, hour by hour.
    Equal weights: nothing is tuned on the test data."""
    return predictions[models].mean(axis=1)


if __name__ == "__main__":
    # A tiny example: three models forecast two hours
    example = pd.DataFrame({"XGBoost": [100.0, 40.0], "LSTM": [90.0, 50.0], "TCN": [80.0, 60.0]})
    example["Hybrid"] = hybrid_forecast(example)
    print(example)
