# config.py
# All project settings live here, so every file uses the same values.

# ---------- File paths ----------
RAW_FILE = "data/PRSA_data_2010.1.1-2014.12.31.csv"   # the UCI Beijing PM2.5 file
CLEAN_FILE = "data/clean_data.csv"                     # made by data_loader.py
FEATURES_FILE = "data/features.csv"                    # made by features.py
RESULTS_FOLDER = "results"                             # metrics, predictions and plots

# ---------- Forecast setup ----------
HORIZON = 6   # we forecast PM2.5 exactly 6 hours ahead

# Weather columns that come with the dataset
WEATHER_COLS = ["DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]

# Lag and rolling-window sizes (in hours) for the XGBoost features
LAGS = [1, 2, 3, 6, 12, 24]
ROLLING_WINDOWS = [6, 24]

# ---------- Rolling chronological backtest ----------
# Each fold trains on ALL data before the test start,
# then tests on the next 6 months. The training window grows each fold.
TEST_PERIODS = [
    ("2013-01-01 00:00", "2013-06-30 23:00"),
    ("2013-07-01 00:00", "2013-12-31 23:00"),
    ("2014-01-01 00:00", "2014-06-30 23:00"),
    ("2014-07-01 00:00", "2014-12-31 23:00"),
]

RANDOM_SEED = 42

# ---------- Phase 2: deep learning (LSTM, temporal CNN, hybrid) ----------
SEQ_LEN = 48            # every sample is a window of the last 48 hours
VAL_FRACTION = 0.1      # the last 10% of each training period is the validation set
BATCH_SIZE = 256
MAX_EPOCHS = 30         # upper limit; early stopping usually ends training sooner
PATIENCE = 4            # stop when validation has not improved for 4 epochs
LEARNING_RATE = 0.001
LSTM_HIDDEN = 64        # size of the LSTM memory
TCN_CHANNELS = 32       # number of filters in each temporal CNN layer
N_SEEDS = 3             # train each network 3 times and average the forecasts

# Models averaged by the hybrid ensemble
HYBRID_MODELS = ["XGBoost", "LSTM", "TCN"]
