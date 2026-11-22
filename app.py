# app.py  (optional)
# A small Streamlit dashboard.
# Run main.py (Phase 1) or main_dl.py (Phase 1 + deep learning) first, then:  streamlit run app.py

import os
import pandas as pd
import streamlit as st
import config
from aqi import pm25_to_category

st.title("Beijing PM2.5 Forecast")
st.write(f"Forecast horizon: {config.HORIZON} hours ahead · AQI standard: US EPA (2024)")

# Use the deep learning results when they exist, otherwise the Phase 1 results
if os.path.exists(f"{config.RESULTS_FOLDER}/dl_predictions.csv"):
    summary_file, preds_file = "dl_metrics_summary.csv", "dl_predictions.csv"
    st.caption("Showing Phase 1 + Phase 2 models (baselines, XGBoost, LSTM, temporal CNN, hybrid).")
else:
    summary_file, preds_file = "metrics_summary.csv", "predictions.csv"
    st.caption("Showing Phase 1 models. Run main_dl.py to add the deep learning models.")

summary = pd.read_csv(f"{config.RESULTS_FOLDER}/{summary_file}", index_col="model")
preds = pd.read_csv(f"{config.RESULTS_FOLDER}/{preds_file}", parse_dates=["forecast_for"])
models = list(summary.index)

st.subheader("Model comparison (average over all folds)")
st.dataframe(summary)

model = st.selectbox("Choose a model", models, index=len(models) - 1)
days = st.slider("Days to show", min_value=1, max_value=30, value=7)

recent = preds.tail(days * 24).set_index("forecast_for")
st.subheader("Forecast vs actual")
st.line_chart(recent[["actual", model]])

latest = recent.iloc[-1]
st.subheader("Latest forecast")
st.metric(f"PM2.5 at {recent.index[-1]}", f"{latest[model]:.1f} µg/m³")
st.write("AQI category:", pm25_to_category(latest[model]))
