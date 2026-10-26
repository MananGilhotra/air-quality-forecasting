# app.py  (optional)
# A small Streamlit dashboard. Run main.py first, then:  streamlit run app.py

import pandas as pd
import streamlit as st
import config
from aqi import pm25_to_category
from plots import MODELS

st.title("Beijing PM2.5 Forecast")
st.write(f"Forecast horizon: {config.HORIZON} hours ahead · AQI standard: US EPA (2024)")

summary = pd.read_csv(f"{config.RESULTS_FOLDER}/metrics_summary.csv", index_col="model")
preds = pd.read_csv(f"{config.RESULTS_FOLDER}/predictions.csv", parse_dates=["forecast_for"])

st.subheader("Model comparison (average over all folds)")
st.dataframe(summary)

model = st.selectbox("Choose a model", MODELS, index=MODELS.index("XGBoost"))
days = st.slider("Days to show", min_value=1, max_value=30, value=7)

recent = preds.tail(days * 24).set_index("forecast_for")
st.subheader("Forecast vs actual")
st.line_chart(recent[["actual", model]])

latest = recent.iloc[-1]
st.subheader("Latest forecast")
st.metric(f"PM2.5 at {recent.index[-1]}", f"{latest[model]:.1f} µg/m³")
st.write("AQI category:", pm25_to_category(latest[model]))
