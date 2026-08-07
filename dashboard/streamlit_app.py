import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Pearls AQI Predictor",
    page_icon="🌫️",
    layout="wide"
)

API_URL = "http://127.0.0.1:5000"

# ── AQI category helper ───────────────────────────────────
def aqi_category(aqi):
    if aqi <= 50:   return "Good", "#00e400"
    if aqi <= 100:  return "Moderate", "#ffff00"
    if aqi <= 150:  return "Unhealthy for Sensitive Groups", "#ff7e00"
    if aqi <= 200:  return "Unhealthy", "#ff0000"
    if aqi <= 300:  return "Very Unhealthy", "#8f3f97"
    return "Hazardous", "#7e0023"

# ── Header ────────────────────────────────────────────────
st.title("🌫️ Pearls AQI Predictor")
st.markdown("Real-time Air Quality Index forecasting for any city worldwide")

# ── City input ────────────────────────────────────────────
city = st.text_input("🔍 Enter City Name", value="Lahore")

if st.button("Get Forecast"):
    with st.spinner("Fetching data..."):

        # ── Current AQI ──────────────────────────────────
        try:
            curr = requests.get(f"{API_URL}/current?city={city}").json()
            aqi_val = curr.get("aqi", "N/A")
            cat, color = aqi_category(aqi_val) if isinstance(aqi_val, int) else ("Unknown", "#ccc")

            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            col1.metric("Current AQI", aqi_val)
            col2.metric("PM2.5", curr.get("pm25", "N/A"))
            col3.metric("Temperature", f"{curr.get('temperature','N/A')}°C")

            col4, col5, col6 = st.columns(3)
            col4.metric("Humidity", f"{curr.get('humidity','N/A')}%")
            col5.metric("Wind Speed", f"{curr.get('wind_speed','N/A')} m/s")
            col6.metric("Weather", curr.get("weather", "N/A"))

            # Alert banner
            st.markdown(
                f"<div style='background:{color};padding:10px;border-radius:8px;"
                f"text-align:center;font-weight:bold;color:#000'>"
                f"AQI Category: {cat}</div>",
                unsafe_allow_html=True
            )

            if aqi_val > 150:
                st.error("⚠️ HAZARD ALERT: Air quality is unhealthy! Avoid outdoor activities.")

        except Exception as e:
            st.error(f"Could not fetch current data: {e}")

        # ── 3-Day Forecast ───────────────────────────────
        try:
            st.markdown("---")
            st.subheader("📈 3-Day AQI Forecast")

            pred = requests.get(f"{API_URL}/predict?city={city}").json()
            df_pred = pd.DataFrame(pred["predictions"])
            df_pred["datetime"] = pd.to_datetime(df_pred["datetime"])

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(df_pred["datetime"], df_pred["predicted_aqi"],
                    color="#ff7e00", linewidth=2)
            ax.fill_between(df_pred["datetime"], df_pred["predicted_aqi"],
                            alpha=0.2, color="#ff7e00")
            ax.set_xlabel("Date & Time")
            ax.set_ylabel("Predicted AQI")
            ax.set_title(f"72-Hour AQI Forecast for {city}")
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

            # Day summary
            st.subheader("📊 Daily Summary")
            summary = df_pred.groupby("day")["predicted_aqi"].mean().round(1).reset_index()
            summary.columns = ["Day", "Avg Predicted AQI"]
            summary["Category"] = summary["Avg Predicted AQI"].apply(
                lambda x: aqi_category(x)[0]
            )
            st.dataframe(summary, use_container_width=True)

        except Exception as e:
            st.error(f"Could not fetch predictions: {e}")

        # ── History chart ────────────────────────────────
        try:
            st.markdown("---")
            st.subheader("📉 Past 7 Days AQI History")

            hist = requests.get(f"{API_URL}/history").json()
            df_hist = pd.DataFrame(hist["history"])
            df_hist["timestamp"] = pd.to_datetime(df_hist["timestamp"])

            fig2, ax2 = plt.subplots(figsize=(12, 3))
            ax2.plot(df_hist["timestamp"], df_hist["aqi"], color="#1f77b4", linewidth=1.5)
            ax2.set_xlabel("Date")
            ax2.set_ylabel("AQI")
            ax2.set_title("Historical AQI (Last 7 Days)")
            ax2.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig2)

        except Exception as e:
            st.error(f"Could not fetch history: {e}")

        # ── SHAP plot ────────────────────────────────────
        try:
            st.markdown("---")
            st.subheader("🧠 Feature Importance (SHAP)")
            if os.path.exists("models/shap_summary.png"):
                st.image("models/shap_summary.png",
                         caption="SHAP Feature Importance",
                         use_container_width=True)
        except Exception as e:
            st.error(f"Could not load SHAP plot: {e}")

st.markdown("---")
st.caption("Pearls AQI Predictor | Powered by AQICN + OpenWeatherMap + ML")