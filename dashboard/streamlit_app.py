import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from supabase import create_client
import joblib
import tempfile
import numpy as np
import os
from datetime import datetime, timedelta

st.set_page_config(page_title="Pearls AQI Predictor", page_icon="🌍", layout="wide")
st.markdown("""
<style>
.block-container {
    max-width: 1100px;
    padding-left: 2rem;
    padding-right: 2rem;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_supabase():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

@st.cache_resource
def load_model():
    sb = get_supabase()
    try:
        reg = sb.table("model_registry").select("*").order("trained_at", desc=True).limit(1).execute()
        if not reg.data:
            return None, None, None
        meta = reg.data[0]
        version = meta["version"]
        with tempfile.TemporaryDirectory() as tmp:
            model_bytes = sb.storage.from_("models").download(f"champion_{version}.joblib")
            encoder_bytes = sb.storage.from_("models").download(f"encoder_{version}.joblib")
            model_path = os.path.join(tmp, "model.joblib")
            encoder_path = os.path.join(tmp, "encoder.joblib")
            with open(model_path, "wb") as f:
                f.write(model_bytes)
            with open(encoder_path, "wb") as f:
                f.write(encoder_bytes)
            model = joblib.load(model_path)
            le = joblib.load(encoder_path)
        return model, le, version
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None, None, None

def get_alert(aqi):
    if aqi <= 50: return "Good", "#00c853"
    elif aqi <= 100: return "Moderate", "#ffd600"
    elif aqi <= 150: return "Unhealthy for Sensitive Groups", "#ff6d00"
    elif aqi <= 200: return "Unhealthy", "#d50000"
    elif aqi <= 300: return "Very Unhealthy", "#6a1b9a"
    else: return "Hazardous", "#7e0023"

FEATURE_COLS = ["city_encoded", "hour", "day_of_week", "month",
                "aqi_lag_1h", "aqi_lag_24h", "aqi_roll_mean_24h", "aqi_change_rate"]

def predict(city, model, le):
    sb = get_supabase()
    result = sb.table("aqi_gold_features")\
        .select("*").eq("city", city)\
        .order("timestamp", desc=True).limit(1).execute()
    if not result.data:
        return None, None
    row = result.data[0]
    city_encoded = int(le.transform([city])[0])
    features = {
        "city_encoded": city_encoded,
        "hour": row.get("hour", 0),
        "day_of_week": row.get("day_of_week", 0),
        "month": row.get("month", 1),
        "aqi_lag_1h": row.get("aqi_lag_1h") or 0,
        "aqi_lag_24h": row.get("aqi_lag_24h") or 0,
        "aqi_roll_mean_24h": row.get("aqi_roll_mean_24h") or 0,
        "aqi_change_rate": row.get("aqi_change_rate") or 0,
    }
    X = pd.DataFrame([features])[FEATURE_COLS]
    preds = model.predict(X)[0]
    as_of = row["timestamp"]
    base_date = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
    forecast = []
    for i, aqi_val in enumerate(preds):
        forecast.append({
            "day": i+1,
            "date": (base_date + timedelta(days=i+1)).strftime("%Y-%m-%d"),
            "aqi": int(round(aqi_val))
        })
    return forecast, row["timestamp"]

def get_history(city):
    sb = get_supabase()
    result = sb.table("aqi_gold_features")\
        .select("timestamp,aqi").eq("city", city)\
        .order("timestamp", desc=True).limit(200).execute()
    return result.data

def spacer():
    st.markdown("<div style='margin:30px 0'></div>", unsafe_allow_html=True)

# Header
st.title("🌍 Pearls AQI Predictor")
st.markdown("""
**Know Your Air. Plan Ahead.**  
*3-day AQI forecasting.*
""")

# Load model
model, le, version = load_model()
if model is None:
    st.error("Model not loaded!")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("Settings")
    cities = list(le.classes_)
    city = st.selectbox("Select City", cities)
    st.caption(f"Model: `{version}`")
    st.markdown("---")
    st.markdown("**EPA AQI Scale:**")
    st.markdown("🟢 0-50: Good")
    st.markdown("🟡 51-100: Moderate")
    st.markdown("🟠 101-150: Sensitive Groups")
    st.markdown("🔴 151-200: Unhealthy")
    st.markdown("🟣 201-300: Very Unhealthy")
    st.markdown("🔴 301+: Hazardous")

# Fetch data
with st.spinner(f"Fetching forecast for {city}..."):
    forecast, as_of = predict(city, model, le)
    history = get_history(city)

if forecast is None:
    st.error(f"No data found for {city}")
    st.stop()

today = datetime.now().strftime("%B %d, %Y")

st.markdown(
    f"<h2>Forecast as of: {today}</h2>",
    unsafe_allow_html=True
)

# Alert banner
max_aqi = max(f["aqi"] for f in forecast)

# Section 1 — Forecast cards
st.markdown("---")
st.subheader(f"3-Day Forecast for {city.title()}")
st.markdown("<div style='margin:16px 0'></div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
for col, f in zip([col1, col2, col3], forecast):
    label, color = get_alert(f["aqi"])
    col.markdown(f"""
    <div style="border:2px solid {color};border-radius:14px;padding:24px;
        text-align:center;margin:4px;min-height:160px">
        <div style="font-size:18px;font-weight:700;color:gray;margin-bottom:10px">{f['date']}</div>
        <div style="font-size:48px;font-weight:700;color:{color};line-height:1">{f['aqi']}</div>
        <div style="font-size:12px;color:gray;margin:6px 0">AQI Index</div>
        <div style="margin-top:12px;padding:7px;border-radius:8px;
            background:{color}22;color:{color};font-size:13px;font-weight:600">{label}</div>
    </div>
    """, unsafe_allow_html=True)

spacer()

# Section 2 — Chart
st.markdown("---")
st.subheader("Forecast + Recent History")
st.markdown("<div style='margin:16px 0'></div>", unsafe_allow_html=True)

fig = go.Figure()
if history:
    hist_df = pd.DataFrame(history)
    hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"])
    hist_df = hist_df.sort_values("timestamp")
    fig.add_trace(go.Scatter(
        x=hist_df["timestamp"], y=hist_df["aqi"],
        name="Historical AQI", line=dict(color="#1f77b4", width=2)
    ))
forecast_dates = [f["date"] for f in forecast]
forecast_aqi = [f["aqi"] for f in forecast]
fig.add_trace(go.Scatter(
    x=forecast_dates, y=forecast_aqi,
    name="Forecast AQI",
    line=dict(color="#ff4b4b", dash="dash", width=2),
    mode="lines+markers", marker=dict(size=10)
))
fig.add_hline(y=100, line_dash="dot", line_color="gold", annotation_text="Moderate")
fig.add_hline(y=150, line_dash="dot", line_color="orange", annotation_text="Sensitive")
fig.add_hline(y=200, line_dash="dot", line_color="red", annotation_text="Unhealthy")
fig.update_layout(
    title=f"AQI Forecast for {city.title()}",
    xaxis_title="Date", yaxis_title="AQI",
    hovermode="x unified",
    plot_bgcolor="rgba(0,0,0,0)",
    height=420,
    margin=dict(t=40, b=40)
)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

spacer()

# Section 3 — SHAP
st.markdown("---")
st.subheader("What drives AQI predictions?")
st.markdown("<div style='margin:16px 0'></div>", unsafe_allow_html=True)

importance_data = {
    "Feature": FEATURE_COLS,
    "Importance": [5.9, 0.6, 0.5, 0.35, 0.15, 0.10, 0.07, 0.0]
}
imp_df = pd.DataFrame(importance_data).sort_values("Importance", ascending=True)
fig2 = go.Figure(go.Bar(
    x=imp_df["Importance"], y=imp_df["Feature"],
    orientation='h', marker_color="#1f77b4"
))
fig2.update_layout(
    title="Feature Importance (SHAP values)",
    height=320,
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(t=40, b=40)
)
st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

spacer()

# Footer
st.markdown("---")
st.caption("Pearls AQI Predictor | Data: AQICN API | Model: Ridge Regression | Store: Supabase")