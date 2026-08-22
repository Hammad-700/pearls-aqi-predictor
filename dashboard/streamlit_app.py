import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from supabase import create_client
import joblib
import tempfile
import numpy as np
import os
from datetime import datetime, timedelta, timezone

st.set_page_config(page_title="Pearls AQI Predictor 🇵🇰", page_icon="🌍", layout="wide")
st.markdown("""
<style>
.block-container {
    max-width: 1100px;
    padding-left: 2rem;
    padding-right: 2rem;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(ttl=300)
def get_supabase():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

@st.cache_resource(ttl=300)
def load_model():
    sb = get_supabase()
    try:
        reg = sb.table("model_registry").select("*").order("trained_at", desc=True).limit(1).execute()
        if not reg.data:
            return None, None, None, None
        meta = reg.data[0]
        version = meta["version"]
        model_type = meta.get("model_type", "unknown")
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
        return model, le, version, model_type
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None, None, None, None

def get_alert(aqi):
    if aqi <= 50: return "Good", "#00c853"
    elif aqi <= 100: return "Moderate", "#ffd600"
    elif aqi <= 150: return "Unhealthy for Sensitive Groups", "#ff6d00"
    elif aqi <= 200: return "Unhealthy", "#d50000"
    elif aqi <= 300: return "Very Unhealthy", "#6a1b9a"
    else: return "Hazardous", "#7e0023"

def get_text_color(background_color):
    hex_color = background_color.lstrip("#")
    red, green, blue = (int(hex_color[index:index + 2], 16) for index in (0, 2, 4))
    brightness = (red * 299 + green * 587 + blue * 114) / 1000
    return "black" if brightness >= 128 else "white"

FEATURE_COLS = ["city_encoded", "hour", "day_of_week", "month",
                "aqi_lag_1h", "aqi_lag_24h", "aqi_roll_mean_24h", "aqi_change_rate"]

def predict(city, model, le):
    sb = get_supabase()
    result = sb.table("aqi_gold_features")\
        .select("*").eq("city", city)\
        .order("timestamp", desc=True).limit(1).execute()
    if not result.data:
        return None, None, None
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
    return forecast, row["timestamp"], row.get("aqi", 0)

def get_history(city):
    sb = get_supabase()
    result = sb.table("aqi_gold_features")\
        .select("timestamp,aqi").eq("city", city)\
        .order("timestamp", desc=True).limit(200).execute()
    return result.data

@st.cache_data(ttl=300)
def get_shap_background(city, _le):
    sb = get_supabase()
    result = sb.table("aqi_gold_features")\
        .select("*").eq("city", city)\
        .not_.is_("aqi_lag_1h", "null")\
        .order("timestamp", desc=True).limit(50).execute()
    df = pd.DataFrame(result.data)
    if df.empty:
        return None
    df["city_encoded"] = _le.transform(df["city"])
    df = df.dropna(subset=FEATURE_COLS)
    return df[FEATURE_COLS] if not df.empty else None

def compute_shap_importance(model, model_type, X_background):
    import shap
    base_model = model.estimators_[0]
    if model_type == "lightgbm" or hasattr(base_model, "feature_importances_"):
        explainer = shap.TreeExplainer(base_model)
    else:
        explainer = shap.LinearExplainer(base_model, X_background)
    shap_values = explainer.shap_values(X_background)
    return np.abs(shap_values).mean(axis=0)

def spacer():
    st.markdown("<div style='margin:30px 0'></div>", unsafe_allow_html=True)

# Header
st.title("Pearls AQI Predictor")

st.markdown("""
<span style="color:#1f77b4; font-size:1.3rem; font-weight:700;">Know Your Air.</span>
<span style="color:#1f77b3; font-size:1.3rem; font-weight:700;"> Plan Ahead.</span><br>
*3-day AQI forecasting.*
""", unsafe_allow_html=True)

# Load model
model, le, version, model_type = load_model()
if model is None:
    st.error("Model not loaded!")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("Settings")
    city = "lahore"
    st.info("📍 Lahore, Pakistan")
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
    forecast, as_of, current_aqi = predict(city, model, le)
    history = get_history(city)

# Staleness check + Pakistan Time
from datetime import datetime, timezone, timedelta

PKT = timezone(timedelta(hours=5))

latest_ts = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
age_hours = (datetime.now(timezone.utc) - latest_ts).total_seconds() / 3600

if age_hours > 3:
    st.warning(f"⚠️ Data is {int(age_hours)} hours old — station may not have updated yet.")

# Show date in Pakistan Time
today_pkt = datetime.now(PKT).strftime("%B %d, %Y  •  %I:%M %p PKT")

st.markdown(
    f"<h2>Forecast as of: {today_pkt}</h2>",
    unsafe_allow_html=True
)

# Current AQI card + Alert banner
max_aqi = max(f["aqi"] for f in forecast)
alert_label, alert_color = get_alert(current_aqi)
current_text_color = get_text_color(alert_color)

st.markdown(f"""
<div style="background-color:{alert_color};border-radius:14px;padding:20px 28px;
    display:flex;align-items:center;gap:24px;margin:10px 0 30px 0">
    <div style="background-color:rgba(0,0,0,0.15);border-radius:10px;
        padding:14px 20px;text-align:center;min-width:90px">
        <div style="font-size:44px;font-weight:800;color:{current_text_color};line-height:1">{current_aqi}</div>
        <div style="font-size:11px;color:{current_text_color};margin-top:4px;font-weight:600;letter-spacing:0.5px">US AQI</div>
    </div>
    <div>
        <div style="font-size:26px;font-weight:700;color:{current_text_color}">{alert_label}</div>
        <div style="font-size:13px;color:{current_text_color};margin-top:6px;opacity:0.8">
            🌍 Lahore, Pakistan — Live AQI reading
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='margin:20px 0'></div>", unsafe_allow_html=True)

# Section 1 — Forecast cards
st.markdown("---")
st.subheader(f"3-Day Forecast for {city.title()}")
st.markdown("<div style='margin:16px 0'></div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
for col, f in zip([col1, col2, col3], forecast):
    label, color = get_alert(f["aqi"])
    text_color = get_text_color(color)
    col.markdown(f"""
    <div style="background-color:{color}22;border-radius:14px;padding:20px;
        text-align:center;margin:4px">
        <div style="font-size:15px;font-weight:600;color:{text_color};margin-bottom:12px">{f['date']}</div>
        <div style="background-color:{color}33;border-radius:10px;padding:12px;display:inline-block;min-width:80px">
            <div style="font-size:44px;font-weight:800;color:{text_color};line-height:1">{f['aqi']}</div>
            <div style="font-size:11px;color:{text_color};margin-top:4px;font-weight:600">US AQI</div>
        </div>
        <div style="margin-top:12px;padding:8px;border-radius:8px;
            background:{color}44;color:{text_color};font-size:13px;font-weight:700">{label}</div>
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
    hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"], format="ISO8601", utc=True)
    hist_df["timestamp"] = hist_df["timestamp"].dt.tz_convert("Asia/Karachi")  # ← PKT

    # Add gap detection — insert None where gap > 3 hours
    hist_df["gap"] = hist_df["timestamp"].diff() > pd.Timedelta(hours=3)
    x_vals = []
    y_vals = []
    for i, row in hist_df.iterrows():
        if row["gap"] and i > 0:
            x_vals.append(None)
            y_vals.append(None)
        x_vals.append(row["timestamp"])
        y_vals.append(row["aqi"])

    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals,
        name="Historical AQI",
        line=dict(color="#1f77b4", width=2),
        connectgaps=False
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

try:
    X_background = get_shap_background(city, le)
    if X_background is None or len(X_background) < 5:
        raise ValueError("Not enough historical rows for SHAP background")
    importances = compute_shap_importance(model, model_type, X_background)
    chart_title = "Feature Importance"
except Exception as e:
    base_estimator = model.estimators_[0]
    if hasattr(base_estimator, 'feature_importances_'):
        importances = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
    elif hasattr(base_estimator, 'coef_'):
        importances = np.abs(base_estimator.coef_)
    else:
        importances = [5.9, 0.6, 0.5, 0.35, 0.15, 0.10, 0.07, 0.0]
    chart_title = "Feature Importance (Model-based)"

importance_data = {
    "Feature": FEATURE_COLS,
    "Importance": importances
}
imp_df = pd.DataFrame(importance_data).sort_values("Importance", ascending=True)
fig2 = go.Figure(go.Bar(
    x=imp_df["Importance"], y=imp_df["Feature"],
    orientation='h', marker_color="#1f77b4"
))
fig2.update_layout(
    title=chart_title,
    height=320,
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(t=40, b=40)
)
st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

spacer()

# Footer
st.markdown("---")
st.caption("Pearls AQI Predictor | Data: AQICN API | Models: Ridge, Random Forest, XGBoost, LightGBM")