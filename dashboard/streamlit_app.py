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
/* Main page spacing */
.block-container {
    max-width: 1100px;
    padding-top: 2rem;          /* ← more breathing room under the header */
    padding-bottom: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* Reduce space below main title */
h1 {
    margin-top: 0rem !important;
    margin-bottom: 0.2rem !important;
}

/* Reduce spacing around the header markdown */
h1 + div {
    margin-top: 0rem !important;
}

/* Reduce paragraph spacing inside header */
h1 + div p {
    margin-top: 0.15rem !important;
    margin-bottom: 0.15rem !important;
}

@media (max-width: 768px) {
    .temperature-card {
        flex-direction: row;
        align-items: center !important;
        gap: 12px !important;
        padding: 16px 20px !important;
    }

    .temperature-value-box {
        flex: 0 0 110px;
        min-width: 0 !important;
        padding: 14px 20px !important;
    }

    .temperature-value {
        font-size: 36px !important;
    }

    .temperature-details {
        flex: 1;
        min-width: 0;
    }

    .temperature-details-title {
        font-size: 20px !important;
    }

    .temperature-details-location {
        overflow-wrap: anywhere;
    }
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
        feature_cols = meta.get("feature_columns")
        if not feature_cols:
            feature_cols = getattr(model, "feature_names_in_", None)
        if not feature_cols and getattr(model, "estimators_", None):
            feature_cols = getattr(model.estimators_[0], "feature_names_in_", None)
        feature_cols = list(feature_cols) if feature_cols is not None else LEGACY_FEATURE_COLS
        return model, le, version, model_type, feature_cols
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None, None, None, None, None

def get_alert(aqi):
    if aqi <= 50: return "Good", "#00c853"
    elif aqi <= 100: return "Moderate", "#ffd600"
    elif aqi <= 150: return "Sensitive", "#ff6d00"
    elif aqi <= 200: return "Unhealthy", "#d50000"
    elif aqi <= 300: return "Very Unhealthy", "#6a1b9a"
    else: return "Hazardous", "#7e0023"

FEATURE_COLS = ["city_encoded", "hour", "day_of_week", "month",
                "aqi_lag_1h", "aqi_lag_24h", "aqi_roll_mean_24h",
                "aqi_change_rate", "temperature", "humidity", "pm25"]
LEGACY_FEATURE_COLS = FEATURE_COLS[:-1]

def predict(city, model, le, feature_cols):
    sb = get_supabase()
    result = sb.table("aqi_gold_features")\
        .select("*").eq("city", city)\
        .order("timestamp", desc=True).limit(1).execute()
    if not result.data:
        return None, None, None, None
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
        "temperature": float(row.get("temperature")) if row.get("temperature") is not None else 0.0,
        "humidity": float(row.get("humidity")) if row.get("humidity") is not None else 0.0,
        "pm25": row.get("pm25") or 0,
    }
    X = pd.DataFrame([features])[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
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
    return forecast, row["timestamp"], row.get("aqi", 0), row.get("temperature")

def get_history(city):
    sb = get_supabase()
    result = sb.table("aqi_gold_features")\
        .select("timestamp,aqi").eq("city", city)\
        .order("timestamp", desc=True).limit(200).execute()
    return result.data

@st.cache_data(ttl=300)
def get_shap_background(city, _le, feature_cols):
    sb = get_supabase()
    result = sb.table("aqi_gold_features")\
        .select("*").eq("city", city)\
        .not_.is_("aqi_lag_1h", "null")\
        .order("timestamp", desc=True).limit(50).execute()
    df = pd.DataFrame(result.data)
    if df.empty:
        return None
    df["city_encoded"] = _le.transform(df["city"])
    for col in feature_cols:
        if col == "city_encoded":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float)
    df = df.dropna(subset=feature_cols)
    return df[feature_cols] if not df.empty else None

def compute_shap_importance(model, model_type, X_background):
    import shap
    horizon_importances = []
    for base_model in model.estimators_:
        if hasattr(base_model, "feature_importances_"):
            explainer = shap.TreeExplainer(base_model)
        else:
            explainer = shap.LinearExplainer(base_model, X_background)
        shap_values = explainer.shap_values(X_background)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        horizon_importances.append(np.abs(shap_values).mean(axis=0))
    return np.mean(horizon_importances, axis=0)

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
model, le, version, model_type, model_feature_cols = load_model()
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
    forecast, as_of, current_aqi, current_temperature = predict(city, model, le, model_feature_cols)
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

aqi_col, weather_col = st.columns(2)
with aqi_col:
    st.markdown(f"""
    <div style="background-color:{alert_color};border-radius:14px;padding:20px 28px;
        display:flex;align-items:center;gap:24px;height:166px;box-sizing:border-box;margin-bottom:16px">
        <div style="background-color:rgba(0,0,0,0.15);border-radius:10px;
            padding:14px 20px;text-align:center;min-width:110px">
            <div style="font-size:44px;font-weight:800;color:black;line-height:1;white-space:nowrap">{current_aqi}</div>
            <div style="font-size:11px;color:black;margin-top:4px;font-weight:600;letter-spacing:0.5px">AQI</div>
        </div>
        <div>
            <div style="font-size:26px;font-weight:700;color:black">{alert_label}</div>
            <div style="font-size:15px;color:black;margin-top:6px;opacity:0.8">
                Lahore - Current AQI
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with weather_col:
    if current_temperature is not None:
        temperature_value = float(current_temperature)
        temperature_number = f"{temperature_value:.0f}"
        temperature_text = f"{temperature_number}°C"
    else:
        temperature_text = "Unavailable"
    st.markdown(f"""
    <div class="temperature-card" style="background:linear-gradient(135deg,#e8f5e9,#c8e6c9);border-radius:14px;
        padding:20px 28px;display:flex;align-items:center;gap:24px;height:166px;
        box-sizing:border-box;color:#173b2a;margin-bottom:16px">
        <div class="temperature-value-box" style="background-color:rgba(0,0,0,0.08);border-radius:10px;
            padding:14px 20px;text-align:center;min-width:110px">
            <div class="temperature-value" style="font-size:44px;font-weight:800;color:#173b2a;line-height:1;white-space:nowrap">
                {temperature_text}
            </div>
            <div style="font-size:11px;color:#173b2a;margin-top:4px;font-weight:600;letter-spacing:0.5px">Temperature</div>
        </div>
        <div class="temperature-details">
            <div class="temperature-details-title" style="font-size:26px;font-weight:700;color:#173b2a">Weather</div>
            <div class="temperature-details-location" style="font-size:15px;color:#173b2a;margin-top:6px;opacity:0.8">
                Lahore - Current Weather
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin:20px 0'></div>", unsafe_allow_html=True)

# Section 1 — Forecast cards
st.markdown("---")
st.markdown(f"<h2>3-Day Forecast for {city.title()}</h2>", unsafe_allow_html=True)
st.markdown("<div style='margin:16px 0'></div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
for col, f in zip([col1, col2, col3], forecast):
    label, color = get_alert(f["aqi"])
    col.markdown(f"""
    <div style="background-color:{color};border-radius:14px;padding:20px 28px;
        display:flex;align-items:center;gap:24px;height:166px;box-sizing:border-box;margin:4px">
        <div style="background-color:rgba(0,0,0,0.15);border-radius:10px;
            padding:14px 20px;text-align:center;min-width:110px">
            <div style="font-size:44px;font-weight:800;color:black;line-height:1;white-space:nowrap">{f['aqi']}</div>
            <div style="font-size:11px;color:black;margin-top:4px;font-weight:600;letter-spacing:0.5px">AQI</div>
        </div>
        <div style="color:black">
            <div style="font-size:26px;font-weight:700">{label}</div>
            <div style="font-size:15px;margin-top:6px;opacity:0.8">
             {f['date']}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

spacer()

# Section 2 — Chart
st.markdown("---")
st.markdown("<h2>Forecast + Recent History</h2>", unsafe_allow_html=True)
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
st.markdown("<h2>What drives AQI predictions?</h2>", unsafe_allow_html=True)
st.markdown("<div style='margin:16px 0'></div>", unsafe_allow_html=True)

try:
    X_background = get_shap_background(city, le, model_feature_cols)
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

feature_labels = {
    "city_encoded": "City",
    "hour": "Hour of Day",
    "day_of_week": "Day of Week",
    "month": "Month",
    "aqi_lag_1h": "AQI 1 Hour Ago",
    "aqi_lag_24h": "AQI 24 Hours Ago",
    "aqi_roll_mean_24h": "24h Rolling Average",
    "aqi_change_rate": "AQI Change Rate",
    "temperature": "Temperature",
    "humidity": "Humidity"
}
importance_data = {
    "Feature": [feature_labels.get(f, f) for f in model_feature_cols],
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