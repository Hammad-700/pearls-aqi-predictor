import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from supabase import create_client
import joblib
import tempfile
import numpy as np
import os
import shap
from datetime import datetime, timedelta, timezone
from sklearn.multioutput import MultiOutputRegressor

st.set_page_config(
    page_title="Pearls AQI Predictor 🇵🇰",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="auto",
)

# ------------------------------------------------------------------------------
# Custom CSS (unchanged)
# ------------------------------------------------------------------------------
st.markdown("""
<style>
.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
}
h1 {
    margin-top: 0rem !important;
    margin-bottom: 0.2rem !important;
}
h1 + div {
    margin-top: 0rem !important;
}
h1 + div p {
    margin-top: 0.15rem !important;
    margin-bottom: 0.15rem !important;
}
@media (max-width: 768px) {
    .forecast-card {
        height: 132px !important;
        min-height: 132px !important;
        padding: 16px !important;
        gap: 14px !important;
        grid-template-columns: 86px max-content !important;
        justify-content: center;
    }
    .forecast-aqi-box {
        flex: 0 0 86px !important;
        min-width: 0 !important;
        padding: 12px 10px !important;
    }
    .forecast-aqi-value {
        font-size: 34px !important;
    }
    .forecast-details {
        min-width: 0;
    }
    .forecast-label {
        font-size: 20px !important;
        overflow-wrap: anywhere;
    }
    .forecast-date {
        font-size: 13px !important;
        overflow-wrap: anywhere;
    }
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
.forecast-card {
    width: 100%;
    height: 166px;
    padding: 16px;
    display: grid;
    grid-template-columns: 96px max-content;
    align-items: center;
    justify-content: center;
    gap: 16px;
    box-sizing: border-box;
    border-radius: 14px;
    margin: 0;
}
.forecast-card > * {
    position: relative;
    left: -3px;
}
.forecast-grid {
    width: 100%;
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 16px;
}
.forecast-aqi-box {
    min-width: 0;
    padding: 12px 10px;
    text-align: center;
    box-sizing: border-box;
    border-radius: 10px;
}
.forecast-aqi-value {
    font-size: 44px;
    font-weight: 800;
    line-height: 1;
    white-space: nowrap;
}
.forecast-aqi-caption {
    font-size: 11px;
    margin-top: 4px;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.forecast-details {
    flex: 1 1 auto;
    min-width: 0;
}
.forecast-label {
    font-size: 22px;
    font-weight: 700;
    line-height: 1.15;
}
.forecast-date {
    font-size: 15px;
    margin-top: 6px;
    opacity: 0.8;
    white-space: nowrap;
}
@media (max-width: 768px) {
    .forecast-grid {
        grid-template-columns: 1fr;
    }
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Cached Supabase client
# ------------------------------------------------------------------------------
@st.cache_resource(ttl=300)
def get_supabase():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

# ------------------------------------------------------------------------------
# Model loader (cached)
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
# SHAP explainer loader (with fallback)
# ------------------------------------------------------------------------------
def load_shap_explainer(model, X_background):
    try:
        if isinstance(model, MultiOutputRegressor):
            base_model = model.estimators_[0]
        elif hasattr(model, 'estimators_'):
            base_model = model.estimators_[0]
        else:
            base_model = model

        if hasattr(base_model, 'feature_importances_') or hasattr(base_model, 'get_booster'):
            return shap.TreeExplainer(base_model)
        elif hasattr(base_model, 'coef_'):
            return shap.LinearExplainer(base_model, X_background)
        else:
            return shap.KernelExplainer(model.predict, X_background)
    except Exception as e:
        st.warning(f"SHAP Tree/LinearExplainer failed: {e}. Trying KernelExplainer...")
        try:
            return shap.KernelExplainer(model.predict, X_background)
        except Exception as e2:
            st.error(f"KernelExplainer also failed: {e2}")
            return None

# ------------------------------------------------------------------------------
# Helper: alert label and color
# ------------------------------------------------------------------------------
def get_alert(aqi):
    if aqi <= 50: return "Good", "#00c853"
    elif aqi <= 100: return "Moderate", "#ffd600"
    elif aqi <= 150: return "Sensitive", "#ff6d00"
    elif aqi <= 200: return "Unhealthy", "#d50000"
    elif aqi <= 300: return "Very Unhealthy", "#6a1b9a"
    else: return "Hazardous", "#7e0023"

# ------------------------------------------------------------------------------
# Feature columns (used as fallback)
# ------------------------------------------------------------------------------
FEATURE_COLS = ["city_encoded", "hour", "day_of_week", "month",
                "aqi_lag_1h", "aqi_lag_24h", "aqi_roll_mean_24h",
                "aqi_change_rate", "temperature", "humidity", "pm25",
                "wind_speed", "wind_direction", "precipitation", "pressure",
                "pm25_raw", "pm10_raw", "no2_raw", "o3_raw"]
LEGACY_FEATURE_COLS = FEATURE_COLS[:10]

# ------------------------------------------------------------------------------
# Prediction function (now handles single‑output models gracefully)
# ------------------------------------------------------------------------------
def predict(city, model, le, feature_cols=None):
    sb = get_supabase()
    result = sb.table("aqi_gold_features")\
        .select("*").eq("city", city)\
        .order("timestamp", desc=True).limit(1).execute()
    if not result.data:
        return None, None, None, None
    row = result.data[0]
    if feature_cols is None:
        feature_cols = getattr(model, "feature_names_in_", None)
        if feature_cols is None:
            feature_cols = FEATURE_COLS
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
        "wind_speed": row.get("wind_speed") or 0,
        "wind_direction": row.get("wind_direction") or 0,
        "precipitation": row.get("precipitation") or 0,
        "pressure": row.get("pressure") or 0,
        "pm25_raw": row.get("pm25_raw") or 0,
        "pm10_raw": row.get("pm10_raw") or 0,
        "no2_raw": row.get("no2_raw") or 0,
        "o3_raw": row.get("o3_raw") or 0,
    }
    X = pd.DataFrame([features]).reindex(columns=feature_cols, fill_value=0)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    try:
        preds = model.predict(X)[0]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None, None, None, None

    # ---- Handle single‑output models ----
    if not hasattr(preds, '__len__') or len(preds) == 1:
        # If only one value, repeat it for 3 days (persistence forecast)
        st.warning("⚠️ Model provides only a 1‑day forecast. We will extend it with the same value for days 2 and 3.")
        preds = [preds] * 3 if not hasattr(preds, '__len__') else [preds[0]] * 3
    else:
        # Ensure we have exactly 3 values (take first 3, or pad with last value)
        preds = list(preds[:3])
        while len(preds) < 3:
            preds.append(preds[-1])

    as_of = row["timestamp"]
    base_date = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
    forecast = []
    for i, aqi_val in enumerate(preds[:3]):
        forecast.append({
            "day": i+1,
            "date": (base_date + timedelta(days=i+1)).strftime("%Y-%m-%d"),
            "aqi": int(round(aqi_val))
        })
    return forecast, row["timestamp"], row, X

# ------------------------------------------------------------------------------
# History (cached)
# ------------------------------------------------------------------------------
@st.cache_data(ttl=300)
def get_history(city):
    sb = get_supabase()
    ten_days_ago = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    result = sb.table("aqi_gold_features")\
        .select("timestamp,aqi").eq("city", city)\
        .gte("timestamp", ten_days_ago)\
        .order("timestamp", desc=False).limit(500).execute()
    if not result.data:
        return []
    df = pd.DataFrame(result.data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)
    df = df.set_index("timestamp").resample("3h").mean().reset_index()
    df["aqi"] = df["aqi"].rolling(window=3, min_periods=1, center=True).mean().round(0)
    # Drop any remaining NaNs
    df = df.dropna(subset=["aqi"])
    return df.to_dict("records")

# ------------------------------------------------------------------------------
# SHAP background (cached)
# ------------------------------------------------------------------------------
@st.cache_data(ttl=300)
def get_shap_background(city, _le, feature_cols):
    sb = get_supabase()
    result = sb.table("aqi_gold_features")\
        .select("*").eq("city", city)\
        .order("timestamp", desc=True).limit(200).execute()
    df = pd.DataFrame(result.data)
    if df.empty:
        return None
    df["city_encoded"] = _le.transform(df["city"])
    for col in feature_cols:
        if col == "city_encoded":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float)
    df = df.reindex(columns=feature_cols, fill_value=0)
    return df

# ------------------------------------------------------------------------------
# Spacer helper
# ------------------------------------------------------------------------------
def spacer():
    st.markdown("<div style='margin:30px 0'></div>", unsafe_allow_html=True)

# ==============================================================================
# MAIN APP
# ==============================================================================

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

# Load SHAP background and explainer
background_df = get_shap_background("lahore", le, model_feature_cols)
shap_explainer = None
if background_df is not None and len(background_df) >= 1:
    shap_explainer = load_shap_explainer(model, background_df)

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

# Fetch forecast
with st.spinner(f"Fetching forecast for {city}..."):
    forecast, as_of, latest_row, X = predict(city, model, le, model_feature_cols)
    history = get_history(city)

if forecast is None or len(forecast) == 0:
    st.error("No forecast available. Please ensure the model is trained and Gold data exists.")
    st.stop()

current_aqi = latest_row.get("aqi", 0)
current_temperature = latest_row.get("temperature")

# Staleness check + Pakistan Time
PKT = timezone(timedelta(hours=5))
latest_ts = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
age_hours = (datetime.now(timezone.utc) - latest_ts).total_seconds() / 3600
if age_hours > 3:
    st.warning(f"⚠️ Data is {int(age_hours)} hours old — station may not have updated yet.")

today_pkt = datetime.now(PKT).strftime("%B %d, %Y  •  %I:%M %p PKT")
st.markdown(f"<h2>Forecast as of: {today_pkt}</h2>", unsafe_allow_html=True)

# Current AQI card + Weather
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
                Lahore - 9 Stations
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:15px;color:gray;margin:8px 0 16px 7px">
        Main pollutant: PM2.5 ({latest_row.get('pm25_raw', '—')} µg/m³)
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
            padding:14px 8px;text-align:center;width:110px;min-width:110px;box-sizing:border-box;
            display:flex;flex-direction:column;align-items:center;justify-content:center">
            <div class="temperature-value" style="font-size:40px;font-weight:800;color:#173b2a;line-height:1;white-space:nowrap">
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

# ------------------------------------------------------------------------------
# 3‑Day Forecast Cards
# ------------------------------------------------------------------------------
st.markdown("---")
st.markdown(f"<h2>3-Day Forecast for {city.title()}</h2>", unsafe_allow_html=True)
st.markdown("<div style='margin:16px 0'></div>", unsafe_allow_html=True)

forecast_cards = []
for f in forecast:
    label, color = get_alert(f["aqi"])
    forecast_cards.append(f"""
<div class="forecast-card" style="background-color:{color};color:black">
<div class="forecast-aqi-box" style="background-color:rgba(0,0,0,0.15)">
<div class="forecast-aqi-value">{f['aqi']}</div>
<div class="forecast-aqi-caption">AQI</div>
</div>
<div class="forecast-details">
<div class="forecast-label">{label}</div>
<div class="forecast-date">{f['date']}</div>
</div>
</div>
""")
st.markdown(
    f'<div class="forecast-grid">{"".join(forecast_cards)}</div>',
    unsafe_allow_html=True,
)

spacer()

# ------------------------------------------------------------------------------
# Chart with robust error handling and debug info
# ------------------------------------------------------------------------------
st.markdown("---")
st.markdown("<h2>Forecast + Recent History</h2>", unsafe_allow_html=True)
st.markdown("<div style='margin:16px 0'></div>", unsafe_allow_html=True)

# Debug expander (only visible to you)
# with st.expander("🔍 Debug info (forecast data)"):
#     # st.write("Forecast list:", forecast)
#     st.write("Number of forecast points:", len(forecast))
#     st.write("History rows:", len(history) if history else 0)

try:
    with st.spinner("Loading historical data..."):
        history = get_history(city)

    fig = go.Figure()

    # ---------- Historical line (simplified) ----------
    if history:
        hist_df = pd.DataFrame(history)
        hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"], format="ISO8601", utc=True)
        hist_df["timestamp"] = hist_df["timestamp"].dt.tz_convert("Asia/Karachi")
        hist_df = hist_df.sort_values("timestamp")
        # Drop any rows with NaN aqi (should already be cleaned, but just in case)
        hist_df_clean = hist_df.dropna(subset=['aqi'])

        if not hist_df_clean.empty:
            fig.add_trace(go.Scatter(
                x=hist_df_clean["timestamp"],
                y=hist_df_clean["aqi"],
                name="Historical AQI",
                line=dict(color="#1f77b4", width=2),
                connectgaps=True
            ))

    # ---------- Forecast line ----------
    as_of_dt = pd.to_datetime(as_of, utc=True).tz_convert("Asia/Karachi")

    # Determine last historical point
    if history and 'hist_df_clean' in locals() and not hist_df_clean.empty:
        last_hist_ts = hist_df_clean["timestamp"].iloc[-1]
        last_hist_aqi = hist_df_clean["aqi"].iloc[-1]
    else:
        last_hist_ts = as_of_dt
        last_hist_aqi = int(latest_row.get("aqi", 0))

    forecast_x = [last_hist_ts]
    forecast_y = [last_hist_aqi]

    for f in forecast:
        f_date = pd.Timestamp(f["date"]).tz_localize("Asia/Karachi") + pd.Timedelta(hours=12)
        forecast_x.append(f_date)
        forecast_y.append(f["aqi"])

    fig.add_trace(go.Scatter(
        x=forecast_x,
        y=forecast_y,
        name="Forecast AQI",
        line=dict(color="#db3838", dash="solid", width=3),
        mode="lines+markers",
        marker=dict(size=9)
    ))

    # Reference lines
    fig.add_hline(y=100, line_dash="dot", line_color="gold", annotation_text="Moderate")
    fig.add_hline(y=150, line_dash="dot", line_color="orange", annotation_text="Sensitive")
    fig.add_hline(y=200, line_dash="dot", line_color="red", annotation_text="Unhealthy")

    fig.update_layout(
        title=f"AQI Forecast for {city.title()}",
        xaxis_title="Date",
        yaxis_title="AQI",
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        height=420,
        margin=dict(t=40, b=40)
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

except Exception as e:
    st.error(f"Chart rendering failed: {e}")
    st.exception(e)

spacer()

# ------------------------------------------------------------------------------
# SHAP Explanations
# ------------------------------------------------------------------------------
st.markdown("---")
st.markdown("<h2>Why this prediction?</h2>", unsafe_allow_html=True)
st.markdown("<div style='margin:16px 0'></div>", unsafe_allow_html=True)

try:
    if shap_explainer is None and X is not None:
        shap_explainer = load_shap_explainer(model, X)

    if shap_explainer is not None and X is not None:
        shap_values = shap_explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_vals = shap_values.flatten() if len(shap_values.shape) > 1 else shap_values
        feature_names = model_feature_cols if model_feature_cols else X.columns.tolist()
        shap_df = pd.DataFrame({
            'Feature': feature_names,
            'Contribution': shap_vals
        }).sort_values('Contribution', ascending=False)

        top_inc = shap_df[shap_df['Contribution'] > 0].head(3)
        top_dec = shap_df[shap_df['Contribution'] < 0].tail(3)

        if not top_inc.empty:
            st.markdown(f"**Top increase:** {top_inc.iloc[0]['Feature']} (+{top_inc.iloc[0]['Contribution']:.2f})")
        if not top_dec.empty:
            st.markdown(f"**Top decrease:** {top_dec.iloc[0]['Feature']} ({top_dec.iloc[0]['Contribution']:.2f})")

        colors = ['#ff4b4b' if c > 0 else '#1f77b4' for c in shap_df['Contribution']]
        fig_shap = go.Figure(go.Bar(
            x=shap_df['Contribution'],
            y=shap_df['Feature'],
            orientation='h',
            marker_color=colors,
            text=[f"{c:+.2f}" for c in shap_df['Contribution']],
            textposition='outside'
        ))
        fig_shap.update_layout(
            title="Feature contributions to the 24‑hour forecast",
            xaxis_title="SHAP value (impact on AQI)",
            height=max(400, 30 * len(shap_df)),
            margin=dict(t=40, b=40),
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_shap, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No SHAP explainer available. Showing global feature importance below.")
except Exception as e:
    st.error(f"SHAP computation failed:\n\n`{e}`")
    st.info("Fallback: global feature importance is shown below.")

spacer()

# ------------------------------------------------------------------------------
# Global Feature Importance (fallback)
# ------------------------------------------------------------------------------
st.markdown("---")
st.markdown("<h2>Global Feature Importance</h2>", unsafe_allow_html=True)
st.markdown("<div style='margin:16px 0'></div>", unsafe_allow_html=True)

try:
    if isinstance(model, MultiOutputRegressor):
        base_estimator = model.estimators_[0]
    else:
        base_estimator = model.estimators_[0] if hasattr(model, 'estimators_') else model

    if hasattr(base_estimator, 'feature_importances_'):
        importances = np.mean([e.feature_importances_ for e in model.estimators_], axis=0)
    elif hasattr(base_estimator, 'coef_'):
        importances = np.abs(base_estimator.coef_)
    else:
        importances = [0.0] * len(FEATURE_COLS)
    importances = importances / importances.sum()
except Exception:
    importances = [0.0] * len(FEATURE_COLS)

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
    "humidity": "Humidity",
    "wind_speed": "Wind Speed",
    "wind_direction": "Wind Direction",
    "precipitation": "Precipitation",
    "pressure": "Pressure",
    "pm25_raw": "PM2.5 Raw",
    "pm10_raw": "PM10 Raw",
    "no2_raw": "NO2 Raw",
    "o3_raw": "O3 Raw",
}

importance_data = {
    "Feature": [feature_labels.get(f, f) for f in FEATURE_COLS],
    "Importance": importances
}

imp_df = pd.DataFrame(importance_data).sort_values("Importance", ascending=True)
fig2 = go.Figure(go.Bar(
    x=imp_df["Importance"], y=imp_df["Feature"],
    orientation="h", marker_color="#1f77b4"
))
fig2.update_layout(
    title="Feature Importance (Random Forest built-in)",
    height=400,
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(t=40, b=40)
)
st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

spacer()

# ------------------------------------------------------------------------------
# Limitations & footer
# ------------------------------------------------------------------------------
st.markdown("---")
st.markdown("<h2>Model Limitations</h2>", unsafe_allow_html=True)
st.markdown("""
- **Training data uses synthetic backfill for first 30 days** – accuracy improves as real hourly data accumulates.
- **Only Lahore supported** – single station means total data loss if station goes offline.
- **AQI station updates every 4–6 hours**, not every minute.
- **Expanded weather features are currently based on a short historical window** and should be re‑evaluated as more real observations accumulate.
""")

spacer()

st.markdown("---")
st.caption("Pearls AQI Predictor | Data: AQICN API | Models: Ridge, Random Forest, XGBoost, LightGBM")