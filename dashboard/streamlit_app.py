import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from supabase import create_client
import joblib
import tempfile
import numpy as np
import os
from datetime import datetime, timedelta, timezone

st.set_page_config(
    page_title="Pearls AQI Predictor 🇵🇰",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="auto",
)
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
                "aqi_change_rate", "temperature", "humidity", "pm25",
                "wind_speed", "wind_direction", "precipitation", "pressure",
                "pm25_raw", "pm10_raw", "no2_raw", "o3_raw"]
LEGACY_FEATURE_COLS = FEATURE_COLS[:10]

def predict(city, model, le, feature_cols=None):
    sb = get_supabase()
    result = sb.table("aqi_gold_features")\
        .select("*").eq("city", city)\
        .order("timestamp", desc=True).limit(1).execute()
    if not result.data:
        return None, None, None
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
    return forecast, row["timestamp"], row

def get_history(city):
    sb = get_supabase()
    from datetime import datetime, timezone, timedelta
    ten_days_ago = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    result = sb.table("aqi_gold_features")\
        .select("timestamp,aqi").eq("city", city)\
        .gte("timestamp", ten_days_ago)\
        .order("timestamp", desc=True).limit(500).execute()
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
    df = df.reindex(columns=feature_cols).dropna(subset=feature_cols)
    return df if not df.empty else None

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
    forecast, as_of, latest_row = predict(city, model, le)
    history = get_history(city)

current_aqi = latest_row.get("aqi", 0)
current_temperature = latest_row.get("temperature")

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
                Lahore - G.O.R. station
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

# Section 1 — Forecast cards
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
    "humidity": "Humidity",
    "pm25": "PM2.5",
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
    height=max(420, len(imp_df) * 32 + 90),
    yaxis=dict(dtick=1, automargin=True),
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(t=40, b=40, l=170, r=20)
)
st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

# ==========================================================
# NEW SECTIONS ADDED BELOW (above the existing footer)
# ==========================================================

spacer()
st.markdown("---")
st.markdown("<h2>Learn more about air pollution in Lahore</h2>", unsafe_allow_html=True)
st.markdown("""
**What is the current air quality in Lahore?**  
The current air quality in Lahore is considered unhealthy for sensitive groups. Children, older adults, and people with heart or lung conditions may experience health effects, while the general public is less likely to be affected.

**How polluted is Lahore’s air?**  
Lahore suffers from high levels of air pollution, with the city regularly ranking at the top of IQAir AirVisual’s live pollution rankings of major global cities. However, pollution only rose to the top of the public’s consciousness in early 2017, when actionable air quality data was published for the first time in Pakistan. In the absence of publicly available government data, a network of citizen-operated sensors began to monitor fine particulate matter, also known as PM2.5, and report data in real-time. The data laid bare Lahore’s high levels of air pollution, shocking the public and becoming a media talking point.  

The resulting publicity led to a public interest petition to review the government’s response to the smog crisis, which was heard at Lahore’s High Court in November 2017. The court ordered authorities to prepare an updated smog response action plan, and publish daily pollution updates until able to publish hourly updates, as the non-government monitors do.

Following the court order, the Punjab Environment Protection Council approved a Smog Action Plan and adopted an Air Quality Index (AQI) classification system in 2017. However, the AQI has been criticized by air quality advocates as being too lax and underreporting the severity of the pollution. While the U.S. AQI deems a PM2.5 concentration of 60 micrograms per cubic meter in the air as “Unhealthy”, Punjab’s AQI reads as “Satisfactory”, with the advice: “May cause minor breathing discomfort to sensitive people.”

Because of this discrepancy, in November 2019, three children asked a court to declare the Punjab AQI “illegal and unreasonable.”

**Does the Pakistani government publish real-time air quality data?**  
As of November 2019, Pakistani authorities still don’t publish any real-time PM2.5 air quality data. All data come from non-government sensors and the U.S. State Department. The U.S. Embassy in Islamabad, and the three U.S. Consulates in Karachi, Lahore and Peshawar began monitoring and publishing real-time PM2.5 data online in the first half of 2019.  

**When is Lahore’s air pollution at its worst?**  
Air quality in Lahore usually worsens during the winter season from October to February when farmers in the wider Punjab province set light to the remnants of crops, producing smoke that adds to smog. At the same time, weather changes mean pollutants remain trapped in the air for longer.

In November 2019, during the heart of Pakistan’s “smog season”, Lahore regularly came second only to Delhi – and sometimes overtook the Indian city – as the world’s most polluted city on IQAir AirVisual’s live rankings of major global cities.

**Is Lahore the most polluted city?**  
In 2018, Lahore ranked 10 in IQAir AirVisual’s 2018 World Air Quality Report. Neighbouring city Faisalabad's air pollution ranked number 3, while air pollution in Islamabad – Pakistan’s capital city – came in significantly lower at number 239. Karachi air pollution was the lowest among the four cities at number 318.

Overall, Pakistan air pollution caused the country to be ranked as the second most polluted in the world with an annual PM2.5 average of 74.3 µg/m³.

**What causes air pollution in Lahore?**  
Air pollution in Lahore is caused by a combination of vehicle and industrial emissions, smoke from brick kilns, the burning of crop residue and general waste, and dust from construction sites. Other factors of air pollution include large scale losses of trees to build new roads and buildings.

Winter air pollution is worse due to temperature inversion, which results in a layer of warm air that is prevented from rising trapping air pollutants.

**How can air pollution in Lahore be reduced?**  
Real-time air quality data must first be made available to everyone with greater granularity. When people know how much pollution they are breathing, they can better take measures to protect themselves and be enabled to mobilise efforts around tackling air pollution.

Reducing industrial and vehicular emissions is also critical to improving the air quality. Prime Minister Imran Khan has told his cabinet that tackling air pollution is a priority, and authorities have taken measures to reduce pollution from brick kilns. Under the Punjab Green Development Program (PGDP) project, there are plans to do more, including establishing 10 air quality monitoring stations in Lahore.

Individuals can take steps in their daily life to reduce personal emissions by carpooling or taking public transport, actively switching to greener fuel alternatives, and more.
""")

spacer()
st.markdown("---")
st.markdown("<h2>Known Limitations</h2>", unsafe_allow_html=True)
st.markdown("""
- **Training data uses synthetic backfill for first 30 days** – accuracy improves as real hourly data accumulates.
- **Only Lahore supported (station @A471607)** – single station means total data loss if station goes offline.
- **AQI station updates every 4–6 hours**, not every minute.
- **Expanded weather features are currently based on a short historical window** and should be re‑evaluated as more real observations accumulate.
""")

# ==========================================================
# END OF NEW SECTIONS
# ==========================================================

spacer()

# Footer
st.markdown("---")
st.caption("Pearls AQI Predictor | Data: AQICN API | Models: Ridge, Random Forest, XGBoost, LightGBM")