# Pearls AQI Predictor

I built this project to predict Air Quality Index (AQI) for the next 3 days using real data from the AQICN API. The idea was to build something end-to-end not just a model in a notebook, but a full pipeline that collects data automatically, retrains itself daily, and shows forecasts on a live website.

**Live app:** https://pearls-aqi-predictor-4hsijrpatnebkrka8uajpj.streamlit.app

---

## What it actually does

Every hour, a GitHub Actions job fetches live AQI data for Lahore and stores it in Supabase. Every day, another job retrains 5 ML models and picks the best one automatically. The Streamlit dashboard reads from Supabase and shows a 3-day forecast with a color-coded alert banner.

The full flow:

```
AQICN API → Bronze (raw) → Silver (cleaned) → Gold (features) → Model → Dashboard
```

All three data layers live in Supabase Postgres. The trained model is saved to Supabase Storage so the dashboard can load it without retraining.

AQI comes from station **@A471607** (G.O.R., Lahore, Punjab, Pakistan). Temperature and humidity come from Lahore, Pakistan coordinates through Open-Meteo.

Latest verified reading: **AQI 97**, **35.0°C**, **53% humidity** at **2026-08-24 15:00 PKT**.

![AQI system architecture](images/AQI-Architecture.png)

---

## Tech I used

- **Python 3.14.7** - developed locally, deployed on 3.11 (hosting platforms haven't caught up yet)
- **Scikit-learn, XGBoost, LightGBM** - five models trained and compared every day (including naive baseline)
- **Explainability** - SHAP feature importance for model predictions
- **Supabase** - Postgres for data storage, Storage bucket as model registry
- **Flask** - REST API with `/predict`, `/history`, `/health` endpoints (local)
- **Streamlit** - live dashboard deployed on Streamlit Cloud
- **GitHub Actions** - hourly data collection + daily retraining, fully automated

---

## Data layers (Bronze / Silver / Gold)

I used the medallion architecture to keep data clean and traceable:

- **Bronze** (`aqi_bronze_raw`) - fetched AQICN JSON with Lahore weather values attached
- **Silver** (`aqi_silver_cleaned`) - nulls handled, negative values dropped, types validated
- **Gold** (`aqi_gold_features`) - lag features, rolling stats, time features, and the 3 forecast targets

Every row has a `(city, timestamp)` - composite key. This prevents duplicate rows and ensures data integrity across all 3 layers.

---

## Models and results

Trained all 5 models on the same features with a **chronological train/test split** (last 20% of dates — no random shuffle, required for time series).

| Model | Avg RMSE | Notes |
|-------|----------|-------|
| Naive Baseline | 20.31 | predict mean — no skill |
| Random Forest | 11.32 | best new candidate; current champion remains 11.29 |
| LightGBM | 11.88 | |
| Ridge Regression | 12.59 | |
| XGBoost | 11.94 | |

**Random Forest reduces RMSE by about 44% versus the naive baseline** — a useful result on the current evaluation data.

Champion is Random Forest with an average RMSE of 11.29. The current training data has complete weather coverage, but historical AQI labels still include backfilled data and should be replaced with a longer period of real observations before treating these metrics as production accuracy.

Per-horizon breakdown for Random Forest champion:

| Day | RMSE | R² |
|-----|------|----|
| Day 1 | 12.02 | 0.63 |
| Day 2 | 11.01 | 0.69 |
| Day 3 | 10.85 | 0.73 |

Champion gate is in place — model only gets promoted if it beats the existing champion's RMSE.

## Features used

| Feature | What it captures |
|---------|-----------------|
| `aqi_lag_1h` | AQI one hour ago |
| `aqi_lag_24h` | AQI yesterday same time |
| `aqi_roll_mean_24h` | Average over last 24 hours |
| `aqi_change_rate` | How fast AQI is changing |
| `hour`, `day_of_week`, `month` | Time patterns (Pakistan time) |
| `city_encoded` | City identifier |
| `temperature` | Lahore temperature in °C |
| `humidity` | Lahore relative humidity percentage |

LightGBM feature importance shows `aqi_lag_1h` and `aqi_roll_mean_24h` are the strongest predictors - recent AQI history matters most.

---

## Running it locally

```bash
git clone https://github.com/Hammad-700/pearls-aqi-predictor.git
cd pearls-aqi-predictor

py -3.14 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file:
```
AQICN_TOKEN=your_token
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_service_role_key
```

Run the pipeline:
```bash
python pipelines/feature_pipeline.py lahore
python pipelines/backfill.py lahore
python pipelines/training_pipeline.py
python api/app.py
streamlit run dashboard/streamlit_app.py
```

---

## API endpoints

```
GET /health               → model version + status
GET /predict?city=lahore  → 3-day forecast
GET /history?city=lahore  → last 24 AQI readings
```

Sample `/predict` response:
```json
{
  "city": "lahore",
  "forecast": [
    {"day": 1, "date": "2026-08-16", "aqi": 135},
    {"day": 2, "date": "2026-08-17", "aqi": 146},
    {"day": 3, "date": "2026-08-18", "aqi": 133}
  ],
  "max_aqi": 146,
  "alert_level": "Unhealthy for Sensitive Groups"
}
```

---

## Project structure

```
pearls-aqi-predictor/
├── .github/workflows/
│   ├── feature_pipeline.yml   ← runs every hour
│   └── training_pipeline.yml  ← runs daily at midnight PKT
├── pipelines/
│   ├── feature_pipeline.py
│   ├── training_pipeline.py
│   ├── backfill.py
│   ├── alert_pipeline.py
│   └── cleanup.py
├── features/
│   ├── fetch_aqi.py
│   └── engineer_features.py
├── models/
│   └── explain.py
├── api/
│   └── app.py
├── dashboard/
│   └── streamlit_app.py
├── test/
│   ├── test_features.py
│   └── check_stations.py
├── requirements.txt
└── README.md
```

---

## Engineering decisions

- **Chronological split** - time series data requires time-aware train/test split, not random shuffle
- **Champion gate** - new model only promoted if it beats existing champion RMSE
- **Staleness detection** - pipeline warns if station data is older than 6 hours
- **UTC storage and PKT features** - timestamps are stored in UTC; calendar features and display use Pakistan time
- **Composite key (city, timestamp)** - prevents duplicate rows at DB level
- **Naive baseline included** - current champion reduces RMSE by about 44% versus the mean predictor

---

## Known limitations

- Training data uses synthetic backfill for first 30 days - accuracy improves as real hourly data accumulates
- Only Lahore supported (station @A471607) - single station means total data loss if station goes offline
- AQI station updates every 4-6 hours, not every minute
- Flask API implemented and tested locally but not publicly deployed - Python 3.14 not yet supported by free hosting platforms
- Model margins between LightGBM/RF/Ridge are small (~0.84 RMSE) - may be noise with current dataset size

---

## Author

Muhammad Hammad Khalid - Data Science Project