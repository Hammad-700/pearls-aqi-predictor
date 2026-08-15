# Pearls AQI Predictor

I built this project to predict Air Quality Index (AQI) for the next 3 days using real data from the AQICN API. The idea was to build something end-to-end — not just a model in a notebook, but a full pipeline that collects data automatically, retrains itself daily, and shows forecasts on a live website.

**Live app:** https://pearls-aqi-predictor-4hsijrpatnebkrka8uajpj.streamlit.app


## What it actually does

Every hour, a GitHub Actions job fetches live AQI data for Lahore and London and stores it in Supabase. Every day, another job retrains 4 different ML models and picks the best one automatically. The Streamlit dashboard reads from Supabase and shows a 3-day forecast with a color-coded alert banner.

The full flow:

AQICN API → Bronze (raw) → Silver (cleaned) → Gold (features) → Model → Dashboard

All three data layers live in Supabase Postgres. The trained model is saved to Supabase Storage so the dashboard can load it without retraining.

## Tech I used

- **Python 3.14.7** — developed locally, deployed on 3.11 (hosting platforms haven't caught up yet)
- **Scikit-learn, XGBoost, LightGBM** — four models trained and compared every day
- **SHAP** — explains *why* the model made a prediction, not just what it predicted
- **Supabase** — Postgres for data storage, Storage bucket as model registry
- **Flask** — REST API with `/predict`, `/history`, `/health` endpoints
- **Streamlit** — live dashboard deployed on Streamlit Cloud
- **GitHub Actions** — hourly data collection + daily retraining, fully automated

## Data layers (Bronze / Silver / Gold)

I used the medallion architecture to keep data clean and traceable:

- **Bronze** (`aqi_bronze_raw`)     — raw JSON from the API, nothing touched
- **Silver** (`aqi_silver_cleaned`) — nulls handled, negative values dropped, types validated
- **Gold**   (`aqi_gold_features`)  — lag features, rolling stats, time features, and the 3 forecast targets

Every row has a `(city, timestamp)` composite key. This prevents Lahore data from mixing with London data when computing lag features — a bug that's easy to miss and painful to debug later.

## Models and results

Trained all 4 models on the same features with the same train/test split (last 20% of dates — no random shuffle, because time series needs time-based splits).

|      Model       | Avg RMSE |             
|------------------|          |
| Ridge Regression | 8.60     | 
| Random Forest    | 8.94     |             
| LightGBM         | 9.50     |             
| XGBoost          | 10.39    | 

Ridge won — surprising at first, but the backfill data has simplified linear patterns so a regularized linear model fits well. Once 30+ days of real data accumulates, I expect the tree models to take over.

Per-horizon breakdown for Ridge:

|  Day  | RMSE |  R²  |
|-------|------|------|
| Day 1 | 8.68 | 0.39 |
| Day 2 | 8.66 | 0.37 |
| Day 3 | 8.46 | 0.41 |

## Features used

|          Feature      | What it captures |
|-----------------------|-----------------|
| `aqi_lag_1h`          | AQI one hour ago |
| `aqi_lag_24h`         | AQI yesterday same time |
| `aqi_roll_mean_24h`   | Average over last 24 hours |
| `aqi_change_rate`     | How fast AQI is changing |
| `hour`, `day_of_week`, `month` | Time patterns |
| `city_encoded`        | Lahore vs London |

SHAP showed `city_encoded` is by far the strongest feature (5.92) — makes sense since Lahore and London have completely different pollution levels. After that, recent AQI history matters most.


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
python pipelines/feature_pipeline.py london
python pipelines/backfill.py lahore london
python pipelines/training_pipeline.py
python api/app.py
streamlit run dashboard/streamlit_app.py
```

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
    {"day": 1, "date": "2026-08-11", "aqi": 135},
    {"day": 2, "date": "2026-08-12", "aqi": 146},
    {"day": 3, "date": "2026-08-13", "aqi": 133}
  ],
  "max_aqi": 36,
  "alert_level": "Good"
}
```

## Project structure

```
pearls-aqi-predictor/
├── .github/workflows/
│   ├── feature_pipeline.yml   ← runs every hour
│   └── training_pipeline.yml  ← runs daily at 2AM
├── pipelines/
│   ├── feature_pipeline.py
│   ├── training_pipeline.py
│   └── backfill.py
├── features/
│   ├── fetch_aqi.py
│   └── engineer_features.py
├── models/
│   └── explain.py
├── api/
│   └── app.py
├── dashboard/
│   └── streamlit_app.py
├── tests/
│   └── test_supabase.py
├── requirements.txt
└── README.md
```


## Known limitations

- Initial training used synthetic backfill, not real historical data. Accuracy will improve as real data accumulates.
- Only Lahore supported right now.
- Flask isn't publicly deployed — Python 3.14 isn't supported by free hosting platforms yet. Dashboard connects to Supabase directly as a workaround.


## Author

Muhammad Hammad Khalid — Data Science Project