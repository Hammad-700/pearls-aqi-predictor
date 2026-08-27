# Pearls AQI Predictor
**Three-Day Air Quality Index (AQI) Forecasting System for Lahore, Pakistan**

## Executive Summary
Pearls AQI Predictor is an end-to-end machine-learning system that forecasts
Lahore's AQI for the next three days. It collects AQI observations, enriches
them with weather data, validates and transforms them through Bronze, Silver,
and Gold data layers, trains several regression models, selects a champion,
and publishes forecasts through a Streamlit dashboard and Flask API.
> **Important:** The initial historical backfill contains synthetic AQI values
> used to bootstrap the pipeline. Current metrics are preliminary and must not
> be treated as final production accuracy.

![AQI system architecture](images/AQI-Architecture.png)

## System Architecture

```text
AQICN API + Open-Meteo
          |
          v
Bronze raw data -> Silver cleaned data -> Gold features and targets
                                              |
                                              v
                         Model training -> Registry and Storage
                                              |
                              Streamlit dashboard / Flask API
```

### Data Layers

| Layer | Storage | Purpose |
|---|---|---|
| Bronze | `aqi_bronze_raw` | Original AQICN response with weather payload attached |
| Silver | `aqi_silver_cleaned` | Cleaned and validated AQI, pollutant, and weather values |
| Gold | `aqi_gold_features` | Time, lag, rolling, weather features, and future targets |
| Training | `training_runs` | Model metrics and run metadata |
| Registry | `model_registry` | Champion model metadata and version |
| Artifacts | Supabase Storage | Serialized models and encoders |
Records use `(city, timestamp)` as a logical composite key, which prevents
duplicates and keeps all three layers aligned.

## Data Sources

### AQICN
AQICN supplies the main AQI, station, and pollutant readings. The default
station is `@A471607` (G.O.R., Lahore, Punjab, Pakistan). The station is
validated against Lahore before data is accepted.

### Open-Meteo
Weather is requested for Lahore using Asia/Karachi local time. The pipeline
uses temperature at two meters, relative humidity, wind speed, wind direction,
hourly precipitation, and mean sea-level pressure. It also stores raw PM2.5,
PM10, NO2, and O3 values where available. If a weather request fails, legacy
AQICN temperature and humidity values can still be used when present.

## Feature Engineering
The model currently uses 19 input features:

| Feature | Purpose |
|---|---|
| `city_encoded` | City context |
| `hour`, `day_of_week`, `month` | Daily, weekly, and seasonal patterns |
| `aqi_lag_1h` | Most recent AQI behavior |
| `aqi_lag_24h` | Previous-day AQI at the comparable time |
| `aqi_roll_mean_24h` | Recent 24-hour AQI average |
| `aqi_change_rate` | AQI direction and rate of change |
| `temperature`, `humidity` | Temperature and relative humidity |
| `pm25` | Fine particulate matter |
| `wind_speed`, `wind_direction` | Wind conditions |
| `precipitation` | Hourly precipitation |
| `pressure` | Mean sea-level pressure |
| `pm25_raw`, `pm10_raw` | Raw particulate concentrations |
| `no2_raw`, `o3_raw` | Raw nitrogen dioxide and ozone concentrations |
Calendar features and display use Pakistan local time; timestamps are stored
in UTC.

## Machine Learning
The target is supervised multi-output regression for:

```text
AQI +24 hours, AQI +48 hours, AQI +72 hours
```

The training pipeline compares:
1. Naive mean baseline
2. Ridge Regression
3. Random Forest
4. XGBoost
5. LightGBM
6. HistGradientBoosting when LightGBM is unavailable
Data is sorted chronologically. The first 80% is used for training and the
last 20% for testing; random shuffling is avoided because this is a time-series
problem. Five-fold `TimeSeriesSplit` cross-validation is also performed.
Evaluation metrics are RMSE, MAE, and R-squared. The primary selection score is the
average RMSE across the three horizons.

### Reported Results

| Model | Average RMSE | Status |
|---|---:|---|
| Naive Baseline | 20.31 | Reference mean predictor |
| Random Forest | 11.34 | Current champion |
| LightGBM | 11.48 | Compared model |
| XGBoost | 11.67 | Compared model |
| Ridge Regression | 12.65 | Compared model |

Random Forest has approximately 44% lower RMSE than the naive baseline in the
reported comparison.

| Forecast horizon | RMSE | R-squared |
|---|---:|---:|
| Day 1 | 11.93 | 0.64 |
| Day 2 | 11.10 | 0.69 |
| Day 3 | 10.97 | 0.72 |

These scores are preliminary because the historical labels include synthetic
backfill. They should be reassessed after verified real observations replace
the bootstrap data.

## Champion Model and Artifacts
A new model is promoted when its average RMSE improves on the current
champion. A required feature-schema migration also promotes a compatible model
so deployed artifacts match the feature code; otherwise the current champion
is retained.
Promotion performs the following steps:
1. Create a model version.
2. Serialize the model with Joblib.
3. Save the label encoder.
4. Upload artifacts to Supabase Storage.
5. Store metrics and metadata in `model_registry`.
The dashboard loads the latest available model and feature data from Supabase.

## Automation and Data Quality
GitHub Actions runs the main workflow:

| Schedule | Job | Work |
|---|---|---|
| Hourly | Feature pipeline | Fetch and process new Lahore AQI and weather data |
| Daily | Training pipeline | Compare models and evaluate champion promotion |

Automation includes staleness checks, secret management, failure alerts, and
champion protection. Data-quality checks cover AQI validity, negative sensor
values, station coordinates, source freshness, duplicate records, weather
coverage, minimum training data, and model availability.
Current operational thresholds:

| Check | Threshold |
|---|---:|
| Minimum training rows | 100 |
| Complete-weather coverage | 80% minimum |
| Stale station warning | 6 hours |
| Dashboard staleness warning | 3 hours |

## Streamlit Dashboard
The dashboard presents:
- Current AQI, temperature, and humidity
- Three-day forecast and forecast dates
- AQI alert category and threshold lines
- Recent AQI history and a historical-plus-forecast chart
- Feature importance
- Data-freshness warning
- Model version and Pakistan local time

## Flask REST API
The local Flask service provides:

| Endpoint | Response |
|---|---|
| `GET /health` | API and model status |
| `GET /predict?city=lahore` | Three forecasts, maximum AQI, and alert level |
| `GET /history?city=lahore` | Latest 24 AQI records |

Example prediction response:

```json
{
  "city": "lahore",
  "as_of": "2026-08-24T10:00:00+00:00",
  "forecast": [
    {"day": 1, "date": "2026-08-25", "aqi": 135},
    {"day": 2, "date": "2026-08-26", "aqi": 146},
    {"day": 3, "date": "2026-08-27", "aqi": 133}
  ],
  "max_aqi": 146,
  "alert_level": "Unhealthy for Sensitive Groups"
}
```

## AQI Categories

| AQI | Category |
|---:|---|
| 0-50 | Good |
| 51-100 | Moderate |
| 101-150 | Unhealthy for Sensitive Groups |
| 151-200 | Unhealthy |
| 201-300 | Very Unhealthy |
| 301+ | Hazardous |

## Explainability
SHAP is used where supported: `TreeExplainer` for tree models and
`LinearExplainer` for linear models. Fallbacks are tree feature importance and
absolute regression coefficients. Feature importance describes model behavior;
it does not prove causation. Recent AQI history, particularly `aqi_lag_1h` and
`aqi_roll_mean_24h`, is currently among the strongest signal sources.

## Project Structure

```text
pearls-aqi-predictor/
|-- api/app.py                         Flask REST API
|-- dashboard/streamlit_app.py         Streamlit dashboard
|-- features/
|   |-- fetch_aqi.py                    AQICN collection
|   |-- fetch_weather.py                Open-Meteo collection
|   `-- engineer_features.py            Feature creation
|-- pipelines/
|   |-- feature_pipeline.py             Hourly processing
|   |-- backfill.py                     Historical AQI backfill
|   |-- backfill_weather.py             Weather backfill
|   |-- training_pipeline.py            Model training
|   `-- cleanup.py                      Maintenance
|-- models/explain.py                   Explainability
|-- test/                               Feature and weather checks
|-- requirements.txt
`-- supabase_weather_migration.sql
```

## Installation and Configuration

```powershell
git clone https://github.com/Hammad-700/pearls-aqi-predictor.git
Set-Location pearls-aqi-predictor
py -3.14 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create `.env` for Flask and pipelines:

```text
AQICN_TOKEN=your_token
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_service_role_key
LAHORE_AQI_STATION_ID=@A471607
```

For Streamlit Cloud, provide `SUPABASE_URL` and `SUPABASE_KEY` through
Streamlit secrets. Never commit tokens, service-role keys, or other secrets.
Before the first weather-enabled run, execute
`supabase_weather_migration.sql` in the Supabase SQL Editor.

## Running the Project

```powershell
# Collect current data and build features
python pipelines/feature_pipeline.py lahore

# Optional historical weather enrichment
python pipelines/backfill_weather.py

# Build and evaluate models
python pipelines/training_pipeline.py

# Start the local API
python api/app.py

# Start the dashboard in another terminal
streamlit run dashboard/streamlit_app.py
```

Run tests and a dashboard compile check:

```powershell
pytest
python -m py_compile dashboard/streamlit_app.py
```

## Limitations and Risks
1. Initial training data uses synthetic AQI values for the first 30 days.
2. Only Lahore and one configured station are currently supported.
3. Station readings may arrive every four to six hours, and an outage can
   remove all current observations.
4. Expanded weather features use a short historical window and need
   re-evaluation as real data accumulates.
5. Missing numeric values are currently filled with zero.
6. The Flask API is tested locally but is not publicly deployed.
7. Predictions are point estimates without uncertainty intervals.

## Future Work
- Replace synthetic history with verified historical AQI observations.
- Add multiple Lahore stations and support for additional cities.
- Add richer pollutant, traffic, satellite, and weather features.
- Improve missing-value handling and model reproducibility metadata.
- Add prediction intervals and threshold-crossing probabilities.
- Add data/model drift monitoring and missing-station alerts.
- Add API authentication, rate limiting, and integration tests.

## Ethical and Safety Note
AQI forecasts can affect health-related decisions. This is an informational
forecasting system, not a medical diagnostic tool. The application should show
the data timestamp, freshness, model limitations, and uncertainty when
available. Users should not treat a prediction as a guarantee of future air
quality.

## Technology Stack and Status

| Technology | Purpose |
|---|---|
| Python, Pandas, NumPy | Application and data processing |
| scikit-learn, XGBoost, LightGBM | Regression and model comparison |
| SHAP | Explainability |
| Supabase/PostgreSQL and Storage | Data and model artifacts |
| AQICN and Open-Meteo | AQI and weather data |
| Flask and Streamlit | API and dashboard |
| GitHub Actions, Git, GitHub | Automation and version control |

**Status:** Working end-to-end prototype and portfolio project. The pipeline
is implemented from collection through forecasting and presentation. Replacing
synthetic backfill with verified historical AQI observations is the most
important next step for stronger scientific evaluation.

**Author:** Muhammad Hammad Khalid  
**Project:** Pearls AQI Predictor  
**Location:** Lahore, Punjab, Pakistan
