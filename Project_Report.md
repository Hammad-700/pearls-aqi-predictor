# Pearls AQI Predictor — Project Report

## 1. Project Overview
**Pearls AQI Predictor** is an end-to-end serverless ML system for forecasting **Lahore AQI for the next 3 days (+24h, +48h, +72h)**. It automates data collection, processing, feature engineering, model training, model selection, and dashboard predictions.

## 2. System Architecture

![AQI Architecture](images/AQI-Architecture.png)

*Figure 1: Pearls AQI Predictor system architecture.*

The system follows a Bronze → Silver → Gold data pipeline, followed by model training, champion-model selection, storage, and prediction through the Flask API and Streamlit dashboard.

```text
AQICN + Open-Meteo
        ↓
Bronze (Raw JSON)
        ↓
Silver (Cleaned Data)
        ↓
Gold (19 Features)
        ↓
Training & Evaluation
        ↓
Champion Gate + Model Registry
        ↓
Supabase Storage
        ↓
Flask API → Streamlit Dashboard

## 3. Technology Choices
| Requirement | Implemented | Reason |
|---|---|---|
| Feature Store | Supabase PostgreSQL + Storage | Simpler and lower operational overhead |
| Orchestration | GitHub Actions | Serverless scheduling and Git integration |
| Deep Learning | XGBoost, LightGBM, Random Forest | Suitable for tabular data; faster and simpler |
| AQI/Weather | AQICN + Open-Meteo | AQI/pollutants plus richer weather features |
| Explainability | SHAP + fallbacks | Model interpretation across model types |
| Dashboard/API | Streamlit + Flask | Lightweight interactive UI and API |

## 4. Data Pipeline
- **Bronze:** raw AQICN/Open-Meteo payloads and station validation.
- **Silver:** cleaning, duplicate removal, null handling, and negative-value filtering.
- **Gold:** feature-ready dataset.
- **Training:** metrics, hyperparameters, timestamps, and model runs.
- **Registry:** model version, metrics, storage URL, and promotion status.

## 5. Feature Engineering — 19 Features
- **Time:** hour, day of week, month
- **Lag:** AQI lag 1h, AQI lag 24h
- **Rolling:** 24h AQI rolling mean
- **Derived:** AQI change rate, city encoding
- **Weather:** temperature, humidity, wind speed/direction, precipitation, pressure
- **Pollutants:** PM2.5, raw PM2.5, PM10, NO2, O3

**Targets:** AQI at +24h, +48h, +72h.

## 6. Model Results
**5-fold TimeSeriesCV**

| Model | Avg. RMSE | Result |
|---|---:|---|
| Naive Baseline | 20.31 | Reference |
| Ridge | 12.65 | Compared |
| Random Forest | **11.34** | **Champion** |
| LightGBM | 11.48 | Compared |
| XGBoost | 11.67 | Compared |

### Champion Performance
| Horizon | RMSE | R² |
|---|---:|---:|
| +24h | 11.93 | 0.64 |
| +48h | 11.10 | 0.69 |
| +72h | 10.97 | 0.72 |

**Important:** Results are preliminary because the initial 30-day history used synthetic backfill. Final scientific validation requires verified real observations.

## 7. Champion Gate
A new model is promoted only when its RMSE improves over the current champion. Otherwise, the existing champion is retained.

## 8. Automation & MLOps
- **Hourly:** AQI/weather collection and Bronze → Silver → Gold pipeline.
- **Daily:** train models, compare metrics, promote champion, update registry.
- **Monitoring:** station staleness and data-quality checks.
- **Alerts:** pipeline failure email notification.
- **Secrets:** stored in GitHub Secrets; no hardcoded tokens.
- **Persistence:** Joblib models with version/schema tracking.

## 9. Web Application
### Streamlit Dashboard
- Current AQI, temperature, humidity
- 3-day forecast
- EPA-style AQI alert categories
- Historical + forecast chart
- Feature importance
- Data-staleness warning
- PKT timezone

### Flask API
- `GET /health` — API/model status
- `GET /predict?city=lahore` — 3-day forecast
- `GET /history?city=lahore` — latest 24 AQI records

## 10. AQI Categories
| AQI | Category |
|---:|---|
| 0–50 | Good |
| 51–100 | Moderate |
| 101–150 | Unhealthy for Sensitive Groups |
| 151–200 | Unhealthy |
| 201–300 | Very Unhealthy |
| 301+ | Hazardous |

## 11. Explainability
- SHAP TreeExplainer/LinearExplainer where supported.
- Feature importance or regression coefficients as fallbacks.
- Strong signals include `aqi_lag_1h` and `aqi_roll_mean_24h`.
- Feature importance indicates model behavior, not causation.

## 12. Key Thresholds
| Check | Threshold |
|---|---:|
| Minimum training rows | 100 |
| Weather coverage | ≥80% |
| Pipeline staleness warning | 6h |
| Dashboard staleness warning | 3h |
| Model promotion | RMSE improvement |

## 13. Challenges & Solutions
- **No historical AQI:** 30-day synthetic backfill; metrics clearly marked preliminary.
- **Weather API failures:** fallback weather data and 80% coverage threshold.
- **Model/schema incompatibility:** version tracking and champion retention.
- **Station outages:** retries and staleness warnings.
- **Layer consistency:** `(city, timestamp)` composite key and quality checks.

## 14. Limitations
- Synthetic 30-day backfill affects current evaluation.
- Single Lahore station/city.
- Some missing numeric values are currently filled with zero.
- No prediction intervals.
- Flask API is local only.
- No drift monitoring yet.

## 15. Future Work
1. Replace synthetic history with verified real observations.
2. Add more Lahore stations and additional cities.
3. Add traffic, satellite, and more pollutant data.
4. Improve missing-value imputation.
5. Add prediction intervals and threshold probabilities.
6. Add data/model drift monitoring.
7. Add API authentication, rate limiting, and integration tests.
8. Deploy Flask API to cloud.

## 16. Ethical & Safety Note
AQI forecasts can influence health decisions. The system is **informational only**, not a medical diagnostic tool. Users should check data freshness, limitations, and treat forecasts as estimates rather than guarantees.

## 17. Deployment Status
| Component | Status |
|---|---|
| Streamlit Dashboard | ✅ Live on Streamlit Cloud |
| Flask API | ✅ Local only |
| Feature Pipeline | ✅ Automated hourly |
| Training Pipeline | ✅ Automated daily |
| Supabase | ✅ Production |
| AQICN | ✅ Live |
| Open-Meteo | ✅ Live |

## 18. Conclusion
Pearls AQI Predictor demonstrates a complete, automated ML pipeline for 3-day Lahore AQI forecasting. The project prioritizes **simplicity, cost-effectiveness, automation, and explainability** using Supabase, GitHub Actions, tree-based models, SHAP, Streamlit, and Flask.

The most important next step is replacing synthetic historical data with verified observations and performing rigorous real-world validation.

**Author:** Muhammad Hammad Khalid  
**Project:** Pearls AQI Predictor  
**Location:** Lahore, Punjab, Pakistan  
**Status:** ✅ Working End-to-End Prototype
