import os
import sys
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'features'))

from fetch_aqi import fetch_aqi
from fetch_weather import fetch_weather
from engineer_features import engineer_features

def predict(city="Lahore"):
    print(f"Generating 72h AQI forecast for {city}...")

    # Load model artifacts
    model        = pickle.load(open("models/best_model.pkl", "rb"))
    feature_cols = pickle.load(open("models/feature_cols.pkl", "rb"))

    # Load historical data for lag/rolling features
    df = pd.read_csv("data/historical_data.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed",utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Get latest real values for lag features
    latest = df.iloc[-1]

    predictions = []
    now = datetime.utcnow()

    for hour in range(1, 73):
        future_time = now + timedelta(hours=hour)

        # Build feature row
        row = {
            "aqi":                  latest["aqi"],
            "pm25":                 latest.get("pm25", 0) or 0,
            "pm10":                 latest.get("pm10", 0) or 0,
            "o3":                   latest.get("o3", 0) or 0,
            "no2":                  latest.get("no2", 0) or 0,
            "so2":                  latest.get("so2", 0) or 0,
            "co":                   latest.get("co", 0) or 0,
            "temperature":          latest.get("temperature", 25),
            "humidity":             latest.get("humidity", 50),
            "pressure":             latest.get("pressure", 1010),
            "wind_speed":           latest.get("wind_speed", 5),
            "hour":                 future_time.hour,
            "day":                  future_time.day,
            "month":                future_time.month,
            "day_of_week":          future_time.weekday(),
            "is_weekend":           int(future_time.weekday() >= 5),
            "aqi_lag_1":            latest.get("aqi_lag_1", latest["aqi"]),
            "aqi_lag_3":            latest.get("aqi_lag_3", latest["aqi"]),
            "aqi_lag_6":            latest.get("aqi_lag_6", latest["aqi"]),
            "aqi_lag_24":           latest.get("aqi_lag_24", latest["aqi"]),
            "aqi_rolling_mean_6":   latest.get("aqi_rolling_mean_6", latest["aqi"]),
            "aqi_rolling_std_6":    latest.get("aqi_rolling_std_6", 0),
            "aqi_rolling_mean_24":  latest.get("aqi_rolling_mean_24", latest["aqi"]),
            "aqi_change_rate":      latest.get("aqi_change_rate", 0),
        }

        # Keep only available feature cols
        X = pd.DataFrame([{k: row[k] for k in feature_cols if k in row}])
        pred = model.predict(X)[0]
        pred = max(0, round(pred, 1))

        predictions.append({
            "datetime": future_time.strftime("%Y-%m-%d %H:%M"),
            "predicted_aqi": pred,
            "day": f"Day {(hour-1)//24 + 1}"
        })

    result = pd.DataFrame(predictions)

    # Save predictions
    os.makedirs("data", exist_ok=True)
    result.to_csv("data/predictions.csv", index=False)
    print("✅ Predictions saved to data/predictions.csv")
    print(result.groupby("day")["predicted_aqi"].mean().round(1))
    return result

if __name__ == "__main__":
    predict("Lahore")