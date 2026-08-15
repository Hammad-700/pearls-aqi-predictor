import pandas as pd

def clean_to_silver(bronze_row: dict) -> dict:
    raw = bronze_row["raw_data"]
    city = bronze_row["city"]
    timestamp = bronze_row["timestamp"]
    try:
        aqi = raw.get("aqi", None)
        if aqi == "-" or aqi is None:
            print(f"[WARN] No AQI for {city}, skipping")
            return None
        aqi = int(aqi)
        iaqi = raw.get("iaqi", {})
        def safe_get(key):
            val = iaqi.get(key, {}).get("v", None)
            if val is not None and float(val) < 0:
                return None
            return float(val) if val is not None else None
        return {
            "city": city,
            "timestamp": timestamp,
            "aqi": aqi,
            "pm25": safe_get("pm25"),
            "pm10": safe_get("pm10"),
            "no2": safe_get("no2"),
            "o3": safe_get("o3"),
            "co": safe_get("co"),
            "so2": safe_get("so2"),
        }
    except Exception as e:
        print(f"[ERROR] Silver cleaning failed: {e}")
        return None


def build_gold_features(silver_rows: list) -> dict:
    if len(silver_rows) < 2:
        print("[WARN] Not enough rows for lag features")
        return None
    df = pd.DataFrame(silver_rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    latest = df.iloc[-1].copy()
    ts = latest["timestamp"]
    ts_utc = ts.tz_localize('UTC') if ts.tzinfo is None else ts.tz_convert('UTC')
    latest["hour"] = ts_utc.hour
    latest["day_of_week"] = ts_utc.dayofweek
    latest["month"] = ts_utc.month
    latest["aqi_lag_1h"] = df.iloc[-2]["aqi"] if len(df) >= 2 else None
    before_24 = df[df["timestamp"] <= ts - pd.Timedelta(hours=24)]
    latest["aqi_lag_24h"] = float(before_24.iloc[-1]["aqi"]) if len(before_24) > 0 else None
    last_24 = df[df["timestamp"] > ts - pd.Timedelta(hours=24)]["aqi"]
    latest["aqi_roll_mean_24h"] = float(last_24.mean()) if len(last_24) > 0 else None
    prev_aqi = df.iloc[-2]["aqi"]
    latest["aqi_change_rate"] = float(latest["aqi"] - prev_aqi) if prev_aqi else None
    latest["aqi_d1"] = None
    latest["aqi_d2"] = None
    latest["aqi_d3"] = None
    gold_row = latest.to_dict()
    gold_row["timestamp"] = ts.isoformat()
    return gold_row