import os
import sys
import time
import random
import math
import requests
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from supabase import create_client
import pandas as pd

# Add project root to path
load_dotenv()
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
TOKEN = os.getenv("AQICN_TOKEN")

if not SUPABASE_URL or not SUPABASE_KEY or not TOKEN:
    raise RuntimeError("Missing Supabase or AQICN_TOKEN in .env")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------- Configuration ----------
CITY = "lahore"
DAYS_BACK = 30
# End date: set to 2026-08-26 23:00 UTC (as requested) or current time if earlier
END_DATE = datetime(2026, 8, 26, 23, 0, 0, tzinfo=timezone.utc)
# If today is earlier than that, use current time (rounded down to hour)
if datetime.now(timezone.utc) < END_DATE:
    END_DATE = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
START_DATE = END_DATE - timedelta(days=DAYS_BACK)

# ---------- Station list (verified Lahore) ----------
LAHORE_STATION_IDS = [
    "-471607",   # G.O.R. (US Consulate)
    "-576565",   # DHA Phase 6
    "-576568",   # Barki Road
    "-576577",   # Egerton Road
    "-576544",   # Shahdara
    "-576559",   # GT Road
    "-576562",   # Kahna Nau
    "-576547",   # Multan Road
    "-576556",   # Punjab University
    "-582631",   # Wagha Border
    "-576550",   # Safari Park
    "-74005",    # Lahore Cantonment
    "-1866349",  # Terapand
    "-538468",   # Naila Road
    "-577588",   # Lathepur
    "11765",     # Lahore US Embassy (older)
]

# ---------- Helper: clean NaN for JSON ----------
def clean_nan_values(obj):
    if isinstance(obj, dict):
        return {k: clean_nan_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan_values(v) for v in obj]
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    else:
        return obj

# ---------- Get freshest AQI ----------
def get_current_aqi():
    """Return the freshest AQI from any Lahore station."""
    best_aqi = None
    best_ts = None
    for sid in LAHORE_STATION_IDS:
        try:
            url = f"https://api.waqi.info/feed/@{sid}/?token={TOKEN}"
            resp = requests.get(url, timeout=10)
            data = resp.json()
            if data.get("status") != "ok":
                continue
            aqi = data["data"].get("aqi")
            if aqi is None or aqi == "-":
                continue
            ts_str = data["data"]["time"].get("iso") or data["data"]["time"].get("s")
            if not ts_str:
                continue
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            if best_ts is None or ts > best_ts:
                best_ts = ts
                best_aqi = int(aqi)
        except Exception:
            continue
    if best_aqi is None:
        raise RuntimeError("Could not fetch current AQI from any station")
    print(f"[INFO] Base AQI: {best_aqi} (from station with timestamp {best_ts})")
    return best_aqi

# ---------- Generate synthetic AQI ----------
def generate_aqi(base_aqi, dt):
    hour = dt.hour
    weekday = dt.weekday()
    if 7 <= hour <= 10:
        time_factor = 1.25
    elif 17 <= hour <= 21:
        time_factor = 1.35
    elif 0 <= hour <= 5:
        time_factor = 0.75
    else:
        time_factor = 1.0
    week_factor = 0.90 if weekday >= 5 else 1.0
    noise = random.uniform(-18, 18)
    aqi = base_aqi * time_factor * week_factor + noise
    return max(15, min(400, int(aqi)))

# ---------- Fetch real weather and pollutants from Open-Meteo ----------
def fetch_weather_bulk(start_date, end_date):
    weather_params = {
        "latitude": 31.5204,
        "longitude": 74.3587,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,precipitation,pressure_msl",
        "timezone": "UTC",
    }
    resp = requests.get("https://archive-api.open-meteo.com/v1/archive", params=weather_params, timeout=60)
    resp.raise_for_status()
    hourly = resp.json().get("hourly", {})

    aq_params = {
        "latitude": 31.5204,
        "longitude": 74.3587,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": "pm2_5,pm10,nitrogen_dioxide,ozone",
        "timezone": "UTC",
    }
    aq_resp = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality", params=aq_params, timeout=60)
    aq_resp.raise_for_status()
    aq_hourly = aq_resp.json().get("hourly", {})

    weather_map = {}
    for i, ts in enumerate(hourly.get("time", [])):
        key = datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%Y-%m-%dT%H:00:00+00:00")
        weather_map[key] = {
            "temperature": float(hourly["temperature_2m"][i]) if hourly["temperature_2m"][i] is not None else None,
            "humidity": float(hourly["relative_humidity_2m"][i]) if hourly["relative_humidity_2m"][i] is not None else None,
            "wind_speed": float(hourly["wind_speed_10m"][i]) if hourly["wind_speed_10m"][i] is not None else None,
            "wind_direction": float(hourly["wind_direction_10m"][i]) if hourly["wind_direction_10m"][i] is not None else None,
            "precipitation": float(hourly["precipitation"][i]) if hourly["precipitation"][i] is not None else None,
            "pressure": float(hourly["pressure_msl"][i]) if hourly["pressure_msl"][i] is not None else None,
            "pm25_raw": float(aq_hourly["pm2_5"][i]) if aq_hourly["pm2_5"][i] is not None else None,
            "pm10_raw": float(aq_hourly["pm10"][i]) if aq_hourly["pm10"][i] is not None else None,
            "no2_raw": float(aq_hourly["nitrogen_dioxide"][i]) if aq_hourly["nitrogen_dioxide"][i] is not None else None,
            "o3_raw": float(aq_hourly["ozone"][i]) if aq_hourly["ozone"][i] is not None else None,
        }
    print(f"[OK] Fetched {len(weather_map)} hours of weather data")
    return weather_map

# ---------- Save Silver rows in batches ----------
def save_silver_rows(rows):
    batch_size = 200
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i+batch_size]
        # Clean NaN
        batch = clean_nan_values(batch)
        try:
            supabase.table("aqi_silver_cleaned").upsert(batch, on_conflict="city,timestamp").execute()
            print(f"  Silver batch {i//batch_size + 1} saved")
        except Exception as e:
            print(f"[ERROR] Silver batch failed: {e}")
        time.sleep(0.1)

# ---------- Build Gold rows from silver ----------
def build_gold_rows(silver_rows):
    df = pd.DataFrame(silver_rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    ts_to_aqi = {row["timestamp"]: row["aqi"] for _, row in df.iterrows()}

    gold_cols = [
        "city", "timestamp", "aqi", "hour", "day_of_week", "month",
        "aqi_lag_1h", "aqi_lag_24h", "aqi_roll_mean_24h", "aqi_change_rate",
        "temperature", "humidity", "pm25", "wind_speed", "wind_direction",
        "precipitation", "pressure", "pm25_raw", "pm10_raw", "no2_raw", "o3_raw",
        "aqi_d1", "aqi_d2", "aqi_d3"
    ]

    gold_rows = []
    for idx in range(1, len(df)):
        row = df.iloc[idx].copy()
        ts = row["timestamp"]
        row["hour"] = ts.hour
        row["day_of_week"] = ts.dayofweek
        row["month"] = ts.month

        # Lags
        row["aqi_lag_1h"] = float(df.iloc[idx-1]["aqi"])
        cutoff_24 = ts - timedelta(hours=24)
        before_24 = df[df["timestamp"] <= cutoff_24]
        row["aqi_lag_24h"] = float(before_24.iloc[-1]["aqi"]) if len(before_24) > 0 else None
        last_24 = df[df["timestamp"] > cutoff_24]["aqi"]
        row["aqi_roll_mean_24h"] = float(last_24.mean()) if len(last_24) > 0 else None
        row["aqi_change_rate"] = float(row["aqi"] - df.iloc[idx-1]["aqi"])

        # Weather fields (already present)
        for col in ["temperature", "humidity", "wind_speed", "wind_direction", "precipitation", "pressure",
                    "pm25_raw", "pm10_raw", "no2_raw", "o3_raw"]:
            if col not in row or pd.isna(row[col]):
                row[col] = None

        # Future targets
        def find_future(hours_ahead):
            target = ts + timedelta(hours=hours_ahead)
            for delta in range(-30, 31, 30):
                check = target + timedelta(minutes=delta)
                if check in ts_to_aqi:
                    return ts_to_aqi[check]
            return None

        d1 = find_future(24)
        d2 = find_future(48)
        d3 = find_future(72)
        if d1 is None or d2 is None or d3 is None:
            continue

        row["aqi_d1"] = float(d1)
        row["aqi_d2"] = float(d2)
        row["aqi_d3"] = float(d3)

        gold_row = {k: row.get(k) for k in gold_cols}
        gold_row["city"] = CITY
        gold_row["timestamp"] = ts.isoformat().replace("+00:00", "Z")
        gold_rows.append(gold_row)

    return gold_rows

# ---------- Save Gold rows in batches ----------
def save_gold_rows(rows):
    batch_size = 200
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i+batch_size]
        # Clean NaN
        batch = clean_nan_values(batch)
        try:
            supabase.table("aqi_gold_features").upsert(batch, on_conflict="city,timestamp").execute()
            print(f"  Gold batch {i//batch_size + 1} saved")
        except Exception as e:
            print(f"[ERROR] Gold batch failed: {e}")
        time.sleep(0.1)

# ---------- Main ----------
def main():
    print(f"\n{'='*60}")
    print(f"Backfill {CITY} from {START_DATE.date()} to {END_DATE.date()}")
    print(f"{'='*60}\n")

    # 1. Get current AQI base
    base_aqi = get_current_aqi()
    random.seed(42)

    # 2. Fetch weather for the period
    weather = fetch_weather_bulk(START_DATE, END_DATE)

    # 3. Generate silver rows
    silver_rows = []
    current = START_DATE
    while current <= END_DATE:
        key = current.strftime("%Y-%m-%dT%H:00:00+00:00")
        w = weather.get(key, {})
        aqi = generate_aqi(base_aqi, current)
        row = {
            "city": CITY,
            "timestamp": key,
            "aqi": aqi,
            "temperature": w.get("temperature"),
            "humidity": w.get("humidity"),
            "wind_speed": w.get("wind_speed"),
            "wind_direction": w.get("wind_direction"),
            "precipitation": w.get("precipitation"),
            "pressure": w.get("pressure"),
            "pm25_raw": w.get("pm25_raw"),
            "pm10_raw": w.get("pm10_raw"),
            "no2_raw": w.get("no2_raw"),
            "o3_raw": w.get("o3_raw"),
            "pm25": None,
            "pm10": None,
            "no2": None,
            "o3": None,
            "co": None,
            "so2": None,
            "station_id": None,
            "station_name": None,
        }
        silver_rows.append(row)
        current += timedelta(hours=1)

    print(f"[INFO] Generated {len(silver_rows)} silver rows")

    # 4. Save silver
    print("[INFO] Saving silver...")
    save_silver_rows(silver_rows)

    # 5. Build gold
    print("[INFO] Building gold features...")
    gold_rows = build_gold_rows(silver_rows)
    print(f"[INFO] Generated {len(gold_rows)} gold rows with targets")

    # 6. Save gold
    if gold_rows:
        print("[INFO] Saving gold...")
        save_gold_rows(gold_rows)
    else:
        print("[WARN] No gold rows generated (not enough data for targets)")

    print("\n[COMPLETE] Backfill finished.")

if __name__ == "__main__":
    main()