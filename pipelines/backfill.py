import os
import sys
import time
import json
import requests
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.engineer_features import clean_to_silver

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
TOKEN = os.getenv("AQICN_TOKEN")

GOLD_COLS = ["city", "timestamp", "aqi", "hour", "day_of_week",
             "month", "aqi_lag_1h", "aqi_lag_24h", "aqi_roll_mean_24h",
             "aqi_change_rate", "aqi_d1", "aqi_d2", "aqi_d3"]

CITY_MAP = {
    "lahore": "@A471607"
}

def fetch_with_backoff(city: str, retries=3, delay=5) -> dict:
    station = CITY_MAP.get(city.lower(), city)
    url = f"https://api.waqi.info/feed/{station}/?token={TOKEN}"
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data["status"] == "ok":
                return data["data"]
            elif "rate" in str(data).lower():
                print(f"[RATE LIMIT] Waiting {delay*2}s...")
                time.sleep(delay * 2)
            else:
                print(f"[WARN] Bad status for {city}: {data.get('status')}")
                return None
        except Exception as e:
            print(f"[ERROR] Attempt {attempt+1} failed: {e}")
            time.sleep(delay)
    return None

def build_gold_from_rows(rows: list, idx: int) -> dict:
    """Build gold features for row at idx using previous rows for lags."""
    if idx < 1:
        return None

    import pandas as pd
    df = pd.DataFrame(rows[:idx+1])
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    df = df.sort_values("timestamp").reset_index(drop=True)

    latest = df.iloc[-1].copy()
    ts = latest["timestamp"]

    latest["hour"] = ts.hour
    latest["day_of_week"] = ts.dayofweek
    latest["month"] = ts.month
    latest["aqi_lag_1h"] = float(df.iloc[-2]["aqi"]) if len(df) >= 2 else None
    before_24 = df[df["timestamp"] <= ts - pd.Timedelta(hours=24)]
    latest["aqi_lag_24h"] = float(before_24.iloc[-1]["aqi"]) if len(before_24) > 0 else None
    last_24 = df[df["timestamp"] > ts - pd.Timedelta(hours=24)]["aqi"]
    latest["aqi_roll_mean_24h"] = float(last_24.mean()) if len(last_24) > 0 else None
    prev_aqi = df.iloc[-2]["aqi"]
    latest["aqi_change_rate"] = float(latest["aqi"] - prev_aqi) if prev_aqi else None
    latest["aqi_d1"] = None
    latest["aqi_d2"] = None
    latest["aqi_d3"] = None

    gold = latest.to_dict()
    gold["timestamp"] = ts.isoformat()
    return {k: v for k, v in gold.items() if k in GOLD_COLS}

def run_backfill(city: str, days: int = 30):
    print(f"\n=== Backfill: {city} for {days} days ===")
    
    # AQICN only provides current data via free API
    # We simulate hourly rows using current AQI with time offsets
    # (Real historical data needs AQICN premium — we build synthetic history)
    
    now = datetime.now(timezone.utc)
    silver_rows = []

    # Fetch current data once
    print(f"[INFO] Fetching current data for {city}...")
    raw = fetch_with_backoff(city)
    if not raw:
        print(f"[ERROR] Could not fetch data for {city}")
        return

    current_aqi = raw.get("aqi", 50)
    if current_aqi == "-":
        current_aqi = 50

    print(f"[INFO] Current AQI: {current_aqi} — generating {days*24} hourly rows")

    import random
    random.seed(55)

    # Generate synthetic hourly rows going back `days` days
    for hour_offset in range(days * 24, 0, -1):
        ts = now - timedelta(hours=hour_offset)
        
        # Add realistic variation (±20% + time-of-day pattern)
        hour = ts.hour
        time_factor = 1.0 + 0.2 * abs(hour - 12) / 12
        noise = random.uniform(-15, 15)
        aqi_val = max(1, int(current_aqi * time_factor + noise))

        silver_row = {
            "city": city,
            "timestamp": ts.isoformat(),
            "aqi": aqi_val,
            "pm25": None, "pm10": None,
            "no2": None, "o3": None,
            "co": None, "so2": None
        }
        silver_rows.append(silver_row)

    # Save silver rows
    print(f"[INFO] Saving {len(silver_rows)} silver rows...")
    batch_size = 100
    for i in range(0, len(silver_rows), batch_size):
        batch = silver_rows[i:i+batch_size]
        try:
            supabase.table("aqi_silver_cleaned").upsert(
                batch, on_conflict="city,timestamp"
            ).execute()
        except Exception as e:
            print(f"[ERROR] Silver batch {i} failed: {e}")
        time.sleep(0.2)  # avoid rate limits

    print(f"[OK] Silver rows saved")

    # Build and save gold rows with targets
    print(f"[INFO] Building Gold rows with targets...")
    gold_saved = 0
    gold_skipped = 0

    for idx in range(1, len(silver_rows)):
        gold_row = build_gold_from_rows(silver_rows, idx)
        if not gold_row:
            gold_skipped += 1
            continue

        # Assign targets: look ahead 24h, 48h, 72h
        ts = datetime.fromisoformat(gold_row["timestamp"])
        
        def find_aqi_at(target_ts):
            for r in silver_rows:
                r_ts = datetime.fromisoformat(r["timestamp"])
                if abs((r_ts - target_ts).total_seconds()) < 1800:  # within 30min
                    return r["aqi"]
            return None

        d1 = find_aqi_at(ts + timedelta(hours=24))
        d2 = find_aqi_at(ts + timedelta(hours=48))
        d3 = find_aqi_at(ts + timedelta(hours=72))

        if d1 is None or d2 is None or d3 is None:
            gold_skipped += 1
            continue

        gold_row["aqi_d1"] = float(d1)
        gold_row["aqi_d2"] = float(d2)
        gold_row["aqi_d3"] = float(d3)

        try:
            supabase.table("aqi_gold_features").upsert(
                gold_row, on_conflict="city,timestamp"
            ).execute()
            gold_saved += 1
        except Exception as e:
            print(f"[ERROR] Gold save failed at idx {idx}: {e}")

        if gold_saved % 100 == 0:
            print(f"[INFO] Gold saved: {gold_saved} rows...")
            time.sleep(0.1)

    print(f"\n[OK] Backfill complete for {city}")
    print(f"     Gold saved: {gold_saved}, skipped: {gold_skipped}")

if __name__ == "__main__":
    cities = sys.argv[1:] if len(sys.argv) > 1 else ["lahore"]
    days = 30
    print(f"⚠️  This will generate {days*24} rows per city")
    print(f"Cities: {cities}")
    for city in cities:
        run_backfill(city, days=days)