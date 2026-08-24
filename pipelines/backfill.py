import os
import sys
import time
import random
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from supabase import create_client
import pandas as pd

load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.engineer_features import clean_to_silver

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
TOKEN = os.getenv("AQICN_TOKEN")

GOLD_COLS = [
    "city", "timestamp", "aqi", "hour", "day_of_week", "month",
    "aqi_lag_1h", "aqi_lag_24h", "aqi_roll_mean_24h",
    "aqi_change_rate", "temperature", "humidity",
    "aqi_d1", "aqi_d2", "aqi_d3"
]

CITY_MAP = {
    "lahore": "@A471607"
}


def fetch_current_aqi(city: str) -> int:
    """Fetch current AQI once (used as base for synthetic history)."""
    station = CITY_MAP.get(city.lower(), city)
    url = f"https://api.waqi.info/feed/{station}/?token={TOKEN}"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if data.get("status") == "ok":
            aqi = data["data"].get("aqi", 80)
            return 80 if aqi == "-" else int(aqi)
    except Exception as e:
        print(f"[WARN] Could not fetch live AQI: {e}")
    return 80  # fallback


def generate_realistic_aqi(base_aqi: int, ts: datetime) -> int:
    """
    Generate more realistic AQI:
    - Higher during day / evening
    - Slight weekly pattern
    - Random noise
    """
    hour = ts.hour
    weekday = ts.weekday()  # 0=Mon ... 6=Sun

    # Time-of-day factor (higher pollution in morning & evening)
    if 7 <= hour <= 10:          # morning peak
        time_factor = 1.25
    elif 17 <= hour <= 21:        # evening peak
        time_factor = 1.35
    elif 0 <= hour <= 5:          # night
        time_factor = 0.75
    else:
        time_factor = 1.0

    # Weekend slightly cleaner
    week_factor = 0.90 if weekday >= 5 else 1.0

    noise = random.uniform(-18, 18)
    aqi = base_aqi * time_factor * week_factor + noise
    return max(15, min(400, int(aqi)))


def build_gold_from_rows(rows: list, idx: int) -> dict | None:
    """Build gold features for row at index idx."""
    if idx < 1:
        return None

    df = pd.DataFrame(rows[:idx + 1])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    latest = df.iloc[-1].copy()
    ts = latest["timestamp"]

    latest["hour"] = ts.hour
    latest["day_of_week"] = ts.dayofweek
    latest["month"] = ts.month

    latest["aqi_lag_1h"] = float(df.iloc[-2]["aqi"]) if len(df) >= 2 else None

    cutoff_24 = ts - pd.Timedelta(hours=24)
    before_24 = df[df["timestamp"] <= cutoff_24]
    latest["aqi_lag_24h"] = float(before_24.iloc[-1]["aqi"]) if len(before_24) > 0 else None

    last_24 = df[df["timestamp"] > cutoff_24]["aqi"]
    latest["aqi_roll_mean_24h"] = float(last_24.mean()) if len(last_24) > 0 else None

    prev_aqi = df.iloc[-2]["aqi"]
    latest["aqi_change_rate"] = float(latest["aqi"] - prev_aqi) if prev_aqi is not None else None

    latest["temperature"] = float(df.iloc[-1].get("temperature")) if pd.notna(df.iloc[-1].get("temperature")) else None
    latest["humidity"] = float(df.iloc[-1].get("humidity")) if pd.notna(df.iloc[-1].get("humidity")) else None

    latest["aqi_d1"] = None
    latest["aqi_d2"] = None
    latest["aqi_d3"] = None

    gold = latest.to_dict()
    gold["timestamp"] = ts.isoformat().replace("+00:00", "Z")
    return {k: v for k, v in gold.items() if k in GOLD_COLS}


def run_backfill(city: str, days: int = 30):
    print(f"\n{'='*60}")
    print(f"Backfill → {city.upper()} | Last {days} days → up to 15 Aug 2026")
    print(f"{'='*60}")

    now = datetime.now(timezone.utc)
    # Force end date to 15 August 2026 (or current time if earlier)
    end_date = datetime(2026, 8, 15, 23, 0, 0, tzinfo=timezone.utc)
    if now < end_date:
        end_date = now.replace(minute=0, second=0, microsecond=0)

    start_date = end_date - timedelta(days=days)

    print(f"Date range : {start_date.date()} → {end_date.date()}")
    print(f"Total hours: {(end_date - start_date).total_seconds() / 3600:.0f}")

    # 1. Get current real AQI as base
    base_aqi = fetch_current_aqi(city)
    print(f"Base AQI   : {base_aqi}")

    random.seed(55)  # reproducible

    # 2. Generate synthetic silver rows
    silver_rows = []
    current = start_date

    while current <= end_date:
        aqi_val = generate_realistic_aqi(base_aqi, current)

        silver_rows.append({
            "city": city,
            "timestamp": current.isoformat().replace("+00:00", "Z"),
            "aqi": aqi_val,
            "pm25": None,
            "pm10": None,
            "no2": None,
            "o3": None,
            "co": None,
            "so2": None,
            "temperature": round(random.uniform(20, 35), 1),
            "humidity": round(random.uniform(40, 80), 1),
        })
        current += timedelta(hours=1)

    print(f"\n[INFO] Generated {len(silver_rows)} silver rows")

    # 3. Save Silver in batches
    print("[INFO] Saving Silver rows...")
    batch_size = 150
    for i in range(0, len(silver_rows), batch_size):
        batch = silver_rows[i:i + batch_size]
        try:
            supabase.table("aqi_silver_cleaned").upsert(
                batch, on_conflict="city,timestamp"
            ).execute()
            print(f"   → Silver batch {i // batch_size + 1} saved")
        except Exception as e:
            print(f"[ERROR] Silver batch failed: {e}")
        time.sleep(0.25)

    print("[OK] Silver saved")

    # 4. Build Gold + targets
    print("\n[INFO] Building Gold features + targets (d1/d2/d3)...")
    gold_saved = 0
    gold_skipped = 0

    # Pre-build timestamp → aqi map for fast target lookup
    ts_to_aqi = {
        datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00")): r["aqi"]
        for r in silver_rows
    }

    for idx in range(1, len(silver_rows)):
        gold_row = build_gold_from_rows(silver_rows, idx)
        if not gold_row:
            gold_skipped += 1
            continue

        ts = datetime.fromisoformat(gold_row["timestamp"].replace("Z", "+00:00"))

        # Find targets (within ±30 min)
        def find_future_aqi(hours_ahead: int):
            target = ts + timedelta(hours=hours_ahead)
            for delta in range(-30, 31, 30):  # check -30, 0, +30 minutes
                check = target + timedelta(minutes=delta)
                if check in ts_to_aqi:
                    return ts_to_aqi[check]
            return None

        d1 = find_future_aqi(24)
        d2 = find_future_aqi(48)
        d3 = find_future_aqi(72)

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
            print(f"[ERROR] Gold save failed (idx={idx}): {e}")

        if gold_saved % 100 == 0 and gold_saved > 0:
            print(f"   → Gold saved so far: {gold_saved}")

        time.sleep(0.05)

    print(f"\n{'='*60}")
    print(f"[DONE] {city.upper()}")
    print(f"   Gold saved   : {gold_saved}")
    print(f"   Gold skipped : {gold_skipped}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import requests  # needed for fetch

    cities = sys.argv[1:] if len(sys.argv) > 1 else ["lahore"]
    days = 30

    print(f"Starting backfill up to 15 August 2026")
    print(f"Cities: {cities} | Days: {days}")

    for city in cities:
        run_backfill(city, days=days)