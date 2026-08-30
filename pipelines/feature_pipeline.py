import os
import sys
import math
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from features.engineer_features import clean_to_silver, build_gold_features
from features.fetch_aqi import fetch_aqi

PKT = timezone(timedelta(hours=5))
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_KEY are required in .env")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

GOLD_COLS = [
    "city", "station_id", "station_name", "timestamp", "aqi",
    "hour", "day_of_week", "month", "aqi_lag_1h", "aqi_lag_24h",
    "aqi_roll_mean_24h", "aqi_change_rate", "temperature", "humidity",
    "pm25", "wind_speed", "wind_direction", "precipitation", "pressure",
    "pm25_raw", "pm10_raw", "no2_raw", "o3_raw", "aqi_d1", "aqi_d2", "aqi_d3",
]


# ---------- Timestamp rounding ----------
def round_to_hour(dt: datetime) -> datetime:
    """Round down to the start of the hour."""
    return dt.replace(minute=0, second=0, microsecond=0)

def normalize_and_round_timestamp(value: str) -> str:
    """Parse ISO timestamp, convert to UTC, and round down to hour."""
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt_utc = dt.astimezone(timezone.utc)
    dt_rounded = round_to_hour(dt_utc)
    return dt_rounded.isoformat()


# ---------- Helper: clean NaN ----------
def clean_nan_values(obj):
    if isinstance(obj, dict):
        return {k: clean_nan_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan_values(v) for v in obj]
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    else:
        return obj


# ---------- Utility functions ----------
def first_value(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None

def to_float(value: Any) -> Optional[float]:
    try:
        return None if value is None or value == "" else float(value)
    except (TypeError, ValueError):
        return None

def extract_weather(bronze: Dict[str, Any]) -> Dict[str, Any]:
    weather = bronze.get("weather") or {}
    current = weather.get("current") if isinstance(weather, dict) else {}
    main = weather.get("main", {}) if isinstance(weather, dict) else {}
    wind = weather.get("wind", {}) if isinstance(weather, dict) else {}
    rain = weather.get("rain", {}) if isinstance(weather, dict) else {}
    if not isinstance(current, dict):
        current = {}

    return {
        "temperature": to_float(first_value(
            current.get("temperature"), current.get("temp"), current.get("temp_c"),
            main.get("temperature"), main.get("temp"),
            weather.get("temperature") if isinstance(weather, dict) else None,
            weather.get("temp") if isinstance(weather, dict) else None,
        )),
        "humidity": to_float(first_value(
            current.get("humidity"), current.get("relative_humidity"),
            main.get("humidity"), weather.get("humidity") if isinstance(weather, dict) else None,
        )),
        "wind_speed": to_float(first_value(
            current.get("wind_speed"), current.get("windspeed"),
            wind.get("speed"), weather.get("wind_speed") if isinstance(weather, dict) else None,
        )),
        "wind_direction": to_float(first_value(
            current.get("wind_direction"), current.get("wind_dir"),
            wind.get("deg"), wind.get("direction"),
        )),
        "precipitation": to_float(first_value(
            current.get("precipitation"), current.get("precip"),
            current.get("precipitation_mm"), rain.get("1h"), rain.get("3h"),
        )),
        "pressure": to_float(first_value(
            current.get("pressure"), current.get("surface_pressure"),
            main.get("pressure"), weather.get("pressure") if isinstance(weather, dict) else None,
        )),
    }

def extract_station(bronze: Dict[str, Any], silver: Dict[str, Any]) -> Dict[str, Any]:
    raw = bronze.get("raw_data") or {}
    data = raw.get("data", {}) if isinstance(raw, dict) else {}
    if not isinstance(data, dict):
        data = {}
    return {
        "station_id": first_value(
            silver.get("station_id"), bronze.get("station_id"),
            raw.get("station_id") if isinstance(raw, dict) else None,
            data.get("station_id"), data.get("idx"),
        ),
        "station_name": first_value(
            silver.get("station_name"), bronze.get("station_name"),
            raw.get("station_name") if isinstance(raw, dict) else None,
            data.get("station_name"),
        ),
    }


# ---------- Bronze ----------
def save_bronze(bronze: Dict[str, Any]) -> None:
    rounded_ts = normalize_and_round_timestamp(bronze["timestamp"])
    base = {
        "city": bronze["city"],
        "timestamp": rounded_ts,                     # rounded to hour
        "raw_data": bronze.get("raw_data", {}),
        "original_timestamp": bronze["timestamp"],   # exact API time (optional)
    }
    with_weather = {**base, "weather": bronze.get("weather") or {}}
    try:
        supabase.table("aqi_bronze_raw").upsert(
            with_weather, on_conflict="city,timestamp"
        ).execute()
        print("[OK] Bronze saved with weather")
    except Exception as exc:
        msg = str(exc).lower()
        if "weather" not in msg or "column" not in msg:
            raise RuntimeError(f"Bronze save failed: {exc}") from exc
        print("[WARN] Bronze has no weather column; retrying without it")
        supabase.table("aqi_bronze_raw").upsert(
            base, on_conflict="city,timestamp"
        ).execute()


# ---------- Silver ----------
def save_silver(bronze: Dict[str, Any]) -> Dict[str, Any]:
    silver = clean_to_silver(bronze)
    if not silver:
        raise RuntimeError("clean_to_silver() returned no row")
    silver = dict(silver)

    # ★ Always update weather from bronze (overwrite)
    weather = extract_weather(bronze)
    for key, value in weather.items():
        if value is not None:
            silver[key] = value

    station = extract_station(bronze, silver)
    for key, value in station.items():
        if silver.get(key) is None and value is not None:
            silver[key] = value

    silver["city"] = bronze["city"]
    silver["timestamp"] = normalize_and_round_timestamp(bronze["timestamp"])

    # Default station_id to avoid NULL in conflict key
    if silver.get("station_id") is None:
        silver["station_id"] = "lahore_avg"
        silver["station_name"] = "Lahore (averaged)"

    supabase.table("aqi_silver_cleaned").upsert(
        silver, on_conflict="city,timestamp"
    ).execute()

    print(
        f"[OK] Silver saved | AQI={silver.get('aqi')} | "
        f"temp={silver.get('temperature')} | humidity={silver.get('humidity')} | "
        f"PM2.5={silver.get('pm25_raw')}"
    )
    return silver


def get_recent_silver(city: str, limit: int = 30):
    result = (
        supabase.table("aqi_silver_cleaned")
        .select("*")
        .eq("city", city)
        .order("timestamp", desc=True)
        .limit(limit)
        .execute()
    )
    return list(reversed(result.data or []))


def partial_gold(silver: Dict[str, Any]) -> Dict[str, Any]:
    ts = datetime.fromisoformat(silver["timestamp"].replace("Z", "+00:00")).astimezone(PKT)
    return {
        "city": silver["city"],
        "station_id": silver.get("station_id"),
        "station_name": silver.get("station_name"),
        "timestamp": silver["timestamp"],
        "aqi": silver.get("aqi"),
        "hour": ts.hour, "day_of_week": ts.weekday(), "month": ts.month,
        "aqi_lag_1h": None, "aqi_lag_24h": None,
        "aqi_roll_mean_24h": None, "aqi_change_rate": None,
        "temperature": silver.get("temperature"), "humidity": silver.get("humidity"),
        "pm25": silver.get("pm25"), "wind_speed": silver.get("wind_speed"),
        "wind_direction": silver.get("wind_direction"),
        "precipitation": silver.get("precipitation"), "pressure": silver.get("pressure"),
        "pm25_raw": silver.get("pm25_raw"), "pm10_raw": silver.get("pm10_raw"),
        "no2_raw": silver.get("no2_raw"), "o3_raw": silver.get("o3_raw"),
        "aqi_d1": None, "aqi_d2": None, "aqi_d3": None,
    }


# ---------- Gold ----------
def save_gold(silver: Dict[str, Any], history: list) -> Dict[str, Any]:
    current_ts = silver["timestamp"]   # already rounded

    history = [r for r in history if r.get("timestamp") != current_ts]
    history = history + [silver]

    gold = build_gold_features(history)

    if not gold:
        if silver.get("aqi") is None:
            raise RuntimeError("No AQI available; cannot save Gold")
        print("[WARN] Not enough history; using partial Gold row")
        gold = partial_gold(silver)
    else:
        gold = dict(gold)

    gold["city"] = silver["city"]
    gold["timestamp"] = silver["timestamp"]   # rounded
    gold["aqi"] = silver.get("aqi")

    station_id = silver.get("station_id") or "lahore_avg"
    station_name = silver.get("station_name") or "Lahore (averaged)"
    gold["station_id"] = None if (isinstance(station_id, float) and math.isnan(station_id)) else station_id
    gold["station_name"] = None if (isinstance(station_name, float) and math.isnan(station_name)) else station_name

    for key in [
        "temperature", "humidity", "pm25",
        "wind_speed", "wind_direction", "precipitation", "pressure",
        "pm25_raw", "pm10_raw", "no2_raw", "o3_raw",
    ]:
        val = gold.get(key)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            silver_val = silver.get(key)
            gold[key] = None if (isinstance(silver_val, float) and math.isnan(silver_val)) else silver_val

    payload = {k: v for k, v in gold.items() if k in GOLD_COLS}
    payload = clean_nan_values(payload)

    print(f"[DEBUG] Cleaned Gold payload (aqi={payload.get('aqi')}, lag1={payload.get('aqi_lag_1h')})")

    # ★ FIX: use only (city, timestamp) for conflict
    result = (
        supabase.table("aqi_gold_features")
        .upsert(payload, on_conflict="city,timestamp")
        .execute()
    )
    print(f"[OK] Gold saved ({len(result.data) if result.data else 'unknown'} row(s))")
    return payload


# ---------- Enhanced fill_past_targets with debugging ----------
def fill_past_targets(city: str):
    """Fill aqi_d1/d2/d3 for Gold rows older than 72h that still have NULL targets."""
    print(f"\n[INFO] Filling past targets for {city}...")
    
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=72)).isoformat()
    print(f"[DEBUG] Cutoff timestamp: {cutoff}")

    # 1. Check how many NULL rows exist (older than 72h)
    count_result = supabase.table("aqi_gold_features")\
        .select("id", count="exact")\
        .eq("city", city)\
        .is_("aqi_d1", "null")\
        .lt("timestamp", cutoff)\
        .execute()
    
    total_null_count = count_result.count
    print(f"[DEBUG] Total Gold rows with NULL targets older than 72h: {total_null_count}")

    if total_null_count == 0:
        print("[WARN] No NULL targets found to fill. Either data is too fresh, or already filled.")
        return

    # 2. Fetch the actual rows (first 200)
    gold_null = supabase.table("aqi_gold_features")\
        .select("id,city,timestamp,aqi")\
        .eq("city", city)\
        .is_("aqi_d1", "null")\
        .lt("timestamp", cutoff)\
        .order("timestamp", desc=False)\
        .limit(200)\
        .execute()

    print(f"[DEBUG] Processing {len(gold_null.data)} rows (out of {total_null_count})")

    # 3. Load Silver data
    silver = supabase.table("aqi_silver_cleaned")\
        .select("timestamp,aqi")\
        .eq("city", city)\
        .order("timestamp", desc=False)\
        .execute()

    if not silver.data:
        print("[ERROR] No Silver data found in DB!")
        return

    silver_df = pd.DataFrame(silver.data)
    # FIX: Use format='ISO8601' to handle fractional seconds and timezone offsets
    silver_df["timestamp"] = pd.to_datetime(silver_df["timestamp"], utc=True, format='ISO8601')
    silver_df = silver_df.set_index("timestamp")
    print(f"[DEBUG] Loaded {len(silver_df)} Silver rows. Date range: {silver_df.index.min()} to {silver_df.index.max()}")

    filled = 0
    skipped = 0

    for idx, row in enumerate(gold_null.data):
        # FIX: Use format='ISO8601' for this conversion as well
        ts = pd.to_datetime(row["timestamp"], utc=True, format='ISO8601')
        print(f"\n[ROW {idx+1}] Base timestamp: {ts}")

        # Define target times
        target_d1 = ts + pd.Timedelta(hours=24)
        target_d2 = ts + pd.Timedelta(hours=48)
        target_d3 = ts + pd.Timedelta(hours=72)
        print(f"  Looking for D1: {target_d1}, D2: {target_d2}, D3: {target_d3}")

        def find_aqi_at(target_ts, window_hours=3):
            # Use a wider window (3 hours) just in case of timezone rounding issues
            start = target_ts - pd.Timedelta(hours=window_hours)
            end = target_ts + pd.Timedelta(hours=window_hours)
            
            # Check if we even have data in this range
            mask = (silver_df.index >= start) & (silver_df.index <= end)
            subset = silver_df[mask]
            
            if subset.empty:
                # Check nearby exact hour without window
                exact_mask = (silver_df.index == target_ts)
                exact_subset = silver_df[exact_mask]
                if not exact_subset.empty:
                    return float(exact_subset["aqi"].mean())
                return None
            
            return float(subset["aqi"].mean())

        d1 = find_aqi_at(target_d1)
        d2 = find_aqi_at(target_d2)
        d3 = find_aqi_at(target_d3)

        print(f"  Found D1: {d1}, D2: {d2}, D3: {d3}")

        if d1 is not None and d2 is not None and d3 is not None:
            # Update the row
            supabase.table("aqi_gold_features").update({
                "aqi_d1": d1, "aqi_d2": d2, "aqi_d3": d3
            }).eq("id", row["id"]).execute()
            filled += 1
            print(f"  ✅ UPDATED row {row['id']}")
        else:
            skipped += 1
            print(f"  ❌ SKIPPED row {row['id']} (missing future AQI data)")

    print(f"\n[OK] Filled targets for {filled} rows. Skipped {skipped} rows due to missing Silver data.")


# ---------- Pipeline runner ----------
def run_pipeline(city: str) -> None:
    city = city.strip().lower()
    print(f"\n{'=' * 60}\nRunning feature pipeline: {city}\n{'=' * 60}")

    bronze = fetch_aqi(city)
    if not bronze:
        raise RuntimeError(f"fetch_aqi() returned no data for {city}")
    if not bronze.get("timestamp"):
        raise RuntimeError("fetch_aqi() returned no timestamp")

    print(f"[DEBUG] Bronze weather = {bronze.get('weather')}")
    save_bronze(bronze)

    silver = save_silver(bronze)
    print(
        f"[DEBUG] Silver weather = temp:{silver.get('temperature')}, "
        f"humidity:{silver.get('humidity')}, wind:{silver.get('wind_speed')}, "
        f"pressure:{silver.get('pressure')}"
    )

    history = get_recent_silver(city, limit=30)
    if not history:
        raise RuntimeError("No Silver history found after current write")
    save_gold(silver, history)

    print("[COMPLETE] Bronze -> Silver -> Gold finished successfully\n")


if __name__ == "__main__":
    city = sys.argv[1] if len(sys.argv) > 1 else "lahore"
    run_pipeline(city)
    fill_past_targets(city)   # Retroactively fill targets for older rows