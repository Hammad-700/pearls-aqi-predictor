import os
import sys
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.fetch_aqi import fetch_aqi
from features.engineer_features import clean_to_silver, build_gold_features

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

GOLD_COLS = ["city", "timestamp", "aqi", "hour", "day_of_week",
             "month", "aqi_lag_1h", "aqi_lag_24h", "aqi_roll_mean_24h",
             "aqi_change_rate", "aqi_d1", "aqi_d2", "aqi_d3"]

def run_pipeline(city: str):
    print(f"\n--- Running pipeline for: {city} ---")

    # Bronze
    bronze_row = fetch_aqi(city)
    if not bronze_row:
        print(f"[ERROR] Fetch failed for {city}")
        return
    try:
        supabase.table("aqi_bronze_raw").upsert({
            "city": bronze_row["city"],
            "timestamp": bronze_row["timestamp"],
            "raw_data": bronze_row["raw_data"]
        }, on_conflict="city,timestamp").execute()
        print(f"[OK] Bronze saved")
    except Exception as e:
        print(f"[ERROR] Bronze save failed: {e}")
        return

    # Silver
    silver_row = clean_to_silver(bronze_row)
    if not silver_row:
        return
    try:
        supabase.table("aqi_silver_cleaned").upsert(
            silver_row, on_conflict="city,timestamp"
        ).execute()
        print(f"[OK] Silver saved")
    except Exception as e:
        print(f"[ERROR] Silver save failed: {e}")
        return

    # Get recent silver rows
    try:
        recent = supabase.table("aqi_silver_cleaned")\
            .select("*")\
            .eq("city", city)\
            .order("timestamp", desc=False)\
            .limit(30)\
            .execute()
        silver_rows = recent.data
    except Exception as e:
        print(f"[ERROR] Fetching silver rows failed: {e}")
        return

    # Gold
    gold_row = build_gold_features(silver_rows)
    if not gold_row:
        print(f"[WARN] Not enough data for Gold yet — need 2+ rows")
        return
    try:
        from datetime import datetime, timezone
        gold_filtered = {k: v for k, v in gold_row.items() if k in GOLD_COLS}
        gold_filtered["timestamp"] = datetime.now(timezone.utc).isoformat()
        supabase.table("aqi_gold_features").upsert(
            gold_filtered, on_conflict="city,timestamp"
        ).execute()
        print(f"[OK] Gold saved")
    except Exception as e:
        print(f"[ERROR] Gold save failed: {e}")