import os
import sys
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard.streamlit_app import PKT
from features.engineer_features import clean_to_silver, build_gold_features
from features.fetch_aqi import fetch_aqi


supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

GOLD_COLS = [
    "city", "timestamp", "aqi", "hour", "day_of_week",
    "month", "aqi_lag_1h", "aqi_lag_24h", "aqi_roll_mean_24h",
    "aqi_change_rate", "aqi_d1", "aqi_d2", "aqi_d3"
]


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

        # Staleness check
        try:
            ts = datetime.fromisoformat(
                bronze_row["timestamp"].replace("Z", "+00:00")
            )
            age_hours = (datetime.now(timezone.utc) - ts).total_seconds() / 3600
            if age_hours > 6:
                print(f"[WARN] Station data is {int(age_hours)}h old — may be stale!")
            else:
                print(f"[OK] Station data is {int(age_hours * 60)} minutes old")
        except Exception as e:
            print(f"[WARN] Staleness check failed: {e}")
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

    # Get recent silver rows (most recent 30, oldest first for feature calc)
    try:
        recent = supabase.table("aqi_silver_cleaned")\
            .select("*")\
            .eq("city", city)\
            .order("timestamp", desc=True)\
            .limit(30)\
            .execute()
        silver_rows = list(reversed(recent.data))
    except Exception as e:
        print(f"[ERROR] Fetching silver rows failed: {e}")
        return

    # Gold
    gold_row = build_gold_features(silver_rows)

    if not gold_row:
        if silver_row.get("aqi") is None:
            print(f"[WARN] No aqi in silver row — skipping Gold")
            return
        else:
            print(f"[WARN] Not enough history for full features — saving partial Gold row")
            ts = datetime.fromisoformat(silver_row["timestamp"].replace("Z", "+00:00"))
            ts_pkt = ts.astimezone(PKT)   # ← add this
            
            gold_row = {
                "city": silver_row["city"],
                "timestamp": silver_row["timestamp"],
                "aqi": silver_row["aqi"],
                "hour": ts_pkt.hour,                 # ← changed
                "day_of_week": ts_pkt.weekday(),     # ← changed
                "month": ts_pkt.month,               # ← changed
                "aqi_lag_1h": None,
                "aqi_lag_24h": None,
                "aqi_roll_mean_24h": None,
                "aqi_change_rate": None,
                "aqi_d1": None,
                "aqi_d2": None,
                "aqi_d3": None,
            }

    try:
        gold_filtered = {k: v for k, v in gold_row.items() if k in GOLD_COLS}
        
        # Force using the silver timestamp
        gold_filtered["timestamp"] = silver_row["timestamp"]
        
        print(f"[DEBUG] Saving Gold → timestamp={gold_filtered['timestamp']} | aqi={gold_filtered['aqi']}")
        
        result = supabase.table("aqi_gold_features").upsert(
            gold_filtered, on_conflict="city,timestamp"
        ).execute()
        
        print(f"[OK] Gold saved (rows affected: {len(result.data) if result.data else 'unknown'})")
        
    except Exception as e:
        print(f"[ERROR] Gold save failed: {e}")
        raise   # Important: make the job fail so you get the alert
if __name__ == "__main__":
    city = sys.argv[1] if len(sys.argv) > 1 else "lahore"
    run_pipeline(city)