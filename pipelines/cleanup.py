import os
import sys
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

def remove_frozen_gold(city: str, hours: int = 6):
    """Remove Gold rows where AQI didn't change for more than `hours` hours."""
    print(f"\n[CLEANUP] Checking frozen rows for {city}...")
    
    r = supabase.table("aqi_gold_features")\
        .select("id,timestamp,aqi")\
        .eq("city", city)\
        .order("timestamp", desc=False)\
        .execute()
    
    rows = r.data
    if len(rows) < 2:
        print("[INFO] Not enough rows to check")
        return

    frozen_ids = []
    prev_aqi = None
    frozen_count = 0

    for row in rows:
        if row["aqi"] == prev_aqi:
            frozen_count += 1
            if frozen_count > hours:
                frozen_ids.append(row["id"])
        else:
            frozen_count = 0
        prev_aqi = row["aqi"]

    if frozen_ids:
        print(f"[INFO] Found {len(frozen_ids)} frozen rows — removing...")
        for fid in frozen_ids:
            supabase.table("aqi_gold_features").delete().eq("id", fid).execute()
        print(f"[OK] Removed {len(frozen_ids)} frozen rows")
    else:
        print(f"[OK] No frozen rows found")

def remove_silver_duplicates(city: str):
    """Remove duplicate Silver rows keeping only first per timestamp."""
    print(f"\n[CLEANUP] Removing Silver duplicates for {city}...")
    supabase.rpc("remove_silver_duplicates", {"p_city": city}).execute()
    print(f"[OK] Silver duplicates removed")

def run_cleanup():
    cities = ["lahore"]
    for city in cities:
        remove_frozen_gold(city)

if __name__ == "__main__":
    run_cleanup()