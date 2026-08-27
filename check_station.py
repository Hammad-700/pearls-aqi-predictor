import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# Fetch the most recent bronze row for Lahore
bronze = (
    supabase.table("aqi_bronze_raw")
    .select("raw_data")
    .eq("city", "lahore")
    .order("timestamp", desc=True)
    .limit(1)
    .execute()
)

if bronze.data:
    raw = bronze.data[0]["raw_data"]
    city_info = raw.get("city", {})
    print(f"Station ID (idx) : {raw.get('idx')}")
    print(f"Station name     : {city_info.get('name')}")
    print(f"Location         : {city_info.get('location')}")
    print(f"Coordinates (geo): {city_info.get('geo')}")
    print(f"URL              : {city_info.get('url')}")
else:
    print("No bronze data found for Lahore.")