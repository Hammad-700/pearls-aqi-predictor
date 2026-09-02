# import os
# from dotenv import load_dotenv
# from supabase import create_client

# load_dotenv()

# url = os.getenv("SUPABASE_URL")
# key = os.getenv("SUPABASE_KEY")

# print("URL:", url)
# print("KEY exists:", bool(key))

# if not url or not key:
#     raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY")

# supabase = create_client(url, key)

# print("Supabase client created")

# result = (
#     supabase
#     .table("aqi_bronze_raw")
#     .select("*")
#     .limit(1)
#     .execute()
# )

# print("SUCCESS")
# print(result.data)

print(f"[DEBUG] Silver payload before save:")
print(f"  city={silver.get('city')}")
print(f"  timestamp={silver.get('timestamp')}")
print(f"  aqi={silver.get('aqi')}")
print(f"  Full payload keys: {list(silver.keys())}")

try:
    supabase.table("aqi_silver_cleaned").upsert(
        silver, on_conflict="city,timestamp"
    ).execute()
except Exception as e:
    print(f"[ERROR] Silver save FAILED: {e}")
    raise