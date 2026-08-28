import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

print("URL:", url)
print("KEY exists:", bool(key))

if not url or not key:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY")

supabase = create_client(url, key)

print("Supabase client created")

result = (
    supabase
    .table("aqi_bronze_raw")
    .select("*")
    .limit(1)
    .execute()
)

print("SUCCESS")
print(result.data)