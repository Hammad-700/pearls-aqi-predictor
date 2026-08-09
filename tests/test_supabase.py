import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

supabase = create_client(url, key)

# Test by reading from bronze table
result = supabase.table("aqi_bronze_raw").select("*").limit(1).execute()
print("[OK] Supabase connected!")
print("Tables accessible. Row count:", len(result.data))