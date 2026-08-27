import os
import sys
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from supabase import create_client
import pandas as pd

load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing Supabase credentials")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

CITY = "lahore"


def fetch_table(table_name, select="*", filters=None):
    """Helper to fetch data from a Supabase table."""
    query = supabase.table(table_name).select(select)
    if filters:
        for key, value in filters.items():
            query = query.eq(key, value)
    # order by timestamp if present
    if "timestamp" in select:
        query = query.order("timestamp", desc=False)
    return query.execute().data


def check_counts():
    """Check total records in each table."""
    bronze = fetch_table("aqi_bronze_raw", select="city,timestamp", filters={"city": CITY})
    silver = fetch_table("aqi_silver_cleaned", select="city,timestamp", filters={"city": CITY})
    gold = fetch_table("aqi_gold_features", select="city,timestamp", filters={"city": CITY})
    print(f"Bronze rows: {len(bronze)}")
    print(f"Silver rows: {len(silver)}")
    print(f"Gold rows  : {len(gold)}")
    return bronze, silver, gold


def check_missing_values(df, table_name):
    """Report null counts for important columns."""
    print(f"\n--- Missing values in {table_name} ---")
    cols = df.columns.tolist()
    for col in cols:
        if col in ["city", "timestamp", "raw_data", "weather"]:
            continue
        nulls = df[col].isna().sum()
        total = len(df)
        pct = (nulls / total * 100) if total > 0 else 0
        if nulls > 0:
            print(f"  {col}: {nulls} / {total} ({pct:.1f}%) missing")
    print()


def check_duplicates(df, table_name):
    """Check for duplicate (city, timestamp) pairs."""
    dup = df.duplicated(subset=["city", "timestamp"]).sum()
    if dup:
        print(f"⚠️ {table_name} has {dup} duplicate (city, timestamp) rows")
    else:
        print(f"✅ {table_name} has no duplicate rows")


def check_value_ranges(df, table_name):
    """Check if numeric values are plausible."""
    print(f"\n--- Value ranges in {table_name} ---")
    if "aqi" in df.columns:
        print(f"  AQI: min={df['aqi'].min():.1f}, max={df['aqi'].max():.1f}, mean={df['aqi'].mean():.1f}")
    if "temperature" in df.columns:
        print(f"  temperature: min={df['temperature'].min():.1f}, max={df['temperature'].max():.1f}")
    if "humidity" in df.columns:
        print(f"  humidity: min={df['humidity'].min():.1f}, max={df['humidity'].max():.1f}")
    if "pm25_raw" in df.columns:
        filled = df["pm25_raw"].notna().sum()
        print(f"  pm25_raw: {filled} / {len(df)} rows filled")


def check_lag_features(df):
    """Check if lag features are mostly populated."""
    print("\n--- Gold lag features ---")
    for col in ["aqi_lag_1h", "aqi_lag_24h", "aqi_roll_mean_24h"]:
        filled = df[col].notna().sum()
        total = len(df)
        pct = (filled / total * 100) if total > 0 else 0
        print(f"  {col}: {filled} / {total} ({pct:.1f}%) filled")


def check_targets(df):
    """Check if future AQI targets are populated."""
    print("\n--- Gold target columns (future AQI) ---")
    for col in ["aqi_d1", "aqi_d2", "aqi_d3"]:
        filled = df[col].notna().sum()
        total = len(df)
        pct = (filled / total * 100) if total > 0 else 0
        print(f"  {col}: {filled} / {total} ({pct:.1f}%) filled")


def check_timestamp_frequency(df):
    """Check that timestamps are hourly and recent."""
    print("\n--- Timestamp frequency ---")
    if "timestamp" not in df.columns:
        print("  No timestamp column")
        return
    # Convert to datetime
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], utc=True)
    # Check if differences are roughly hourly
    diffs = df["timestamp_dt"].diff().dropna()
    if len(diffs) > 0:
        # Convert to seconds and round to nearest hour
        diff_hours = diffs.dt.total_seconds() / 3600
        # Count how many are within 0.8-1.2 hours
        on_hour = ((diff_hours >= 0.8) & (diff_hours <= 1.2)).sum()
        total = len(diff_hours)
        pct = (on_hour / total * 100) if total > 0 else 0
        print(f"  {on_hour} / {total} gaps are ~1 hour ({pct:.1f}%)")
    # Check latest timestamp
    latest = df["timestamp_dt"].max()
    now = datetime.now(timezone.utc)
    age = (now - latest).total_seconds() / 3600
    print(f"  Latest timestamp: {latest} (UTC) – {age:.1f} hours ago")


def main():
    print(f"Data validation for city: {CITY}\n{'='*60}")

    # 1. Bronze (optional, but we can check raw_data structure)
    bronze = fetch_table("aqi_bronze_raw", filters={"city": CITY})
    print(f"Bronze rows: {len(bronze)}")
    if bronze:
        # sample raw_data keys
        sample = bronze[0].get("raw_data", {})
        print(f"Sample raw_data keys: {list(sample.keys())[:10]}")

    # 2. Silver
    silver_df = pd.DataFrame(fetch_table("aqi_silver_cleaned", filters={"city": CITY}))
    print(f"\nSilver rows: {len(silver_df)}")
    if len(silver_df) == 0:
        print("⚠️ No silver data found!")
        return
    check_missing_values(silver_df, "aqi_silver_cleaned")
    check_duplicates(silver_df, "aqi_silver_cleaned")
    check_value_ranges(silver_df, "aqi_silver_cleaned")
    check_timestamp_frequency(silver_df)

    # 3. Gold
    gold_df = pd.DataFrame(fetch_table("aqi_gold_features", filters={"city": CITY}))
    print(f"\nGold rows: {len(gold_df)}")
    if len(gold_df) == 0:
        print("⚠️ No gold data found!")
        return
    check_missing_values(gold_df, "aqi_gold_features")
    check_duplicates(gold_df, "aqi_gold_features")
    check_value_ranges(gold_df, "aqi_gold_features")
    check_lag_features(gold_df)
    check_targets(gold_df)
    check_timestamp_frequency(gold_df)

    # 4. Cross-check: Silver vs Gold row count
    print("\n--- Cross-check ---")
    silver_ts = set(pd.to_datetime(silver_df["timestamp"], utc=True))
    gold_ts = set(pd.to_datetime(gold_df["timestamp"], utc=True))
    missing_in_gold = silver_ts - gold_ts
    if missing_in_gold:
        print(f"⚠️ {len(missing_in_gold)} silver timestamps missing in gold")
    else:
        print("✅ All silver timestamps are present in gold")

    # 5. Summary
    print("\n" + "="*60)
    print("Validation complete.")
    print("Review the missing values report above.")
    print("If pm25_raw, lag features, and targets are >95% filled, your data is ready for training.")


if __name__ == "__main__":
    main()