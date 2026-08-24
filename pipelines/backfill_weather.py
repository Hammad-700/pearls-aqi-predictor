import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

LATITUDE = 31.5204
LONGITUDE = 74.3587
CITY = "lahore"


def get_supabase():
    return create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))


def timestamp_key(value):
    timestamp = pd.to_datetime(value, utc=True)
    return timestamp.strftime("%Y-%m-%dT%H:00:00+00:00")


def fetch_weather(start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": "temperature_2m,relative_humidity_2m",
        "timezone": "UTC",
    }
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    hourly = response.json().get("hourly", {})
    weather = {}
    for timestamp, temperature, humidity in zip(
        hourly.get("time", []),
        hourly.get("temperature_2m", []),
        hourly.get("relative_humidity_2m", []),
    ):
        if temperature is not None and humidity is not None:
            weather[timestamp_key(timestamp)] = {
                "temperature": float(temperature),
                "humidity": float(humidity),
            }
    return weather


def update_table(supabase, table, rows, weather):
    updated = 0
    missing = 0
    for row in rows:
        key = timestamp_key(row["timestamp"])
        values = weather.get(key)
        if not values:
            missing += 1
            continue
        supabase.table(table).update(values).eq("city", CITY).eq(
            "timestamp", row["timestamp"]
        ).execute()
        updated += 1
        if updated % 100 == 0:
            print(f"[INFO] {table}: updated {updated} rows")
    print(f"[OK] {table}: updated={updated}, missing_weather={missing}")


def main():
    supabase = get_supabase()
    gold_rows = supabase.table("aqi_gold_features").select(
        "city,timestamp,temperature,humidity"
    ).eq("city", CITY).not_.is_("aqi_d1", "null").order(
        "timestamp", desc=False
    ).execute().data
    if not gold_rows:
        raise RuntimeError("No labeled Lahore Gold rows found")

    silver_rows = supabase.table("aqi_silver_cleaned").select(
        "city,timestamp,temperature,humidity"
    ).eq("city", CITY).order("timestamp", desc=False).execute().data

    start = pd.to_datetime(gold_rows[0]["timestamp"], utc=True).date()
    end = pd.to_datetime(gold_rows[-1]["timestamp"], utc=True).date()
    print(f"[INFO] Fetching Lahore weather: {start} -> {end}")
    weather = fetch_weather(start, end)
    print(f"[OK] Retrieved {len(weather)} hourly weather values")

    update_table(supabase, "aqi_silver_cleaned", silver_rows, weather)
    update_table(supabase, "aqi_gold_features", gold_rows, weather)


if __name__ == "__main__":
    main()
