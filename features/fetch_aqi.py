import os
import requests
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from features.fetch_weather import fetch_weather_lahore

load_dotenv()
TOKEN = os.getenv("AQICN_TOKEN")

# Primary station (most reliable in Lahore)
PRIMARY_STATION = "-471607"   # G.O.R. / US Consulate

# Backup stations (only used if primary fails)
BACKUP_STATIONS = [
    "-576565",   # DHA Phase 6
    "-576568",   # Barki Road
    "-576577",   # Egerton Road
    "-576544",   # Shahdara
    "-576559",   # GT Road
    "-576562",   # Kahna Nau
    "-576547",   # Multan Road
    "-576556",   # Punjab University
]

MAX_AGE_HOURS = 6   # Reject readings older than this


def station_matches_lahore_pakistan(data: dict) -> bool:
    city = data.get("data", {}).get("city", {})
    geo = city.get("geo")
    location = str(city.get("location", "")).lower()

    if not geo or len(geo) != 2:
        return False

    lat = float(geo[0])
    lon = float(geo[1])

    if any(token in location for token in ["usa", "united states", "oregon", "bend", "fairway heights"]):
        return False

    PAKISTAN_LAHORE = {"lat": 31.5204, "lon": 74.3587, "radius_deg": 3.0}
    return (abs(lat - PAKISTAN_LAHORE["lat"]) <= PAKISTAN_LAHORE["radius_deg"]
            and abs(lon - PAKISTAN_LAHORE["lon"]) <= PAKISTAN_LAHORE["radius_deg"])


def _fetch_one_station(sid: str):
    """Try a single station. Returns (data, timestamp) or (None, None)."""
    url = f"https://api.waqi.info/feed/@{sid}/?token={TOKEN}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "ok":
            return None, None

        if not station_matches_lahore_pakistan(data):
            return None, None

        aqi = data["data"].get("aqi")
        if aqi is None or aqi == "-":
            return None, None

        station_time = data["data"]["time"].get("iso") or data["data"]["time"].get("s")
        if not station_time:
            return None, None

        ts = datetime.fromisoformat(station_time.replace("Z", "+00:00"))

        # Reject very old readings
        age = datetime.now(timezone.utc) - ts.astimezone(timezone.utc)
        if age > timedelta(hours=MAX_AGE_HOURS):
            print(f"[WARN] Station {sid} reading is {age} old — skipped")
            return None, None

        return data, ts

    except Exception as e:
        print(f"[WARN] Station {sid} failed: {e}")
        return None, None


def fetch_aqi(city: str) -> dict:
    city = city.lower()
    if city != "lahore":
        raise NotImplementedError("Only Lahore is supported right now")

    best_data = None
    best_ts = None
    used_station = None

    # 1. Try primary station first
    data, ts = _fetch_one_station(PRIMARY_STATION)
    if data is not None:
        best_data = data
        best_ts = ts
        used_station = PRIMARY_STATION
        print(f"[OK] Using primary station {PRIMARY_STATION}")
    else:
        # 2. Fall back to other stations (pick the newest valid one)
        print("[WARN] Primary station failed — trying backups...")
        for sid in BACKUP_STATIONS:
            data, ts = _fetch_one_station(sid)
            if data is None:
                continue
            if best_ts is None or ts > best_ts:
                best_data = data
                best_ts = ts
                used_station = sid

    if best_data is None:
        print(f"[ERROR] No station returned valid data for {city}")
        return None

    # Weather
    try:
        weather = fetch_weather_lahore()
        print(f"[OK] Lahore weather: {weather['temperature']}C, {weather['humidity']}% humidity")
    except Exception as e:
        print(f"[WARN] Lahore weather fetch failed: {e}")
        weather = {}

    # Build timestamp
    station_time = best_data["data"]["time"].get("iso") or best_data["data"]["time"].get("s")
    try:
        reading_ts = datetime.fromisoformat(station_time.replace("Z", "+00:00"))
        reading_ts = reading_ts.astimezone(timezone.utc)
    except Exception:
        reading_ts = datetime.now(timezone.utc)

    # If the station time is older than 30 minutes, use current time instead
    # (prevents old station readings from being treated as "latest")
    now = datetime.now(timezone.utc)
    if (now - reading_ts) > timedelta(minutes=30):
        print(f"[WARN] Station reading is old ({reading_ts.isoformat()[:16]}). Using current time.")
        reading_ts = now

    timestamp = reading_ts.isoformat()

    bronze_row = {
        "city": city,
        "timestamp": timestamp,
        "raw_data": best_data["data"],
        "weather": weather,
    }

    print(
        f"[OK] Fetched AQI for {city}: {best_data['data']['aqi']} "
        f"(reading time: {timestamp[:16]}) from station {used_station}"
    )
    return bronze_row