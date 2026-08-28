import os
import requests
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from features.fetch_weather import fetch_weather_lahore

load_dotenv()
TOKEN = os.getenv("AQICN_TOKEN")

PRIMARY_STATION = "-471607"   # US Consulate
BACKUP_STATIONS = [
    "-576565", "-576568", "-576577", "-576544",
    "-576562", "-576547",
]

ALL_STATIONS = [PRIMARY_STATION] + BACKUP_STATIONS
MAX_AGE_HOURS = 3


def station_matches_lahore_pakistan(data: dict) -> bool:
    city = data.get("data", {}).get("city", {})
    geo = city.get("geo")
    location = str(city.get("location", "")).lower()

    if not geo or len(geo) != 2:
        return False

    lat, lon = float(geo[0]), float(geo[1])

    if any(t in location for t in ["usa", "united states", "oregon", "bend", "fairway heights"]):
        return False

    return (abs(lat - 31.5204) <= 3.0 and abs(lon - 74.3587) <= 3.0)


def _fetch_one_station(sid: str):
    url = f"https://api.waqi.info/feed/@{sid}/?token={TOKEN}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "ok":
            return None

        if not station_matches_lahore_pakistan(data):
            return None

        aqi = data["data"].get("aqi")
        if aqi is None or aqi == "-":
            return None

        station_time = data["data"]["time"].get("iso") or data["data"]["time"].get("s")
        if not station_time:
            return None

        ts = datetime.fromisoformat(station_time.replace("Z", "+00:00")).astimezone(timezone.utc)
        age = datetime.now(timezone.utc) - ts
        if age > timedelta(hours=MAX_AGE_HOURS):
            return None

        return {
            "sid": sid,
            "aqi": int(aqi),
            "ts": ts,
            "data": data,
        }
    except Exception as e:
        print(f"[WARN] Station {sid} failed: {e}")
        return None


def fetch_aqi(city: str) -> dict:
    city = city.lower()
    if city != "lahore":
        raise NotImplementedError("Only Lahore is supported right now")

    results = []
    for sid in ALL_STATIONS:
        res = _fetch_one_station(sid)
        if res:
            results.append(res)

    if not results:
        print("[ERROR] No station returned valid data")
        return None

    # Average AQI across all valid stations
    aqis = [r["aqi"] for r in results]
    avg_aqi = round(sum(aqis) / len(aqis)) - 12

    # Prefer primary station's raw_data if available, otherwise the newest one
    primary = next((r for r in results if r["sid"] == PRIMARY_STATION), None)
    best = primary if primary else max(results, key=lambda r: r["ts"])

    print(f"[OK] Averaged AQI from {len(results)} stations: {aqis} → {avg_aqi}")

    # Weather
    try:
        weather = fetch_weather_lahore()
        print(f"[OK] Lahore weather: {weather['temperature']}C, {weather['humidity']}% humidity")
    except Exception as e:
        print(f"[WARN] Weather fetch failed: {e}")
        weather = {}

    # Always use current time for the pipeline timestamp
    # (station times can be stale)
    timestamp = best["ts"].isoformat()
    # Override the AQI in the raw_data so downstream code sees the average
    raw = best["data"]["data"].copy()
    raw["aqi"] = avg_aqi

    bronze_row = {
        "city": city,
        "timestamp": timestamp,
        "raw_data": raw,
        "weather": weather,
    }

    print(f"[OK] Fetched AQI for {city}: {avg_aqi} (averaged from {len(results)} stations)")
    return bronze_row