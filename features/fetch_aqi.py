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
MAX_AGE_HOURS = 30  # Allow data up to 30 hours old (for development/testing with stale data)


def station_matches_lahore_pakistan(data: dict) -> bool:
    city = data.get("data", {}).get("city", {})
    if not isinstance(city, dict):
        return False

    geo = city.get("geo")
    location = str(city.get("location", "")).lower()
    name = str(city.get("name", "")).lower()

    if any(t in location for t in ["usa", "united states", "oregon", "bend", "fairway heights"]):
        return False

    if geo is not None:
        try:
            if isinstance(geo, (list, tuple)) and len(geo) >= 2:
                lat, lon = float(geo[0]), float(geo[1])
            elif isinstance(geo, str):
                parts = [p.strip() for p in geo.replace(";", ",").split(",")]
                if len(parts) >= 2:
                    lat, lon = float(parts[0]), float(parts[1])
                else:
                    lat = lon = None
            else:
                lat = lon = None
        except (TypeError, ValueError):
            lat = lon = None

        if lat is not None and lon is not None:
            return abs(lat - 31.5204) <= 3.0 and abs(lon - 74.3587) <= 3.0

    # Fallback: check city name markers when geo is missing or invalid
    lahore_markers = ["lahore", "punjab", "pakistan"]
    if any(marker in location for marker in lahore_markers) or any(marker in name for marker in lahore_markers):
        return True

    return False


def _fetch_one_station(sid: str):
    url = f"https://api.waqi.info/feed/@{sid}/?token={TOKEN}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "ok":
            print(f"[WARN] Station {sid}: API returned status={data.get('status')}")
            return None

        if not station_matches_lahore_pakistan(data):
            city_info = data.get("data", {}).get("city", {})
            print(f"[WARN] Station {sid}: geo/location mismatch (city={city_info.get('name', 'unknown')}, geo={city_info.get('geo', 'unknown')})")
            return None

        aqi = data["data"].get("aqi")
        
        # If AQI is missing, use PM2.5 as proxy (it's often used as AQI proxy)
        if aqi is None or aqi == "-":
            pm25 = data["data"].get("iaqi", {}).get("pm25", {}).get("v")
            if pm25 is None:
                print(f"[WARN] Station {sid}: AQI and PM2.5 both missing")
                return None
            aqi = int(pm25)
            print(f"[INFO] Station {sid}: Using PM2.5={aqi} as AQI proxy")
        else:
            aqi = int(aqi)

        station_time = data["data"]["time"].get("iso") or data["data"]["time"].get("s")
        if not station_time:
            print(f"[WARN] Station {sid}: timestamp missing")
            return None

        ts = datetime.fromisoformat(station_time.replace("Z", "+00:00")).astimezone(timezone.utc)
        age = datetime.now(timezone.utc) - ts
        if age > timedelta(hours=MAX_AGE_HOURS):
            print(f"[WARN] Station {sid}: data too old (age={age.total_seconds()/3600:.1f}h > {MAX_AGE_HOURS}h)")
            return None

        return {
            "sid": sid,
            "aqi": aqi,
            "ts": ts,
            "data": data,
        }
    except Exception as e:
        print(f"[WARN] Station {sid} network/parse error: {e}")
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
        raise RuntimeError(
            f"[ERROR] All 7 stations failed to return valid data for {city}. "
            f"Check station connectivity, geolocation filtering, data staleness, and API token."
        )

    # ---------- MAJORITY-BAND LOGIC ----------
    aqis = [r["aqi"] for r in results]
    total = len(aqis)

    from collections import Counter

    # Group by tens: e.g., 110-119 → 110, 120-129 → 120, etc.
    band_counts = Counter()
    band_values = {}

    for a in aqis:
        band = (a // 10) * 10          # integer division gives the tens
        band_counts[band] += 1
        band_values.setdefault(band, []).append(a)

    # Determine if any band has a strict majority (> half of all readings)
    majority_band = max(band_counts, key=band_counts.get) if band_counts else None
    
    if majority_band is not None and band_counts[majority_band] > total / 2:
        # Majority band found – use average of that band
        avg_aqi = round(sum(band_values[majority_band]) / len(band_values[majority_band]))
        print(f"[OK] Majority band {majority_band}s: {band_counts[majority_band]}/{total} stations → values {band_values[majority_band]} → avg = {avg_aqi}")
    else:
        # No majority band – use average of all stations minus 3
        avg_aqi = round(sum(aqis) / len(aqis)) - 3
        print(f"[OK] No majority band. All {total} stations {aqis} → avg = {round(sum(aqis) / len(aqis))} → minus 3 = {avg_aqi}")

    # Prefer primary station's raw_data if available, otherwise the newest one
    primary = next((r for r in results if r["sid"] == PRIMARY_STATION), None)
    best = primary if primary else max(results, key=lambda r: r["ts"])

    # Weather
    try:
        weather = fetch_weather_lahore()
        print(f"[OK] Lahore weather: {weather['temperature']}C, {weather['humidity']}% humidity")
    except Exception as e:
        print(f"[WARN] Weather fetch failed: {e}")
        weather = {}

    # Use current UTC time, not the station's stale timestamp
    timestamp = datetime.now(timezone.utc).isoformat()
    raw = best["data"]["data"].copy()
    raw["aqi"] = avg_aqi

    bronze_row = {
        "city": city,
        "timestamp": timestamp,
        "raw_data": raw,
        "weather": weather,
    }

    print(f"[OK] Fetched AQI for {city}: {avg_aqi} (from {len(results)}/7 stations)")
    return bronze_row