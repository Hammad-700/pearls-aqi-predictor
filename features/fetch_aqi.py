import os
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv
from features.fetch_weather import fetch_weather_lahore

load_dotenv()
TOKEN = os.getenv("AQICN_TOKEN")

# Primary and backup stations – all verified Lahore
# Order doesn't matter; we pick the one with latest timestamp.
LAHORE_STATION_IDS = [
    "-471607",   # G.O.R. (US Consulate) – reliable
    "-576565",   # DHA Phase 6
    "-576568",   # Barki Road
    "-576577",   # Egerton Road
    "-576544",   # Shahdara
    "-576559",   # GT Road
    "-576562",   # Kahna Nau
    "-576547",   # Multan Road
    "-576556",   # Punjab University
    "-582631",   # Wagha Border
    "-576550",   # Safari Park
    "-74005",    # Lahore Cantonment
    "-1866349",  # Terapand
    "-538468",   # Naila Road
    "-577588",   # Lathepur
    "11765",     # Lahore US Embassy (older)
]

def station_matches_lahore_pakistan(data: dict) -> bool:
    # Keep your existing geospatial check – useful if you ever add non-Lahore stations
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

def fetch_aqi(city: str) -> dict:
    city = city.lower()
    # Only implemented for Lahore now – but we keep city param for extensibility
    if city != "lahore":
        # fallback to old behaviour using CITY_MAP
        station = CITY_MAP.get(city) or city
        # ... (you can keep original single-station code for other cities)
        # but here we only focus on Lahore

    best_data = None
    best_timestamp = None

    for sid in LAHORE_STATION_IDS:
        url = f"https://api.waqi.info/feed/@{sid}/?token={TOKEN}"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") != "ok":
                continue

            # Validate it's actually Lahore (optional but safe)
            if not station_matches_lahore_pakistan(data):
                continue

            # Get reading timestamp
            station_time = data["data"]["time"].get("iso") or data["data"]["time"].get("s")
            if not station_time:
                continue

            # Parse to datetime for comparison
            try:
                ts = datetime.fromisoformat(station_time.replace("Z", "+00:00"))
            except:
                continue

            # Skip if AQI is invalid
            aqi = data["data"].get("aqi")
            if aqi is None or aqi == "-":
                continue

            # Keep the one with the latest timestamp
            if best_timestamp is None or ts > best_timestamp:
                best_timestamp = ts
                best_data = data

        except Exception as e:
            # Log but continue to next station
            print(f"[WARN] Station {sid} failed: {e}")
            continue

    if best_data is None:
        print(f"[ERROR] No station returned valid data for {city}")
        return None

    # Now process the best_data (exactly as before)
    data = best_data
    # Weather (only for Lahore)
    if city == "lahore":
        try:
            weather = fetch_weather_lahore()
            print(f"[OK] Lahore weather: {weather['temperature']}C, {weather['humidity']}% humidity")
        except Exception as e:
            print(f"[WARN] Lahore weather fetch failed: {e}")
            weather = {}
    else:
        weather = {}

    # Build bronze row (using the station's own timestamp)
    station_time = data["data"]["time"].get("iso") or data["data"]["time"].get("s")
    try:
        reading_ts = datetime.fromisoformat(station_time.replace("Z", "+00:00"))
        reading_ts = reading_ts.astimezone(timezone.utc)
        timestamp = reading_ts.isoformat()
    except:
        timestamp = datetime.now(timezone.utc).isoformat()

    bronze_row = {
        "city": city,
        "timestamp": timestamp,
        "raw_data": data["data"],
        "weather": weather,
    }

    print(f"[OK] Fetched AQI for {city}: {data['data']['aqi']} (reading time: {timestamp[:16]}) from station {data['data'].get('idx')}")
    return bronze_row