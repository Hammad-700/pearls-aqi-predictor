import requests
import os
import json
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("AQICN_TOKEN")

# Allow an explicit Lahore station override from the environment without breaking the current cron job.
# Example: LAHORE_AQI_STATION_ID=@A123456
LAHORE_AQI_STATION_ID = os.getenv("LAHORE_AQI_STATION_ID", "@A471607")

# Keep the current station as default for compatibility with the existing pipeline.
CITY_MAP = {
    "lahore": LAHORE_AQI_STATION_ID,
}

PAKISTAN_LAHORE = {
    "lat": 31.5204,
    "lon": 74.3587,
    "radius_deg": 3.0,
}


def fetch_lahore_weather() -> dict:
    url = (
        "https://api.open-meteo.com/v1/forecast"
        "?latitude=31.5204&longitude=74.3587"
        "&current=temperature_2m,relative_humidity_2m"
        "&timezone=Asia%2FKarachi"
    )
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    current = response.json().get("current", {})
    return {
        "temperature": float(current["temperature_2m"]),
        "humidity": float(current["relative_humidity_2m"]),
    }


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

    return (
        abs(lat - PAKISTAN_LAHORE["lat"]) <= PAKISTAN_LAHORE["radius_deg"]
        and abs(lon - PAKISTAN_LAHORE["lon"]) <= PAKISTAN_LAHORE["radius_deg"]
    )


def fetch_aqi(city: str) -> dict:
    station = CITY_MAP.get(city.lower()) or city

    url = f"https://api.waqi.info/feed/{station}/?token={TOKEN}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data["status"] != "ok":
            print(f"[ERROR] API returned status: {data['status']}")
            return None

        if city.lower() == "lahore" and not station_matches_lahore_pakistan(data):
            raise ValueError(
                f"Lahore station {station} does not match Lahore, Pakistan coordinates"
            )

        if city.lower() == "lahore":
            try:
                weather = fetch_lahore_weather()
                data["data"].setdefault("iaqi", {})["t"] = {"v": weather["temperature"]}
                data["data"]["iaqi"]["h"] = {"v": weather["humidity"]}
                print(f"[OK] Lahore weather: {weather['temperature']}C, {weather['humidity']}% humidity")
            except (requests.exceptions.RequestException, KeyError, TypeError, ValueError) as e:
                print(f"[WARN] Lahore weather fetch failed: {e}; using AQICN weather values")

        # Use station's own reading time (not fetch time)
        station_time = data["data"]["time"].get("iso") or data["data"]["time"].get("s")
        if station_time:
            try:
                reading_ts = datetime.fromisoformat(station_time.replace("Z", "+00:00"))
                # Convert to UTC
                reading_ts = reading_ts.astimezone(timezone.utc)
                timestamp = reading_ts.isoformat()
            except:
                timestamp = datetime.now(timezone.utc).isoformat()
        else:
            timestamp = datetime.now(timezone.utc).isoformat()

        bronze_row = {
            "city": city,
            "timestamp": timestamp,
            "raw_data": data["data"],
        }

        print(f"[OK] Fetched AQI for {city}: {data['data']['aqi']} (reading time: {timestamp[:16]})")
        return bronze_row

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Request failed for {city}: {e}")
        return None

if __name__ == "__main__":
    city = input("Enter city name: ")
    result = fetch_aqi(city)
    if result:
        print(json.dumps(result, indent=2, default=str))