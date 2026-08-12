import requests
import os
import json
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("AQICN_TOKEN")

CITY_MAP = {
    "lahore": "@A471607"
}

def fetch_aqi(city: str) -> dict:
    station = CITY_MAP.get(city.lower(), city)
    url = f"https://api.waqi.info/feed/{station}/?token={TOKEN}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data["status"] != "ok":
            print(f"[ERROR] API returned status: {data['status']}")
            return None

        bronze_row = {
            "city": city,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "raw_data": data["data"]
        }

        print(f"[OK] Fetched AQI for {city}: {data['data']['aqi']}")
        return bronze_row

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Request failed for {city}: {e}")
        return None

if __name__ == "__main__":
    city = input("Enter city name: ")
    result = fetch_aqi(city)
    if result:
        print(json.dumps(result, indent=2, default=str))