import requests
import os
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("AQICN_TOKEN")

stations = [
    "@A471607",
    "@A471608",
    "@A471609",
    "@A471610",
    "@A471611",
    "@A471612",
    "@A471613",
    "@A471614",
    "@A471615",
]

for station_id in stations:

    url = f"https://api.waqi.info/feed/{station_id}/?token={token}"

    try:
        response = requests.get(url, timeout=10)
        data = response.json()

        if data.get("status") != "ok":
            print(f"❌ {station_id} → inactive / unavailable")
            continue

        station = data["data"]

        aqi = station.get("aqi")
        name = station["city"]["name"]
        update_time = station["time"].get("iso")

        if aqi == "-":
            print(f"⚠️ {station_id} | {name} | No AQI")
        else:
            print(
                f"✅ {station_id} | "
                f"{name} | "
                f"AQI={aqi} | "
                f"Updated={update_time}"
            )

    except Exception as e:
        print(f"❌ {station_id} → ERROR: {e}")