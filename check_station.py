import os
import requests
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("AQICN_TOKEN")

# List of station IDs you provided
station_ids = [
    "-576544", "-576568", "-538468", "-576577", "-576559",
    "-577588", "-576565", "-576562", "-74005", "-1866349",
    "-471607", "-582631", "-576550", "-576547", "-576556", "11765"
]

print("Checking station IDs:\n")

for sid in station_ids:
    # The API expects the ID with a leading @
    url = f"https://api.waqi.info/feed/@{sid}/?token={TOKEN}"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if data.get("status") == "ok":
            d = data["data"]
            city_info = d.get("city", {})
            name = city_info.get("name", "Unknown")
            location = city_info.get("location", "Unknown")
            geo = city_info.get("geo")
            aqi = d.get("aqi", "N/A")
            lat, lon = geo if geo else ("N/A", "N/A")
            # Check if it's Lahore, Pakistan
            is_lahore = "lahore" in name.lower() or "lahore" in location.lower()
            print(f"✅ ID: @{sid}")
            print(f"   Name     : {name}")
            print(f"   Location : {location}")
            print(f"   Coords   : ({lat}, {lon})")
            print(f"   AQI      : {aqi}")
            print(f"   Lahore?  : {'YES' if is_lahore else 'NO'}")
            print()
        else:
            print(f"❌ ID: @{sid} – invalid or no data (status: {data.get('status')})")
    except Exception as e:
        print(f"⚠️  ID: @{sid} – error: {e}")

print("Done.")