import requests
from datetime import datetime, timezone, timedelta

LAHORE_LAT = 31.5204
LAHORE_LON = 74.3587

def fetch_weather_lahore() -> dict:
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={LAHORE_LAT}&longitude={LAHORE_LON}"
            f"&hourly=temperature_2m,relative_humidity_2m,"
            f"wind_speed_10m,wind_direction_10m,precipitation,pressure_msl"
            f"&forecast_days=1&timezone=Asia/Karachi"
        )
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        d = r.json()

        current_hour = datetime.now(timezone(timedelta(hours=5))).hour
        hourly = d["hourly"]

        weather = {
            "temperature": hourly["temperature_2m"][current_hour],
            "humidity": hourly["relative_humidity_2m"][current_hour],
            "wind_speed": hourly["wind_speed_10m"][current_hour],
            "wind_direction": hourly["wind_direction_10m"][current_hour],
            "precipitation": hourly["precipitation"][current_hour],
            "pressure": hourly["pressure_msl"][current_hour],
        }
        print(f"[OK] Weather fetched: temp={weather['temperature']}°C "
              f"wind={weather['wind_speed']}km/h")
        return weather

    except Exception as e:
        print(f"[WARN] Weather fetch failed: {e}")
        return {
            "temperature": None, "humidity": None,
            "wind_speed": None, "wind_direction": None,
            "precipitation": None, "pressure": None,
        }