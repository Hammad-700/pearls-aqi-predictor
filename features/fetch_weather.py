import requests
from datetime import datetime, timezone, timedelta

LAHORE_LAT = 31.5204
LAHORE_LON = 74.3587

def fetch_weather_lahore() -> dict:
    try:
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={LAHORE_LAT}&longitude={LAHORE_LON}"
            f"&hourly=temperature_2m,relative_humidity_2m,"
            f"wind_speed_10m,wind_direction_10m,precipitation,pressure_msl"
            f"&forecast_days=1&timezone=Asia/Karachi"
        )
        weather_response = requests.get(weather_url, timeout=10)
        weather_response.raise_for_status()
        weather_data = weather_response.json()

        air_quality_url = (
            f"https://air-quality-api.open-meteo.com/v1/air-quality?"
            f"latitude={LAHORE_LAT}&longitude={LAHORE_LON}"
            f"&hourly=pm2_5,pm10,nitrogen_dioxide,ozone"
            f"&forecast_days=1&timezone=Asia%2FKarachi"
        )
        air_quality_response = requests.get(air_quality_url, timeout=10)
        air_quality_response.raise_for_status()
        air_quality_data = air_quality_response.json()

        current_hour = datetime.now(timezone(timedelta(hours=5))).hour
        weather_hourly = weather_data["hourly"]
        air_quality_hourly = air_quality_data["hourly"]

        weather = {
            "temperature": weather_hourly["temperature_2m"][current_hour],
            "humidity": weather_hourly["relative_humidity_2m"][current_hour],
            "wind_speed": weather_hourly["wind_speed_10m"][current_hour],
            "wind_direction": weather_hourly["wind_direction_10m"][current_hour],
            "precipitation": weather_hourly["precipitation"][current_hour],
            "pressure": weather_hourly["pressure_msl"][current_hour],
            "pm25_raw": air_quality_hourly["pm2_5"][current_hour],
            "pm10_raw": air_quality_hourly["pm10"][current_hour],
            "no2_raw": air_quality_hourly["nitrogen_dioxide"][current_hour],
            "o3_raw": air_quality_hourly["ozone"][current_hour],
        }
        print(f"[OK] Weather+AQ fetched: temp={weather['temperature']}°C "
              f"pm25={weather['pm25_raw']}ug/m3 wind={weather['wind_speed']}km/h")
        return weather

    except Exception as e:
        print(f"[WARN] Weather fetch failed: {e}")
        return {
            "temperature": None, "humidity": None,
            "wind_speed": None, "wind_direction": None,
            "precipitation": None, "pressure": None,
            "pm25_raw": None, "pm10_raw": None,
            "no2_raw": None, "o3_raw": None,
        }