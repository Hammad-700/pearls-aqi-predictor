import requests
from datetime import datetime

lat, lon = 31.5204, 74.3587  # Lahore

url = (
    f"https://api.open-meteo.com/v1/forecast?"
    f"latitude={lat}&longitude={lon}"
    f"&hourly=temperature_2m,relative_humidity_2m,"
    f"wind_speed_10m,wind_direction_10m,precipitation,pressure_msl"
    f"&forecast_days=1&timezone=Asia/Karachi"
)

r = requests.get(url)
d = r.json()

# Get current hour index
current_hour = datetime.now().hour
hourly = d["hourly"]

print("Temperature:", hourly["temperature_2m"][current_hour])
print("Humidity:", hourly["relative_humidity_2m"][current_hour])
print("Wind speed:", hourly["wind_speed_10m"][current_hour])
print("Wind direction:", hourly["wind_direction_10m"][current_hour])
print("Precipitation:", hourly["precipitation"][current_hour])
print("Pressure:", hourly["pressure_msl"][current_hour])