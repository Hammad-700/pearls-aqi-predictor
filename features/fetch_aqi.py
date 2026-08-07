import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("AQICN_API_KEY")

def fetch_aqi(city="Lahore"):
    url = f"https://api.waqi.info/feed/{city}/?token={API_KEY}"
    response = requests.get(url)
    data = response.json()

    if data["status"] != "ok":
        print(f"Error fetching AQI for {city}: {data}")
        return None

    aqi_data = {
        "city": city,
        "aqi": data["data"]["aqi"],
        "pm25": data["data"]["iaqi"].get("pm25", {}).get("v", None),
        "pm10": data["data"]["iaqi"].get("pm10", {}).get("v", None),
        "o3":   data["data"]["iaqi"].get("o3",   {}).get("v", None),
        "no2":  data["data"]["iaqi"].get("no2",  {}).get("v", None),
        "so2":  data["data"]["iaqi"].get("so2",  {}).get("v", None),
        "co":   data["data"]["iaqi"].get("co",   {}).get("v", None),
        "timestamp": data["data"]["time"]["s"],
    }

    print(aqi_data)
    return aqi_data

if __name__ == "__main__":
    fetch_aqi("Lahore")