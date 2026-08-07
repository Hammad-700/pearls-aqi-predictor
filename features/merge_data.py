import pandas as pd
from datetime import datetime
from fetch_aqi import fetch_aqi
from fetch_weather import fetch_weather

def fetch_combined(city="Lahore"):
    aqi = fetch_aqi(city)
    weather = fetch_weather(city)

    if not aqi or not weather:
        print("Failed to fetch data")
        return None

    combined = {**aqi, **weather}
    combined["fetched_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    df = pd.DataFrame([combined])
    print(df.to_string())
    return df

if __name__ == "__main__":
    fetch_combined("Lahore")