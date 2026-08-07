import os
import sys
import pandas as pd
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'features'))
load_dotenv()

from fetch_aqi import fetch_aqi
from fetch_weather import fetch_weather

def run_feature_pipeline(city="Lahore"):
    print(f"Fetching data for {city}...")

    aqi     = fetch_aqi(city)
    weather = fetch_weather(city)

    if not aqi or not weather:
        print("Failed to fetch data")
        return

    combined = {**aqi, **weather}
    df = pd.DataFrame([combined])

    path = "data/historical_data.csv"
    if os.path.exists(path):
        existing = pd.read_csv(path)
        df = pd.concat([existing, df], ignore_index=True)
        df = df.drop_duplicates(subset=["timestamp"])

    df.to_csv(path, index=False)
    print(f"✅ Data saved! Total rows: {len(df)}")

if __name__ == "__main__":
    run_feature_pipeline("Lahore")