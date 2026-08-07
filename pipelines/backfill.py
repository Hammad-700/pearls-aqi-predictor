import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'features'))

from engineer_features import engineer_features

def generate_historical_data(city="Lahore", days=90):
    print(f"Generating {days} days of historical data for {city}...")

    # Generate hourly timestamps for past 90 days
    dates = pd.date_range(end=pd.Timestamp.utcnow(), periods=days*24, freq="h")

    np.random.seed(42)
    n = len(dates)

    # Simulate realistic AQI pattern for Lahore
    # Lahore has high pollution - AQI typically 100-300
    base_aqi = 150
    hourly_pattern = np.sin(np.linspace(0, 2*np.pi, 24))  # daily cycle
    hourly_cycle = np.tile(hourly_pattern, days) * 30

    seasonal_noise = np.random.normal(0, 20, n)
    aqi_values = base_aqi + hourly_cycle + seasonal_noise
    aqi_values = np.clip(aqi_values, 50, 350).astype(int)

    df = pd.DataFrame({
        "city": city,
        "aqi": aqi_values,
        "pm25": (aqi_values * 0.8 + np.random.normal(0, 5, n)).clip(0),
        "pm10": (aqi_values * 0.6 + np.random.normal(0, 5, n)).clip(0),
        "o3": np.random.uniform(10, 60, n),
        "no2": np.random.uniform(5, 80, n),
        "so2": np.random.uniform(1, 30, n),
        "co": np.random.uniform(0.5, 5, n),
        "temperature": np.random.uniform(15, 42, n),
        "humidity": np.random.uniform(30, 90, n),
        "pressure": np.random.uniform(990, 1015, n),
        "wind_speed": np.random.uniform(0, 12, n),
        "timestamp": dates,
    })

    # Run feature engineering
    df = engineer_features(df)

    # Save to CSV
    os.makedirs("data", exist_ok=True)
    output_path = "data/historical_data.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ Saved {len(df)} rows to {output_path}")
    return df

if __name__ == "__main__":
    generate_historical_data("Lahore", days=90)