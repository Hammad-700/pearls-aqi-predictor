import pandas as pd
import numpy as np

def engineer_features(df):
    df = df.copy()

    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Time-based features
    df["hour"]       = df["timestamp"].dt.hour
    df["day"]        = df["timestamp"].dt.day
    df["month"]      = df["timestamp"].dt.month
    df["day_of_week"]= df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Fill missing pollutants with 0
    for col in ["pm25", "pm10", "o3", "no2", "so2", "co"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Lag features (requires sorted historical data)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["aqi_lag_1"]  = df["aqi"].shift(1)
    df["aqi_lag_3"]  = df["aqi"].shift(3)
    df["aqi_lag_6"]  = df["aqi"].shift(6)
    df["aqi_lag_24"] = df["aqi"].shift(24)

    # Rolling statistics
    df["aqi_rolling_mean_6"]  = df["aqi"].rolling(window=6).mean()
    df["aqi_rolling_std_6"]   = df["aqi"].rolling(window=6).std()
    df["aqi_rolling_mean_24"] = df["aqi"].rolling(window=24).mean()

    # AQI change rate
    df["aqi_change_rate"] = df["aqi"].diff()

    # Target: AQI 72 hours ahead
    df["target_aqi_72h"] = df["aqi"].shift(-72)

    # Drop rows with missing target
    df = df.dropna(subset=["target_aqi_72h"])

    print(f"Features created! Shape: {df.shape}")
    print(df.columns.tolist())
    return df

if __name__ == "__main__":
    # Test with dummy data
    dummy = pd.DataFrame({
        "city": ["Lahore"] * 100,
        "aqi": np.random.randint(50, 200, 100),
        "pm25": np.random.randint(10, 100, 100),
        "pm10": np.random.randint(10, 100, 100),
        "o3": np.random.randint(5, 50, 100),
        "no2": np.random.randint(5, 50, 100),
        "so2": np.random.randint(1, 20, 100),
        "co": np.random.randint(1, 10, 100),
        "temperature": np.random.uniform(15, 40, 100),
        "humidity": np.random.randint(30, 90, 100),
        "pressure": np.random.randint(990, 1020, 100),
        "wind_speed": np.random.uniform(0, 10, 100),
        "timestamp": pd.date_range("2024-01-01", periods=100, freq="h")
    })
    engineer_features(dummy)