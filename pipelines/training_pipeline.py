import pandas as pd
import numpy as np
import os
import pickle
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

def load_data():
    df = pd.read_csv("data/historical_data.csv")
    df = df.dropna(subset=["target_aqi_72h"])

    feature_cols = [
        "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
        "temperature", "humidity", "pressure", "wind_speed",
        "hour", "day", "month", "day_of_week", "is_weekend",
        "aqi_lag_1", "aqi_lag_3", "aqi_lag_6", "aqi_lag_24",
        "aqi_rolling_mean_6", "aqi_rolling_std_6", "aqi_rolling_mean_24",
        "aqi_change_rate"
    ]

    # Keep only available columns
    feature_cols = [c for c in feature_cols if c in df.columns]
    df = df.dropna(subset=feature_cols)

    X = df[feature_cols]
    y = df["target_aqi_72h"]
    return X, y, feature_cols

def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)
    print(f"\n{name}")
    print(f"  RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.4f}")
    return rmse, mae, r2

def train():
    print("Loading data...")
    X, y, feature_cols = load_data()
    print(f"Dataset: {X.shape[0]} rows, {X.shape[1]} features")

    # Time-based split (no shuffle — respects time order)
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    results = {}

    # Model 1: Ridge Regression
    print("\nTraining Ridge Regression...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    rmse, mae, r2 = evaluate("Ridge Regression", ridge, X_test_scaled, y_test)
    results["ridge"] = (ridge, rmse)

    # Model 2: Random Forest
    print("Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rmse, mae, r2 = evaluate("Random Forest", rf, X_test, y_test)
    results["random_forest"] = (rf, rmse)

    # Model 3: MLP Neural Network
    print("Training MLP Neural Network...")
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    mlp.fit(X_train_scaled, y_train)
    rmse, mae, r2 = evaluate("MLP Neural Network", mlp, X_test_scaled, y_test)
    results["mlp"] = (mlp, rmse)

    # Pick best model
    best_name = min(results, key=lambda k: results[k][1])
    best_model = results[best_name][0]
    print(f"\n✅ Best model: {best_name} (RMSE: {results[best_name][1]:.2f})")

    # Save models
    os.makedirs("models", exist_ok=True)
    pickle.dump(best_model, open("models/best_model.pkl", "wb"))
    pickle.dump(scaler, open("models/scaler.pkl", "wb"))
    pickle.dump(feature_cols, open("models/feature_cols.pkl", "wb"))
    print("✅ Models saved to models/")

if __name__ == "__main__":
    train()
    