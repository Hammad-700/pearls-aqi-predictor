import os
import sys
import joblib
import tempfile
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from supabase import create_client

load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
CORS(app)

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

FEATURE_COLS = ["city_encoded", "hour", "day_of_week", "month",
                "aqi_lag_1h", "aqi_lag_24h", "aqi_roll_mean_24h", "aqi_change_rate"]

model = None
le = None
model_version = None

def get_alert_level(aqi):
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Moderate"
    elif aqi <= 150: return "Unhealthy for Sensitive Groups"
    elif aqi <= 200: return "Unhealthy"
    elif aqi <= 300: return "Very Unhealthy"
    else: return "Hazardous"

def load_model():
    global model, le, model_version
    try:
        reg = supabase.table("model_registry")\
            .select("*").order("trained_at", desc=True).limit(1).execute()
        if not reg.data:
            print("[ERROR] No model in registry")
            return False
        meta = reg.data[0]
        model_version = meta["version"]

        with tempfile.TemporaryDirectory() as tmp:
            model_bytes = supabase.storage.from_("models")\
                .download(f"champion_{model_version}.joblib")
            encoder_bytes = supabase.storage.from_("models")\
                .download(f"encoder_{model_version}.joblib")

            model_path = os.path.join(tmp, "model.joblib")
            encoder_path = os.path.join(tmp, "encoder.joblib")

            with open(model_path, "wb") as f:
                f.write(model_bytes)
            with open(encoder_path, "wb") as f:
                f.write(encoder_bytes)

            model = joblib.load(model_path)
            le = joblib.load(encoder_path)

        print(f"[OK] Model loaded: {model_version}")
        return True
    except Exception as e:
        print(f"[ERROR] Model load failed: {e}")
        return False

# Load model on startup
load_model()

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_version": model_version,
        "model_loaded": model is not None
    })

@app.route("/predict")
def predict():
    city = request.args.get("city", "").lower().strip()
    if not city:
        return jsonify({"error": "city parameter required"}), 400

    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        # Check city is known
        if city not in le.classes_:
            return jsonify({"error": f"Unknown city: {city}. Known: {list(le.classes_)}"}), 400

        # Get latest gold row for city
        result = supabase.table("aqi_gold_features")\
            .select("*")\
            .eq("city", city)\
            .order("timestamp", desc=True)\
            .limit(1)\
            .execute()

        if not result.data:
            return jsonify({"error": f"No data for city: {city}"}), 404

        row = result.data[0]
        city_encoded = int(le.transform([city])[0])

        features = {
            "city_encoded": city_encoded,
            "hour": row.get("hour", 0),
            "day_of_week": row.get("day_of_week", 0),
            "month": row.get("month", 1),
            "aqi_lag_1h": row.get("aqi_lag_1h") or 0,
            "aqi_lag_24h": row.get("aqi_lag_24h") or 0,
            "aqi_roll_mean_24h": row.get("aqi_roll_mean_24h") or 0,
            "aqi_change_rate": row.get("aqi_change_rate") or 0,
        }

        X = pd.DataFrame([features])[FEATURE_COLS]
        preds = model.predict(X)[0]

        as_of = row["timestamp"]
        base_date = datetime.fromisoformat(as_of.replace("Z", "+00:00"))

        forecast = []
        for i, aqi_val in enumerate(preds):
            day_date = (base_date + timedelta(days=i+1)).strftime("%Y-%m-%d")
            forecast.append({
                "day": i+1,
                "date": day_date,
                "aqi": int(round(aqi_val))
            })

        max_aqi = max(f["aqi"] for f in forecast)

        return jsonify({
            "city": city,
            "as_of": as_of,
            "forecast": forecast,
            "max_aqi": max_aqi,
            "alert_level": get_alert_level(max_aqi)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/history")
def history():
    city = request.args.get("city", "").lower().strip()
    if not city:
        return jsonify({"error": "city parameter required"}), 400
    try:
        result = supabase.table("aqi_gold_features")\
            .select("timestamp,aqi")\
            .eq("city", city)\
            .order("timestamp", desc=True)\
            .limit(24)\
            .execute()
        return jsonify({"city": city, "history": result.data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)