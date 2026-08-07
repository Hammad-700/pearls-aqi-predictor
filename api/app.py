import os
import sys
import pickle
import pandas as pd
from flask import Flask, jsonify, request
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pipelines'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'features'))

app = Flask(__name__)
os.chdir(os.path.join(os.path.dirname(__file__), '..'))


def get_predictions(city="Lahore"):
    from inference_pipeline import predict
    return predict(city)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat()})


@app.route("/predict", methods=["GET"])
def predict_aqi():
    city = request.args.get("city", "Lahore")
    try:
        df = get_predictions(city)
        return jsonify({
            "city": city,
            "predictions": df.to_dict(orient="records")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/history", methods=["GET"])
def history():
    try:
        df = pd.read_csv("data/historical_data.csv")
        df = df.tail(168)
        df = df[["timestamp", "aqi"]].dropna()
        df["timestamp"] = df["timestamp"].astype(str)
        return jsonify({
            "history": df.to_dict(orient="records")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/current", methods=["GET"])
def current():
    try:
        from fetch_aqi import fetch_aqi
        from fetch_weather import fetch_weather
        city = request.args.get("city", "Lahore")
        aqi = fetch_aqi(city)
        weather = fetch_weather(city)
        return jsonify({**aqi, **weather})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
