import os
import sys
import uuid
import joblib
import tempfile
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv
from supabase import create_client
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

try:
    from xgboost import XGBRegressor
    XGB_OK = True
except ImportError:
    XGB_OK = False
    print("[WARN] XGBoost not available")

try:
    from lightgbm import LGBMRegressor
    LGB_OK = True
except ImportError:
    LGB_OK = False
    print("[WARN] LightGBM not available - using HistGradientBoosting fallback")

FEATURE_COLS = ["city_encoded", "hour", "day_of_week", "month",
                "aqi_lag_1h", "aqi_lag_24h", "aqi_roll_mean_24h", "aqi_change_rate"]
TARGET_COLS = ["aqi_d1", "aqi_d2", "aqi_d3"]


def load_gold_data():
    print("[INFO] Loading Gold features from Supabase...")
    all_rows = []
    page_size = 1000
    offset = 0
    while True:
        result = supabase.table("aqi_gold_features")\
            .select("*")\
            .not_.is_("aqi_d1", "null")\
            .not_.is_("aqi_d2", "null")\
            .not_.is_("aqi_d3", "null")\
            .order("timestamp", desc=False)\
            .range(offset, offset + page_size - 1)\
            .execute()
        rows = result.data
        all_rows.extend(rows)
        if len(rows) < page_size:
            break
        offset += page_size
    print(f"[OK] Loaded {len(all_rows)} rows")
    return pd.DataFrame(all_rows)


def prepare_data(df):
    le = LabelEncoder()
    df["city_encoded"] = le.fit_transform(df["city"])
    df = df.dropna(subset=FEATURE_COLS + TARGET_COLS)
    df = df.sort_values("timestamp").reset_index(drop=True)
    X = df[FEATURE_COLS]
    y = df[TARGET_COLS]
    split = int(len(df) * 0.8)
    return X[:split], X[split:], y[:split], y[split:], le


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    metrics = {}
    for i, col in enumerate(TARGET_COLS):
        rmse = float(np.sqrt(mean_squared_error(y_test.iloc[:, i], preds[:, i])))
        mae = float(mean_absolute_error(y_test.iloc[:, i], preds[:, i]))
        r2 = float(r2_score(y_test.iloc[:, i], preds[:, i]))
        metrics[f"rmse_{col}"] = rmse
        metrics[f"mae_{col}"] = mae
        metrics[f"r2_{col}"] = r2
        print(f"  {col}: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.3f}")
    metrics["avg_rmse"] = float(np.mean([metrics[f"rmse_{c}"] for c in TARGET_COLS]))
    print(f"  AVG RMSE: {metrics['avg_rmse']:.2f}")
    return metrics


def get_models():
    models = {
        "ridge": MultiOutputRegressor(Ridge()),
        "random_forest": MultiOutputRegressor(
            RandomForestRegressor(n_estimators=30, max_depth=6, random_state=42)
        ),
    }
    if XGB_OK:
        models["xgboost"] = MultiOutputRegressor(
            XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        )
    if LGB_OK:
        models["lightgbm"] = MultiOutputRegressor(
            LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        )
    else:
        models["histgradient"] = MultiOutputRegressor(
            HistGradientBoostingRegressor(random_state=42)
        )
    return models


def save_champion(model, le, version, metrics, model_type):
    print(f"\n[INFO] Saving champion ({model_type}) to Supabase Storage...")
    with tempfile.TemporaryDirectory() as tmp:
        model_path = os.path.join(tmp, f"champion_{version}.joblib")
        encoder_path = os.path.join(tmp, f"encoder_{version}.joblib")
        joblib.dump(model, model_path)
        joblib.dump(le, encoder_path)

        with open(model_path, "rb") as f:
            supabase.storage.from_("models").upload(
                f"champion_{version}.joblib", f.read(),
                {"content-type": "application/octet-stream", "upsert": "true"}
            )
        with open(encoder_path, "rb") as f:
            supabase.storage.from_("models").upload(
                f"encoder_{version}.joblib", f.read(),
                {"content-type": "application/octet-stream", "upsert": "true"}
            )

    supabase.table("model_registry").insert({
        "version": version,
        "model_type": model_type,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "feature_columns": FEATURE_COLS,
        "lib_versions": f"sklearn={__import__('sklearn').__version__}"
    }).execute()

    print(f"[OK] Champion saved as champion_{version}.joblib")


def run_training():
    df = load_gold_data()
    if len(df) < 100:
        print("[ERROR] Not enough data - run backfill first")
        return

    X_train, X_test, y_train, y_test, le = prepare_data(df)
    print(f"[INFO] Train: {len(X_train)} rows, Test: {len(X_test)} rows")

    models = get_models()
    results = {}
    run_id = str(uuid.uuid4())[:8]

    for name, model in models.items():
        print(f"\n[TRAINING] {name}...")
        model.fit(X_train, y_train)
        print(f"[EVAL] {name}:")
        metrics = evaluate(model, X_test, y_test)
        results[name] = {"model": model, "metrics": metrics}

        supabase.table("training_runs").insert({
            "run_id": run_id,
            "model_type": name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "rmse_d1": metrics["rmse_aqi_d1"],
            "rmse_d2": metrics["rmse_aqi_d2"],
            "rmse_d3": metrics["rmse_aqi_d3"],
            "mae_d1": metrics["mae_aqi_d1"],
            "mae_d2": metrics["mae_aqi_d2"],
            "mae_d3": metrics["mae_aqi_d3"],
            "r2_d1": metrics["r2_aqi_d1"],
            "r2_d2": metrics["r2_aqi_d2"],
            "r2_d3": metrics["r2_aqi_d3"],
        }).execute()

    champion_name = min(results, key=lambda k: results[k]["metrics"]["avg_rmse"])
    champion = results[champion_name]
    print(f"\n[CHAMPION] {champion_name} avg RMSE: {champion['metrics']['avg_rmse']:.2f}")

    # Champion gate — only save if better than existing champion
    try:
        existing = supabase.table("model_registry")\
            .select("metrics,version")\
            .order("trained_at", desc=True)\
            .limit(1).execute()
        
        if existing.data:
            existing_rmse = existing.data[0]["metrics"].get("avg_rmse", 999)
            new_rmse = champion["metrics"]["avg_rmse"]
            print(f"\n[GATE] Existing RMSE: {existing_rmse:.2f} | New RMSE: {new_rmse:.2f}")
            
            if new_rmse < existing_rmse:
                print(f"[GATE] New model is better — promoting to champion!")
                version = f"v{datetime.now(timezone.utc).strftime('%Y%m%d%H%M')}"
                save_champion(champion["model"], le, version, champion["metrics"], champion_name)
                print(f"\n[DONE] New champion saved! Version: {version}")
            else:
                print(f"[GATE] Existing model is better — keeping current champion!")
                print(f"\n[DONE] Training complete — no champion update needed.")
        else:
            version = f"v{datetime.now(timezone.utc).strftime('%Y%m%d%H%M')}"
            save_champion(champion["model"], le, version, champion["metrics"], champion_name)
            print(f"\n[DONE] First champion saved! Version: {version}")
    except Exception as e:
        print(f"[WARN] Champion gate failed: {e} — saving anyway")
        version = f"v{datetime.now(timezone.utc).strftime('%Y%m%d%H%M')}"
        save_champion(champion["model"], le, version, champion["metrics"], champion_name)
        print(f"\n[DONE] Training complete! Version: {version}")


if __name__ == "__main__":
    run_training()