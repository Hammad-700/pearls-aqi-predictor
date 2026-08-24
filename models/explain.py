import os
import sys
import joblib
import tempfile
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

FEATURE_COLS = ["city_encoded", "hour", "day_of_week", "month",
                "aqi_lag_1h", "aqi_lag_24h", "aqi_roll_mean_24h",
                "aqi_change_rate", "temperature", "humidity"]
TARGET_COLS = ["aqi_d1", "aqi_d2", "aqi_d3"]

def load_champion():
    print("[INFO] Loading champion model from Supabase...")
    reg = supabase.table("model_registry")\
        .select("*").order("trained_at", desc=True).limit(1).execute()
    if not reg.data:
        print("[ERROR] No model in registry")
        return None, None, None
    meta = reg.data[0]
    version = meta["version"]
    print(f"[OK] Found champion: {meta['model_type']} version {version}")

    with tempfile.TemporaryDirectory() as tmp:
        model_bytes = supabase.storage.from_("models")\
            .download(f"champion_{version}.joblib")
        encoder_bytes = supabase.storage.from_("models")\
            .download(f"encoder_{version}.joblib")

        model_path = os.path.join(tmp, "model.joblib")
        encoder_path = os.path.join(tmp, "encoder.joblib")

        with open(model_path, "wb") as f:
            f.write(model_bytes)
        with open(encoder_path, "wb") as f:
            f.write(encoder_bytes)

        model = joblib.load(model_path)
        le = joblib.load(encoder_path)

    return model, le, meta

def load_test_data(le):
    result = supabase.table("aqi_gold_features")\
        .select("*")\
        .not_.is_("aqi_d1", "null")\
        .order("timestamp", desc=False)\
        .execute()
    df = pd.DataFrame(result.data)
    df["city_encoded"] = le.transform(df["city"])
    for col in FEATURE_COLS:
        if col == "city_encoded":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float)
    df = df.dropna(subset=FEATURE_COLS)
    split = int(len(df) * 0.8)
    return df[FEATURE_COLS].iloc[split:]

def explain(model, X_test):
    try:
        import shap
        print("[INFO] Using SHAP explainer...")

        horizon_importances = []
        for base_model in model.estimators_:
            if hasattr(base_model, "feature_importances_"):
                explainer = shap.TreeExplainer(base_model)
            else:
                explainer = shap.LinearExplainer(base_model, X_test)
            shap_values = explainer.shap_values(X_test)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            horizon_importances.append(np.abs(shap_values).mean(axis=0))
        mean_importance = np.mean(horizon_importances, axis=0)

        print("\n[SHAP] Mean feature importance across forecast horizons:")
        importance = pd.DataFrame({
            "feature": FEATURE_COLS,
            "mean_abs_shap": mean_importance
        }).sort_values("mean_abs_shap", ascending=False)

        for _, row in importance.iterrows():
            bar = "#" * int(row["mean_abs_shap"] * 2)
            print(f"  {row['feature']:25s} {row['mean_abs_shap']:.3f} {bar}")

        return importance

    except Exception as e:
        print(f"[WARN] SHAP failed: {e}")
        print("[INFO] Using fallback: Ridge coefficients as importance...")

        horizon_importances = []
        for base_model in model.estimators_:
            if hasattr(base_model, "feature_importances_"):
                horizon_importances.append(base_model.feature_importances_)
            elif hasattr(base_model, "coef_"):
                horizon_importances.append(np.abs(base_model.coef_))
        importance = pd.DataFrame({
            "feature": FEATURE_COLS,
            "mean_abs_shap": np.mean(horizon_importances, axis=0)
        }).sort_values("mean_abs_shap", ascending=False)

        print("\n[FALLBACK] Feature importance for aqi_d1 forecast:")
        for _, row in importance.iterrows():
            bar = "#" * int(row["mean_abs_shap"] * 2)
            print(f"  {row['feature']:25s} {row['mean_abs_shap']:.3f} {bar}")

        return importance

if __name__ == "__main__":
    model, le, meta = load_champion()
    if model:
        X_test = load_test_data(le)
        print(f"[INFO] Explaining on {len(X_test)} test rows")
        importance = explain(model, X_test)
        print("\n[DONE] Explainability complete!")