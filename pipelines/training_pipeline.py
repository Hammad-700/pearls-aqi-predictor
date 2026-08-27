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
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import TimeSeriesSplit

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set.")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

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


# ---------------------------------------------------------------------
# Model schema
# ---------------------------------------------------------------------

FEATURE_COLS = [
    "city_encoded",
    "hour",
    "day_of_week",
    "month",
    "aqi_lag_1h",
    "aqi_lag_24h",
    "aqi_roll_mean_24h",
    "aqi_change_rate",
    "temperature",
    "humidity",
    "pm25",
    "wind_speed",
    "wind_direction",
    "precipitation",
    "pressure",
    "pm25_raw",
    "pm10_raw",
    "no2_raw",
    "o3_raw",
]

TARGET_COLS = ["aqi_d1", "aqi_d2", "aqi_d3"]

WEATHER_COLS = [
    "temperature",
    "humidity",
    "wind_speed",
    "wind_direction",
    "precipitation",
    "pressure",
    "pm25_raw",
    "pm10_raw",
    "no2_raw",
    "o3_raw",
]

MIN_WEATHER_COVERAGE = 0.80


# ---------------------------------------------------------------------
# Load labeled Gold data
# ---------------------------------------------------------------------

def load_gold_data():
    print("[INFO] Loading labeled Gold features from Supabase...")

    all_rows = []
    page_size = 1000
    offset = 0

    while True:
        result = (
            supabase.table("aqi_gold_features")
            .select("*")
            .not_.is_("aqi_d1", "null")
            .not_.is_("aqi_d2", "null")
            .not_.is_("aqi_d3", "null")
            .order("timestamp", desc=False)
            .range(offset, offset + page_size - 1)
            .execute()
        )

        rows = result.data or []
        all_rows.extend(rows)

        if len(rows) < page_size:
            break

        offset += page_size

    print(f"[OK] Loaded {len(all_rows)} labeled rows")
    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------
# Prepare data
# ---------------------------------------------------------------------

def prepare_data(df):
    df = df.copy()

    if df.empty:
        raise RuntimeError("Gold dataset is empty.")

    # Stable city encoder.
    le = LabelEncoder()
    df["city_encoded"] = le.fit_transform(df["city"].astype(str))

    # Convert timestamp for chronological sorting.
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    df = df.dropna(subset=["timestamp"])

    # Numeric conversion.
    for col in FEATURE_COLS:
        if col == "city_encoded":
            continue

        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Targets must exist for supervised training.
    df = df.dropna(subset=TARGET_COLS)

    if df.empty:
        raise RuntimeError("No rows remain after removing NULL targets.")

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Chronological 80/20 split.
    split = int(len(df) * 0.8)

    if split < 2 or len(df) - split < 1:
        raise RuntimeError("Not enough rows for train/test split.")

    X_train_raw = df.iloc[:split][FEATURE_COLS].copy()
    X_test_raw = df.iloc[split:][FEATURE_COLS].copy()

    y_train = df.iloc[:split][TARGET_COLS].copy()
    y_test = df.iloc[split:][TARGET_COLS].copy()

    # Median imputation learned ONLY from training data.
    # This avoids treating missing temperature/pressure/etc. as zero.
    feature_medians = X_train_raw.median(numeric_only=True)

    X_train = X_train_raw.fillna(feature_medians)
    X_test = X_test_raw.fillna(feature_medians)

    # Any column that is entirely missing in training gets zero only
    # as a final fallback.
    X_train = X_train.fillna(0).astype(float)
    X_test = X_test.fillna(0).astype(float)

    return X_train, X_test, y_train, y_test, le, feature_medians


# ---------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------

def cross_validate_model(model, X, y, name):
    if len(X) < 10:
        print(f"  [WARN] Too few rows for 5-fold CV: {len(X)}")
        return float("nan"), float("nan")

    n_splits = min(5, len(X) - 1)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmse_scores = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_tr = X.iloc[train_idx]
        X_te = X.iloc[test_idx]
        y_tr = y.iloc[train_idx]
        y_te = y.iloc[test_idx]

        # Fresh model for each fold when possible.
        # sklearn estimators support get_params/deep cloning.
        from sklearn.base import clone
        fold_model = clone(model)

        fold_model.fit(X_tr, y_tr)
        preds = fold_model.predict(X_te)

        rmse = float(np.sqrt(mean_squared_error(y_te, preds)))
        rmse_scores.append(rmse)

        print(f"    Fold {fold}: RMSE={rmse:.2f}")

    mean_rmse = float(np.mean(rmse_scores))
    std_rmse = float(np.std(rmse_scores))

    print(
        f"  CV RMSE: {mean_rmse:.2f} ± {std_rmse:.2f} "
        f"({n_splits}-fold TimeSeriesSplit)"
    )

    return mean_rmse, std_rmse


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)

    metrics = {}

    for i, col in enumerate(TARGET_COLS):
        rmse = float(
            np.sqrt(
                mean_squared_error(
                    y_test.iloc[:, i],
                    preds[:, i],
                )
            )
        )

        mae = float(
            mean_absolute_error(
                y_test.iloc[:, i],
                preds[:, i],
            )
        )

        r2 = float(
            r2_score(
                y_test.iloc[:, i],
                preds[:, i],
            )
        )

        metrics[f"rmse_{col}"] = rmse
        metrics[f"mae_{col}"] = mae
        metrics[f"r2_{col}"] = r2

        print(
            f"  {col}: "
            f"RMSE={rmse:.2f}, "
            f"MAE={mae:.2f}, "
            f"R2={r2:.3f}"
        )

    metrics["avg_rmse"] = float(
        np.mean(
            [
                metrics[f"rmse_{col}"]
                for col in TARGET_COLS
            ]
        )
    )

    print(f"  AVG RMSE: {metrics['avg_rmse']:.2f}")

    return metrics


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------

def get_models():
    models = {
        "naive_baseline": MultiOutputRegressor(
            DummyRegressor(strategy="mean")
        ),

        "ridge": MultiOutputRegressor(
            Ridge()
        ),

        "random_forest": MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                random_state=42,
                n_jobs=-1,
            )
        ),
    }

    if XGB_OK:
        models["xgboost"] = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                verbosity=0,
            )
        )

    if LGB_OK:
        models["lightgbm"] = MultiOutputRegressor(
            LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42,
                verbose=-1,
            )
        )
    else:
        models["histgradient"] = MultiOutputRegressor(
            HistGradientBoostingRegressor(
                random_state=42
            )
        )

    return models


# ---------------------------------------------------------------------
# Save champion
# ---------------------------------------------------------------------

def save_champion(
    model,
    le,
    version,
    metrics,
    model_type,
    feature_medians,
):
    print(
        f"\n[INFO] Saving champion ({model_type}) "
        f"to Supabase Storage..."
    )

    with tempfile.TemporaryDirectory() as tmp:
        model_path = os.path.join(
            tmp,
            f"champion_{version}.joblib",
        )

        encoder_path = os.path.join(
            tmp,
            f"encoder_{version}.joblib",
        )

        imputer_path = os.path.join(
            tmp,
            f"imputer_{version}.joblib",
        )

        joblib.dump(model, model_path)
        joblib.dump(le, encoder_path)
        joblib.dump(feature_medians.to_dict(), imputer_path)

        with open(model_path, "rb") as f:
            supabase.storage.from_("models").upload(
                f"champion_{version}.joblib",
                f.read(),
                {
                    "content-type": "application/octet-stream",
                    "upsert": "true",
                },
            )

        with open(encoder_path, "rb") as f:
            supabase.storage.from_("models").upload(
                f"encoder_{version}.joblib",
                f.read(),
                {
                    "content-type": "application/octet-stream",
                    "upsert": "true",
                },
            )

        with open(imputer_path, "rb") as f:
            supabase.storage.from_("models").upload(
                f"imputer_{version}.joblib",
                f.read(),
                {
                    "content-type": "application/octet-stream",
                    "upsert": "true",
                },
            )

    registry_payload = {
        "version": version,
        "model_type": model_type,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "feature_columns": FEATURE_COLS,
        "lib_versions": (
            f"sklearn={__import__('sklearn').__version__}"
        ),
    }

    supabase.table("model_registry").insert(
        registry_payload
    ).execute()

    print(
        f"[OK] Champion saved: champion_{version}.joblib"
    )


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def run_training():
    df = load_gold_data()

    if len(df) < 100:
        print(
            f"[ERROR] Not enough labeled data ({len(df)} rows). "
            "Run backfill first."
        )
        return

    # Weather completeness check.
    missing_weather = (
        df[WEATHER_COLS]
        .isna()
        .all(axis=1)
    )

    weather_coverage = float(
        (~missing_weather).mean()
    )

    print(
        f"[INFO] Weather feature coverage: "
        f"{weather_coverage:.1%}"
    )

    # Keep the original strict requirement that most rows have complete
    # weather observations. Individual missing feature values are later
    # median-imputed from the training set.
    complete_weather_coverage = float(
        df[WEATHER_COLS]
        .notna()
        .all(axis=1)
        .mean()
    )

    print(
        f"[INFO] Complete weather coverage: "
        f"{complete_weather_coverage:.1%}"
    )

    if complete_weather_coverage < MIN_WEATHER_COVERAGE:
        raise RuntimeError(
            f"Only {complete_weather_coverage:.1%} of labeled rows "
            f"have complete weather features; need at least "
            f"{MIN_WEATHER_COVERAGE:.0%} before training."
        )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        le,
        feature_medians,
    ) = prepare_data(df)

    print(
        f"[INFO] Train: {len(X_train)} rows | "
        f"Test: {len(X_test)} rows"
    )

    models = get_models()
    results = {}
    run_id = str(uuid.uuid4())[:8]

    # ---------------------------------------------------------------
    # Train and evaluate each model
    # ---------------------------------------------------------------

    for name, model in models.items():
        print(f"\n[TRAINING] {name}...")

        model.fit(X_train, y_train)

        print(f"[EVAL] {name}:")
        metrics = evaluate(
            model,
            X_test,
            y_test,
        )

        # CV uses a fresh clone for every fold.
        cv_mean, cv_std = cross_validate_model(
            model,
            pd.concat(
                [X_train, X_test],
                ignore_index=True,
            ),
            pd.concat(
                [y_train, y_test],
                ignore_index=True,
            ),
            name,
        )

        metrics["cv_rmse_mean"] = cv_mean
        metrics["cv_rmse_std"] = cv_std

        results[name] = {
            "model": model,
            "metrics": metrics,
        }

        supabase.table("training_runs").insert(
            {
                "run_id": run_id,
                "model_type": name,
                "timestamp": datetime.now(
                    timezone.utc
                ).isoformat(),
                "rmse_d1": metrics["rmse_aqi_d1"],
                "rmse_d2": metrics["rmse_aqi_d2"],
                "rmse_d3": metrics["rmse_aqi_d3"],
                "mae_d1": metrics["mae_aqi_d1"],
                "mae_d2": metrics["mae_aqi_d2"],
                "mae_d3": metrics["mae_aqi_d3"],
                "r2_d1": metrics["r2_aqi_d1"],
                "r2_d2": metrics["r2_aqi_d2"],
                "r2_d3": metrics["r2_aqi_d3"],
            }
        ).execute()

    # ---------------------------------------------------------------
    # Select champion
    # ---------------------------------------------------------------

    champion_name = min(
        results,
        key=lambda name: results[name]["metrics"]["avg_rmse"],
    )

    champion = results[champion_name]

    print(
        f"\n[CHAMPION] {champion_name} | "
        f"avg RMSE: "
        f"{champion['metrics']['avg_rmse']:.2f}"
    )

    # ---------------------------------------------------------------
    # IMPORTANT:
    # Refit the selected champion on ALL labeled data.
    #
    # The old version could save a model left fitted on the final
    # cross-validation fold. This version trains the final champion
    # on every available labeled row.
    # ---------------------------------------------------------------

    X_full = pd.concat(
        [X_train, X_test],
        ignore_index=True,
    )

    y_full = pd.concat(
        [y_train, y_test],
        ignore_index=True,
    )

    champion["model"].fit(
        X_full,
        y_full,
    )

    print(
        f"[OK] Champion refitted on "
        f"{len(X_full)} labeled rows."
    )

    # ---------------------------------------------------------------
    # Champion gate
    # ---------------------------------------------------------------

    try:
        existing = (
            supabase.table("model_registry")
            .select("metrics,version,feature_columns")
            .order("trained_at", desc=True)
            .limit(1)
            .execute()
        )

        if existing.data:
            existing_row = existing.data[0]

            existing_metrics = (
                existing_row.get("metrics") or {}
            )

            existing_rmse = float(
                existing_metrics.get(
                    "avg_rmse",
                    999,
                )
            )

            existing_features = (
                existing_row.get("feature_columns") or []
            )

            new_rmse = float(
                champion["metrics"]["avg_rmse"]
            )

            print(
                f"\n[GATE] Existing RMSE: "
                f"{existing_rmse:.2f} | "
                f"New RMSE: {new_rmse:.2f}"
            )

            feature_schema_changed = (
                existing_features != FEATURE_COLS
            )

            if feature_schema_changed or new_rmse < existing_rmse:
                if feature_schema_changed:
                    print(
                        "[GATE] Feature schema changed — "
                        "promoting new champion."
                    )
                else:
                    print(
                        "[GATE] New model is better — "
                        "promoting champion."
                    )

                version = (
                    f"v{datetime.now(timezone.utc).strftime('%Y%m%d%H%M')}"
                )

                save_champion(
                    champion["model"],
                    le,
                    version,
                    champion["metrics"],
                    champion_name,
                    feature_medians,
                )

                print(
                    f"\n[DONE] New champion saved: "
                    f"{version}"
                )

            else:
                print(
                    "[GATE] Existing model is better — "
                    "keeping current champion."
                )

                print(
                    "\n[DONE] Training complete — "
                    "no champion update needed."
                )

        else:
            version = (
                f"v{datetime.now(timezone.utc).strftime('%Y%m%d%H%M')}"
            )

            save_champion(
                champion["model"],
                le,
                version,
                champion["metrics"],
                champion_name,
                feature_medians,
            )

            print(
                f"\n[DONE] First champion saved: "
                f"{version}"
            )

    except Exception as exc:
        print(
            f"[WARN] Champion gate failed: {exc} "
            "— saving anyway."
        )

        version = (
            f"v{datetime.now(timezone.utc).strftime('%Y%m%d%H%M')}"
        )

        save_champion(
            champion["model"],
            le,
            version,
            champion["metrics"],
            champion_name,
            feature_medians,
        )

        print(
            f"\n[DONE] Training complete: "
            f"{version}"
        )


if __name__ == "__main__":
    run_training()
