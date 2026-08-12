from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("india_forest_fire_dataset.csv")
MODEL_PATH = Path("models/fire_classifier.joblib")
METRICS_PATH = Path("models/metrics.json")

FEATURES = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
    "wind_speed_10m", "wind_direction_10m", "surface_pressure", "cloud_cover",
    "soil_moisture_0_to_7cm", "rain_24h", "rain_72h", "rain_168h",
    "avg_temp_24h", "avg_humidity_24h", "max_wind_24h",
]


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}. Run prepare_india_dataset.py first.")
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    missing = [c for c in FEATURES + ["fire"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df["fire"] = pd.to_numeric(df["fire"], errors="coerce")
    df = df.dropna(subset=FEATURES + ["fire"])
    df["fire"] = df["fire"].astype(int)
    if set(df["fire"].unique()) != {0, 1}:
        raise ValueError("Training target must contain both fire=0 and fire=1.")
    return df


def train_models():
    df = load_data()
    X, y = df[FEATURES], df["fire"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    candidates = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, max_depth=8, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
    }

    results, fitted = {}, {}
    for name, estimator in candidates.items():
        estimator.fit(X_train, y_train)
        probabilities = estimator.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        results[name] = {
            "accuracy": round(accuracy_score(y_test, predictions), 4),
            "precision": round(precision_score(y_test, predictions, zero_division=0), 4),
            "recall": round(recall_score(y_test, predictions, zero_division=0), 4),
            "f1": round(f1_score(y_test, predictions, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_test, probabilities), 4),
        }
        fitted[name] = estimator

    best_name = max(results, key=lambda name: results[name]["f1"])
    best_model = fitted[best_name]
    best_model.fit(X, y)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    metrics = {"selected_model": best_name, "features": FEATURES, "test_size": 0.20, "random_state": 42, "models": results}
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return best_model, metrics


def load_or_train_model():
    if MODEL_PATH.exists() and METRICS_PATH.exists():
        return joblib.load(MODEL_PATH), json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return train_models()


if __name__ == "__main__":
    model, metrics = train_models()
    print(f"Selected model: {metrics['selected_model']}")
    print(metrics["models"][metrics["selected_model"]])
