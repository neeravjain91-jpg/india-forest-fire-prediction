from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("J2_VIIRS_C2_SouthEast_Asia_48h_with_weather.csv")
MODEL_PATH = Path("models/fire_classifier.joblib")
METRICS_PATH = Path("models/metrics.json")

# Prototype prediction features available in the FIRMS + weather file.
# Satellite fire-intensity fields are deliberately excluded because they
# describe an already-detected fire and would leak the target in a
# fire-occurrence classifier.
FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "cloud_cover",
    "soil_moisture_0_to_7cm",
    "rain_24h",
    "rain_72h",
    "rain_168h",
    "avg_temp_24h",
    "avg_humidity_24h",
    "max_wind_24h",
]


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. "
            "Copy the India FIRMS + weather CSV into the repository root."
        )

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    missing = [column for column in FEATURES if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    # The current FIRMS prototype contains detections only, so there is no
    # legitimate negative class yet. Stop instead of silently training a
    # meaningless one-class classifier.
    if "fire" not in df.columns:
        raise ValueError(
            "The prototype dataset contains FIRMS fire detections only. "
            "Create fire=0 non-fire samples before training this classifier."
        )

    df["fire"] = pd.to_numeric(df["fire"], errors="coerce")
    df = df.dropna(subset=FEATURES + ["fire"])
    df["fire"] = df["fire"].astype(int)

    classes = set(df["fire"].unique())
    if not classes.issuperset({0, 1}):
        raise ValueError(
            "The target must contain both fire=0 and fire=1 classes before training."
        )

    return df


def train_models():
    df = load_data()
    X = df[FEATURES]
    y = df["fire"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    candidates = {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = {}
    fitted = {}

    for name, estimator in candidates.items():
        estimator.fit(X_train, y_train)
        probabilities = estimator.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        results[name] = {
            "accuracy": round(accuracy_score(y_test, predictions), 4),
            "precision": round(
                precision_score(y_test, predictions, zero_division=0), 4
            ),
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

    metrics = {
        "selected_model": best_name,
        "features": FEATURES,
        "test_size": 0.20,
        "random_state": 42,
        "models": results,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return best_model, metrics


if __name__ == "__main__":
    model, metrics = train_models()
    print(f"Selected model: {metrics['selected_model']}")
    print(metrics["models"][metrics["selected_model"]])
