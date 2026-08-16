from pathlib import Path

from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

MODEL_PATH = Path("models/final_hgb_model.joblib")
METRICS_PATH = Path("models/final_metrics.json")

FEATURES = [
    "grid_lat", "grid_lon", "hour", "year", "month",
    "temp_1d", "rh_1d", "wind_1d", "pressure_1d", "soil_1d", "rain_1d",
    "temp_3d_mean", "temp_3d_max", "temp_3d_min", "rh_3d_mean", "rh_3d_min",
    "wind_3d_mean", "wind_3d_max", "pressure_3d_mean", "soil_3d_mean", "rain_3d_total",
    "temp_7d_mean", "temp_7d_max", "temp_7d_min", "rh_7d_mean", "rh_7d_min",
    "wind_7d_mean", "wind_7d_max", "pressure_7d_mean", "soil_7d_mean", "rain_7d_total",
]

MODEL = joblib.load(MODEL_PATH)

try:
    import json
    METRICS = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}
except Exception:
    METRICS = {}


def numeric_form(name: str) -> float:
    return float(request.form[name])


@app.route("/", methods=["GET"])
def index():
    return render_template("research_model.html", result=None, metrics=METRICS)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = {name: numeric_form(name) for name in FEATURES}
        probability = float(MODEL.predict_proba([[values[name] for name in FEATURES]])[0][1])
        prediction = int(probability >= 0.5)
        result = {
            "prediction": prediction,
            "probability": probability * 100.0,
            "label": "Fire Detected Risk" if prediction else "Lower Fire Risk",
        }
        return render_template("research_model.html", result=result, metrics=METRICS)
    except (KeyError, ValueError, TypeError) as exc:
        return render_template(
            "research_model.html",
            result={"error": f"Invalid input: {exc}"},
            metrics=METRICS,
        ), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
