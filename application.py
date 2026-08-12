from pathlib import Path

from flask import Flask, render_template, request

from model_training import load_or_train_model

app = Flask(__name__)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

model, metrics = load_or_train_model()


@app.route("/")
def index():
    return render_template("index.html", metrics=metrics)


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    result = None

    if request.method == "POST":
        values = {
            "temperature_2m": float(request.form["temperature_2m"]),
            "relative_humidity_2m": float(request.form["relative_humidity_2m"]),
            "dew_point_2m": float(request.form["dew_point_2m"]),
            "precipitation": float(request.form["precipitation"]),
            "wind_speed_10m": float(request.form["wind_speed_10m"]),
            "wind_direction_10m": float(request.form["wind_direction_10m"]),
            "surface_pressure": float(request.form["surface_pressure"]),
            "cloud_cover": float(request.form["cloud_cover"]),
            "soil_moisture_0_to_7cm": float(request.form["soil_moisture_0_to_7cm"]),
            "rain_24h": float(request.form["rain_24h"]),
            "rain_72h": float(request.form["rain_72h"]),
            "rain_168h": float(request.form["rain_168h"]),
            "avg_temp_24h": float(request.form["avg_temp_24h"]),
            "avg_humidity_24h": float(request.form["avg_humidity_24h"]),
            "max_wind_24h": float(request.form["max_wind_24h"]),
        }

        probability = float(model.predict_proba([list(values.values())])[0][1])
        prediction = int(probability >= 0.5)

        result = {
            "prediction": prediction,
            "probability": probability * 100,
            "label": "Fire Risk Detected" if prediction else "Low Fire Risk",
        }

    return render_template("home.html", result=result, metrics=metrics)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
