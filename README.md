# India Forest Fire Prediction

A mini-project that adapts the original forest-fire classification workflow from the Algerian Forest Fires dataset to India-focused satellite fire observations and weather data.

## Project objective

The objective is to build a practical machine-learning classifier that estimates whether fire risk is present from local weather and recent weather conditions. The project keeps the original application's Flask architecture and compares Logistic Regression with Random Forest.

This is a **mini-project / prototype**, not an operational nationwide fire-warning system.

## Dataset

### Fire observations

Fire observations are sourced from **NASA FIRMS VIIRS Suomi-NPP** data for the India region, covering 2018–2025.

The raw FIRMS archive used for the project contains approximately **7.9 million observations**. A reproducible subset was prepared for the mini-project and combined with historical weather observations.

### Weather data

Historical hourly weather conditions were obtained from the **Open-Meteo Historical Weather API** for the sampled fire and non-fire observations.

The current trained model uses weather variables rather than satellite fire-intensity variables such as FRP, which would leak information about an already detected fire.

## Features

The classifier uses 15 weather/environmental features:

- `temperature_2m`
- `relative_humidity_2m`
- `dew_point_2m`
- `precipitation`
- `wind_speed_10m`
- `wind_direction_10m`
- `surface_pressure`
- `cloud_cover`
- `soil_moisture_0_to_7cm`
- `rain_24h`
- `rain_72h`
- `rain_168h`
- `avg_temp_24h`
- `avg_humidity_24h`
- `max_wind_24h`

The rolling weather features represent recent weather conditions associated with each observation.

## Machine-learning models

Two models are evaluated:

1. **Logistic Regression** with feature standardization and balanced class weighting.
2. **Random Forest Classifier** with balanced class weighting.

The model with the better F1-score is selected for the application.

### Current result

Using the partial weather-matched mini-project dataset, Random Forest was selected:

| Metric | Score |
|---|---:|
| Accuracy | **84.74%** |
| Precision | **67.12%** |
| Recall | **93.51%** |
| F1-score | **78.15%** |
| ROC-AUC | **93.17%** |

These are **prototype results** from the current mini-project dataset and should not be interpreted as nationwide operational accuracy.

## Application

The project includes a Flask web application where the user supplies the weather inputs and receives:

- Fire-risk classification
- Predicted probability

## Project structure

```text
india-forest-fire-prediction/
├── application.py
├── model_training.py
├── prepare_india_dataset.py
├── build_training_dataset.py
├── requirements.txt
├── README.md
├── models/
│   ├── fire_classifier.joblib
│   └── metrics.json
└── templates/
    ├── index.html
    └── home.html
```

The historical raw FIRMS CSV, weather cache, generated datasets, Python caches, and local environment files are intentionally kept out of the GitHub repository because they are large/generated artifacts.

## How to run

### 1. Clone the repository

```bash
git clone https://github.com/neeravjain91-jpg/india-forest-fire-prediction.git
cd india-forest-fire-prediction
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Run the trained application

```bash
python application.py
```

Open:

```text
http://127.0.0.1:5000
```

### 4. Retrain the model

The training script expects `india_forest_fire_dataset.csv` to be present locally.

```bash
python model_training.py
```

## Screenshots

Add the following screenshots to `docs/screenshots/` and update this section with the actual image links:

- `home.png` — project landing page
- `prediction-form.png` — weather input form
- `prediction-result.png` — prediction result

Example Markdown:

```markdown
![Home page](docs/screenshots/home.png)
![Prediction form](docs/screenshots/prediction-form.png)
![Prediction result](docs/screenshots/prediction-result.png)
```

## Limitations

- The current mini-project uses a sampled training dataset rather than all 7.9 million raw FIRMS observations.
- The current evaluation is a standard stratified train/test split; it is not a strict geographic or time-forward validation study.
- The system is a prototype and should not be used as an operational emergency fire-warning system.

## Future scope

The separate advanced track is being developed independently using the full multi-year FIRMS archive and a larger historical weather dataset. That research version can later add stronger temporal/geographic validation, larger training data, explainability, and more advanced models.

## License

Use according to the licensing and attribution requirements of the underlying NASA FIRMS and weather data sources.
