# India Forest Fire Prediction

India-focused adaptation of the existing forest-fire classification project.

This repository preserves the original project's overall structure and model approach while replacing the Algerian Forest Fires dataset with India/South Asia FIRMS-derived data.

## Approach

- Keep the existing Flask application structure.
- Keep Logistic Regression and Random Forest as the initial candidate models.
- Adapt the feature layer to the India FIRMS + weather dataset.
- Avoid using satellite fire-intensity variables such as FRP as prediction inputs when they would leak the target.
- Keep model evaluation with accuracy, precision, recall, F1, and ROC-AUC.

## Data

The prototype India dataset currently available is:

`J2_VIIRS_C2_SouthEast_Asia_48h_with_weather.csv`

Historical multi-year FIRMS data is being downloaded separately and is not required for this repository's initial adaptation.
