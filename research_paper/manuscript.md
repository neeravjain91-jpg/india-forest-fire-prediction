# India Forest Fire Prediction Using VIIRS FIRMS and Meteorological Features

## Abstract

Forest fires are a major environmental hazard in India, and timely identification of meteorological conditions associated with fire occurrence can support risk assessment. This study develops an India-focused machine-learning classification framework using satellite-derived fire observations from NASA's Fire Information for Resource Management System (FIRMS) and historical meteorological variables. The current implementation evaluates Logistic Regression and Random Forest classifiers using fifteen weather and weather-derived features, including temperature, relative humidity, precipitation, wind, soil moisture, and multi-timescale rainfall indicators. In the prototype experiment, Random Forest achieved an accuracy of 84.74%, precision of 67.12%, recall of 93.51%, F1-score of 78.15%, and ROC-AUC of 93.17%. A Flask web application was developed to expose the trained classifier through a user-facing prediction interface. The current results are preliminary because the prototype uses a partial weather-matched dataset and a stratified random train/test split. The research contribution and final claims will be established only after a focused literature-gap analysis and, if required by the selected research question, additional controlled experiments.

**Keywords:** forest fire prediction, machine learning, VIIRS, NASA FIRMS, meteorological data, Random Forest, India

## 1. Introduction

Forest fires threaten ecosystems, biodiversity, air quality, infrastructure, and livelihoods. India experiences recurring fire activity across multiple climatic and ecological regions, making data-driven fire-risk assessment an important application of remote sensing and machine learning.

Satellite fire products provide a scalable source of historical fire observations, while meteorological variables describe environmental conditions associated with fire occurrence. This study investigates an India-focused classification workflow that integrates these information sources and evaluates whether standard machine-learning models can distinguish fire-associated conditions from matched non-fire conditions.

The study is intentionally scoped as a practical machine-learning framework rather than an operational emergency-warning system. A separate advanced research track will address larger-scale historical data, stronger validation, and additional environmental variables.

## 2. Related Work

Indian forest-fire research has already applied machine learning to susceptibility and occurrence modelling. Sharma et al. (2022) compared six machine-learning algorithms using MODIS fire hotspots from 2001–2020 together with forest, climatic, and topographic predictors. Their study reported strong ROC-AUC performance for SVM and ANN and produced susceptibility patterns for Indian forests.

More recent work has also examined pan-India prediction using combinations of weather and socio-economic or environmental variables. For example, the 2025 AutoML-Fire study used a pan-India dataset spanning 2003–2018 and incorporated variables including cloud cover, humidity, NDVI, soil moisture, temperature, wind speed, precipitation, and socio-economic factors.

These studies establish that machine learning and environmental predictors are viable for Indian fire modelling. Therefore, this paper will not claim that machine learning for Indian forest fires is itself novel. The exact research gap for this study must instead be defined from the literature review around the specific data period, satellite product, feature construction, target definition, and evaluation protocol used here.

## 3. Research Gap and Proposed Contribution

**Status: to be finalized after literature-gap verification.**

The working hypothesis is that an India-focused framework based on the 2018–2025 VIIRS SNPP FIRMS archive, integrated with historical meteorological observations and recent-weather-derived features, may provide a useful and reproducible fire-occurrence classification baseline. The paper will only present this as a novelty claim if the literature review demonstrates that the exact combination and experimental formulation has not already been reported.

The proposed contribution is intentionally limited to **one primary contribution**. The final wording will distinguish between (a) the dataset period and data integration and (b) the actual methodological contribution demonstrated by experiments.

## 4. Data and Materials

### 4.1 FIRMS fire observations

The project uses NASA FIRMS VIIRS Suomi-NPP fire observations for India. The available raw archive covers 2018–2025 and contains approximately 7.9 million raw observations. A reproducible subset was prepared for the prototype training dataset.

### 4.2 Historical meteorology

Historical hourly meteorological data were obtained from the Open-Meteo Historical Weather API. Weather variables were matched to spatially gridded fire and non-fire observations.

### 4.3 Target construction

Fire observations are represented by `fire = 1`. The prototype creates matched non-fire observations represented by `fire = 0`, subject to the constraints implemented in the dataset-construction pipeline.

## 5. Feature Engineering

The classifier uses fifteen features:

1. temperature_2m
2. relative_humidity_2m
3. dew_point_2m
4. precipitation
5. wind_speed_10m
6. wind_direction_10m
7. surface_pressure
8. cloud_cover
9. soil_moisture_0_to_7cm
10. rain_24h
11. rain_72h
12. rain_168h
13. avg_temp_24h
14. avg_humidity_24h
15. max_wind_24h

The multi-timescale variables are intended to represent recent environmental conditions rather than only the instantaneous weather state.

## 6. Methodology

The workflow consists of data preparation, spatial/temporal matching of fire and non-fire observations, historical-weather matching, feature construction, model training, evaluation, and deployment.

Two classifiers are evaluated:

- Logistic Regression with standardization and balanced class weights.
- Random Forest with balanced class weights, 400 trees, maximum depth 8, and minimum leaf size 2.

The prototype uses an 80/20 stratified train/test split with `random_state=42`. The model with the higher F1-score is selected and then refit on the complete prototype dataset for application use.

## 7. Results

### 7.1 Prototype performance

| Metric | Random Forest |
|---|---:|
| Accuracy | 84.74% |
| Precision | 67.12% |
| Recall | 93.51% |
| F1-score | 78.15% |
| ROC-AUC | 93.17% |

The high recall indicates that the prototype detects a large proportion of positive fire-labelled observations, while the lower precision indicates a non-trivial false-positive rate. The ROC-AUC indicates strong discrimination under the prototype test split.

### 7.2 Interpretation

These metrics must be interpreted as **prototype results**. They were obtained from the current partial weather-matched dataset using a random stratified split and therefore do not establish nationwide or future-year predictive performance.

## 8. Web Application

A Flask application exposes the trained Random Forest classifier through a web interface. Users provide the required meteorological inputs and receive a fire-risk classification and predicted probability.

The application demonstrates how the trained model can be integrated into a simple decision-support interface. It is not intended to replace official fire-alert systems.

## 9. Discussion

The prototype demonstrates the feasibility of combining satellite-derived fire observations with meteorological variables in a conventional tabular machine-learning pipeline. The recall-oriented performance is potentially useful for screening applications where missing a positive event is costly; however, the false-positive burden must be considered.

The principal methodological limitation is the current evaluation protocol. Random splitting can produce optimistic estimates when observations are correlated in space or time. A stronger research evaluation should use chronological and/or geographic holdouts before any final research claim is made.

## 10. Limitations

- The prototype does not use all 7.9 million raw FIRMS observations for training.
- The current weather-matched dataset is partial.
- The evaluation uses a random stratified split rather than strict future-year or geographic validation.
- FIRMS detections are satellite thermal-anomaly observations and should not automatically be interpreted as confirmed forest fires without appropriate filtering or ancillary land-use information.
- The model is a prototype and should not be used as an operational emergency-warning system.

## 11. Conclusion

This study presents an India-focused forest-fire classification prototype integrating VIIRS FIRMS observations with historical meteorological variables. Random Forest provided the strongest prototype performance among the evaluated models, achieving 84.74% accuracy and 93.17% ROC-AUC on the current test split. The work provides a reproducible baseline and a working web application. The final research contribution will be determined after completing the literature-gap analysis and, where necessary, strengthening the experimental design.

## 12. Future Work

The advanced research track will investigate the full multi-year FIRMS archive, larger historical weather coverage, stronger spatial and temporal validation, additional environmental predictors, explainability, and more comprehensive model comparisons.

## References

1. Sharma, L. K., Gupta, R., & Fatima, N. (2022). Assessing the predictive efficacy of six machine learning algorithms for the susceptibility of Indian forests to fire. *International Journal of Wildland Fire*, 31(8), 735–758. https://doi.org/10.1071/WF22016
2. NASA FIRMS. Fire Information for Resource Management System. https://firms.modaps.eosdis.nasa.gov/
3. Open-Meteo. Historical Weather API documentation. https://open-meteo.com/en/docs/historical-weather-api
4. AutoML-Fire: Automated machine-learning approach to predict forest fires. *Environmental Modelling & Software*, 193 (2025), 106578. https://doi.org/10.1016/j.envsoft.2025.106578
