# Project 2 — Advanced Research Pipeline

This directory is separate from the working mini-project application.

## Objective
Develop a multi-year India forest-fire occurrence dataset and evaluate machine-learning models using VIIRS SNPP FIRMS observations and historical meteorological variables.

## Data already available
- Raw VIIRS SNPP FIRMS India observations: 2018–2025
- Approximately 7.9 million raw fire detections
- NASA FIRMS MAP_KEY

## Research principles
1. Keep raw FIRMS data immutable.
2. Use a reproducible spatial grid and time definition.
3. Construct matched non-fire samples rather than arbitrary negatives.
4. Use historical weather matched to the prediction time.
5. Use chronological and geographic holdout tests to reduce leakage.
6. Report precision, recall, F1, ROC-AUC and PR-AUC in addition to accuracy.
7. Treat Project 1 as the baseline; do not modify its working Flask application.

## First research dataset
The first advanced dataset will use a stratified sample of FIRMS detections across 2018–2025. Weather will be downloaded only for the sampled spatial/time cells and cached locally. The sample size can be expanded after the pipeline is validated.
