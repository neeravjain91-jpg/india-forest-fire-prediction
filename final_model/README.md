# Final research-model inference

The working Flask application uses the final 31-feature HistGradientBoosting model from the research project.

Expected local files:

- `models/final_hgb_model.joblib`
- `models/final_metrics.json`

The model expects these features in this exact order:

1. `grid_lat`
2. `grid_lon`
3. `hour`
4. `year`
5. `month`
6. `temp_1d`
7. `rh_1d`
8. `wind_1d`
9. `pressure_1d`
10. `soil_1d`
11. `rain_1d`
12. `temp_3d_mean`
13. `temp_3d_max`
14. `temp_3d_min`
15. `rh_3d_mean`
16. `rh_3d_min`
17. `wind_3d_mean`
18. `wind_3d_max`
19. `pressure_3d_mean`
20. `soil_3d_mean`
21. `rain_3d_total`
22. `temp_7d_mean`
23. `temp_7d_max`
24. `temp_7d_min`
25. `rh_7d_mean`
26. `rh_7d_min`
27. `wind_7d_mean`
28. `wind_7d_max`
29. `pressure_7d_mean`
30. `soil_7d_mean`
31. `rain_7d_total`

Model configuration: HistGradientBoostingClassifier with `max_iter=300`, `learning_rate=0.05`, `max_leaf_nodes=31`, `l2_regularization=1.0`, `random_state=42`.
