# Supervised Learning Methodology

> All current experiments use **synthetic development data**. Their scores demonstrate the software pipeline only; they are not estimates of real-world performance.

## Target and prediction-time inputs

The target is `consumed_kg`, a continuous quantity, so this is a regression task. The prediction interface accepts `date`, `expected_students`, `meal_period`, `menu_category`, `exam_day`, `holiday`, `special_event`, `temperature_c`, and `rainfall_mm`. Date becomes weekday and month. Weather values are assumed to represent forecasts available before the meal; in the starter CSV they are synthetic examples and must be replaced by forecast inputs for a real evaluation.

The feature builder does not use `actual_students`, `prepared_kg`, `consumed_kg`, `edible_leftover_kg`, or `discarded_waste_kg`. These are outcomes or post-meal measurements unavailable to a pre-meal forecast. Automated tests protect the feature allowlist and reject unsupported prediction inputs. Numeric features pass through unchanged and categorical features use one-hot encoding, which ignores categories absent from training. No imputation is used because the current validator and prediction interface require complete inputs and the starter data has no missing values.

The prediction input validator also rejects unsupported meal names, known menu categories paired with the wrong meal, non-finite numbers, negative weather values, non-whole student counts, and temperature values outside the data validator's broad physical bounds. An unseen non-empty menu category remains accepted because the encoder supports new categories; prediction emits a warning that its category indicators are all zero. If the model returns a negative estimate, prediction emits the raw value in a warning and returns a physical floor of 0 kg. A non-finite model output raises an error instead of being hidden.

The selected Random Forest is only used for `expected_students` values within its fitted synthetic-data interval, currently 57–730 students. Outside that range the prediction interface warns and withholds an estimate, since tree-based models do not extrapolate reliably. The interval is saved in model metadata and refreshed by each training run. See [prediction range audit](prediction_range_audit.md) for the linear-regression comparison and its limitations.

## Chronological evaluation

Rows are grouped by distinct date and sorted chronologically before splitting, so all meal records from one date stay together. The split proportions are 70% train, 15% validation, and 15% test by distinct dates; exact boundaries are saved to `models/synthetic/metadata.json`. Candidate estimators fit only on training dates. The model with the lowest validation MAE is selected. That selected pipeline is then refit on train plus validation dates and evaluated once on the later, untouched test dates.

Candidates:

- `DummyRegressor`: mean-prediction baseline.
- `LinearRegression`: simple linear baseline with one-hot categorical inputs.
- `DecisionTreeRegressor`: non-linear decision rules, with a minimum leaf size.
- `RandomForestRegressor`: average of randomized trees.
- `HistGradientBoostingRegressor`: boosted trees for non-linear patterns.

All candidates use the same preprocessing. Randomized estimators use seed 395 by default. MAE and RMSE are in kg; R² is unitless. Lower MAE/RMSE is better. R² can be negative when performance is worse than predicting a constant mean.

## Artifacts

- `models/synthetic/<Estimator>.joblib`: each complete estimator and preprocessing pipeline, fitted on training dates for comparison.
- `models/synthetic/selected_model.joblib`: selected pipeline refit on train plus validation dates.
- `models/synthetic/metadata.json`: target, safe feature list, dataset label, date boundaries, model selection, metrics, and limitations.
- `outputs/synthetic/validation_model_comparison.csv`: candidate validation scores.
- `outputs/synthetic/selected_test_predictions.csv`: held-out selected-model predictions.

Load Joblib artifacts only from trusted sources and use a compatible Python/scikit-learn environment.

## Limitations

The target is generated from simulated actual attendance and meal-specific assumptions. Actual attendance is intentionally excluded from prediction, leaving expected attendance as a proxy. One generated year and simulated weather do not represent operational variation, measurement error, policy changes, or seasonal differences. Scores on this data cannot establish operational accuracy. Real performance requires representative historical data, temporal evaluation, and monitoring.

## Reproduce (Windows PowerShell)

Run from the project root:

```powershell
.\.venv\Scripts\python.exe -m src.train data/synthetic/mess_food_synthetic.csv
.\.venv\Scripts\python.exe -m src.predict --model models/synthetic/selected_model.joblib --input-json '{"date":"2026-01-15","expected_students":420,"meal_period":"Lunch","menu_category":"North Indian","exam_day":false,"holiday":false,"special_event":false,"temperature_c":27.0,"rainfall_mm":0.0}'
```
