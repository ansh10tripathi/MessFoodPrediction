# Synthetic Model Run Results

> **These are synthetic-data results only.** They must not be represented as actual university mess performance or real-world prediction accuracy.

## Run details

- Dataset: `data/synthetic/mess_food_synthetic.csv` (synthetic, 1,460 records, 365 dates)
- Target: continuous `consumed_kg`
- Seed: 395
- Split method: chronological by whole date; no date appears in more than one partition
- Feature allowlist: expected attendance, meal period, menu category, weekday, month, exam/holiday/event flags, and synthetic temperature/rainfall fields assumed available as forecasts
- Explicitly excluded: actual attendance, prepared quantity, target consumption, edible leftovers, discarded waste

| Partition | Dates | Rows | Inclusive date boundaries |
|---|---:|---:|---|
| Train (70%) | 255 | 1,020 | 2025-01-01 to 2025-09-12 |
| Validation (15%) | 55 | 220 | 2025-09-13 to 2025-11-06 |
| Test (15%) | 55 | 220 | 2025-11-07 to 2025-12-31 |

Each candidate was fitted on training dates and compared by validation MAE. RandomForestRegressor was selected because it had the lowest validation MAE. Its pipeline was then refitted on train plus validation and evaluated on the test period, which was not used for selection.

## Candidate comparison on validation dates

| Model | MAE (kg) | RMSE (kg) | R² |
|---|---:|---:|---:|
| RandomForestRegressor (selected) | 12.935 | 17.488 | 0.970 |
| HistGradientBoostingRegressor | 13.205 | 17.867 | 0.968 |
| LinearRegression | 13.247 | 17.560 | 0.969 |
| DecisionTreeRegressor | 15.033 | 20.316 | 0.959 |
| DummyRegressor | 92.699 | 100.387 | -0.000 |

## Final test result for selected model

| Model | MAE (kg) | RMSE (kg) | R² |
|---|---:|---:|---:|
| RandomForestRegressor | 12.969 | 17.509 | 0.968 |

The scores summarize performance against this generator's patterns only. The generator uses simulated actual attendance to produce consumption, while actual attendance is intentionally withheld from the prediction features. Expected attendance and meal identity remain correlated with the simulated target by design. This synthetic test therefore demonstrates the code path and split procedure; it cannot establish accuracy at a real mess. The synthetic weather values also do not model real forecast errors.

## Output files

- `models/synthetic/metadata.json`: complete machine-readable model metadata and exact metrics.
- `models/synthetic/selected_model.joblib`: selected preprocessing-plus-model pipeline refit on train and validation dates.
- `models/synthetic/<Estimator>.joblib`: each candidate pipeline fitted on training dates.
- `outputs/synthetic/validation_model_comparison.csv`: unrounded validation metrics.
- `outputs/synthetic/selected_test_predictions.csv`: test-period predictions, with every row labeled synthetic.
