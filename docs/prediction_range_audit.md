# Synthetic prediction range audit

This is an audit of the pre-meal `expected_students` path using the synthetic dataset and the existing chronological partitions. It does not establish real-world prediction accuracy.

## Finding

`expected_students` is passed from the Streamlit number input into the prediction input mapping, validated and retained by `build_prediction_input`, and delivered to `model.predict` in the pipeline's `FEATURE_COLUMNS` order. There is no prediction cache, attendance overwrite, scaling step, or output cap. The model returns a finite value and only negative predictions are floored to zero with a warning.

The apparent high-attendance plateau is a Random Forest limitation. The selected model was fit on expected attendance from **57 to 730 students**. Decision-tree models return averages of training targets in their terminal leaves, so they do not continue a linear trend outside the range they observed. At high attendance, this input scenario falls into the same terminal leaves and returns the same estimate. The current metadata records the fitted attendance interval, and future training runs write that interval from the rows used to fit the final pipeline.

## Fixed-input illustration

The following comparison holds date, meal, menu, flags, and weather constant: 2025-07-01; Lunch; North Indian; exam/holiday/event false; 27 °C; 0 mm rain.

| Expected students | Random Forest before range guard | Linear regression diagnostic |
|---:|---:|---:|
| 200 | 73.39 kg | 45.96 kg |
| 420 | 156.31 kg | 177.76 kg |
| 650 | 310.26 kg | 315.55 kg |
| 2,200 | 358.72 kg | 1,244.13 kg |
| 6,200 | 358.72 kg | 3,640.46 kg |

Values depend on the other input settings; this fixed scenario does not reproduce every reported 420-student observation. With the current application, values from **57 through 730** use the selected Random Forest. Values outside that fitted interval produce a warning and no estimate. The app does not present a tree-leaf plateau as a supported forecast.

## Model comparison and decision

All values below are synthetic-data results. Splits remain chronological by whole date: 255 training dates, 55 validation dates, and 55 untouched test dates.

| Candidate | Validation MAE (kg) | Validation RMSE (kg) | Validation R² | Test MAE (kg) | Test RMSE (kg) | Test R² |
|---|---:|---:|---:|---:|---:|---:|
| RandomForestRegressor | 12.935 | 17.488 | 0.970 | 12.969 | 17.509 | 0.968 |
| LinearRegression | 13.247 | 17.560 | 0.969 | 13.704 | 18.059 | 0.966 |

The Random Forest remains selected because it has the lower validation MAE and slightly lower synthetic test errors. Linear regression was examined as an extrapolating alternative, but its high-attendance outputs are unsupported by observations and its held-out scores are worse. Selecting it solely to avoid a plateau would replace one limitation with unverified extrapolation. The safer behavior is to withhold predictions outside the observed attendance interval.

## Limitations

The synthetic target is generated from actual attendance, while the model must use expected attendance, an imperfect proxy. There is no training evidence for expected values above 730 students. The validation/test metrics and illustrative estimates describe only this generated dataset; real operational use requires representative historical data and prospective evaluation.
