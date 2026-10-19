# University Mess Food Demand & Waste Prediction System

An academic project for exploring meal demand, food preparation, edible leftovers, and discarded food. The starter dataset is **synthetic** and exists to develop and check the software pipeline. It is not a record of any university's operations and cannot establish real-world model accuracy.

## Current phase

The project includes a reproducible synthetic-data generator, validation and EDA, a leakage-safe regression workflow, and a local Streamlit dashboard. Uploaded CSVs are held and analyzed in the current app session; the dashboard does not save them to disk.

## Setup (Windows PowerShell)

Use Python 3.12 from the project directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, invoke `.\.venv\Scripts\python.exe` directly. To install only the ML dependencies for Phase 4:

```powershell
.\.venv\Scripts\python.exe -m pip install "pandas>=2.2,<4" "numpy>=1.26,<3" "scikit-learn>=1.4,<2" "joblib>=1.3,<2"
```

The dashboard dependencies are included in `requirements.txt`. Install them locally with `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.

## Launch the dashboard

From the project directory in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit prints a local address to open in your browser. The sidebar contains Overview, Demand Prediction, Waste Analytics, Data Management, Model Performance, and Reports. Select the active dataset explicitly in the sidebar. The prediction and model-performance sections identify the current selected model as trained on synthetic data.

CSV uploads are checked against the project's documented analytics columns and the four-meal-per-date structure. Choose `real` or `synthetic` explicitly before validation; there is no implicit real-data default. If `dataset_type` exists, it must agree with that choice. An unlabelled file receives the declared source only in its in-memory analysis copy, with a warning that the classification is user-declared and unverified. Valid uploads stay in the browser session and the original bytes are not changed or written into `data/real/`. A failed upload is not activated, does not replace a prior active upload, and shows the first ten validation errors with an option to inspect the rest. The dashboard's light theme is configured in `.streamlit/config.toml` using Streamlit's supported theme settings.

## Troubleshooting

- **`streamlit` or `plotly` is missing:** run `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` from the project root.
- **Model unavailable:** check that `models/synthetic/selected_model.joblib` and `models/synthetic/metadata.json` exist. The app does not retrain models automatically.
- **Dataset unavailable:** check that `data/synthetic/mess_food_synthetic.csv` exists and passes validation. The app displays a load error instead of substituting another dataset.
- **Upload rejected:** review the displayed column, value, date-coverage, meal-period, source-label, and food-accounting errors. The upload must use the project's supported analytics schema.
- **Port already in use:** launch with `.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502`.

## Generate the development dataset

```powershell
.\.venv\Scripts\python.exe -m src.data_generator --output data/synthetic/mess_food_synthetic.csv --days 365 --seed 395
```

The CSV contains four rows per date: Breakfast, Lunch, Snacks, and Dinner. An existing output file is refused to prevent overwriting. See the data dictionary and assumptions below.

## Validate, explore, and train

```powershell
.\.venv\Scripts\python.exe -m src.data_validation data/synthetic/mess_food_synthetic.csv --expected-days 365
.\.venv\Scripts\python.exe -m src.explore_data data/synthetic/mess_food_synthetic.csv --output docs/eda_synthetic_report.md
.\.venv\Scripts\python.exe -m src.train data/synthetic/mess_food_synthetic.csv
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Training splits complete dates chronologically into 70% train, 15% validation, and 15% test. It chooses the model on validation MAE, refits that selected pipeline on train plus validation, and evaluates it once on later, untouched test dates. Current method and actual run details are in [ML methodology](docs/ml_methodology.md) and [synthetic results](docs/ml_synthetic_results.md).

On the current generated dataset, RandomForestRegressor was selected (validation MAE 12.935 kg). Its untouched synthetic test scores were MAE 12.969 kg, RMSE 17.509 kg, and R² 0.968. These results demonstrate behavior on generated data only and are not evidence of real-world accuracy. The fitted model observed expected attendance from 57 to 730 students; the prediction page warns and withholds an estimate outside that interval because the selected Random Forest does not extrapolate reliably. See [prediction range audit](docs/prediction_range_audit.md) for a fixed-input comparison with linear regression.

## Make a sample prediction

The interface takes pre-meal values only. Replace the illustrative weather values with forecasts available at prediction time for a real use case.

```powershell
.\.venv\Scripts\python.exe -m src.predict --model models/synthetic/selected_model.joblib --input-json '{"date":"2026-01-15","expected_students":420,"meal_period":"Lunch","menu_category":"North Indian","exam_day":false,"holiday":false,"special_event":false,"temperature_c":27.0,"rainfall_mm":0.0}'
```

The result is a consumption estimate, not a guaranteed amount to prepare. Predicted consumption and preparation quantity are different planning decisions.

## Data layout

- `data/synthetic/`: labeled development data.
- `data/real/`: reserved for user-supplied data; no sample real records are included.
- `src/`: generator, validation, EDA, features, training, and prediction modules.
- `tests/`: unit tests.
- `models/synthetic/`: fitted complete pipelines and metadata.
- `outputs/synthetic/`: validation comparison and held-out prediction CSVs.
- `docs/`: project and evaluation documentation.

## Synthetic data and accounting

Attendance and per-student consumption are simulated from hand-chosen ranges and meal-specific assumptions. Temperature and rainfall are illustrative values, not observations or forecasts. For every row, `prepared_kg = consumed_kg + edible_leftover_kg + discarded_waste_kg`; the accepted absolute accounting tolerance is 1e-6 kg. Edible leftovers remain distinct from discarded waste. Real operations may need categories for inedible scraps, safe retention, and measurement losses.

| Column | Meaning |
|---|---|
| `date` | ISO calendar date |
| `meal_period` | Breakfast, Lunch, Snacks, or Dinner |
| `menu_category` | Synthetic meal style |
| `expected_students` | Pre-meal attendance estimate |
| `actual_students` | Simulated attendance; excluded from prediction features |
| `exam_day`, `holiday`, `special_event` | Boolean indicators |
| `temperature_c`, `rainfall_mm` | Synthetic weather context |
| `prepared_kg`, `consumed_kg` | Simulated quantities |
| `edible_leftover_kg`, `discarded_waste_kg` | Separate post-meal quantities |
| `dataset_type` | `synthetic` for all starter records |

All generated EDA/model metrics are synthetic-data results. Real-world predictive accuracy requires representative real observations and prospective evaluation.
