"""Train, compare, select, evaluate, and save synthetic consumption models."""

from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

from src.data_validation import read_csv, validate_rows
from src.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET, assert_leakage_safe, build_features

RANDOM_SEED = 395
TRAIN_PROPORTION = 0.70
VALIDATION_PROPORTION = 0.15
MINIMUM_DATES = 20
MODEL_NAMES = ("DummyRegressor", "LinearRegression", "DecisionTreeRegressor", "RandomForestRegressor", "HistGradientBoostingRegressor")


@dataclass
class DateSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    boundaries: dict[str, dict[str, str]]


def chronological_date_split(frame: pd.DataFrame) -> DateSplits:
    """Split whole dates into chronological 70/15/15 partitions."""
    if "date" not in frame:
        raise ValueError("Training data must contain a date column")
    dated = frame.copy()
    dated["_split_date"] = pd.to_datetime(dated["date"], errors="coerce")
    if dated["_split_date"].isna().any():
        raise ValueError("Training data contains invalid dates")
    unique_dates = sorted(dated["_split_date"].dt.normalize().unique())
    if len(unique_dates) < MINIMUM_DATES:
        raise ValueError(f"At least {MINIMUM_DATES} distinct dates are required; found {len(unique_dates)}")
    train_end = int(len(unique_dates) * TRAIN_PROPORTION)
    validation_end = int(len(unique_dates) * (TRAIN_PROPORTION + VALIDATION_PROPORTION))
    train_dates = set(unique_dates[:train_end])
    validation_dates = set(unique_dates[train_end:validation_end])
    test_dates = set(unique_dates[validation_end:])
    if min(map(len, (train_dates, validation_dates, test_dates))) == 0:
        raise ValueError("Not enough dates to create non-empty train, validation, and test sets")

    dated_days = dated["_split_date"].dt.normalize()
    partitions = [dated.loc[dated_days.isin(day_set)].drop(columns="_split_date").copy() for day_set in (train_dates, validation_dates, test_dates)]
    names = ("train", "validation", "test")
    boundaries = {
        name: {
            "start": pd.Timestamp(min(day_set)).date().isoformat(),
            "end": pd.Timestamp(max(day_set)).date().isoformat(),
            "dates": len(day_set),
            "rows": len(partition),
        }
        for name, day_set, partition in zip(names, (train_dates, validation_dates, test_dates), partitions)
    }
    return DateSplits(*partitions, boundaries)


def _make_preprocessor() -> ColumnTransformer:
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer([
        ("numeric", "passthrough", list(NUMERIC_FEATURES)),
        ("categorical", categorical, list(CATEGORICAL_FEATURES)),
    ], remainder="drop", verbose_feature_names_out=False)


def build_model_pipelines(seed: int = RANDOM_SEED, random_forest_trees: int = 200) -> dict[str, Pipeline]:
    """Return all required estimators wrapped with their own preprocessing."""
    assert_leakage_safe(FEATURE_COLUMNS)
    estimators = {
        "DummyRegressor": DummyRegressor(strategy="mean"),
        "LinearRegression": LinearRegression(),
        "DecisionTreeRegressor": DecisionTreeRegressor(random_state=seed, min_samples_leaf=4),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=random_forest_trees, random_state=seed, n_jobs=-1, min_samples_leaf=2),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(max_iter=120, learning_rate=0.08, l2_regularization=1.0, random_state=seed),
    }
    return {
        name: Pipeline([("preprocessor", _make_preprocessor()), ("model", estimator)])
        for name, estimator in estimators.items()
    }


def _metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def train_and_evaluate(
    csv_path: Path,
    *,
    models_dir: Path = Path("models/synthetic"),
    output_dir: Path = Path("outputs/synthetic"),
    seed: int = RANDOM_SEED,
    random_forest_trees: int = 200,
) -> dict[str, Any]:
    """Fit all candidates, select on validation MAE, then evaluate on test once."""
    columns, raw_rows = read_csv(csv_path)
    validation = validate_rows(columns, raw_rows, expected_dataset_type="synthetic")
    if not validation.is_valid:
        raise ValueError("Training CSV failed validation: " + "; ".join(validation.errors[:8]))
    frame = pd.DataFrame(raw_rows)
    if TARGET not in frame:
        raise ValueError(f"Training CSV is missing target '{TARGET}'")
    # Target and future-only measurements are read for supervision/validation only.
    target = pd.to_numeric(frame[TARGET], errors="coerce")
    if target.isna().any() or not np.isfinite(target.to_numpy()).all() or (target < 0).any():
        raise ValueError("Target consumed_kg must contain finite non-negative numbers")
    features = build_features(frame)
    safe_model_frame = features.copy()
    safe_model_frame["date"] = frame["date"].to_numpy()
    safe_model_frame[TARGET] = target.to_numpy()
    splits = chronological_date_split(safe_model_frame)
    if min(len(splits.train), len(splits.validation), len(splits.test)) < 2:
        raise ValueError("Each partition must contain at least two observations")

    x_train, y_train = splits.train.loc[:, FEATURE_COLUMNS], splits.train[TARGET]
    x_validation, y_validation = splits.validation.loc[:, FEATURE_COLUMNS], splits.validation[TARGET]
    x_test, y_test = splits.test.loc[:, FEATURE_COLUMNS], splits.test[TARGET]
    candidate_pipelines = build_model_pipelines(seed, random_forest_trees)
    comparison: list[dict[str, Any]] = []
    validation_predictions: dict[str, np.ndarray] = {}
    for model_name, pipeline in candidate_pipelines.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            pipeline.fit(x_train, y_train)
        prediction = pipeline.predict(x_validation)
        validation_predictions[model_name] = prediction
        metric_row: dict[str, Any] = {"model": model_name, **_metrics(y_validation, prediction)}
        comparison.append(metric_row)
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, models_dir / f"{model_name}.joblib")

    selected_name = min(comparison, key=lambda item: (item["mae"], item["model"]))["model"]
    # Refit selected pipeline on train+validation only after model selection.
    selected_pipeline = build_model_pipelines(seed, random_forest_trees)[selected_name]
    x_train_validation = pd.concat([x_train, x_validation], axis=0)
    y_train_validation = pd.concat([y_train, y_validation], axis=0)
    expected_students_training_range = {
        "min": int(x_train_validation["expected_students"].min()),
        "max": int(x_train_validation["expected_students"].max()),
    }
    selected_pipeline.fit(x_train_validation, y_train_validation)
    test_prediction = selected_pipeline.predict(x_test)
    test_metrics = _metrics(y_test, test_prediction)
    models_dir.mkdir(parents=True, exist_ok=True)
    selected_artifact = models_dir / "selected_model.joblib"
    joblib.dump(selected_pipeline, selected_artifact)

    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "validation_model_comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("model", "mae", "rmse", "r2"))
        writer.writeheader()
        writer.writerows(sorted(comparison, key=lambda row: row["mae"]))
    test_predictions_path = output_dir / "selected_test_predictions.csv"
    test_export = pd.DataFrame({
        "date": splits.test["date"].to_numpy(),
        "meal_period": frame.loc[splits.test.index, "meal_period"].to_numpy(),
        "actual_consumed_kg": y_test.to_numpy(),
        "predicted_consumed_kg": test_prediction,
        "dataset_type": "synthetic",
    })
    test_export.to_csv(test_predictions_path, index=False)

    first_test_model_date = min(pd.to_datetime(splits.test["date"]).dt.date)
    metadata = {
        "result_label": "synthetic-data model results; not real-world accuracy",
        "target_variable": TARGET,
        "target_type": "continuous regression",
        "feature_list": list(FEATURE_COLUMNS),
        "raw_prediction_inputs": ["date", "expected_students", "meal_period", "menu_category", "exam_day", "holiday", "special_event", "temperature_c", "rainfall_mm"],
        "supported_input_ranges": {"expected_students": expected_students_training_range},
        "forbidden_features": sorted({"actual_students", "prepared_kg", "consumed_kg", "edible_leftover_kg", "discarded_waste_kg"}),
        "dataset_source": str(csv_path),
        "dataset_status": "synthetic",
        "random_seed": seed,
        "split_proportions_by_distinct_date": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "split_boundaries": splits.boundaries,
        "selection_metric": "validation MAE (lower is better)",
        "validation_model_comparison": sorted(comparison, key=lambda row: row["mae"]),
        "selected_model": selected_name,
        "selected_model_refit": "train + validation dates only; test dates excluded",
        "test_evaluation_started_after_selection": str(first_test_model_date),
        "final_test_metrics": test_metrics,
        "metrics_note": "MAE and RMSE are kg; R2 is unitless. All values describe synthetic data only.",
        "limitations": [
            "The dataset is generated from assumptions and does not establish real-world predictive accuracy.",
            "Actual attendance is a major latent driver in the generator and is intentionally excluded as unavailable before the meal.",
            "Weather inputs must be replaced by genuine forecast values available at prediction time for a real deployment.",
            "One synthetic year is too limited to establish season-to-season robustness.",
        ],
        "artifacts": {"selected_pipeline": str(selected_artifact), "comparison_csv": str(comparison_path), "test_predictions_csv": str(test_predictions_path)},
    }
    metadata_path = models_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"metadata": metadata, "metadata_path": metadata_path, "comparison": comparison, "selected_pipeline": selected_artifact}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", nargs="?", type=Path, default=Path("data/synthetic/mess_food_synthetic.csv"))
    parser.add_argument("--models-dir", type=Path, default=Path("models/synthetic"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/synthetic"))
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--trees", type=int, default=200)
    args = parser.parse_args()
    result = train_and_evaluate(args.csv_path, models_dir=args.models_dir, output_dir=args.output_dir, seed=args.seed, random_forest_trees=args.trees)
    print("All validation metrics and test metrics below are synthetic-data results only.")
    for metric in sorted(result["comparison"], key=lambda row: row["mae"]):
        print(f"{metric['model']}: validation MAE={metric['mae']:.3f} kg, RMSE={metric['rmse']:.3f} kg, R2={metric['r2']:.3f}")
    meta = result["metadata"]
    print(f"Selected on validation MAE: {meta['selected_model']}")
    test = meta["final_test_metrics"]
    print(f"Final untouched test: MAE={test['mae']:.3f} kg, RMSE={test['rmse']:.3f} kg, R2={test['r2']:.3f}")
    print(f"Metadata: {result['metadata_path']}")


if __name__ == "__main__":
    main()
