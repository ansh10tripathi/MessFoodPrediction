"""Reusable, session-independent data and metric helpers for the dashboard."""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_validation import ValidationReport, read_csv, validate_rows
from src.features import FEATURE_COLUMNS, TARGET

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_CSV = PROJECT_ROOT / "data" / "synthetic" / "mess_food_synthetic.csv"
SYNTHETIC_MODEL_DIR = PROJECT_ROOT / "models" / "synthetic"
MODEL_PATH = SYNTHETIC_MODEL_DIR / "selected_model.joblib"
MODEL_METADATA_PATH = SYNTHETIC_MODEL_DIR / "metadata.json"
MODEL_COMPARISON_PATH = PROJECT_ROOT / "outputs" / "synthetic" / "validation_model_comparison.csv"
TEST_PREDICTIONS_PATH = PROJECT_ROOT / "outputs" / "synthetic" / "selected_test_predictions.csv"

FOOD_COLUMNS = ("prepared_kg", "consumed_kg", "edible_leftover_kg", "discarded_waste_kg")
NUMERIC_COLUMNS = (
    "expected_students", "actual_students", "temperature_c", "rainfall_mm",
    *FOOD_COLUMNS,
)


@dataclass
class UploadResult:
    """In-memory CSV validation result. Uploaded bytes are never written to disk."""

    data: pd.DataFrame | None
    report: ValidationReport
    source_type: str
    added_source_label_in_memory: bool = False
    parse_errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.data is not None and self.report.is_valid and not self.parse_errors


def validate_uploaded_csv(contents: bytes, source_type: str) -> UploadResult:
    """Validate uploaded CSV bytes against the project's analytics schema.

    For uploads lacking a `dataset_type` column, the user-selected source label
    is added to the in-memory rows only. The original upload is never modified
    or saved to the project directory.
    """
    normalized_type = source_type.strip().lower()
    if normalized_type not in {"real", "synthetic"}:
        raise ValueError("source_type must be 'real' or 'synthetic'")
    try:
        text = contents.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        report = ValidationReport(errors=[f"CSV must be UTF-8 encoded: {error}"])
        return UploadResult(None, report, normalized_type, parse_errors=report.errors.copy())

    try:
        reader = csv.reader(io.StringIO(text, newline=""))
        columns = next(reader, None)
        if columns is None:
            report = ValidationReport(errors=["CSV is empty or has no header"])
            return UploadResult(None, report, normalized_type, parse_errors=report.errors.copy())
        duplicate_headers = sorted({name for name in columns if columns.count(name) > 1})
        if duplicate_headers:
            message = f"CSV contains duplicate column name(s): {', '.join(duplicate_headers)}"
            report = ValidationReport(errors=[message])
            return UploadResult(None, report, normalized_type, parse_errors=[message])
        rows: list[dict[str, str]] = []
        parse_errors: list[str] = []
        for row_number, values in enumerate(reader, start=2):
            if len(values) != len(columns):
                parse_errors.append(
                    f"Row {row_number} has {len(values)} fields; the header has {len(columns)}"
                )
                continue
            rows.append(dict(zip(columns, values)))
    except csv.Error as error:
        report = ValidationReport(errors=[f"Could not parse CSV: {error}"])
        return UploadResult(None, report, normalized_type, parse_errors=report.errors.copy())

    if parse_errors:
        report = ValidationReport(rows_checked=len(rows), errors=parse_errors)
        return UploadResult(None, report, normalized_type, parse_errors=parse_errors)

    added_label = "dataset_type" not in columns
    if added_label:
        columns.append("dataset_type")
        for row in rows:
            row["dataset_type"] = normalized_type
    else:
        source_labels = sorted({(row.get("dataset_type") or "").strip().lower() for row in rows})
        conflicting = [label for label in source_labels if label != normalized_type]
        if conflicting:
            report = ValidationReport(
                rows_checked=len(rows),
                errors=[
                    "Selected source is "
                    f"'{normalized_type}', but the file's dataset_type column contains: "
                    f"{', '.join(repr(label or '<blank>') for label in conflicting)}. "
                    "Choose the matching source only if that classification is correct; "
                    "the upload will not be relabeled automatically."
                ],
            )
            return UploadResult(None, report, normalized_type)
    report = validate_rows(columns, rows, expected_dataset_type=normalized_type, expected_days=None)
    if added_label and report.is_valid:
        report.warnings.append(
            f"Source classification '{normalized_type}' is user-declared because the file has no "
            "dataset_type column; it was not independently verified."
        )
    if not report.is_valid:
        return UploadResult(None, report, normalized_type, added_label)
    return UploadResult(pd.DataFrame(rows, columns=columns), report, normalized_type, added_label)


def load_validated_dataset(path: Path, *, expected_source: str, expected_days: int | None = None) -> tuple[pd.DataFrame, ValidationReport]:
    """Load a project CSV and validate it without modifying the source file."""
    columns, rows = read_csv(path)
    report = validate_rows(
        columns,
        rows,
        expected_dataset_type=expected_source,
        expected_days=expected_days,
    )
    if not report.is_valid:
        raise ValueError("; ".join(report.errors[:12]))
    return pd.DataFrame(rows, columns=columns), report


def source_status(data: pd.DataFrame) -> str:
    """Return an explicit source badge label for a dataset frame."""
    if "dataset_type" not in data:
        return "SOURCE UNKNOWN"
    labels = {str(value).strip().lower() for value in data["dataset_type"].dropna().unique()}
    if labels == {"synthetic"}:
        return "SYNTHETIC DATA"
    if labels == {"real"}:
        return "REAL DATA"
    if labels:
        return "MIXED DATA SOURCES"
    return "SOURCE UNKNOWN"


def prepare_analysis_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Convert validated date and numeric fields for analysis on a copy."""
    frame = data.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in NUMERIC_COLUMNS:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def summary_totals(data: pd.DataFrame) -> dict[str, float | int | str | None]:
    """Return safe aggregate metrics; waste rate uses discarded/prepared × 100."""
    if data.empty:
        return {"records": 0, "date_start": None, "date_end": None, **{c: 0.0 for c in FOOD_COLUMNS}, "discarded_pct_prepared": 0.0}
    frame = prepare_analysis_frame(data)
    totals = {column: float(frame[column].sum()) for column in FOOD_COLUMNS}
    prepared = totals["prepared_kg"]
    return {
        "records": len(frame),
        "date_start": frame["date"].min().date().isoformat(),
        "date_end": frame["date"].max().date().isoformat(),
        **totals,
        "discarded_pct_prepared": 100.0 * totals["discarded_waste_kg"] / prepared if prepared > 0 else 0.0,
    }


def filter_records(data: pd.DataFrame, start_date: Any, end_date: Any, meal_periods: list[str] | None = None) -> pd.DataFrame:
    """Filter a copy of the data; invalid or empty ranges return an empty frame."""
    frame = prepare_analysis_frame(data)
    if start_date is None or end_date is None or start_date > end_date:
        return frame.iloc[0:0].copy()
    selected = frame[frame["date"].dt.date.between(start_date, end_date)]
    if meal_periods:
        selected = selected[selected["meal_period"].isin(meal_periods)]
    return selected.copy()


def daily_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate recorded quantities by date, keeping edible and discarded separate."""
    if data.empty:
        return pd.DataFrame(columns=["date", *FOOD_COLUMNS, "discarded_pct_prepared"])
    frame = prepare_analysis_frame(data)
    daily = frame.groupby("date", as_index=False)[list(FOOD_COLUMNS)].sum()
    daily["discarded_pct_prepared"] = np.where(
        daily["prepared_kg"] > 0,
        100.0 * daily["discarded_waste_kg"] / daily["prepared_kg"],
        0.0,
    )
    return daily


def meal_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate recorded quantities by meal period."""
    frame = prepare_analysis_frame(data)
    if frame.empty:
        return pd.DataFrame(columns=["meal_period", *FOOD_COLUMNS, "discarded_pct_prepared"])
    summary = frame.groupby("meal_period", as_index=False)[list(FOOD_COLUMNS)].sum()
    summary["discarded_pct_prepared"] = np.where(
        summary["prepared_kg"] > 0,
        100.0 * summary["discarded_waste_kg"] / summary["prepared_kg"],
        0.0,
    )
    return summary


def menu_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate recorded quantities by menu category."""
    frame = prepare_analysis_frame(data)
    if frame.empty:
        return pd.DataFrame(columns=["menu_category", *FOOD_COLUMNS, "discarded_pct_prepared"])
    summary = frame.groupby("menu_category", as_index=False)[list(FOOD_COLUMNS)].sum()
    summary["discarded_pct_prepared"] = np.where(
        summary["prepared_kg"] > 0,
        100.0 * summary["discarded_waste_kg"] / summary["prepared_kg"],
        0.0,
    )
    return summary


def regression_metrics(actual: Any, predicted: Any) -> dict[str, float]:
    """Compute MAE, RMSE, and R² for numeric sequences."""
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    if actual_values.ndim != 1 or predicted_values.ndim != 1 or len(actual_values) != len(predicted_values) or len(actual_values) == 0:
        raise ValueError("actual and predicted must be non-empty 1D arrays of equal length")
    if not np.isfinite(actual_values).all() or not np.isfinite(predicted_values).all():
        raise ValueError("metrics inputs must be finite")
    return {
        "mae": float(mean_absolute_error(actual_values, predicted_values)),
        "rmse": float(np.sqrt(mean_squared_error(actual_values, predicted_values))),
        "r2": float(r2_score(actual_values, predicted_values)) if len(actual_values) >= 2 else float("nan"),
    }


def load_model_bundle(model_path: Path = MODEL_PATH, metadata_path: Path = MODEL_METADATA_PATH) -> tuple[Any, dict[str, Any]]:
    """Load the selected saved pipeline and check it against its metadata."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Selected model artifact was not found: {model_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Model metadata was not found: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read model metadata: {error}") from error
    pipeline = joblib.load(model_path)
    if metadata.get("dataset_status") != "synthetic":
        raise ValueError("Selected dashboard model metadata must identify a synthetic-trained model")
    if metadata.get("target_variable") != TARGET:
        raise ValueError(f"Model metadata target must be {TARGET!r}")
    if metadata.get("feature_list") != list(FEATURE_COLUMNS):
        raise ValueError("Saved model metadata feature list does not match the dashboard prediction inputs")
    ranges_metadata = metadata.get("supported_input_ranges")
    expected_range = ranges_metadata.get("expected_students") if isinstance(ranges_metadata, dict) else None
    try:
        range_min = float(expected_range["min"])
        range_max = float(expected_range["max"])
    except (TypeError, KeyError, ValueError) as error:
        raise ValueError("Saved model metadata is missing a valid expected_students training range") from error
    if not math.isfinite(range_min) or not math.isfinite(range_max) or range_min > range_max:
        raise ValueError("Saved model metadata has an invalid expected_students training range")
    model_step = getattr(pipeline, "named_steps", {}).get("model")
    if model_step is None or type(model_step).__name__ != metadata.get("selected_model"):
        raise ValueError("Saved model artifact does not match the selected model named in metadata")
    pipeline_features = tuple(getattr(pipeline, "feature_names_in_", ()))
    if pipeline_features != FEATURE_COLUMNS:
        raise ValueError("Saved pipeline input features do not match model metadata")
    return pipeline, metadata


def load_model_comparison(path: Path = MODEL_COMPARISON_PATH) -> pd.DataFrame:
    """Read saved validation comparison metrics without retraining."""
    if not path.is_file():
        raise FileNotFoundError(f"Saved model comparison metrics were not found: {path}")
    metrics = pd.read_csv(path)
    required = {"model", "mae", "rmse", "r2"}
    if not required.issubset(metrics.columns):
        raise ValueError(f"Model comparison file is missing columns: {', '.join(sorted(required - set(metrics.columns)))}")
    return metrics


def load_test_predictions(path: Path = TEST_PREDICTIONS_PATH) -> pd.DataFrame:
    """Read saved held-out predictions when available."""
    if not path.is_file():
        raise FileNotFoundError(f"Saved test predictions were not found: {path}")
    frame = pd.read_csv(path)
    required = {"date", "meal_period", "actual_consumed_kg", "predicted_consumed_kg", "dataset_type"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Test prediction file is missing columns: {', '.join(sorted(required - set(frame.columns)))}")
    if not frame.empty and set(frame["dataset_type"].astype(str).str.lower()) != {"synthetic"}:
        raise ValueError("Saved test predictions are not labeled synthetic")
    return frame


def analytics_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Create a downloadable, source-agnostic meal summary."""
    summary = meal_summary(data)
    if not summary.empty:
        summary.insert(0, "dataset_type", source_status(data).replace(" DATA", "").lower())
        summary["discarded_percentage_denominator"] = "prepared_kg"
    return summary
