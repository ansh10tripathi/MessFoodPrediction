"""Load the selected model pipeline and predict pre-meal consumption."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any
import warnings

import joblib

from src.features import build_prediction_input


def expected_students_support_warning(
    value: int | float,
    supported_range: dict[str, float] | None,
) -> str | None:
    """Describe attendance values outside the observed model-training interval."""
    if supported_range is None:
        return "The model's expected_students training range is unavailable; no supported estimate can be returned."
    try:
        minimum = float(supported_range["min"])
        maximum = float(supported_range["max"])
        numeric_value = float(value)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Model expected_students training range metadata is invalid") from error
    if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum > maximum:
        raise ValueError("Model expected_students training range metadata is invalid")
    if numeric_value < minimum or numeric_value > maximum:
        return (
            f"Expected students ({numeric_value:g}) is outside the synthetic training range "
            f"({minimum:g} to {maximum:g}). The fitted model cannot extrapolate reliably; "
            "no consumption estimate is supported for this input."
        )
    return None


def predict_with_pipeline(
    model: Any,
    inputs: dict[str, Any],
    *,
    supported_expected_students_range: dict[str, float] | None = None,
) -> float:
    """Return an estimate from an already loaded pipeline and pre-meal inputs."""
    features = build_prediction_input(inputs)
    range_warning = expected_students_support_warning(
        inputs["expected_students"], supported_expected_students_range
    )
    if range_warning:
        raise ValueError(range_warning)
    encoder = model.named_steps["preprocessor"].named_transformers_["categorical"]
    known_menu_categories = set(encoder.categories_[1])
    menu_category = str(inputs["menu_category"]).strip()
    if menu_category not in known_menu_categories:
        warnings.warn(
            f"Menu category {menu_category!r} was not seen during training; "
            "the one-hot encoder will represent it with all-zero menu-category indicators.",
            RuntimeWarning,
            stacklevel=2,
        )
    prediction = float(model.predict(features)[0])
    if not math.isfinite(prediction):
        raise RuntimeError(f"Model returned a non-finite consumption estimate: {prediction}")
    if prediction < 0:
        warnings.warn(
            f"Model returned an implausible negative estimate ({prediction:.3f} kg); "
            "returning the physical floor of 0 kg. Review the model and inputs.",
            RuntimeWarning,
            stacklevel=2,
        )
        prediction = 0.0
    return prediction


def predict_consumption(model_path: Path, inputs: dict[str, Any]) -> float:
    """Load a saved pipeline and return estimated consumed kilograms."""
    model = joblib.load(model_path)
    metadata_path = model_path.with_name("metadata.json")
    if not metadata_path.is_file():
        raise ValueError(f"Model training-range metadata was not found: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        ranges_metadata = metadata.get("supported_input_ranges")
        supported_range = ranges_metadata.get("expected_students") if isinstance(ranges_metadata, dict) else None
    except (OSError, json.JSONDecodeError, AttributeError) as error:
        raise ValueError(f"Could not read model training-range metadata: {error}") from error
    return predict_with_pipeline(
        model,
        inputs,
        supported_expected_students_range=supported_range,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/synthetic/selected_model.joblib"))
    parser.add_argument("--input-json", required=True, help="JSON object with pre-meal fields only")
    args = parser.parse_args()
    try:
        inputs = json.loads(args.input_json)
        if not isinstance(inputs, dict):
            raise ValueError("--input-json must decode to an object of prediction fields")
        prediction = predict_consumption(args.model, inputs)
    except (json.JSONDecodeError, ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Synthetic-data model estimate: {prediction:.2f} kg consumed (estimate, not a guaranteed amount).")
    print("Predicted consumption is not the same as the amount that must be prepared.")


if __name__ == "__main__":
    main()
