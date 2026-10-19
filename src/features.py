"""Leakage-safe feature construction and input validation for consumption models."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
import math

import pandas as pd

from src.data_generator import MEALS, MENU_CATEGORIES

TARGET = "consumed_kg"
FORBIDDEN_FEATURES = frozenset({
    "actual_students", "prepared_kg", "consumed_kg", "edible_leftover_kg", "discarded_waste_kg",
})
RAW_INPUT_COLUMNS = (
    "date", "expected_students", "meal_period", "menu_category", "exam_day", "holiday",
    "special_event", "temperature_c", "rainfall_mm",
)
NUMERIC_FEATURES = ("expected_students", "day_of_week", "month", "temperature_c", "rainfall_mm", "exam_day", "holiday", "special_event")
CATEGORICAL_FEATURES = ("meal_period", "menu_category")
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def assert_leakage_safe(feature_names: tuple[str, ...] | list[str] = FEATURE_COLUMNS) -> None:
    """Fail fast if a target or post-meal observation enters the feature list."""
    leaked = sorted(set(feature_names) & FORBIDDEN_FEATURES)
    if leaked:
        raise ValueError(f"Target leakage detected in feature list: {', '.join(leaked)}")


def _parse_boolean_column(series: pd.Series, name: str) -> pd.Series:
    """Convert true/false, bool, or 0/1 values to numeric indicators."""
    mapping = {"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0}
    normalized = series.astype("string").str.strip().str.lower()
    converted = normalized.map(mapping)
    invalid = converted.isna()
    if invalid.any():
        values = sorted(normalized[invalid].dropna().unique().tolist())
        raise ValueError(f"Invalid values in {name}: {values or ['missing']}")
    return converted.astype(float)


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Derive model features from known-before-meal fields without target leakage.

    Temperature and rainfall fields are assumed to represent forecast values at
    prediction time. The starter CSV contains synthetic illustrative values.
    """
    missing = [column for column in RAW_INPUT_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing prediction input columns: {', '.join(missing)}")
    assert_leakage_safe()

    dates = pd.to_datetime(data["date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("date must contain valid calendar dates")

    features = pd.DataFrame(index=data.index)
    features["expected_students"] = pd.to_numeric(data["expected_students"], errors="coerce")
    features["day_of_week"] = dates.dt.dayofweek
    features["month"] = dates.dt.month
    features["temperature_c"] = pd.to_numeric(data["temperature_c"], errors="coerce")
    features["rainfall_mm"] = pd.to_numeric(data["rainfall_mm"], errors="coerce")
    for flag in ("exam_day", "holiday", "special_event"):
        features[flag] = _parse_boolean_column(data[flag], flag)
    features["meal_period"] = data["meal_period"].astype("string").str.strip()
    features["menu_category"] = data["menu_category"].astype("string").str.strip()

    invalid_numeric = [name for name in NUMERIC_FEATURES if features[name].isna().any()]
    if invalid_numeric:
        raise ValueError(f"Missing or invalid numeric feature values: {', '.join(invalid_numeric)}")
    non_finite = [name for name in NUMERIC_FEATURES if not all(math.isfinite(float(value)) for value in features[name])]
    if non_finite:
        raise ValueError(f"Numeric inputs must be finite: {', '.join(non_finite)}")
    if (features["expected_students"] < 0).any() or (features["expected_students"] % 1 != 0).any():
        raise ValueError("expected_students must be a non-negative whole number")
    if (features["temperature_c"] < -90).any() or (features["temperature_c"] > 65).any() or (features["rainfall_mm"] < 0).any():
        raise ValueError("Input contains an impossible expected attendance or weather value")
    for name in CATEGORICAL_FEATURES:
        if features[name].isna().any() or (features[name].str.len() == 0).any():
            raise ValueError(f"{name} cannot be missing or blank")
    invalid_meals = sorted(set(features["meal_period"].dropna()) - set(MEALS))
    if invalid_meals:
        raise ValueError(f"Unsupported meal_period value(s): {', '.join(invalid_meals)}")
    known_menu_categories = {category for categories in MENU_CATEGORIES.values() for category in categories}
    invalid_known_pairs = sorted({
        (meal, menu)
        for meal, menu in zip(features["meal_period"], features["menu_category"])
        if menu in known_menu_categories
        and menu not in MENU_CATEGORIES[meal]
    })
    if invalid_known_pairs:
        pairs = ", ".join(f"{meal}/{menu}" for meal, menu in invalid_known_pairs)
        raise ValueError(f"Menu category is not valid for the selected meal: {pairs}")
    return features.loc[:, FEATURE_COLUMNS]


def build_prediction_input(values: Mapping[str, object]) -> pd.DataFrame:
    """Validate the user-facing input mapping and return one feature row."""
    unknown = set(values) - set(RAW_INPUT_COLUMNS)
    missing = set(RAW_INPUT_COLUMNS) - set(values)
    if unknown:
        raise ValueError(f"Unsupported prediction inputs: {', '.join(sorted(unknown))}")
    if missing:
        raise ValueError(f"Missing prediction inputs: {', '.join(sorted(missing))}")
    row = dict(values)
    if isinstance(row["date"], date):
        row["date"] = row["date"].isoformat()
    return build_features(pd.DataFrame([row]))
