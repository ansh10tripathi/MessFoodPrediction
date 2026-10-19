"""Validation helpers for mess meal CSV files (standard library only)."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from src.data_generator import ACCOUNTING_TOLERANCE_KG, FIELDS, MEALS

NUMERIC_COLUMNS = (
    "expected_students", "actual_students", "temperature_c", "rainfall_mm",
    "prepared_kg", "consumed_kg", "edible_leftover_kg", "discarded_waste_kg",
)
NON_NEGATIVE_COLUMNS = (
    "expected_students", "actual_students", "rainfall_mm", "prepared_kg", "consumed_kg",
    "edible_leftover_kg", "discarded_waste_kg",
)
BOOLEAN_COLUMNS = ("exam_day", "holiday", "special_event")
VALID_BOOLEAN_VALUES = {"true", "false"}


@dataclass
class ValidationReport:
    """Issues collected while validating one CSV dataset."""

    rows_checked: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Load a CSV while preserving cell values as strings for validation."""
    with path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            return [], []
        return list(reader.fieldnames), list(reader)


def validate_rows(
    columns: list[str],
    rows: list[dict[str, str]],
    *,
    expected_dataset_type: str = "synthetic",
    expected_days: int | None = None,
    tolerance_kg: float = ACCOUNTING_TOLERANCE_KG,
) -> ValidationReport:
    """Validate schema, cells, uniqueness, coverage, quantities, labels, and accounting.

    A date range must be contiguous, and every represented date must have exactly
    one record for each configured meal. `expected_days`, when supplied, also
    checks the number of distinct dates (for example, 365 for the starter CSV).
    """
    report = ValidationReport(rows_checked=len(rows))
    missing_columns = [name for name in FIELDS if name not in columns]
    if missing_columns:
        report.errors.append(f"Missing required columns: {', '.join(missing_columns)}")
    extra_columns = [name for name in columns if name not in FIELDS]
    if extra_columns:
        report.warnings.append(f"Extra columns are present: {', '.join(extra_columns)}")
    if tolerance_kg < 0:
        report.errors.append("Accounting tolerance must be non-negative")
        return report
    if not rows:
        report.errors.append("Dataset contains no data rows")
        return report
    if missing_columns:
        # Continue with available fields to report all schema-level omissions,
        # but skip row checks that cannot be performed safely.
        return report

    seen_records: set[tuple[str, ...]] = set()
    dates: set[date] = set()
    meal_counts: Counter[tuple[date, str]] = Counter()
    date_errors: set[str] = set()

    for row_number, row in enumerate(rows, start=2):
        normalized = tuple((row.get(column) or "").strip() for column in columns)
        if normalized in seen_records:
            report.errors.append(f"Row {row_number}: duplicate record")
        seen_records.add(normalized)

        for column in FIELDS:
            if not (row.get(column) or "").strip():
                report.errors.append(f"Row {row_number}: missing value in '{column}'")

        raw_date = (row.get("date") or "").strip()
        parsed_date: date | None = None
        try:
            parsed_date = date.fromisoformat(raw_date)
            dates.add(parsed_date)
        except ValueError:
            date_errors.add(raw_date)
            report.errors.append(f"Row {row_number}: date '{raw_date}' is not a valid ISO date")

        meal = (row.get("meal_period") or "").strip()
        if meal not in MEALS:
            report.errors.append(f"Row {row_number}: invalid meal_period '{meal}'")
        elif parsed_date is not None:
            meal_counts[(parsed_date, meal)] += 1

        if (row.get("dataset_type") or "").strip().lower() != expected_dataset_type.lower():
            report.errors.append(
                f"Row {row_number}: dataset_type must be '{expected_dataset_type}'"
            )

        for column in BOOLEAN_COLUMNS:
            value = (row.get(column) or "").strip().lower()
            if value not in VALID_BOOLEAN_VALUES:
                report.errors.append(f"Row {row_number}: '{column}' must be true or false")

        values: dict[str, float] = {}
        for column in NUMERIC_COLUMNS:
            raw_value = (row.get(column) or "").strip()
            try:
                value = float(raw_value)
                if not math.isfinite(value):
                    raise ValueError
                values[column] = value
            except ValueError:
                if raw_value:
                    report.errors.append(f"Row {row_number}: '{column}' must be a finite number")
                continue
            if column in NON_NEGATIVE_COLUMNS and value < 0:
                report.errors.append(f"Row {row_number}: '{column}' cannot be negative")
            if column in {"expected_students", "actual_students"} and not value.is_integer():
                report.errors.append(f"Row {row_number}: '{column}' must be a whole number")
        if values.get("temperature_c", 0.0) < -90 or values.get("temperature_c", 0.0) > 65:
            report.errors.append(f"Row {row_number}: temperature_c is outside plausible bounds (-90..65)")

        food_columns = ("prepared_kg", "consumed_kg", "edible_leftover_kg", "discarded_waste_kg")
        if all(name in values for name in food_columns):
            prepared = values["prepared_kg"]
            accounted = sum(values[name] for name in food_columns[1:])
            difference = abs(prepared - accounted)
            if difference > tolerance_kg:
                report.errors.append(
                    f"Row {row_number}: food accounting differs by {difference:.9g} kg "
                    f"(tolerance {tolerance_kg:g} kg)"
                )

    # Date coverage and exact per-day meal representation.
    if dates:
        first, last = min(dates), max(dates)
        expected_dates: set[date] = set()
        current = first
        while current <= last:
            expected_dates.add(current)
            current += timedelta(days=1)
        gaps = sorted(expected_dates - dates)
        if gaps:
            report.errors.append(f"Date coverage has {len(gaps)} missing date(s), from {gaps[0]} to {gaps[-1]}")
        if expected_days is not None and len(dates) != expected_days:
            report.errors.append(f"Expected {expected_days} distinct dates; found {len(dates)}")
        for current in sorted(dates):
            for expected_meal in MEALS:
                count = meal_counts[(current, expected_meal)]
                if count != 1:
                    report.errors.append(
                        f"Date {current}: expected exactly one {expected_meal} record; found {count}"
                    )
    elif not date_errors:
        report.errors.append("No valid dates found")

    return report


def validate_csv(path: Path, *, expected_days: int | None = None) -> ValidationReport:
    """Read and validate a CSV path."""
    columns, rows = read_csv(path)
    return validate_rows(columns, rows, expected_days=expected_days)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", nargs="?", type=Path, default=Path("data/synthetic/mess_food_synthetic.csv"))
    parser.add_argument("--expected-days", type=int, default=365)
    args = parser.parse_args()
    try:
        report = validate_csv(args.csv_path, expected_days=args.expected_days)
    except (OSError, UnicodeError, csv.Error) as error:
        print(f"Could not read {args.csv_path}: {error}")
        raise SystemExit(2) from error

    print(f"Dataset: {args.csv_path} (expected synthetic)")
    print(f"Rows checked: {report.rows_checked}")
    print(f"Status: {'VALID' if report.is_valid else 'INVALID'}")
    for warning in report.warnings:
        print(f"WARNING: {warning}")
    for error in report.errors:
        print(f"ERROR: {error}")
    raise SystemExit(0 if report.is_valid else 1)


if __name__ == "__main__":
    main()
