"""Tests for CSV schema, quality, coverage, and accounting validation."""

import unittest
from datetime import date

from src.data_generator import FIELDS, generate_records
from src.data_validation import validate_rows


def rows_as_strings(days: int = 2) -> list[dict[str, str]]:
    return [{key: str(value) for key, value in row.items()} for row in generate_records(days=days, start_date=date(2025, 1, 1))]


class DataValidationTests(unittest.TestCase):
    def test_accepts_complete_valid_synthetic_data(self) -> None:
        report = validate_rows(list(FIELDS), rows_as_strings(), expected_days=2)
        self.assertTrue(report.is_valid, report.errors)

    def test_reports_missing_values(self) -> None:
        rows = rows_as_strings()
        rows[0]["menu_category"] = ""
        report = validate_rows(list(FIELDS), rows, expected_days=2)
        self.assertTrue(any("missing value in 'menu_category'" in error for error in report.errors))

    def test_reports_duplicate_records_and_duplicate_meal_slot(self) -> None:
        rows = rows_as_strings()
        rows.append(rows[0].copy())
        report = validate_rows(list(FIELDS), rows, expected_days=2)
        self.assertTrue(any("duplicate record" in error for error in report.errors))
        self.assertTrue(any("expected exactly one Breakfast" in error for error in report.errors))

    def test_reports_negative_quantity(self) -> None:
        rows = rows_as_strings()
        rows[0]["edible_leftover_kg"] = "-0.25"
        report = validate_rows(list(FIELDS), rows, expected_days=2)
        self.assertTrue(any("edible_leftover_kg' cannot be negative" in error for error in report.errors))

    def test_reports_accounting_violation(self) -> None:
        rows = rows_as_strings()
        rows[0]["prepared_kg"] = "999.0"
        report = validate_rows(list(FIELDS), rows, expected_days=2)
        self.assertTrue(any("food accounting differs" in error for error in report.errors))

    def test_reports_invalid_meal_label_and_synthetic_label(self) -> None:
        rows = rows_as_strings()
        rows[0]["meal_period"] = "Brunch"
        rows[1]["dataset_type"] = "real"
        report = validate_rows(list(FIELDS), rows, expected_days=2)
        self.assertTrue(any("invalid meal_period" in error for error in report.errors))
        self.assertTrue(any("dataset_type must be 'synthetic'" in error for error in report.errors))

    def test_reports_missing_date_and_required_meal_slot(self) -> None:
        rows = rows_as_strings()
        rows = [row for row in rows if not (row["date"] == "2025-01-01" and row["meal_period"] == "Lunch")]
        report = validate_rows(list(FIELDS), rows, expected_days=2)
        self.assertTrue(any("expected exactly one Lunch" in error for error in report.errors))


if __name__ == "__main__":
    unittest.main()
