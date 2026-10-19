"""Tests for dashboard source, validation, filtering, and metric utilities."""

import csv
import io
import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.dashboard_utils import (
    filter_records,
    load_model_bundle,
    regression_metrics,
    source_status,
    summary_totals,
    validate_uploaded_csv,
)
from src.data_generator import FIELDS, generate_records


def make_csv_bytes(*, include_source_column: bool = True, days: int = 2) -> bytes:
    rows = generate_records(days=days)
    fields = list(FIELDS)
    if not include_source_column:
        fields.remove("dataset_type")
        rows = [{key: value for key, value in row.items() if key != "dataset_type"} for row in rows]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


class DashboardUploadTests(unittest.TestCase):
    def test_valid_synthetic_upload_is_labeled_synthetic(self) -> None:
        result = validate_uploaded_csv(make_csv_bytes(), "synthetic")
        self.assertTrue(result.is_valid, result.report.errors)
        self.assertEqual(source_status(result.data), "SYNTHETIC DATA")
        self.assertFalse(result.added_source_label_in_memory)

    def test_real_upload_without_source_column_is_labeled_in_memory(self) -> None:
        original = make_csv_bytes(include_source_column=False)
        result = validate_uploaded_csv(original, "real")
        self.assertTrue(result.is_valid, result.report.errors)
        self.assertTrue(result.added_source_label_in_memory)
        self.assertEqual(source_status(result.data), "REAL DATA")
        self.assertNotIn(b"dataset_type", original)
        self.assertTrue(any("user-declared" in item for item in result.report.warnings))

    def test_project_synthetic_csv_passes_with_explicit_synthetic_source(self) -> None:
        csv_path = Path(__file__).resolve().parents[1] / "data" / "synthetic" / "mess_food_synthetic.csv"
        result = validate_uploaded_csv(csv_path.read_bytes(), "synthetic")
        self.assertTrue(result.is_valid, result.report.errors[:5])
        self.assertEqual(result.report.rows_checked, 365 * 4)
        self.assertEqual(source_status(result.data), "SYNTHETIC DATA")

    def test_synthetic_file_cannot_be_declared_real(self) -> None:
        original = make_csv_bytes()
        result = validate_uploaded_csv(original, "real")
        self.assertFalse(result.is_valid)
        self.assertIn("file's dataset_type column contains", result.report.errors[0])
        self.assertIn("synthetic", result.report.errors[0])
        self.assertEqual(source_status(pd.DataFrame([{"dataset_type": "synthetic"}])), "SYNTHETIC DATA")
        self.assertIn(b"dataset_type", original)

    def test_unlabelled_file_requires_a_valid_explicit_source(self) -> None:
        original = make_csv_bytes(include_source_column=False)
        with self.assertRaisesRegex(ValueError, "source_type must be 'real' or 'synthetic'"):
            validate_uploaded_csv(original, "")
        result = validate_uploaded_csv(original, "synthetic")
        self.assertTrue(result.is_valid, result.report.errors)
        self.assertEqual(source_status(result.data), "SYNTHETIC DATA")
        self.assertTrue(any("not independently verified" in warning for warning in result.report.warnings))

    def test_failure_does_not_modify_uploaded_bytes(self) -> None:
        original = make_csv_bytes()
        before = original[:]
        result = validate_uploaded_csv(original, "real")
        self.assertFalse(result.is_valid)
        self.assertEqual(original, before)

    def test_duplicate_csv_headers_are_reported(self) -> None:
        result = validate_uploaded_csv(b"date,date\n2025-01-01,2025-01-02\n", "synthetic")
        self.assertFalse(result.is_valid)
        self.assertIn("duplicate column name", result.report.errors[0])

    def test_invalid_upload_reports_required_schema_errors(self) -> None:
        result = validate_uploaded_csv(b"date,meal_period\n2025-01-01,Lunch\n", "real")
        self.assertFalse(result.is_valid)
        self.assertTrue(any("Missing required columns" in error for error in result.report.errors))

    def test_ragged_csv_rows_are_rejected(self) -> None:
        result = validate_uploaded_csv(b"a,b\n1\n", "real")
        self.assertFalse(result.is_valid)
        self.assertTrue(any("header has" in error for error in result.report.errors))

    def test_missing_model_artifacts_raise_clear_messages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaisesRegex(FileNotFoundError, "Selected model artifact was not found"):
                load_model_bundle(root / "missing.joblib", root / "metadata.json")
            (root / "present.joblib").write_bytes(b"placeholder")
            with self.assertRaisesRegex(FileNotFoundError, "Model metadata was not found"):
                load_model_bundle(root / "present.joblib", root / "missing.json")


class DashboardCalculationTests(unittest.TestCase):
    def test_summary_uses_discarded_waste_over_prepared_denominator(self) -> None:
        data = pd.DataFrame([{
            "date": "2025-01-01", "prepared_kg": "10", "consumed_kg": "7",
            "edible_leftover_kg": "2", "discarded_waste_kg": "1",
        }])
        summary = summary_totals(data)
        self.assertEqual(summary["discarded_pct_prepared"], 10.0)
        self.assertEqual(summary["edible_leftover_kg"], 2.0)

    def test_zero_prepared_total_has_zero_rate(self) -> None:
        data = pd.DataFrame([{
            "date": "2025-01-01", "prepared_kg": "0", "consumed_kg": "0",
            "edible_leftover_kg": "0", "discarded_waste_kg": "0",
        }])
        self.assertEqual(summary_totals(data)["discarded_pct_prepared"], 0.0)

    def test_regression_metrics_are_computed_in_expected_units(self) -> None:
        metrics = regression_metrics([1, 2, 3], [2, 2, 4])
        self.assertAlmostEqual(metrics["mae"], 2 / 3)
        self.assertAlmostEqual(metrics["rmse"], math.sqrt(2 / 3))
        self.assertAlmostEqual(metrics["r2"], 0.0)

    def test_invalid_date_filter_returns_empty_frame(self) -> None:
        data = pd.DataFrame([{
            "date": "2025-01-01", "meal_period": "Lunch", "prepared_kg": "10",
            "consumed_kg": "8", "edible_leftover_kg": "1", "discarded_waste_kg": "1",
        }])
        filtered = filter_records(data, pd.Timestamp("2025-01-02").date(), pd.Timestamp("2025-01-01").date())
        self.assertTrue(filtered.empty)


if __name__ == "__main__":
    unittest.main()
