"""Tests for leakage-safe features, date splits, training, and prediction."""

import csv
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd
import joblib
from unittest.mock import patch

from src.data_generator import FIELDS, generate_records
from src.features import FEATURE_COLUMNS, FORBIDDEN_FEATURES, build_features, build_prediction_input, assert_leakage_safe
from src.predict import expected_students_support_warning, predict_consumption, predict_with_pipeline
from src.train import chronological_date_split, train_and_evaluate


class FeatureEngineeringTests(unittest.TestCase):
    def test_feature_generation_uses_date_derived_fields_and_no_leakage(self) -> None:
        rows = generate_records(days=2, start_date=date(2025, 1, 1))
        frame = pd.DataFrame(rows)
        features = build_features(frame)
        self.assertEqual(tuple(features.columns), FEATURE_COLUMNS)
        self.assertEqual(features.iloc[0]["day_of_week"], 2)
        self.assertEqual(features.iloc[0]["month"], 1)
        self.assertFalse(set(features.columns) & FORBIDDEN_FEATURES)

    def test_leakage_guard_rejects_target_and_actual_attendance(self) -> None:
        with self.assertRaisesRegex(ValueError, "Target leakage"):
            assert_leakage_safe(["expected_students", "consumed_kg", "actual_students"])

    def test_prediction_input_rejects_outcome_fields(self) -> None:
        allowed = {
            "date": "2025-04-01", "expected_students": 400, "meal_period": "Lunch",
            "menu_category": "Mixed", "exam_day": False, "holiday": False,
            "special_event": False, "temperature_c": 25, "rainfall_mm": 0,
        }
        features = build_prediction_input(allowed)
        self.assertEqual(len(features), 1)
        with self.assertRaisesRegex(ValueError, "Unsupported prediction inputs"):
            build_prediction_input({**allowed, "actual_students": 400})

    def test_prediction_input_rejects_invalid_meal_and_non_finite_numbers(self) -> None:
        valid = {
            "date": "2025-04-01", "expected_students": 400, "meal_period": "Lunch",
            "menu_category": "Mixed", "exam_day": False, "holiday": False,
            "special_event": False, "temperature_c": 25, "rainfall_mm": 0,
        }
        with self.assertRaisesRegex(ValueError, "Unsupported meal_period"):
            build_prediction_input({**valid, "meal_period": "Brunch"})
        with self.assertRaisesRegex(ValueError, "not valid for the selected meal"):
            build_prediction_input({**valid, "meal_period": "Breakfast", "menu_category": "North Indian"})
        with self.assertRaisesRegex(ValueError, "finite"):
            build_prediction_input({**valid, "rainfall_mm": float("inf")})
        with self.assertRaisesRegex(ValueError, "whole number"):
            build_prediction_input({**valid, "expected_students": 400.5})

    def test_expected_students_changes_constructed_feature(self) -> None:
        base = {
            "date": "2025-04-01", "meal_period": "Lunch", "menu_category": "Mixed",
            "exam_day": False, "holiday": False, "special_event": False,
            "temperature_c": 25, "rainfall_mm": 0,
        }
        low = build_prediction_input({**base, "expected_students": 200})
        high = build_prediction_input({**base, "expected_students": 600})
        self.assertEqual(low.iloc[0]["expected_students"], 200)
        self.assertEqual(high.iloc[0]["expected_students"], 600)

    def test_inference_receives_each_current_attendance_and_recomputes(self) -> None:
        calls: list[int] = []

        class Encoder:
            categories_ = [["Breakfast", "Lunch", "Snacks", "Dinner"], ["Mixed"]]

        class Preprocessor:
            named_transformers_ = {"categorical": Encoder()}

        class SpyPipeline:
            named_steps = {"preprocessor": Preprocessor()}

            def predict(self, features: pd.DataFrame) -> list[float]:
                attendance = int(features.iloc[0]["expected_students"])
                calls.append(attendance)
                return [attendance * 0.4]

        base = {
            "date": "2025-04-01", "meal_period": "Lunch", "menu_category": "Mixed",
            "exam_day": False, "holiday": False, "special_event": False,
            "temperature_c": 25, "rainfall_mm": 0,
        }
        supported = {"min": 0, "max": 1000}
        first = predict_with_pipeline(SpyPipeline(), {**base, "expected_students": 200}, supported_expected_students_range=supported)
        second = predict_with_pipeline(SpyPipeline(), {**base, "expected_students": 420}, supported_expected_students_range=supported)
        self.assertEqual(calls, [200, 420])
        self.assertEqual((first, second), (80.0, 168.0))

    def test_out_of_range_attendance_gets_transparent_warning(self) -> None:
        warning = expected_students_support_warning(2200, {"min": 57, "max": 730})
        self.assertIsNotNone(warning)
        self.assertIn("2200", warning)
        self.assertIn("57 to 730", warning)
        self.assertIn("no consumption estimate is supported", warning)
        valid_inputs = {
            "date": "2025-04-01", "expected_students": 2200, "meal_period": "Lunch",
            "menu_category": "Mixed", "exam_day": False, "holiday": False,
            "special_event": False, "temperature_c": 25, "rainfall_mm": 0,
        }
        with self.assertRaisesRegex(ValueError, "no consumption estimate is supported"):
            predict_with_pipeline(object(), valid_inputs, supported_expected_students_range={"min": 57, "max": 730})


class ChronologicalSplitTests(unittest.TestCase):
    def test_keeps_all_records_from_a_date_in_one_partition(self) -> None:
        rows = generate_records(days=40)
        frame = pd.DataFrame(rows)
        splits = chronological_date_split(frame)
        date_sets = [set(pd.to_datetime(part["date"]).dt.date) for part in (splits.train, splits.validation, splits.test)]
        self.assertFalse(date_sets[0] & date_sets[1])
        self.assertFalse(date_sets[0] & date_sets[2])
        self.assertFalse(date_sets[1] & date_sets[2])
        self.assertTrue(max(date_sets[0]) < min(date_sets[1]) < min(date_sets[2]))
        self.assertTrue(all(len(group) % 4 == 0 for group in (splits.train, splits.validation, splits.test)))

    def test_rejects_too_few_dates(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least 20"):
            chronological_date_split(pd.DataFrame(generate_records(days=10)))


class TrainingAndPredictionTests(unittest.TestCase):
    def test_training_metrics_artifacts_and_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            csv_path = root / "small_synthetic.csv"
            rows = generate_records(days=40, seed=22, start_date=date(2025, 1, 1))
            with csv_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            result = train_and_evaluate(
                csv_path,
                models_dir=root / "models",
                output_dir=root / "outputs",
                random_forest_trees=8,
            )
            metadata = result["metadata"]
            self.assertEqual(metadata["dataset_status"], "synthetic")
            self.assertEqual(len(result["comparison"]), 5)
            self.assertTrue((root / "models" / "selected_model.joblib").is_file())
            self.assertTrue((root / "models" / "metadata.json").is_file())
            self.assertTrue((root / "outputs" / "validation_model_comparison.csv").is_file())
            self.assertTrue((root / "outputs" / "selected_test_predictions.csv").is_file())
            self.assertEqual(json.loads((root / "models" / "metadata.json").read_text(encoding="utf-8"))["target_variable"], "consumed_kg")
            self.assertEqual(metadata["feature_list"], list(FEATURE_COLUMNS))
            loaded_pipeline = joblib.load(result["selected_pipeline"])
            self.assertEqual(type(loaded_pipeline.named_steps["model"]).__name__, metadata["selected_model"])
            self.assertEqual(tuple(loaded_pipeline.feature_names_in_), FEATURE_COLUMNS)
            fitted_rows = pd.DataFrame(rows).iloc[:136]
            self.assertEqual(metadata["supported_input_ranges"]["expected_students"]["min"], int(fitted_rows["expected_students"].min()))
            self.assertEqual(metadata["supported_input_ranges"]["expected_students"]["max"], int(fitted_rows["expected_students"].max()))
            for metrics in [*result["comparison"], metadata["final_test_metrics"]]:
                self.assertGreaterEqual(metrics["mae"], 0)
                self.assertGreaterEqual(metrics["rmse"], 0)
                self.assertTrue(-1 <= metrics["r2"] <= 1)
            prediction_inputs = {
                "date": "2026-01-15", "expected_students": 420, "meal_period": "Lunch",
                "menu_category": "North Indian", "exam_day": False, "holiday": False,
                "special_event": False, "temperature_c": 27.0, "rainfall_mm": 0.0,
            }
            prediction = predict_consumption(result["selected_pipeline"], prediction_inputs)
            self.assertGreaterEqual(prediction, 0)

    def test_negative_model_prediction_is_warned_before_physical_clamp(self) -> None:
        class NegativePredictionModel:
            def predict(self, features: pd.DataFrame) -> list[float]:
                return [-2.5]

        class Encoder:
            categories_ = [["Breakfast", "Lunch", "Snacks", "Dinner"], ["North Indian"]]

        class Preprocessor:
            named_transformers_ = {"categorical": Encoder()}

        class PipelineStub:
            named_steps = {"preprocessor": Preprocessor()}

            def predict(self, features: pd.DataFrame) -> list[float]:
                return NegativePredictionModel().predict(features)

        inputs = {
            "date": "2025-04-01", "expected_students": 400, "meal_period": "Lunch",
            "menu_category": "North Indian", "exam_day": False, "holiday": False,
            "special_event": False, "temperature_c": 25, "rainfall_mm": 0,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "unused.joblib"
            (Path(temporary_directory) / "metadata.json").write_text(
                json.dumps({"supported_input_ranges": {"expected_students": {"min": 0, "max": 730}}}),
                encoding="utf-8",
            )
            with patch("src.predict.joblib.load", return_value=PipelineStub()):
                with self.assertWarnsRegex(RuntimeWarning, "negative estimate.*Review the model"):
                    result = predict_consumption(model_path, inputs)
        self.assertEqual(result, 0.0)


if __name__ == "__main__":
    unittest.main()
