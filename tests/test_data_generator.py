"""Basic tests for synthetic data generation and food accounting."""

import unittest
from datetime import date

from src.data_generator import MEALS, generate_records, validate_food_accounting


class SyntheticDataGeneratorTests(unittest.TestCase):
    def test_generates_four_synthetic_meals_per_day(self) -> None:
        rows = generate_records(days=3, seed=17, start_date=date(2025, 1, 1))
        self.assertEqual(len(rows), 12)
        self.assertEqual([row["meal_period"] for row in rows[:4]], list(MEALS))
        self.assertTrue(all(row["dataset_type"] == "synthetic" for row in rows))

    def test_same_seed_produces_same_records(self) -> None:
        self.assertEqual(generate_records(days=2, seed=10), generate_records(days=2, seed=10))

    def test_rejects_non_positive_days(self) -> None:
        with self.assertRaises(ValueError):
            generate_records(days=0)

    def test_generated_food_accounting_is_consistent(self) -> None:
        rows = generate_records(days=30)
        validate_food_accounting(rows)
        for row in rows:
            prepared = float(row["prepared_kg"])
            accounted = sum(float(row[key]) for key in ("consumed_kg", "edible_leftover_kg", "discarded_waste_kg"))
            self.assertLessEqual(abs(prepared - accounted), 1e-6)

    def test_food_accounting_rejects_inconsistent_row(self) -> None:
        row = {
            "prepared_kg": "10.0",
            "consumed_kg": "7.0",
            "edible_leftover_kg": "2.0",
            "discarded_waste_kg": "0.5",
        }
        with self.assertRaises(ValueError):
            validate_food_accounting([row])


if __name__ == "__main__":
    unittest.main()
