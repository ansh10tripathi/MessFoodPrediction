"""Generate reproducible, clearly labeled synthetic mess meal records.

This module intentionally uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import random
from datetime import date, timedelta
from pathlib import Path

MEALS = ("Breakfast", "Lunch", "Snacks", "Dinner")
MENU_CATEGORIES = {
    "Breakfast": ("Indian", "Continental", "Light"),
    "Lunch": ("North Indian", "South Indian", "Rice-based", "Mixed"),
    "Snacks": ("Savory", "Fried", "Beverage and snack"),
    "Dinner": ("North Indian", "South Indian", "Rice-based", "Mixed"),
}
EXPECTED_BY_MEAL = {"Breakfast": 420, "Lunch": 650, "Snacks": 300, "Dinner": 570}
CONSUMPTION_KG_PER_STUDENT = {"Breakfast": 0.32, "Lunch": 0.48, "Snacks": 0.22, "Dinner": 0.44}
ACCOUNTING_TOLERANCE_KG = 1e-6
FIELDS = (
    "date", "meal_period", "menu_category", "expected_students", "actual_students",
    "exam_day", "holiday", "special_event", "temperature_c", "rainfall_mm",
    "prepared_kg", "consumed_kg", "edible_leftover_kg", "discarded_waste_kg", "dataset_type",
)


def generate_records(days: int = 365, seed: int = 395, start_date: date = date(2025, 1, 1)) -> list[dict[str, object]]:
    """Return `days * 4` synthetic meal rows, deterministically for a given seed."""
    if days <= 0:
        raise ValueError("days must be a positive integer")

    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    for offset in range(days):
        current = start_date + timedelta(days=offset)
        weekday = current.weekday()
        holiday = rng.random() < 0.035
        exam_day = (not holiday) and rng.random() < 0.08
        special_event = rng.random() < 0.025
        temperature = round(rng.uniform(17.0, 36.0), 1)
        rainfall = round(0.0 if rng.random() < 0.70 else rng.uniform(0.2, 24.0), 1)

        for meal in MEALS:
            expected = EXPECTED_BY_MEAL[meal]
            attendance_factor = 0.88 if weekday >= 5 else 1.0
            if holiday:
                attendance_factor *= 0.22
            if exam_day:
                attendance_factor *= 0.87
            if special_event:
                attendance_factor *= 1.10
            actual = max(0, round(expected * attendance_factor * rng.uniform(0.91, 1.08)))
            expected_for_meal = max(0, round(expected * attendance_factor * rng.uniform(0.97, 1.03)))
            menu = rng.choice(MENU_CATEGORIES[meal])
            per_student = CONSUMPTION_KG_PER_STUDENT[meal] * rng.uniform(0.88, 1.12)
            if menu in {"Fried", "North Indian", "Mixed"}:
                per_student *= 1.04
            consumed = actual * per_student

            surplus_rate = rng.uniform(0.06, 0.20)
            surplus = consumed * surplus_rate
            edible_share = rng.uniform(0.45, 0.78)
            edible_leftover = surplus * edible_share
            discarded_waste = surplus - edible_leftover
            # Define preparation as the exact sum of stored components.
            consumed = round(consumed, 6)
            edible_leftover = round(edible_leftover, 6)
            discarded_waste = round(discarded_waste, 6)
            prepared = round(consumed + edible_leftover + discarded_waste, 6)

            rows.append({
                "date": current.isoformat(),
                "meal_period": meal,
                "menu_category": menu,
                "expected_students": expected_for_meal,
                "actual_students": actual,
                "exam_day": str(exam_day).lower(),
                "holiday": str(holiday).lower(),
                "special_event": str(special_event).lower(),
                "temperature_c": temperature,
                "rainfall_mm": rainfall,
                "prepared_kg": f"{prepared:.6f}",
                "consumed_kg": f"{consumed:.6f}",
                "edible_leftover_kg": f"{edible_leftover:.6f}",
                "discarded_waste_kg": f"{discarded_waste:.6f}",
                "dataset_type": "synthetic",
            })
    return rows


def validate_food_accounting(rows: list[dict[str, object]], tolerance_kg: float = ACCOUNTING_TOLERANCE_KG) -> None:
    """Raise ValueError if any row violates the prepared-food accounting identity."""
    if tolerance_kg < 0:
        raise ValueError("tolerance_kg must be non-negative")
    for index, row in enumerate(rows, start=1):
        prepared = float(row["prepared_kg"])
        accounted = sum(float(row[key]) for key in ("consumed_kg", "edible_leftover_kg", "discarded_waste_kg"))
        if abs(prepared - accounted) > tolerance_kg:
            raise ValueError(f"row {index} food accounting differs by {abs(prepared - accounted):.9f} kg")


def write_dataset(output: Path, rows: list[dict[str, object]]) -> None:
    """Write CSV without overwriting an existing file."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/synthetic/mess_food_synthetic.csv"))
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--seed", type=int, default=395)
    parser.add_argument("--start-date", type=date.fromisoformat, default=date(2025, 1, 1))
    args = parser.parse_args()

    rows = generate_records(days=args.days, seed=args.seed, start_date=args.start_date)
    validate_food_accounting(rows)
    write_dataset(args.output, rows)
    print(f"Wrote {len(rows)} synthetic records to {args.output}")
    print(f"Accounting checked with absolute tolerance {ACCOUNTING_TOLERANCE_KG:g} kg")


if __name__ == "__main__":
    main()
