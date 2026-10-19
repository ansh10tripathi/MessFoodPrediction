"""Create a readable EDA report for the synthetic mess-food CSV."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from src.data_validation import NUMERIC_COLUMNS, read_csv, validate_rows

FOOD_COLUMNS = ("prepared_kg", "consumed_kg", "edible_leftover_kg", "discarded_waste_kg")


def _as_float(value: str) -> float:
    return float(value)


def _format_number(value: float) -> str:
    return f"{value:,.3f}"


def build_report(columns: list[str], rows: list[dict[str, str]]) -> str:
    """Return a Markdown EDA report; measurements remain explicitly synthetic."""
    validation = validate_rows(columns, rows, expected_days=365)
    report: list[str] = [
        "# Exploratory Data Analysis: Synthetic Mess Food Dataset",
        "",
        "> **Synthetic development data only.** These summaries describe generated sample records, "
        "not measured university operations and not real-world predictive performance.",
        "",
        "## Dataset overview",
        "",
        f"- Dimensions: **{len(rows):,} rows × {len(columns)} columns**",
        f"- Columns: {', '.join(f'`{name}`' for name in columns)}",
    ]

    parsed_dates = []
    for row in rows:
        try:
            parsed_dates.append(date.fromisoformat(row["date"]))
        except (KeyError, ValueError):
            pass
    if parsed_dates:
        report.append(f"- Date range: **{min(parsed_dates)} to {max(parsed_dates)}** ({len(set(parsed_dates))} distinct dates)")
    meal_counts = Counter(row.get("meal_period", "") for row in rows)
    report.extend(["", "## Records and data quality", "", "### Records per meal", "", "| Meal | Records |", "|---|---:|"])
    for meal, count in sorted(meal_counts.items()):
        report.append(f"| {meal or '(blank)'} | {count:,} |")
    missing = {column: sum(not (row.get(column) or "").strip() for row in rows) for column in columns}
    missing_nonzero = {column: count for column, count in missing.items() if count}
    duplicates = len(rows) - len({tuple((row.get(c) or "").strip() for c in columns) for row in rows})
    report.extend([
        "", f"- Missing cells: **{sum(missing.values())}**" + (f" ({missing_nonzero})" if missing_nonzero else ""),
        f"- Exact duplicate records: **{duplicates}**",
        f"- Validator status: **{'VALID' if validation.is_valid else 'INVALID'}**",
    ])
    if validation.errors:
        report.extend(["", "Validation findings:"])
        report.extend(f"- {issue}" for issue in validation.errors)
    if validation.warnings:
        report.extend(["", "Validation warnings:"])
        report.extend(f"- {issue}" for issue in validation.warnings)

    report.extend(["", "## Numerical summary statistics", "", "Values are shown in the source units; student counts are people, temperatures °C, rainfall mm, and food quantities kg.", "", "| Column | Count | Mean | Std. dev. | Min | Median | Max |", "|---|---:|---:|---:|---:|---:|---:|"])
    for column in NUMERIC_COLUMNS:
        values = []
        for row in rows:
            try:
                values.append(_as_float(row.get(column, "")))
            except ValueError:
                continue
        if values:
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            report.append(
                f"| `{column}` | {len(values):,} | {_format_number(statistics.mean(values))} | "
                f"{_format_number(std)} | {_format_number(min(values))} | "
                f"{_format_number(statistics.median(values))} | {_format_number(max(values))} |"
            )

    totals = {column: sum(float(row[column]) for row in rows) for column in FOOD_COLUMNS}
    prepared = totals["prepared_kg"]
    discarded_pct = 100.0 * totals["discarded_waste_kg"] / prepared if prepared else 0.0
    report.extend([
        "", "## Food totals", "",
        "Waste percentage is defined here as **total discarded_waste_kg ÷ total prepared_kg × 100**. "
        "Edible leftovers remain a separate category and are not included in discarded waste.",
        "", "| Quantity | Total (kg) |", "|---|---:|",
        f"| Prepared | {_format_number(totals['prepared_kg'])} |",
        f"| Consumed | {_format_number(totals['consumed_kg'])} |",
        f"| Edible leftovers | {_format_number(totals['edible_leftover_kg'])} |",
        f"| Discarded waste | {_format_number(totals['discarded_waste_kg'])} |",
        f"| Discarded waste / prepared | {discarded_pct:.2f}% |",
        "", "## Average consumption and discarded waste by meal", "",
        "| Meal | Records | Mean consumed (kg/record) | Mean discarded (kg/record) |",
        "|---|---:|---:|---:|",
    ])
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"consumed_kg": [], "discarded_waste_kg": []})
    for row in rows:
        meal = row.get("meal_period", "")
        for column in grouped[meal]:
            try:
                grouped[meal][column].append(float(row[column]))
            except (ValueError, KeyError):
                pass
    for meal, values in sorted(grouped.items()):
        consumed = values["consumed_kg"]
        discarded = values["discarded_waste_kg"]
        report.append(
            f"| {meal} | {max(len(consumed), len(discarded)):,} | "
            f"{_format_number(statistics.mean(consumed)) if consumed else 'n/a'} | "
            f"{_format_number(statistics.mean(discarded)) if discarded else 'n/a'} |"
        )
    report.extend([
        "", "## Interpretation and limitations", "",
        "The dataset is programmatically generated with a fixed seed. Its totals and averages describe only the generator's assumptions. "
        "They must not be presented as actual mess measurements or as evidence of model accuracy. "
        "Synthetic weather values are illustrative, not observed forecasts or records.", "",
    ])
    return "\n".join(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", nargs="?", type=Path, default=Path("data/synthetic/mess_food_synthetic.csv"))
    parser.add_argument("--output", type=Path, default=Path("docs/eda_synthetic_report.md"))
    args = parser.parse_args()
    columns, rows = read_csv(args.csv_path)
    report = build_report(columns, rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote synthetic EDA report for {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
