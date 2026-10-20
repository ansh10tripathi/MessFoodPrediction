"""Load and query the printed mess menu separately from ML datasets.

Menu entries are transcriptions of source material, not meal-consumption
observations. Weekly rotations remain weekday-based unless a source provides
an exact calendar date.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

MENU_ITEMS_PATH = Path(__file__).resolve().parents[1] / "data" / "menu" / "menu_items.csv"
SEASONAL_ITEMS_PATH = Path(__file__).resolve().parents[1] / "data" / "menu" / "seasonal_vegetables.csv"

MEAL_TYPES = ("Breakfast", "Lunch", "Evening Snacks", "Dinner")
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
SEASONS = ("Aug-Oct", "Nov-Feb", "Mar-May")
VERIFICATION_STATUSES = ("verified", "needs_verification")
ROTATION_LABELS = tuple(f"{day} {week}" for week in (1, 2) for day in WEEKDAYS) + tuple(f"Column {number}" for number in range(1, 9))
TIMINGS = {
    "working_day": {
        "Breakfast": "07:30 AM–09:30 AM",
        "Lunch": "11:30 AM–03:00 PM",
        "Evening Snacks": "05:00 PM–06:30 PM",
        "Dinner": "07:30 PM–09:30 PM",
    },
    "sunday_holiday": {
        "Breakfast": "08:00 AM–10:00 AM",
        "Lunch": "12:30 PM–03:00 PM",
        "Evening Snacks": "05:00 PM–06:30 PM",
        "Dinner": "07:30 PM–09:30 PM",
    },
}

MENU_FIELDS = (
    "schedule_type", "menu_date", "day_of_week", "rotation_label", "meal_type", "food_item",
    "cuisine", "category", "verification_status", "source_reference", "notes",
)
SEASONAL_FIELDS = (
    "season", "cuisine", "meal_type", "day_label", "food_item",
    "verification_status", "source_reference", "notes",
)
EXPORT_FIELDS = (
    "record_type", "selected_month", "schedule_type", "menu_date", "day_of_week",
    "rotation_label", "meal_type", "day_label", "food_item", "cuisine", "category", "season", "verification_status",
    "source_reference", "notes",
)


class MenuDataError(ValueError):
    """A menu source is missing, malformed, or contains invalid entries."""


@dataclass(frozen=True)
class MenuData:
    """Separate digitized printed-menu items and seasonal-vegetable tables."""

    items: tuple[dict[str, str], ...]
    seasonal_items: tuple[dict[str, str], ...]


def university_today() -> date:
    """Return today's date in India Standard Time (UTC+05:30; no DST)."""
    india_time = timezone(timedelta(hours=5, minutes=30), name="Asia/Kolkata")
    return datetime.now(india_time).date()


def season_for_date(selected_date: date) -> str | None:
    """Return a season named in the supplied menu brief, or None if unspecified."""
    month = selected_date.month
    if month in (8, 9, 10):
        return "Aug-Oct"
    if month in (11, 12, 1, 2):
        return "Nov-Feb"
    if month in (3, 4, 5):
        return "Mar-May"
    return None


def service_day_type(selected_date: date, *, is_holiday: bool = False) -> str:
    """Choose the provided Sunday/holiday timing when Sunday or user-marked holiday."""
    return "sunday_holiday" if selected_date.weekday() == 6 or is_holiday else "working_day"


def meal_timing(selected_date: date, meal_type: str, *, is_holiday: bool = False) -> str:
    """Return the timing supplied in the feature request for one meal and date."""
    if meal_type not in MEAL_TYPES:
        raise ValueError(f"Unsupported meal type: {meal_type}")
    return TIMINGS[service_day_type(selected_date, is_holiday=is_holiday)][meal_type]


def _read_table(path: Path, expected_fields: tuple[str, ...], label: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise MenuDataError(f"{label} file was not found: {path}")
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames is None:
                raise MenuDataError(f"{label} file is empty and has no header: {path}")
            missing = [field for field in expected_fields if field not in reader.fieldnames]
            if missing:
                raise MenuDataError(f"{label} file is missing columns: {', '.join(missing)}")
            return [
                {field: (row.get(field) or "").strip() for field in expected_fields}
                for row in reader
            ]
    except (OSError, UnicodeError, csv.Error) as error:
        raise MenuDataError(f"Could not read {label} file {path}: {error}") from error


def _validate_menu_items(rows: list[dict[str, str]]) -> None:
    seen: set[tuple[str, ...]] = set()
    for row_number, row in enumerate(rows, start=2):
        schedule = row["schedule_type"]
        if schedule not in {"date_specific", "weekly_rotation"}:
            raise MenuDataError(f"Menu row {row_number}: schedule_type must be date_specific or weekly_rotation")
        if row["meal_type"] not in MEAL_TYPES:
            raise MenuDataError(f"Menu row {row_number}: invalid meal_type {row['meal_type']!r}")
        if not row["food_item"]:
            raise MenuDataError(f"Menu row {row_number}: food_item cannot be blank")
        if row["verification_status"] not in VERIFICATION_STATUSES:
            raise MenuDataError(f"Menu row {row_number}: verification_status must be verified or needs_verification")
        if schedule == "date_specific":
            if not row["menu_date"] or row["day_of_week"] or row["rotation_label"]:
                raise MenuDataError(f"Menu row {row_number}: date_specific rows need menu_date and blank rotation labels")
            try:
                parsed = date.fromisoformat(row["menu_date"])
            except ValueError as error:
                raise MenuDataError(f"Menu row {row_number}: menu_date must be an ISO date") from error
            if row["menu_date"] != parsed.isoformat():
                raise MenuDataError(f"Menu row {row_number}: menu_date must use YYYY-MM-DD format")
        else:
            if row["menu_date"] or row["rotation_label"] not in ROTATION_LABELS:
                raise MenuDataError(f"Menu row {row_number}: weekly_rotation rows need a valid rotation_label and blank menu_date")
            if row["rotation_label"].startswith("Column "):
                if row["day_of_week"]:
                    raise MenuDataError(f"Menu row {row_number}: unlabeled source columns must not be assigned a weekday")
            elif row["day_of_week"] not in WEEKDAYS or row["rotation_label"].rsplit(" ", 1)[0] != row["day_of_week"]:
                raise MenuDataError(f"Menu row {row_number}: day_of_week must agree with rotation_label")
        key = tuple(row[field] for field in ("schedule_type", "menu_date", "day_of_week", "rotation_label", "meal_type", "food_item", "cuisine", "category"))
        if key in seen:
            raise MenuDataError(f"Menu row {row_number}: duplicate menu entry")
        seen.add(key)


def _validate_seasonal_items(rows: list[dict[str, str]]) -> None:
    seen: set[tuple[str, ...]] = set()
    for row_number, row in enumerate(rows, start=2):
        if row["season"] not in SEASONS:
            raise MenuDataError(f"Seasonal row {row_number}: season must be one of {', '.join(SEASONS)}")
        if row["cuisine"] not in {"North Cuisine", "South Cuisine"}:
            raise MenuDataError(f"Seasonal row {row_number}: cuisine must be North Cuisine or South Cuisine")
        if row["meal_type"] not in {"Lunch", "Dinner"}:
            raise MenuDataError(f"Seasonal row {row_number}: meal_type must be Lunch or Dinner")
        if row["day_label"] not in ROTATION_LABELS:
            raise MenuDataError(f"Seasonal row {row_number}: day_label must be a printed weekday rotation label")
        if not row["food_item"]:
            raise MenuDataError(f"Seasonal row {row_number}: food_item cannot be blank")
        if row["verification_status"] not in VERIFICATION_STATUSES:
            raise MenuDataError(f"Seasonal row {row_number}: verification_status must be verified or needs_verification")
        key = tuple(row[field] for field in ("season", "cuisine", "meal_type", "day_label", "food_item"))
        if key in seen:
            raise MenuDataError(f"Seasonal row {row_number}: duplicate seasonal entry")
        seen.add(key)


def _is_displayable(row: dict[str, str]) -> bool:
    """Only named dishes with a supported verification state are displayable."""
    return bool(row.get("food_item", "").strip()) and row.get("verification_status") in VERIFICATION_STATUSES


def load_menu_data(
    menu_path: Path = MENU_ITEMS_PATH,
    seasonal_path: Path = SEASONAL_ITEMS_PATH,
) -> MenuData:
    """Load tables and ignore blank/pending rows that contain no displayable dish."""
    raw_items = _read_table(menu_path, MENU_FIELDS, "Menu")
    raw_seasonal = _read_table(seasonal_path, SEASONAL_FIELDS, "Seasonal menu")
    # The source pack may contain placeholder rows for unreadable cells. They
    # have no food name and are not menu entries; keep them out of all views.
    items = [row for row in raw_items if row["food_item"] and row["verification_status"] != "pending_transcription"]
    seasonal = [row for row in raw_seasonal if row["food_item"] and row["verification_status"] != "pending_transcription"]
    _validate_menu_items(items)
    _validate_seasonal_items(seasonal)
    return MenuData(tuple(items), tuple(seasonal))


def filtered_items(
    rows: tuple[dict[str, str], ...] | list[dict[str, str]],
    *,
    meal_types: set[str] | None = None,
    cuisines: set[str] | None = None,
) -> list[dict[str, str]]:
    """Filter menu rows without changing their source ordering."""
    return [
        row for row in rows
        if _is_displayable(row)
        and (meal_types is None or row.get("meal_type") in meal_types)
        and (cuisines is None or row.get("cuisine") in cuisines or row.get("category") in cuisines)
    ]


def menu_for_date(menu: MenuData, selected_date: date) -> dict[str, dict[str, object]]:
    """Return each meal's exact-date entry, else its weekday rotation if available."""
    day_name = selected_date.strftime("%A")
    result: dict[str, dict[str, object]] = {}
    for meal_type in MEAL_TYPES:
        exact = [
            row for row in menu.items
            if row["schedule_type"] == "date_specific"
            and _is_displayable(row)
            and row["menu_date"] == selected_date.isoformat()
            and row["meal_type"] == meal_type
        ]
        rotation = [
            row for row in menu.items
            if row["schedule_type"] == "weekly_rotation"
            and _is_displayable(row)
            and row["day_of_week"] == day_name
            and row["meal_type"] == meal_type
        ]
        rows = exact or rotation
        result[meal_type] = {
            "items": rows,
            "schedule_type": "date_specific" if exact else "weekly_rotation" if rotation else None,
            "verified": bool(rows) and all(row["verification_status"] == "verified" for row in rows),
        }
    return result


def month_items(
    menu: MenuData,
    year: int,
    month: int,
    *,
    meal_types: set[str] | None = None,
    cuisines: set[str] | None = None,
) -> list[dict[str, str]]:
    """Return only date-specific menu rows in the selected month/year."""
    if month not in range(1, 13):
        raise ValueError("month must be between 1 and 12")
    prefix = f"{year:04d}-{month:02d}-"
    rows = [
        row for row in menu.items
        if row["schedule_type"] == "date_specific" and row["menu_date"].startswith(prefix)
    ]
    return filtered_items(rows, meal_types=meal_types, cuisines=cuisines)


def weekly_rotation_items(
    menu: MenuData,
    *,
    meal_types: set[str] | None = None,
    cuisines: set[str] | None = None,
) -> list[dict[str, str]]:
    """Return weekday rotation rows without mapping them onto calendar dates."""
    rows = [row for row in menu.items if row["schedule_type"] == "weekly_rotation"]
    return filtered_items(rows, meal_types=meal_types, cuisines=cuisines)


def seasonal_items_for_date(menu: MenuData, selected_date: date) -> list[dict[str, str]]:
    """Return seasonal vegetable entries for the explicitly described season."""
    season = season_for_date(selected_date)
    if season is None:
        return []
    return [row for row in menu.seasonal_items if _is_displayable(row) and row["season"] == season]


def export_month_csv(
    menu: MenuData,
    year: int,
    month: int,
    *,
    meal_types: set[str] | None = None,
    cuisines: set[str] | None = None,
) -> bytes:
    """Export selected-month entries, rotation patterns, and that month's season."""
    selected_month = f"{year:04d}-{month:02d}"
    dated = month_items(menu, year, month, meal_types=meal_types, cuisines=cuisines)
    rotation = weekly_rotation_items(menu, meal_types=meal_types, cuisines=cuisines)
    season = season_for_date(date(year, month, 1))
    seasonal = [
        row for row in menu.seasonal_items
        if _is_displayable(row) and row["season"] == season and (cuisines is None or row["cuisine"] in cuisines)
    ]
    output: list[dict[str, str]] = []
    for record_type, rows in (("date_specific", dated), ("weekly_rotation_pattern", rotation), ("seasonal_vegetable", seasonal)):
        for row in rows:
            output.append({
                "record_type": record_type,
                "selected_month": selected_month,
                **{field: row.get(field, "") for field in EXPORT_FIELDS if field not in {"record_type", "selected_month"}},
            })
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=EXPORT_FIELDS)
    writer.writeheader()
    writer.writerows(output)
    return buffer.getvalue().encode("utf-8")
