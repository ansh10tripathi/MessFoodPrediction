"""Tests for the separate printed-menu data and query utilities."""

import csv
from datetime import date
from pathlib import Path

import pytest

from src.menu import (
    MENU_FIELDS,
    SEASONAL_FIELDS,
    MenuData,
    MenuDataError,
    export_month_csv,
    filtered_items,
    load_menu_data,
    meal_timing,
    menu_for_date,
    month_items,
    seasonal_items_for_date,
    university_today,
    weekly_rotation_items,
)


def write_table(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_project_menu_files_load_transcribed_rotation_and_seasonal_tables() -> None:
    menu = load_menu_data()
    assert isinstance(menu, MenuData)
    assert len(menu.items) == 256
    assert len(menu.seasonal_items) == 39
    assert {row["meal_type"] for row in menu.items} == {"Breakfast", "Lunch", "Evening Snacks", "Dinner"}
    assert {row["season"] for row in menu.seasonal_items} == {"Aug-Oct", "Nov-Feb", "Mar-May"}
    assert {row["cuisine"] for row in menu.seasonal_items} == {"North Cuisine", "South Cuisine"}
    assert all(row["food_item"].strip() for row in menu.items + menu.seasonal_items)
    assert any(row["rotation_label"] == "Sunday 2" for row in menu.items)
    assert not any(row["rotation_label"].startswith("Column ") for row in menu.items)
    assert len(filtered_items(menu.items, cuisines={"North Indian"})) > 0
    today_preview = menu_for_date(menu, date(2026, 10, 20))
    assert any(row["food_item"] == "Poori" and row["rotation_label"] == "Tuesday 1" for row in today_preview["Breakfast"]["items"])
    assert {row["rotation_label"] for row in today_preview["Lunch"]["items"]} == {"Tuesday 1"}
    assert {row["rotation_label"] for row in today_preview["Evening Snacks"]["items"]} == {"Tuesday 1"}
    sunday_preview = menu_for_date(menu, date(2026, 10, 25))["Breakfast"]["items"]
    assert {row["rotation_label"] for row in sunday_preview} == {"Sunday 1", "Sunday 2"}


def test_all_four_meals_map_to_their_printed_weekday_rotation() -> None:
    menu = load_menu_data()
    preview = menu_for_date(menu, date(2026, 10, 19))  # Monday
    assert set(preview) == {"Breakfast", "Lunch", "Evening Snacks", "Dinner"}
    for meal, detail in preview.items():
        assert detail["items"], meal
        assert {row["rotation_label"] for row in detail["items"]} == {"Monday 1"}
        assert {row["day_of_week"] for row in detail["items"]} == {"Monday"}


def test_date_mapping_leaves_absent_rotation_meal_empty() -> None:
    menu = MenuData(items=({
        "schedule_type": "weekly_rotation", "menu_date": "", "day_of_week": "Monday", "rotation_label": "Monday 1",
        "meal_type": "Breakfast", "food_item": "Only breakfast", "cuisine": "", "category": "Main",
        "verification_status": "verified", "source_reference": "test", "notes": "",
    },), seasonal_items=())
    preview = menu_for_date(menu, date(2026, 8, 3))
    assert preview["Breakfast"]["items"][0]["food_item"] == "Only breakfast"
    assert all(not preview[meal]["items"] for meal in ("Lunch", "Evening Snacks", "Dinner"))


def test_weekday_rotation_matching_and_meal_cuisine_filters() -> None:
    menu = load_menu_data()
    monday_lunch = [row for row in weekly_rotation_items(menu, meal_types={"Lunch"}, cuisines={"South Indian"}) if row["rotation_label"] == "Monday 1"]
    assert monday_lunch
    assert all(row["meal_type"] == "Lunch" and row["cuisine"] == "South Indian" for row in monday_lunch)
    assert not [row for row in weekly_rotation_items(menu, meal_types={"Dinner"}) if row["rotation_label"] == "Monday 1" and row["meal_type"] != "Dinner"]


def test_loads_valid_weekly_and_seasonal_rows_from_temporary_files(tmp_path: Path) -> None:
    menu_path, seasonal_path = tmp_path / "menu.csv", tmp_path / "seasonal.csv"
    write_table(menu_path, MENU_FIELDS, [{
        "schedule_type": "weekly_rotation", "menu_date": "", "day_of_week": "Monday", "rotation_label": "Monday 1",
        "meal_type": "Breakfast", "food_item": "Test item", "cuisine": "North Indian", "category": "Main",
        "verification_status": "needs_verification", "source_reference": "temporary fixture", "notes": "",
    }])
    write_table(seasonal_path, SEASONAL_FIELDS, [{
        "season": "Aug-Oct", "cuisine": "South Cuisine", "meal_type": "Lunch", "day_label": "Thursday 1",
        "food_item": "Test vegetable", "verification_status": "verified", "source_reference": "temporary fixture", "notes": "",
    }])
    menu = load_menu_data(menu_path, seasonal_path)
    assert menu.items[0]["verification_status"] == "needs_verification"
    assert menu.seasonal_items[0]["cuisine"] == "South Cuisine"


def test_blank_pending_transcription_placeholders_are_ignored(tmp_path: Path) -> None:
    menu_path, seasonal_path = tmp_path / "menu.csv", tmp_path / "seasonal.csv"
    write_table(menu_path, MENU_FIELDS, [{
        "schedule_type": "", "menu_date": "", "day_of_week": "", "rotation_label": "", "meal_type": "Breakfast",
        "food_item": "", "cuisine": "", "category": "", "verification_status": "pending_transcription",
        "source_reference": "enhanced photo", "notes": "text not legible",
    }])
    write_table(seasonal_path, SEASONAL_FIELDS, [{
        "season": "Aug-Oct", "cuisine": "North Cuisine", "meal_type": "Lunch", "day_label": "Monday 1", "food_item": "",
        "verification_status": "pending_transcription", "source_reference": "enhanced photo",
        "notes": "text not legible",
    }])
    menu = load_menu_data(menu_path, seasonal_path)
    assert menu.items == ()
    assert menu.seasonal_items == ()


def test_named_pending_rows_never_appear_in_queries_or_exports() -> None:
    placeholder = {
        "schedule_type": "weekly_rotation", "menu_date": "", "day_of_week": "Monday", "rotation_label": "Monday 1",
        "meal_type": "Breakfast", "food_item": "Unverified placeholder", "cuisine": "North Cuisine", "category": "Main",
        "verification_status": "pending_transcription", "source_reference": "test", "notes": "pending",
    }
    menu = MenuData(items=(placeholder,), seasonal_items=())
    assert weekly_rotation_items(menu) == []
    assert menu_for_date(menu, date(2026, 8, 3))["Breakfast"]["items"] == []
    assert "Unverified placeholder" not in export_month_csv(menu, 2026, 8).decode("utf-8")


def test_rejects_duplicate_rows_instead_of_silently_deduplicating(tmp_path: Path) -> None:
    menu_path, seasonal_path = tmp_path / "menu.csv", tmp_path / "seasonal.csv"
    row = {"schedule_type": "weekly_rotation", "menu_date": "", "day_of_week": "Monday", "rotation_label": "Monday 1",
           "meal_type": "Breakfast", "food_item": "Test item", "cuisine": "North Indian", "category": "Main",
           "verification_status": "needs_verification", "source_reference": "temporary fixture", "notes": ""}
    write_table(menu_path, MENU_FIELDS, [row, row])
    write_table(seasonal_path, SEASONAL_FIELDS, [])
    with pytest.raises(MenuDataError, match="duplicate menu entry"):
        load_menu_data(menu_path, seasonal_path)


def test_missing_or_malformed_menu_files_raise_clear_error(tmp_path: Path) -> None:
    with pytest.raises(MenuDataError, match="not found"):
        load_menu_data(tmp_path / "missing.csv", tmp_path / "seasonal.csv")
    malformed = tmp_path / "menu.csv"
    malformed.write_text("wrong,header\n", encoding="utf-8")
    seasonal = tmp_path / "seasonal.csv"
    write_table(seasonal, SEASONAL_FIELDS, [])
    with pytest.raises(MenuDataError, match="missing columns"):
        load_menu_data(malformed, seasonal)


def test_invalid_meal_category_is_rejected(tmp_path: Path) -> None:
    menu_path, seasonal_path = tmp_path / "menu.csv", tmp_path / "seasonal.csv"
    write_table(menu_path, MENU_FIELDS, [{
        "schedule_type": "weekly_rotation", "menu_date": "", "day_of_week": "Monday", "rotation_label": "Monday 1",
        "meal_type": "Brunch", "food_item": "Sample", "cuisine": "North Indian", "category": "Main",
        "verification_status": "needs_verification", "source_reference": "test image", "notes": "temporary test row",
    }])
    write_table(seasonal_path, SEASONAL_FIELDS, [])
    with pytest.raises(MenuDataError, match="invalid meal_type"):
        load_menu_data(menu_path, seasonal_path)


def test_date_month_rotation_season_and_timing_queries() -> None:
    menu = MenuData(
        items=(
            {"schedule_type": "date_specific", "menu_date": "2026-08-03", "day_of_week": "", "rotation_label": "", "meal_type": "Lunch", "food_item": "Sample lunch", "cuisine": "North Cuisine", "category": "Main", "verification_status": "verified", "source_reference": "photo 1", "notes": ""},
            {"schedule_type": "weekly_rotation", "menu_date": "", "day_of_week": "Monday", "rotation_label": "Monday 1", "meal_type": "Breakfast", "food_item": "Sample breakfast", "cuisine": "South Cuisine", "category": "Main", "verification_status": "needs_verification", "source_reference": "photo 2", "notes": "unclear cell"},
        ),
        seasonal_items=({"season": "Aug-Oct", "cuisine": "North Cuisine", "meal_type": "Lunch", "day_label": "Monday 1", "food_item": "Seasonal sample", "verification_status": "verified", "source_reference": "photo 3", "notes": ""},),
    )
    assert len(month_items(menu, 2026, 8)) == 1
    assert month_items(menu, 2025, 8) == []
    assert filtered_items(menu.items, meal_types={"Breakfast"})[0]["day_of_week"] == "Monday"
    assert menu_for_date(menu, date(2026, 8, 3))["Lunch"]["schedule_type"] == "date_specific"
    assert menu_for_date(menu, date(2026, 8, 3))["Breakfast"]["verified"] is False
    assert len(weekly_rotation_items(menu)) == 1
    assert seasonal_items_for_date(menu, date(2026, 8, 3))[0]["cuisine"] == "North Cuisine"
    assert meal_timing(date(2026, 8, 3), "Breakfast") == "07:30 AM–09:30 AM"
    assert meal_timing(date(2026, 8, 2), "Breakfast") == "08:00 AM–10:00 AM"
    assert meal_timing(date(2026, 8, 3), "Lunch", is_holiday=True) == "12:30 PM–03:00 PM"
    assert isinstance(university_today(), date)


def test_empty_entries_and_utf8_month_export() -> None:
    menu = MenuData(items=(), seasonal_items=())
    assert menu_for_date(menu, date(2026, 8, 3))["Dinner"]["items"] == []
    exported = export_month_csv(menu, 2026, 8)
    assert exported.startswith(b"record_type,selected_month")
    assert exported.decode("utf-8").count("\n") == 1

    populated = MenuData(items=({
        "schedule_type": "weekly_rotation", "menu_date": "", "day_of_week": "Monday", "rotation_label": "Monday 1",
        "meal_type": "Breakfast", "food_item": "Poha – café style", "cuisine": "North Indian", "category": "Main",
        "verification_status": "needs_verification", "source_reference": "temporary test input", "notes": "",
    },), seasonal_items=())
    populated_csv = export_month_csv(populated, 2026, 8).decode("utf-8")
    assert "2026-08" in populated_csv
    assert "Poha – café style" in populated_csv
