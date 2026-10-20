"""Headless smoke tests for each Streamlit dashboard section."""

from pathlib import Path
import re

from streamlit.testing.v1 import AppTest
from src.menu import load_menu_data, menu_for_date, university_today


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
PAGES = (
    "Overview",
    "Demand Prediction",
    "Waste Analytics",
    "Data Management",
    "Model Performance",
    "Reports",
    "Mess Menu",
)


def test_all_dashboard_pages_render_without_exceptions() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    assert not app.exception
    for page in PAGES[1:]:
        app.radio(key="dashboard_page").set_value(page).run()
        assert not app.exception, f"{page} raised: {app.exception}"
        if page == "Data Management":
            assert app.selectbox(key="upload_source_type").value == "Choose source…"


def test_prediction_form_uses_updated_attendance_on_each_submission() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    app.radio(key="dashboard_page").set_value("Demand Prediction").run()
    estimates: list[float] = []
    for attendance in (200, 420):
        app.number_input(key="prediction_expected_students").set_value(attendance)
        app.form_submit_button[0].click().run()
        assert not app.exception, f"Prediction submission raised: {app.exception}"
        messages = [item.value for item in app.success if "Estimated consumption" in item.value]
        assert len(messages) == 1
        estimates.append(float(re.search(r"([\d,]+\.\d{2}) kg", messages[0]).group(1).replace(",", "")))
    assert estimates[0] != estimates[1]


def test_prediction_page_warns_and_withholds_estimate_outside_training_range() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    app.radio(key="dashboard_page").set_value("Demand Prediction").run()
    app.number_input(key="prediction_expected_students").set_value(2200)
    app.form_submit_button[0].click().run()
    assert not app.exception
    assert any("outside the synthetic training range" in item.value for item in app.warning)
    assert not any("Estimated consumption" in item.value for item in app.success)


def test_overview_shows_menu_and_menu_page_is_accessible() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    assert not app.exception
    assert any("Today's Mess Menu" in item.value for item in app.subheader)
    assert app.date_input(key="menu_preview_date").value == university_today()
    expected_meals = menu_for_date(load_menu_data(), university_today())
    expected_breakfast_items = expected_meals["Breakfast"]["items"]
    if expected_breakfast_items:
        assert any(expected_breakfast_items[0]["food_item"] in str(item.value) for item in app.markdown)
    assert "Mess Menu" in app.radio(key="dashboard_page").options
    app.radio(key="dashboard_page").set_value("Mess Menu").run()
    assert not app.exception
    assert any(item.value == "Mess Menu" for item in app.title)
    assert any(item.value == "Weekly rotation" for item in app.subheader)


def test_overview_maps_all_weekday_meals_when_the_rotation_exists() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    app.date_input(key="menu_preview_date").set_value(__import__("datetime").date(2026, 10, 19)).run()
    assert not app.exception
    visible = "\n".join(str(item.value) for collection in (app.markdown, app.caption, app.subheader, app.info) for item in collection)
    for dish in ("Aloo Prantha", "Muthu Pappu", "Bread Roll", "Donda Kaya Fry"):
        assert dish in visible
    for implementation_detail in ("South Indian", "Main Veg 1", "Monday 1", "Needs verification", "source_reference"):
        assert implementation_detail not in visible


def test_menu_page_renders_monthly_rotation_as_dish_lists_without_metadata() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    app.radio(key="dashboard_page").set_value("Mess Menu").run()
    assert not app.exception
    visible_text = "\n".join(str(item.value) for collection in (app.markdown, app.caption, app.subheader, app.info) for item in collection)
    headings = [item.value for item in app.subheader]
    assert any(value.endswith("menu") and " " in value for value in headings)
    assert "Date-specific menus" in headings
    assert "No date-specific menus are recorded for this month." in visible_text
    assert "Weekly rotation" in headings
    assert any("Breakfast" in str(item.value) for item in app.markdown)
    assert any("Seasonal vegetables" in item.label for item in app.expander)
    for dish in ("Poori", "Muthu Pappu", "Bread Roll", "Donda Kaya Fry"):
        assert dish in visible_text
    for implementation_detail in ("South Indian", "North Indian", "Main Veg 1", "Monday 1", "needs verification", "schedule_type", "source_reference"):
        assert implementation_detail not in visible_text
