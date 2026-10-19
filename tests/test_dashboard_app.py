"""Headless smoke tests for each Streamlit dashboard section."""

from pathlib import Path
import re

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
PAGES = (
    "Overview",
    "Demand Prediction",
    "Waste Analytics",
    "Data Management",
    "Model Performance",
    "Reports",
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
