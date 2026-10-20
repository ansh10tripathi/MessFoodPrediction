"""Streamlit page renderers for the university mess analytics dashboard."""

from __future__ import annotations

import hashlib
import html
import warnings
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_generator import MENU_CATEGORIES, MEALS
from src.dashboard_utils import (
    MODEL_COMPARISON_PATH,
    TEST_PREDICTIONS_PATH,
    analytics_summary,
    daily_summary,
    filter_records,
    load_model_comparison,
    load_test_predictions,
    meal_summary,
    menu_summary,
    prepare_analysis_frame,
    regression_metrics,
    source_status,
    summary_totals,
    validate_uploaded_csv,
)
from src.predict import expected_students_support_warning, predict_with_pipeline
from src.menu import (
    MEAL_TYPES,
    ROTATION_LABELS,
    MenuData,
    export_month_csv,
    meal_timing,
    menu_for_date,
    month_items,
    university_today,
    weekly_rotation_items,
)

SYNTHETIC_MODEL_WARNING = (
    "Experimental model trained and evaluated on synthetic data. These results do not establish "
    "real-world accuracy. Use the estimate for demonstration only."
)
FOOD_LABELS = {
    "prepared_kg": "Prepared food",
    "consumed_kg": "Consumed food",
    "edible_leftover_kg": "Edible leftovers",
    "discarded_waste_kg": "Discarded waste",
}


def _source_badge(data: pd.DataFrame) -> None:
    status = source_status(data)
    color = "green" if status == "REAL DATA" else "blue" if status == "SYNTHETIC DATA" else "orange"
    st.badge(status, color=color)


def _open_menu_page() -> None:
    st.session_state["dashboard_page"] = "Mess Menu"


def _shift_menu_month(offset: int) -> None:
    import calendar
    today = university_today()
    month = int(st.session_state.get("menu_month", today.month))
    year = int(st.session_state.get("menu_year", today.year))
    shifted = (year * 12 + month - 1 + offset)
    shifted_year, month_zero = divmod(shifted, 12)
    st.session_state["menu_month"] = month_zero + 1
    st.session_state["menu_year"] = shifted_year


def _chart_layout(figure: go.Figure, height: int = 350) -> go.Figure:
    figure.update_layout(
        template="plotly_white",
        height=height,
        margin={"l": 54, "r": 24, "t": 64, "b": 48},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="white",
        font={"family": "Inter, Segoe UI, sans-serif", "color": "#1d2939", "size": 12},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0, "entrywidth": 110},
        autosize=True,
    )
    return figure


def _menu_dish_names(rows: list[dict[str, str]]) -> list[str]:
    """Return unique dish names in source order, without exposing row metadata."""
    return list(dict.fromkeys(row["food_item"].strip() for row in rows if row.get("food_item", "").strip()))


def _render_dish_bullets(rows: list[dict[str, str]], *, empty_message: str = "No dishes listed.") -> None:
    names = _menu_dish_names(rows)
    if names:
        st.markdown("\n".join(f"- {html.escape(name)}" for name in names))
    else:
        st.caption(empty_message)


def _render_menu_day(rows: list[dict[str, str]], heading: str) -> None:
    """Render one date or rotation day as four dish-only meal sections."""
    with st.container(border=True):
        st.markdown(f"### {html.escape(heading)}")
        for meal in MEAL_TYPES:
            st.markdown(f"**{meal}**")
            _render_dish_bullets([row for row in rows if row["meal_type"] == meal])


def _rotation_heading(rotation_label: str) -> str:
    """Render a concise human heading without exposing the CSV field label."""
    day, cycle = rotation_label.rsplit(" ", 1)
    return day if cycle == "1" else f"{day} · alternate menu"


def _render_today_menu(menu: MenuData | None, menu_error: str | None) -> None:
    st.subheader("Today's Mess Menu")
    today = university_today()
    selected = st.date_input("Preview menu date", value=today, key="menu_preview_date")
    is_holiday = st.checkbox("Use Sunday/holiday meal timings", value=selected.weekday() == 6, key="menu_preview_holiday")
    st.caption(f"Menu preview for {selected.strftime('%A, %d %B %Y')} · Serving may vary.")
    if menu_error:
        st.error(f"Menu data is unavailable: {menu_error}")
        return
    if menu is None:
        st.info("Menu transcription is pending verification. No dish names are shown until they can be read reliably from the supplied source.")
        return
    selected_items = menu_for_date(menu, selected)
    cols = st.columns(4)
    for column, meal in zip(cols, MEAL_TYPES):
        detail = selected_items[meal]
        with column.container(border=True):
            st.markdown(f"**{meal}**")
            st.caption(meal_timing(selected, meal, is_holiday=is_holiday))
            _render_dish_bullets(detail["items"], empty_message="No dishes listed for this day.")
    st.button("Open complete Mess Menu", type="primary", key="open_menu_page", on_click=_open_menu_page)
    st.caption("Menu entries follow the printed pattern; actual service may change.")


def render_mess_menu(menu: MenuData | None, menu_error: str | None) -> None:
    import calendar
    st.title("Mess Menu")
    st.caption("Printed menu pattern for the selected month. Actual servings may change.")
    if menu_error:
        st.error(f"Menu data could not be loaded: {menu_error}")
        return
    if menu is None:
        st.info("Menu transcription is pending verification.")
        return
    today = university_today()
    previous, next_month = st.columns(2)
    previous.button("← Previous month", on_click=_shift_menu_month, args=(-1,), key="menu_previous_month")
    next_month.button("Next month →", on_click=_shift_menu_month, args=(1,), key="menu_next_month")
    col1, col2 = st.columns(2)
    with col1:
        if "menu_month" not in st.session_state:
            st.session_state["menu_month"] = today.month
        month = st.selectbox("Month", range(1, 13), format_func=lambda value: calendar.month_name[value], key="menu_month")
    with col2:
        if "menu_year" not in st.session_state:
            st.session_state["menu_year"] = today.year
        selected_year = int(st.session_state["menu_year"])
        years = range(min(today.year - 2, selected_year - 1), max(today.year + 6, selected_year + 2))
        year = st.selectbox("Year", years, key="menu_year")
    st.subheader(f"{calendar.month_name[month]} {year} menu")
    dated = month_items(menu, year, month)
    st.subheader("Date-specific menus")
    if dated:
        dates = sorted({row["menu_date"] for row in dated})
        for menu_date in dates:
            current = [row for row in dated if row["menu_date"] == menu_date]
            label = date.fromisoformat(menu_date).strftime("%A, %d %B")
            _render_menu_day(current, label)
    else:
        st.info("No date-specific menus are recorded for this month.")

    rotation = weekly_rotation_items(menu)
    st.subheader("Weekly rotation")
    if rotation:
        for label in ROTATION_LABELS:
            current = [row for row in rotation if row["rotation_label"] == label]
            if current:
                _render_menu_day(current, _rotation_heading(label))
    else:
        st.info("No weekly rotation entries are available.")

    if menu.seasonal_items:
        with st.expander("Seasonal vegetables"):
            for season in ("Aug-Oct", "Nov-Feb", "Mar-May"):
                season_names = list(dict.fromkeys(
                    row["food_item"] for row in menu.seasonal_items if row["season"] == season
                ))
                if season_names:
                    st.markdown(f"**{season}**")
                    _render_dish_bullets([{"food_item": name} for name in season_names])
    st.download_button("Download menu CSV", data=export_month_csv(menu, year, month), file_name=f"mess_menu_{year}-{month:02d}.csv", mime="text/csv")


def render_overview(data: pd.DataFrame | None, source_name: str, menu: MenuData | None = None, menu_error: str | None = None) -> None:
    st.title("Mess Food Demand & Waste")
    st.write("Monitor recorded meal quantities and explore an experimental pre-meal consumption estimate.")
    _render_today_menu(menu, menu_error)
    if data is None:
        st.error("No valid dataset is active. Check the data source or load a valid CSV in Data Management.")
        return
    _source_badge(data)
    totals = summary_totals(data)
    st.caption(f"Source: {source_name} · {totals['records']:,} records · {totals['date_start']} to {totals['date_end']}")

    cols = st.columns(4)
    cards = [
        ("Prepared", totals["prepared_kg"], "kg"),
        ("Consumed", totals["consumed_kg"], "kg"),
        ("Edible leftovers", totals["edible_leftover_kg"], "kg"),
        ("Discarded waste", totals["discarded_waste_kg"], "kg"),
    ]
    for column, (label, value, unit) in zip(cols, cards):
        column.metric(label, f"{value:,.1f} {unit}", border=True)
    st.caption(f"Discarded-waste rate: {totals['discarded_pct_prepared']:.2f}% of prepared food (discarded_waste_kg ÷ prepared_kg). Edible leftovers are separate.")

    daily = daily_summary(data)
    meal = meal_summary(data)
    left, right = st.columns([1.35, 1])
    with left:
        st.subheader("Daily food quantities")
        if daily.empty:
            st.info("No daily records to chart.")
        else:
            chart_data = daily.melt(id_vars="date", value_vars=["prepared_kg", "consumed_kg"], var_name="quantity", value_name="kg")
            chart_data["quantity"] = chart_data["quantity"].map(FOOD_LABELS)
            fig = px.line(chart_data, x="date", y="kg", color="quantity", color_discrete_sequence=["#315efb", "#16a6a0"])
            _chart_layout(fig, 360)
            fig.update_xaxes(title=None)
            fig.update_yaxes(title="Food (kg)")
            st.plotly_chart(fig, width="stretch", alt="Daily prepared and consumed food quantities in kilograms.")
    with right:
        st.subheader("Meal-wise quantities")
        if meal.empty:
            st.info("No meal records to chart.")
        else:
            chart_data = meal.melt(id_vars="meal_period", value_vars=["prepared_kg", "consumed_kg"], var_name="quantity", value_name="kg")
            chart_data["quantity"] = chart_data["quantity"].map(FOOD_LABELS)
            fig = px.bar(chart_data, x="meal_period", y="kg", color="quantity", barmode="group", color_discrete_sequence=["#315efb", "#16a6a0"])
            _chart_layout(fig, 360)
            fig.update_xaxes(title=None)
            fig.update_yaxes(title="Total food (kg)")
            st.plotly_chart(fig, width="stretch", alt="Prepared and consumed quantities compared across meal periods.")

    st.subheader("Recorded waste by meal")
    if meal.empty:
        st.info("No meal records available.")
    else:
        waste_data = meal.melt(id_vars="meal_period", value_vars=["edible_leftover_kg", "discarded_waste_kg"], var_name="category", value_name="kg")
        waste_data["category"] = waste_data["category"].map(FOOD_LABELS)
        fig = px.bar(waste_data, x="meal_period", y="kg", color="category", barmode="group", color_discrete_sequence=["#8a6ff0", "#e1844a"])
        _chart_layout(fig, 320)
        fig.update_xaxes(title=None)
        fig.update_yaxes(title="Recorded quantity (kg)")
        st.plotly_chart(fig, width="stretch", alt="Edible leftovers and discarded waste totals by meal period.")


def render_prediction(model_bundle: tuple[Any, dict[str, Any]] | None, model_error: str | None) -> None:
    st.title("Demand Prediction")
    st.warning(SYNTHETIC_MODEL_WARNING)
    st.write("Estimate consumed food from information available before service. This estimate is not a guaranteed quantity and is not itself a preparation order.")
    if model_bundle is None:
        st.error(model_error or "The selected model is unavailable. Check the model artifact and metadata files.")
        return
    model, metadata = model_bundle
    st.caption(f"Model: {metadata['selected_model']} · target: {metadata['target_variable']} · synthetic training data")

    with st.form("prediction_form"):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            prediction_date = st.date_input("Meal date", value=date.today())
            expected_students = st.number_input("Expected students", min_value=0, max_value=10000, value=420, step=1, key="prediction_expected_students")
            meal_period = st.selectbox("Meal period", MEALS)
        with col_b:
            menu_category = st.selectbox("Menu category", MENU_CATEGORIES[meal_period])
            exam_day = st.checkbox("Exam day")
            holiday = st.checkbox("Holiday")
            special_event = st.checkbox("Special event")
        with col_c:
            temperature_c = st.number_input("Temperature forecast (°C)", min_value=-90.0, max_value=65.0, value=27.0, step=0.5)
            rainfall_mm = st.number_input("Rainfall forecast (mm)", min_value=0.0, max_value=1000.0, value=0.0, step=1.0)
        use_buffer = st.checkbox("Show a planning suggestion with a safety buffer", value=False)
        buffer_pct = st.slider("Planning buffer (%)", min_value=0, max_value=50, value=10, disabled=not use_buffer)
        submitted = st.form_submit_button("Estimate consumption", type="primary")

    if submitted:
        inputs = {
            "date": prediction_date.isoformat(),
            "expected_students": int(expected_students),
            "meal_period": meal_period,
            "menu_category": menu_category,
            "exam_day": exam_day,
            "holiday": holiday,
            "special_event": special_event,
            "temperature_c": float(temperature_c),
            "rainfall_mm": float(rainfall_mm),
        }
        supported_range = metadata.get("supported_input_ranges", {}).get("expected_students")
        range_warning = expected_students_support_warning(expected_students, supported_range)
        if range_warning:
            advice = "" if supported_range is None else f" Adjust expected students to {supported_range['min']} to {supported_range['max']} to receive an estimate within the model's observed attendance range."
            st.warning(range_warning + advice)
            return
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                prediction = predict_with_pipeline(
                    model,
                    inputs,
                    supported_expected_students_range=supported_range,
                )
            except (ValueError, RuntimeError, KeyError) as error:
                st.error(f"Prediction could not be produced: {error}")
                return
        for item in caught:
            st.warning(str(item.message))
        st.success(f"Estimated consumption: **{prediction:,.2f} kg**")
        if use_buffer:
            planning_amount = prediction * (1 + buffer_pct / 100)
            st.info(f"Optional planning suggestion with {buffer_pct}% buffer: **{planning_amount:,.2f} kg**. This heuristic is separate from predicted consumption and should be reviewed by mess staff.")
        st.caption("Experimental estimate from synthetic-trained data. It is not guaranteed; predicted consumption does not equal the amount that must be prepared.")


def _date_filter(data: pd.DataFrame, key_prefix: str) -> tuple[date | None, date | None, list[str]]:
    frame = prepare_analysis_frame(data)
    if frame.empty or frame["date"].isna().all():
        st.info("No valid dates are available for filtering.")
        return None, None, []
    min_date = frame["date"].min().date()
    max_date = frame["date"].max().date()
    col1, col2 = st.columns([1, 1])
    with col1:
        value = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date, key=f"{key_prefix}_dates")
    with col2:
        meals = st.multiselect("Meal periods", options=MEALS, default=list(MEALS), key=f"{key_prefix}_meals")
    if isinstance(value, tuple) and len(value) == 2:
        return value[0], value[1], meals
    if isinstance(value, date):
        return value, value, meals
    return None, None, meals


def render_waste_analytics(data: pd.DataFrame | None, source_name: str) -> None:
    st.title("Waste Analytics")
    st.write("Review recorded prepared food, edible leftovers, and discarded waste as separate quantities.")
    if data is None:
        st.error("No valid dataset is active. Load a valid dataset in Data Management.")
        return
    _source_badge(data)
    start_date, end_date, meals = _date_filter(data, "waste")
    if start_date is None or end_date is None:
        st.info("Choose a valid date range to continue.")
        return
    filtered = filter_records(data, start_date, end_date, meals)
    if filtered.empty:
        st.info("No records match the selected date and meal filters.")
        return
    totals = summary_totals(filtered)
    st.caption(f"Source: {source_name} · {len(filtered):,} records · Discarded waste percentage = discarded_waste_kg ÷ prepared_kg × 100.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Prepared food", f"{totals['prepared_kg']:,.1f} kg")
    c2.metric("Edible leftovers", f"{totals['edible_leftover_kg']:,.1f} kg")
    c3.metric("Discarded waste", f"{totals['discarded_waste_kg']:,.1f} kg", f"{totals['discarded_pct_prepared']:.2f}% of prepared")

    daily = daily_summary(filtered)
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Daily recorded leftovers and waste")
        trend = daily.melt(id_vars="date", value_vars=["edible_leftover_kg", "discarded_waste_kg"], var_name="category", value_name="kg")
        trend["category"] = trend["category"].map(FOOD_LABELS)
        fig = px.line(trend, x="date", y="kg", color="category", color_discrete_sequence=["#8a6ff0", "#e1844a"])
        _chart_layout(fig)
        fig.update_yaxes(title="Recorded quantity (kg)")
        fig.update_xaxes(title=None)
        st.plotly_chart(fig, width="stretch", alt="Daily edible leftovers and discarded waste over the selected dates.")
    with col_b:
        st.subheader("Meal-wise recorded quantities")
        meals_data = meal_summary(filtered).melt(id_vars="meal_period", value_vars=["prepared_kg", "consumed_kg", "edible_leftover_kg", "discarded_waste_kg"], var_name="quantity", value_name="kg")
        meals_data["quantity"] = meals_data["quantity"].map(FOOD_LABELS)
        fig = px.bar(meals_data, x="meal_period", y="kg", color="quantity", barmode="group")
        _chart_layout(fig)
        fig.update_xaxes(title=None)
        fig.update_yaxes(title="Total quantity (kg)")
        st.plotly_chart(fig, width="stretch", alt="Prepared, consumed, leftover, and discarded quantities by meal period.")

    st.subheader("Menu-category summary")
    menu_data = menu_summary(filtered)
    fig = px.bar(menu_data, x="menu_category", y="discarded_waste_kg", color="discarded_pct_prepared", color_continuous_scale="Blues", hover_data=["prepared_kg", "edible_leftover_kg"])
    _chart_layout(fig, 320)
    fig.update_xaxes(title=None)
    fig.update_yaxes(title="Recorded discarded waste (kg)")
    st.plotly_chart(fig, width="stretch", alt="Recorded discarded waste by menu category, with waste percentage of prepared food.")
    st.caption("Only recorded discarded_waste_kg is counted as discarded waste. No waste is inferred from food surplus; edible leftovers remain separate.")


def render_data_management(data: pd.DataFrame | None, source_name: str) -> None:
    st.title("Data Management")
    st.write("Inspect the active dataset or validate a local CSV in memory before using it for analysis.")
    st.info("Uploads are processed in this app session only. They are not written to data/real, do not alter the source file, and are never sent to an external service.")
    st.subheader("Active dataset")
    if data is None:
        st.warning("No valid dataset is active.")
    else:
        _source_badge(data)
        st.caption(f"Source: {source_name} · {len(data):,} rows × {len(data.columns)} columns")
        st.dataframe(data.head(12), width="stretch", hide_index=True, alt="First 12 rows of the active validated dataset.")

    st.subheader("Upload and validate CSV")
    st.caption("The current analytics schema is listed in the README. A file must contain the supported project columns. Existing files are never overwritten.")
    source_type = st.selectbox(
        "Declare the upload source",
        ["Choose source…", "real", "synthetic"],
        key="upload_source_type",
        help="The default is intentionally unselected. A source declaration is required before validation.",
    )
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"], key="dashboard_csv_upload")
    if uploaded is not None:
        contents = uploaded.getvalue()
        if source_type == "Choose source…":
            st.info("Select real or synthetic explicitly to validate this upload.")
        else:
            signature = hashlib.sha256(contents + source_type.encode("utf-8")).hexdigest()
            if signature != st.session_state.get("dashboard_upload_signature"):
                result = validate_uploaded_csv(contents, source_type)
                st.session_state["dashboard_upload_signature"] = signature
                st.session_state["dashboard_last_upload_name"] = Path(uploaded.name).name
                st.session_state["dashboard_last_upload_type"] = source_type
                st.session_state["dashboard_upload_errors"] = result.report.errors
                st.session_state["dashboard_upload_warnings"] = result.report.warnings
                if result.is_valid:
                    st.session_state["dashboard_uploaded_frame"] = result.data
                    st.session_state["dashboard_upload_name"] = Path(uploaded.name).name
                    st.session_state["dashboard_upload_type"] = source_type
                    st.session_state["dashboard_upload_label_added"] = result.added_source_label_in_memory
                    st.session_state["dashboard_upload_status"] = "valid"
                    st.session_state["dashboard_pending_source"] = "Uploaded file"
                else:
                    # Preserve any previously active upload; the failed candidate is never activated.
                    st.session_state["dashboard_upload_status"] = "invalid"
                st.rerun()

    status = st.session_state.get("dashboard_upload_status", "none")
    if status == "valid" and st.session_state.get("dashboard_upload_signature"):
        label = str(st.session_state.get("dashboard_upload_type", "unknown")).upper()
        st.success(f"{label} CSV passed validation and is available as the uploaded dataset.")
        for warning in st.session_state.get("dashboard_upload_warnings", []):
            st.warning(warning)
        frame = st.session_state.get("dashboard_uploaded_frame")
        if frame is not None:
            st.dataframe(frame.head(20), width="stretch", hide_index=True, alt="Preview of the successfully validated uploaded dataset.")
    elif status == "invalid":
        st.error(f"{st.session_state.get('dashboard_last_upload_name', 'Uploaded CSV')} failed validation and was not activated. Any previously active dataset remains unchanged.")
        errors = st.session_state.get("dashboard_upload_errors", [])
        st.caption(f"{len(errors)} validation error(s). Showing the first {min(10, len(errors))}.")
        for error in errors[:10]:
            st.write(f"- {error}")
        if len(errors) > 10:
            with st.expander(f"Inspect all {len(errors)} validation errors"):
                for error in errors[10:]:
                    st.write(f"- {error}")
    elif uploaded is None:
        st.caption("No file selected. The project synthetic sample remains clearly labeled and available as a separate source.")


def render_model_performance(model_bundle: tuple[Any, dict[str, Any]] | None, model_error: str | None) -> None:
    st.title("Model Performance")
    st.warning(SYNTHETIC_MODEL_WARNING)
    if model_bundle is None:
        st.error(model_error or "Saved model metadata is unavailable.")
        return
    _, metadata = model_bundle
    st.caption(f"Selected model: {metadata.get('selected_model', 'Unknown')} · target: {metadata.get('target_variable', 'Unknown')}")
    try:
        comparison = load_model_comparison(MODEL_COMPARISON_PATH)
        predictions = load_test_predictions(TEST_PREDICTIONS_PATH)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        st.error(f"Saved evaluation outputs are unavailable or malformed: {error}")
        return

    st.subheader("Validation comparison")
    st.caption("MAE/RMSE are in kg; lower is better. R² is unitless; values closer to 1 indicate a closer fit to these synthetic outcomes.")
    st.dataframe(comparison.style.format({"mae": "{:.3f}", "rmse": "{:.3f}", "r2": "{:.3f}"}), width="stretch", hide_index=True, alt="Synthetic validation metrics for the candidate regression models.")
    c1, c2, c3 = st.columns(3)
    test_metrics = metadata.get("final_test_metrics", {})
    c1.metric("Final test MAE", f"{test_metrics.get('mae', float('nan')):.3f} kg")
    c2.metric("Final test RMSE", f"{test_metrics.get('rmse', float('nan')):.3f} kg")
    c3.metric("Final test R²", f"{test_metrics.get('r2', float('nan')):.3f}")
    st.caption("Model selection used validation MAE. The selected pipeline was refit on training plus validation dates before this final test evaluation.")

    boundaries = metadata.get("split_boundaries", {})
    boundary_frame = pd.DataFrame([
        {"Partition": label.title(), "Dates": detail.get("dates"), "Rows": detail.get("rows"), "Start": detail.get("start"), "End": detail.get("end")}
        for label, detail in boundaries.items()
    ])
    if not boundary_frame.empty:
        st.subheader("Chronological date boundaries")
        st.dataframe(boundary_frame, width="stretch", hide_index=True, alt="Chronological train, validation, and test date boundaries.")

    if predictions.empty:
        st.info("The saved test prediction file has no observations to chart.")
        return
    chart_data = predictions.copy()
    chart_data["residual_kg"] = chart_data["actual_consumed_kg"] - chart_data["predicted_consumed_kg"]
    computed_metrics = regression_metrics(chart_data["actual_consumed_kg"], chart_data["predicted_consumed_kg"])
    st.caption(f"Recomputed from saved held-out predictions: MAE {computed_metrics['mae']:.3f} kg · RMSE {computed_metrics['rmse']:.3f} kg · R² {computed_metrics['r2']:.3f}")
    left, right = st.columns(2)
    with left:
        st.subheader("Actual vs predicted")
        scatter = px.scatter(chart_data, x="actual_consumed_kg", y="predicted_consumed_kg", color="meal_period", color_discrete_sequence=px.colors.qualitative.Set2)
        min_axis = min(chart_data["actual_consumed_kg"].min(), chart_data["predicted_consumed_kg"].min())
        max_axis = max(chart_data["actual_consumed_kg"].max(), chart_data["predicted_consumed_kg"].max())
        scatter.add_trace(go.Scatter(x=[min_axis, max_axis], y=[min_axis, max_axis], mode="lines", name="Perfect estimate", line={"dash": "dash", "color": "#667085"}))
        scatter.update_xaxes(title="Actual consumption (kg)")
        scatter.update_yaxes(title="Predicted consumption (kg)")
        _chart_layout(scatter)
        st.plotly_chart(scatter, width="stretch", alt="Actual versus predicted synthetic test consumption by meal; diagonal indicates a perfect estimate.")
    with right:
        st.subheader("Residuals")
        residual_fig = px.scatter(chart_data, x="predicted_consumed_kg", y="residual_kg", color="meal_period", color_discrete_sequence=px.colors.qualitative.Set2)
        residual_fig.add_hline(y=0, line_dash="dash", line_color="#667085")
        residual_fig.update_xaxes(title="Predicted consumption (kg)")
        residual_fig.update_yaxes(title="Actual − predicted (kg)")
        _chart_layout(residual_fig)
        st.plotly_chart(residual_fig, width="stretch", alt="Prediction residuals, computed as actual minus predicted synthetic consumption.")


def render_reports(data: pd.DataFrame | None, source_name: str) -> None:
    st.title("Reports & Downloads")
    st.write("Export the active, validated source with the selected filters. Downloads retain clear source labels.")
    if data is not None:
        _source_badge(data)
        start_date, end_date, meals = _date_filter(data, "reports")
        if start_date is not None and end_date is not None:
            filtered = filter_records(data, start_date, end_date, meals)
            st.caption(f"Export source: {source_name} · {len(filtered):,} matching records")
            if filtered.empty:
                st.info("No records match the selected filters.")
            else:
                file_label = source_status(data).lower().replace(" ", "_")
                st.download_button(
                    "Download filtered records as CSV",
                    data=filtered.to_csv(index=False).encode("utf-8"),
                    file_name=f"mess_food_{file_label}_filtered.csv",
                    mime="text/csv",
                    type="primary",
                )
                summary = analytics_summary(filtered)
                st.download_button(
                    "Download analytics summary as CSV",
                    data=summary.to_csv(index=False).encode("utf-8"),
                    file_name=f"mess_food_{file_label}_analytics_summary.csv",
                    mime="text/csv",
                )
                st.dataframe(filtered.head(20), width="stretch", hide_index=True, alt="Preview of the filtered records ready for download.")
    else:
        st.info("Dataset export is unavailable until a valid dataset is active.")

    st.subheader("Model comparison export")
    try:
        comparison = load_model_comparison(MODEL_COMPARISON_PATH)
        st.caption("Source: synthetic-data validation results. These metrics do not establish real-world accuracy.")
        st.download_button(
            "Download model comparison CSV",
            data=comparison.to_csv(index=False).encode("utf-8"),
            file_name="synthetic_model_comparison.csv",
            mime="text/csv",
        )
    except (OSError, ValueError, pd.errors.ParserError) as error:
        st.warning(f"Model comparison download unavailable: {error}")
