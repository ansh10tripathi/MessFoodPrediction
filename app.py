"""Entry point for the University Mess Food Streamlit dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.dashboard_pages import (
    render_data_management,
    render_model_performance,
    render_overview,
    render_prediction,
    render_reports,
    render_waste_analytics,
)
from src.dashboard_utils import (
    MODEL_METADATA_PATH,
    MODEL_PATH,
    SYNTHETIC_CSV,
    load_model_bundle,
    load_validated_dataset,
)

st.set_page_config(
    page_title="Mess Food Demand & Waste",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_data(show_spinner=False)
def _load_synthetic_dataset(path: str, file_modified_time: float) -> pd.DataFrame:
    """Cache only the fixed project dataset, invalidating when its file changes."""
    del file_modified_time
    frame, _ = load_validated_dataset(Path(path), expected_source="synthetic", expected_days=365)
    return frame


@st.cache_resource(show_spinner="Loading the saved model...")
def _load_model(model_path: str, model_mtime: float, metadata_path: str, metadata_mtime: float) -> tuple[Any, dict[str, Any]]:
    """Cache the static trained model bundle, keyed by the artifact mtimes."""
    del model_mtime, metadata_mtime
    return load_model_bundle(Path(model_path), Path(metadata_path))


def _resolve_dataset() -> tuple[pd.DataFrame | None, str]:
    """Resolve the explicitly selected dataset for this browser session."""
    pending = st.session_state.pop("dashboard_pending_source", None)
    if pending is not None:
        st.session_state["dashboard_selected_source"] = pending

    choices = ["Synthetic sample"]
    if st.session_state.get("dashboard_uploaded_frame") is not None:
        choices.append("Uploaded file")
    selected = st.session_state.get("dashboard_selected_source", "Synthetic sample")
    if selected not in choices:
        selected = "Synthetic sample" if "Synthetic sample" in choices else choices[0]
        st.session_state["dashboard_selected_source"] = selected

    selection = st.sidebar.radio("Active dataset", choices, key="dashboard_selected_source")
    if selection == "Uploaded file":
        uploaded = st.session_state.get("dashboard_uploaded_frame")
        if uploaded is None:
            st.sidebar.error("The uploaded dataset is unavailable. Select another source or validate the file again.")
            return None, "Uploaded file (unavailable)"
        name = st.session_state.get("dashboard_upload_name", "Uploaded CSV")
        label = "real" if st.session_state.get("dashboard_upload_type") == "real" else "synthetic"
        return uploaded.copy(), f"{name} · {label} upload"

    if not SYNTHETIC_CSV.is_file():
        st.sidebar.error("Synthetic dataset file is missing. No fallback dataset was loaded.")
        return None, "Synthetic sample (missing)"
    try:
        modified = SYNTHETIC_CSV.stat().st_mtime
        frame = _load_synthetic_dataset(str(SYNTHETIC_CSV), modified)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        st.sidebar.error(f"Synthetic dataset could not be loaded: {error}")
        return None, "Synthetic sample (invalid)"
    return frame, "data/synthetic/mess_food_synthetic.csv"


def _resolve_model() -> tuple[tuple[Any, dict[str, Any]] | None, str | None]:
    if not MODEL_PATH.is_file():
        return None, f"Selected model artifact is missing: {MODEL_PATH}"
    if not MODEL_METADATA_PATH.is_file():
        return None, f"Model metadata is missing: {MODEL_METADATA_PATH}"
    try:
        return _load_model(
            str(MODEL_PATH), MODEL_PATH.stat().st_mtime,
            str(MODEL_METADATA_PATH), MODEL_METADATA_PATH.stat().st_mtime,
        ), None
    except (OSError, ValueError, KeyError, TypeError) as error:
        return None, f"Saved model could not be loaded or did not match its metadata: {error}"


def main() -> None:
    st.sidebar.markdown("## Mess Food Analytics")
    st.sidebar.caption("Demand estimation · Waste tracking · Academic project")
    st.sidebar.divider()
    data, source_name = _resolve_dataset()
    page = st.sidebar.radio(
        "Workspace",
        ["Overview", "Demand Prediction", "Waste Analytics", "Data Management", "Model Performance", "Reports"],
        key="dashboard_page",
    )
    if st.session_state.get("dashboard_upload_status") == "invalid" and page != "Data Management":
        st.warning("The latest CSV upload failed validation and was not activated. The previously active source is unchanged. Open Data Management to review the errors.")

    if page == "Overview":
        render_overview(data, source_name)
    elif page == "Demand Prediction":
        render_prediction(*_resolve_model())
    elif page == "Waste Analytics":
        render_waste_analytics(data, source_name)
    elif page == "Data Management":
        render_data_management(data, source_name)
    elif page == "Model Performance":
        render_model_performance(*_resolve_model())
    else:
        render_reports(data, source_name)


if __name__ == "__main__":
    main()
