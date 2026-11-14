"""
Streamlit research dashboard entrypoint for FlexPulse-ML.
"""

from __future__ import annotations

import streamlit as st
from flexpulse_ml.config import get_app_config

st.set_page_config(
    page_title="FlexPulse-ML Research Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_synthetic_notice() -> None:
    st.warning(
        "⚠️ **SIMULATION ONLY / SYNTHETIC DATA NOTICE:** "
        "All measurements, signals, and results presented in this application are generated "
        "using mathematical models. This system does NOT connect to or control real physical USB-C hardware."
    )


def page_overview() -> None:
    st.title("⚡ FlexPulse-ML: Project Overview")
    render_synthetic_notice()
    st.markdown(
        """
        ### Research Framework
        **FlexPulse-ML** is a simulation-based machine learning framework investigating whether 
        **confidence-gated adaptive baseline calibration** improves the early detection of intermittent 
        contact resistance faults in USB-C-inspired electrical power paths.

        #### Academic & Scientific Integrity Principles
        * **Pure Mathematical Simulation:** No physical microcontroller, sensor, or oscilloscope is required.
        * **Zero Hardware Claims:** The system is evaluated exclusively on controlled synthetic scenarios.
        * **Run-Level Isolation:** Overlapping time windows from the same simulation run are never split across train/test sets.
        * **No Target Leakage:** Feature pipelines only access observed voltage and current signals.
        """
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mathematical Model", "Resistive Equivalent Circuit")
    with col2:
        st.metric("Detection Paradigms", "4 Compared Approaches")
    with col3:
        st.metric("Operating Scenarios", "6 Synthetic Regimes")


def page_simulator_controls() -> None:
    st.title("🎛️ Simulator Controls (Phase 1 Placeholder)")
    render_synthetic_notice()
    st.info("The mathematical electrical power-path simulation engine will be implemented in Phase 1.")


def page_data_explorer() -> None:
    st.title("📈 Data Explorer (Phase 2 Placeholder)")
    render_synthetic_notice()
    st.info("Synthetic time-series dataset visualization and scenario comparisons will be active in Phase 2.")


def page_feature_analysis() -> None:
    st.title("🔍 Feature Analysis (Phase 3 Placeholder)")
    render_synthetic_notice()
    st.info("Time-window feature extraction, statistical moments, and scatter views will be active in Phase 3.")


def page_model_training() -> None:
    st.title("🤖 Model Training (Phase 4 Placeholder)")
    render_synthetic_notice()
    st.info("Classifiers (Logistic Regression, Decision Tree, Random Forest, SVM) will be trained here in Phase 4.")


def page_self_calibration() -> None:
    st.title("🔄 Self-Calibration Engine (Phase 5 Placeholder)")
    render_synthetic_notice()
    st.info("Confidence-gated adaptive baseline updates and freeze dynamics will be demonstrated here in Phase 5.")


def page_evaluation() -> None:
    st.title("📊 Evaluation & Benchmarking (Phase 6 Placeholder)")
    render_synthetic_notice()
    st.info("Run-level metrics, detection delays, and out-of-distribution benchmarks will be evaluated in Phase 6.")


def page_export_and_reports() -> None:
    st.title("📄 Export and Reports (Phase 7 Placeholder)")
    render_synthetic_notice()
    st.info("Exporting publication figures, LaTeX tables, and JSON manifests will be enabled in Phase 7.")


def main() -> None:
    cfg = get_app_config()

    st.sidebar.title(cfg.project_name)
    st.sidebar.caption(f"Version {cfg.version} (Simulation Prototype)")

    pages = {
        "1. Overview": page_overview,
        "2. Simulator Controls": page_simulator_controls,
        "3. Data Explorer": page_data_explorer,
        "4. Feature Analysis": page_feature_analysis,
        "5. Model Training": page_model_training,
        "6. Self-Calibration": page_self_calibration,
        "7. Evaluation": page_evaluation,
        "8. Export and Reports": page_export_and_reports,
    }

    selection = st.sidebar.radio("Navigation", list(pages.keys()))
    st.sidebar.divider()
    st.sidebar.caption("B.Tech Computer Science Research Prototype")

    pages[selection]()


if __name__ == "__main__":
    main()
