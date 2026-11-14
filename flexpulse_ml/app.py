"""
Streamlit research dashboard entrypoint for FlexPulse-ML.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

from flexpulse_ml.config import PROJECT_ROOT, get_app_config
from flexpulse_ml.data.generator import (
    DatasetGeneratorConfig,
    SyntheticDatasetGenerator,
    generate_synthetic_dataset,
)
from flexpulse_ml.data.storage import SYNTHETIC_DATA_ROOT, DatasetStorage
from flexpulse_ml.data.validator import validate_synthetic_dataset
from flexpulse_ml.simulation.fault_injector import (
    PowerPathSimulator,
    SimulationParams,
)
from flexpulse_ml.simulation.plotting import plot_simulation_scenario

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
    st.title("🎛️ Simulator Controls")
    render_synthetic_notice()
    st.markdown("Configure and inspect individual synthetic runs from the mathematical equivalent circuit model.")

    col1, col2 = st.columns(2)
    with col1:
        scenario = st.selectbox(
            "Simulation Scenario",
            PowerPathSimulator.ALL_SCENARIOS,
            index=2,
        )
        duration_s = st.slider("Duration (seconds)", min_value=10.0, max_value=120.0, value=60.0, step=5.0)
        sampling_rate_hz = st.selectbox("Sampling Frequency (Hz)", [50.0, 100.0, 200.0], index=1)

    with col2:
        seed = st.number_input("Random Seed", min_value=0, max_value=999999, value=42)
        noise_mult = st.slider("Noise Floor Multiplier", min_value=0.5, max_value=5.0, value=1.0, step=0.5)

    if st.button("Run Simulation", type="primary"):
        params = SimulationParams(
            duration_seconds=duration_s,
            sampling_rate_hz=sampling_rate_hz,
            measurement_noise_voltage_std_v=0.005 * noise_mult,
            measurement_noise_current_std_a=0.008 * noise_mult,
        )
        sim = PowerPathSimulator()
        result = sim.run_simulation(
            scenario=scenario,
            params=params,
            seed=int(seed),
        )

        st.success(f"Simulation completed for {scenario} ({len(result.observations.timestamp_s)} samples generated).")
        fig_path = plot_simulation_scenario(result)
        st.image(str(fig_path))


def page_data_explorer() -> None:
    st.title("📈 Synthetic Dataset Generator & Explorer")
    render_synthetic_notice()
    st.markdown("Generate, partition, validate, and inspect multi-run synthetic datasets for model training and benchmarking.")

    st.subheader("1. Dataset Generation Controls")
    gcol1, gcol2, gcol3 = st.columns(3)
    with gcol1:
        dataset_id = st.text_input("Dataset Identifier", value="flexpulse_demo_dataset")
        mode = st.radio("Generation Mode", ["SCENARIO_BALANCED", "SCENARIO_IMBALANCED"])
    with gcol2:
        runs_per_sc = st.number_input("Nominal Runs Per Scenario", min_value=1, max_value=50, value=3)
        ood_runs_per_sc = st.number_input("OOD Runs Per Scenario", min_value=0, max_value=20, value=1)
    with gcol3:
        root_seed = st.number_input("Root Master Seed", min_value=0, max_value=999999, value=42)
        overwrite = st.checkbox("Allow Overwrite", value=True)

    if st.button("Generate Synthetic Dataset", type="primary"):
        with st.spinner("Generating independent simulation runs and computing hashes..."):
            manifest = generate_synthetic_dataset(
                dataset_id=dataset_id,
                runs_per_scenario=int(runs_per_sc),
                ood_runs_per_scenario=int(ood_runs_per_sc),
                root_seed=int(root_seed),
                mode=mode,
                allow_overwrite=overwrite,
            )
            st.session_state["active_dataset_id"] = dataset_id
            st.success(f"Generated dataset '{dataset_id}' with {manifest.total_simulation_runs} runs and {manifest.total_samples} samples.")

    st.divider()

    st.subheader("2. Dataset Validation Engine")
    val_dataset_id = st.text_input("Dataset to Validate", value=st.session_state.get("active_dataset_id", dataset_id))
    dataset_path = SYNTHETIC_DATA_ROOT / val_dataset_id

    if st.button("Validate Dataset"):
        if not dataset_path.exists():
            st.error(f"Dataset path does not exist: {dataset_path}")
        else:
            with st.spinner("Running 20 scientific validation checks..."):
                report = validate_synthetic_dataset(dataset_path)

            if report.is_valid:
                st.success(f"Validation PASSED ({len(report.checks)} checks completed).")
            else:
                st.error(f"Validation FAILED with {len(report.errors)} errors.")

            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            with mcol1:
                st.metric("Total Runs", report.summary.get("total_runs", 0))
            with mcol2:
                st.metric("Total Samples", report.summary.get("total_samples", 0))
            with mcol3:
                st.metric("Fault Ratio", f"{report.summary.get('fault_ratio', 0.0):.1%}")
            with mcol4:
                st.metric("Fault Events", report.summary.get("total_events", 0))

            with st.expander("Detailed Validation Checks", expanded=True):
                check_rows = []
                for chk in report.checks:
                    icon = "✅" if chk.status == "PASS" else ("⚠️" if chk.status == "WARNING" else "❌")
                    check_rows.append({"Status": f"{icon} {chk.status}", "Check": chk.name, "Message": chk.message})
                st.dataframe(pd.DataFrame(check_rows), use_container_width=True)

            if report.errors:
                with st.expander("Validation Errors", expanded=True):
                    for err in report.errors:
                        st.error(err)

    st.divider()

    st.subheader("3. Dataset Run Inspector")
    if dataset_path.exists():
        storage = DatasetStorage(dataset_id=val_dataset_id)
        splits_dict = storage.load_splits() if storage.splits_path.exists() else {}

        selected_split = st.selectbox("Select Split Partition", list(storage.ALL_SPLITS))
        available_runs = splits_dict.get(selected_split, [])

        if available_runs:
            selected_run = st.selectbox("Select Run", available_runs)
            run_data = storage.load_run(selected_split, selected_run)

            meta = run_data["metadata"]
            st.caption(f"Scenario: **{meta.get('scenario')}** | Seed: **{meta.get('random_seed')}** | Duration: **{meta.get('simulation_duration_s')}s**")

            rcol1, rcol2 = st.columns(2)
            with rcol1:
                st.markdown("**Observations Preview (Publicly Observable):**")
                st.dataframe(run_data["observations"].head(10), use_container_width=True)
            with rcol2:
                st.markdown("**Ground Truth Preview (Hidden / Quarantined):**")
                st.dataframe(run_data["ground_truth"].head(10), use_container_width=True)

            st.markdown("**Observed Signals:**")
            obs_plot_df = run_data["observations"].set_index("timestamp_s")[["voltage_measured_v", "current_measured_a"]]
            st.line_chart(obs_plot_df)
        else:
            st.info(f"No runs found in split '{selected_split}'.")
    else:
        st.info(f"Generate a dataset above or create records under data/synthetic/{val_dataset_id} to inspect runs.")


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
