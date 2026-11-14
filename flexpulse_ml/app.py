"""
Streamlit research dashboard entrypoint for FlexPulse-ML.
"""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
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
from flexpulse_ml.features.dataset import (
    FeatureDatasetStorage,
    PROCESSED_FEATURES_ROOT,
    generate_feature_dataset,
)
from flexpulse_ml.calibration.adaptive import (
    AdaptiveCalibrationConfig,
    ConfidenceGatedAdaptiveCalibrator,
)
from flexpulse_ml.calibration.baseline import InitialBaselineEstimator
from flexpulse_ml.calibration.experiment import run_phase5_benchmark
from flexpulse_ml.features.extractor import FEATURE_NAMES, FEATURE_UNITS
from flexpulse_ml.features.validator import validate_feature_dataset
from flexpulse_ml.ml.experiment import run_phase4_benchmark
from flexpulse_ml.ml.models import SUPPORTED_MODELS
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
        dataset_id = st.text_input("Dataset Identifier", value="flexpulse_research_v1")
        mode = st.radio("Generation Mode", ["SCENARIO_BALANCED", "SCENARIO_IMBALANCED"])
    with gcol2:
        runs_per_sc = st.number_input("Nominal Runs Per Scenario", min_value=1, max_value=50, value=20)
        ood_runs_per_sc = st.number_input("OOD Runs Per Scenario", min_value=0, max_value=20, value=5)
    with gcol3:
        root_seed = st.number_input("Root Master Seed", min_value=0, max_value=999999, value=42)
        overwrite = st.checkbox("Allow Overwrite", value=True)

    if st.button("Generate Synthetic Dataset", type="primary"):
        with st.spinner("Generating independent simulation runs and computing hashes..."):
            manifest = generate_synthetic_dataset(
                dataset_id=dataset_id,
                runs_per_scenario=int(runs_per_sc),
                ood_runs_per_scenario=int(ood_runs_per_sc),
                train_runs_per_scenario=14 if int(runs_per_sc) == 20 else None,
                val_runs_per_scenario=3 if int(runs_per_sc) == 20 else None,
                id_test_runs_per_scenario=3 if int(runs_per_sc) == 20 else None,
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
            with st.spinner("Running scientific validation checks..."):
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
    st.title("🔍 Electrical Feature Analysis & Extraction Engine")
    render_synthetic_notice()
    st.markdown(
        "Extract, validate, and inspect sliding-window electrical and statistical features "
        "derived strictly from observable voltage and current telemetry."
    )

    st.subheader("1. Feature Extraction Pipeline")
    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        dataset_id = st.text_input("Source Synthetic Dataset", value="flexpulse_research_v1")
    with fcol2:
        window_duration = st.slider("Window Duration (seconds)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
    with fcol3:
        overlap = st.slider("Overlap Fraction", min_value=0.0, max_value=0.8, value=0.5, step=0.1)

    if st.button("Generate Feature Dataset", type="primary"):
        with st.spinner("Extracting 26 features across sliding windows..."):
            manifest = generate_feature_dataset(
                dataset_id=dataset_id,
                window_duration_s=window_duration,
                overlap_fraction=overlap,
            )
            st.success(
                f"Generated feature dataset for '{dataset_id}' with {manifest.total_windows} windows "
                f"({manifest.total_healthy_windows} healthy, {manifest.total_fault_windows} faulty)."
            )

    st.divider()

    st.subheader("2. Feature Dataset Validation")
    feature_dir = PROCESSED_FEATURES_ROOT / dataset_id
    if st.button("Validate Feature Dataset"):
        if not feature_dir.exists():
            st.error(f"Feature dataset not found at {feature_dir}. Please generate it above first.")
        else:
            with st.spinner("Executing 10 feature validation checks..."):
                report = validate_feature_dataset(feature_dir)

            if report.is_valid:
                st.success(f"Feature Validation PASSED ({len(report.checks)} checks completed).")
            else:
                st.error(f"Feature Validation FAILED with {len(report.errors)} errors.")

            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            with mcol1:
                st.metric("Total Windows", report.summary.get("total_windows", 0))
            with mcol2:
                st.metric("Healthy Windows", report.summary.get("total_healthy_windows", 0))
            with mcol3:
                st.metric("Fault Windows", report.summary.get("total_fault_windows", 0))
            with mcol4:
                st.metric("Fault Ratio", f"{report.summary.get('fault_ratio', 0.0):.1%}")

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

    st.subheader("3. Feature Exploration and Statistical Comparison")
    if feature_dir.exists():
        storage = FeatureDatasetStorage(dataset_id=dataset_id)
        selected_split = st.selectbox("Select Partition", list(storage.SPLIT_FILES.keys()), index=0)
        split_df = storage.load_partition(selected_split)

        st.caption(f"Loaded partition **{selected_split}**: {len(split_df)} windows across {len(split_df['run_id'].unique())} runs.")

        tab1, tab2, tab3, tab4 = st.tabs([
            "Feature Distributions",
            "Healthy vs. Faulty Comparison",
            "Scenario Breakdown",
            "Correlation Matrix",
        ])

        with tab1:
            st.markdown("**Single Feature Distribution Histogram**")
            selected_feature = st.selectbox("Select Feature", FEATURE_NAMES, index=2)
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.hist(split_df[selected_feature].dropna(), bins=40, color="#1f77b4", edgecolor="black", alpha=0.7)
            ax.set_title(f"Distribution of {selected_feature} ({selected_split} partition) [SYNTHETIC]")
            ax.set_xlabel(f"{selected_feature} [{FEATURE_UNITS.get(selected_feature, '')}]")
            ax.set_ylabel("Frequency")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

        with tab2:
            st.markdown("**Healthy vs. Faulty Class Separation**")
            compare_feature = st.selectbox("Feature to Compare", FEATURE_NAMES, index=10, key="comp_feat")
            healthy_vals = split_df[split_df["fault_label"] == 0][compare_feature].dropna()
            faulty_vals = split_df[split_df["fault_label"] == 1][compare_feature].dropna()

            fig, ax = plt.subplots(figsize=(9, 4))
            ax.hist(healthy_vals, bins=35, alpha=0.6, label="Healthy (y=0)", color="#2ca02c", density=True)
            ax.hist(faulty_vals, bins=35, alpha=0.6, label="Faulty (y=1)", color="#d62728", density=True)
            ax.set_title(f"Class Separation: {compare_feature} [SYNTHETIC]")
            ax.set_xlabel(f"{compare_feature} [{FEATURE_UNITS.get(compare_feature, '')}]")
            ax.set_ylabel("Normalized Density")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

        with tab3:
            st.markdown("**Scenario-Wise Feature Means and Standard Deviations**")
            sc_feature = st.selectbox("Feature by Scenario", FEATURE_NAMES, index=12, key="sc_feat")
            grouped = split_df.groupby("scenario")[sc_feature].agg(["mean", "std"]).reset_index()

            fig, ax = plt.subplots(figsize=(10, 4.5))
            ax.bar(grouped["scenario"], grouped["mean"], yerr=grouped["std"], capsize=5, color="#3b528b", alpha=0.85)
            ax.set_title(f"Scenario-Wise {sc_feature} [SYNTHETIC]")
            ax.set_ylabel(f"Mean {sc_feature} [{FEATURE_UNITS.get(sc_feature, '')}]")
            plt.xticks(rotation=25, ha="right")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

        with tab4:
            st.markdown("**Top Feature Cross-Correlation Matrix**")
            top_features = [
                "voltage_mean_v",
                "voltage_std_v",
                "voltage_range_v",
                "voltage_mad_v",
                "voltage_max_abs_diff_v",
                "current_mean_a",
                "current_std_a",
                "current_range_a",
                "current_mad_a",
                "current_max_abs_diff_a",
            ]
            corr = split_df[top_features].corr()

            fig, ax = plt.subplots(figsize=(8, 6.5))
            cax = ax.matshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
            fig.colorbar(cax)
            ax.set_xticks(range(len(top_features)))
            ax.set_yticks(range(len(top_features)))
            ax.set_xticklabels(top_features, rotation=45, ha="left", fontsize=8)
            ax.set_yticklabels(top_features, fontsize=8)
            ax.set_title("Correlation Heatmap [SYNTHETIC]", pad=20)
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.info(f"Generate feature dataset for '{dataset_id}' above to enable feature exploration and visualizations.")


def page_model_training() -> None:
    st.title("🤖 Machine Learning Model Training")
    render_synthetic_notice()
    st.markdown(
        "Train and tune candidate classifiers on the verified synthetic feature partitions. "
        "Strict allowlist enforcement guarantees no unobservable or ground-truth columns are leaked."
    )

    storage_root = PROCESSED_FEATURES_ROOT
    available_datasets = [d.name for d in storage_root.iterdir() if d.is_dir()] if storage_root.exists() else []
    if not available_datasets:
        st.error("No processed feature datasets found. Please extract features in the Feature Analysis page.")
        return

    col1, col2 = st.columns(2)
    with col1:
        dataset_id = st.selectbox("Select Feature Dataset", available_datasets, index=0)
    with col2:
        seed = st.number_input("Random Seed", min_value=0, max_value=999999, value=42)

    with st.expander("Approved Feature Allowlist (26 Features)", expanded=False):
        st.markdown(
            "Every classifier receives strictly these 26 observable electrical and statistical features. "
            "No ground-truth labels, scenario names, or unobservable physical parameters are ever provided."
        )
        allowlist_df = pd.DataFrame({
            "Feature Name": FEATURE_NAMES,
            "Unit": [FEATURE_UNITS.get(f, "") for f in FEATURE_NAMES],
        })
        st.dataframe(allowlist_df, use_container_width=True)

    selected_models = st.multiselect(
        "Candidate Classification Models",
        options=SUPPORTED_MODELS,
        default=SUPPORTED_MODELS,
    )

    if st.button("Train Selected Models & Benchmark", type="primary"):
        if not selected_models:
            st.warning("Please select at least one candidate model to train.")
            return

        with st.spinner("Training models, tuning hyperparameters on validation set, and freezing..."):
            summary = run_phase4_benchmark(
                dataset_id=dataset_id,
                random_seed=seed,
                models_to_train=selected_models,
            )
            st.success(f"Phase 4 Benchmark Complete! Best model: **{summary.best_model_name.replace('_', ' ').title()}** (Validation F1 = {summary.best_model_validation_f1:.4f})")

    st.subheader("Saved Model Artifacts")
    models_dir = PROJECT_ROOT / "models" / "phase4"
    if models_dir.exists():
        joblib_files = list(models_dir.glob("*.joblib"))
        if joblib_files:
            rows = []
            for jf in sorted(joblib_files):
                meta_path = models_dir / f"{jf.stem}_metadata.json"
                meta_data = {}
                if meta_path.exists():
                    try:
                        with open(meta_path, "r", encoding="utf-8") as f:
                            meta_data = json.load(f)
                    except Exception:
                        pass
                rows.append({
                    "Artifact File": jf.name,
                    "Size [KB]": round(jf.stat().st_size / 1024, 2),
                    "Validation F1": round(meta_data.get("validation_f1", 0.0), 4) if meta_data else "N/A",
                    "Validation Bal Acc": round(meta_data.get("validation_balanced_accuracy", 0.0), 4) if meta_data else "N/A",
                    "Train Time [s]": round(meta_data.get("train_duration_s", 0.0), 4) if meta_data else "N/A",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            best_meta_path = models_dir / "best_model_metadata.json"
            if best_meta_path.exists():
                with open(best_meta_path, "r", encoding="utf-8") as f:
                    best_meta = json.load(f)
                st.info(
                    f"🏆 **Primary Deployed Model:** `{best_meta.get('model_name', '').replace('_', ' ').title()}` | "
                    f"Validation F1: **{best_meta.get('validation_f1', 0.0):.4f}** | "
                    f"Optimal Parameters: `{best_meta.get('best_hyperparameters', {})}`"
                )
        else:
            st.info("No trained models saved yet. Run training above.")
    else:
        st.info("No trained models directory found.")


def page_self_calibration() -> None:
    st.title("🔄 Confidence-Gated Adaptive Self-Calibration Engine")
    render_synthetic_notice()
    st.markdown(
        "Demonstrates initial known-healthy baseline estimation, confidence-gated online adaptation, "
        "and freeze-on-suspicion safeguards to prevent baseline contamination during true fault events."
    )

    tab_run, tab_benchmark, tab_ablation = st.tabs([
        "1. Interactive Live Run Inspector",
        "2. Method Benchmarking (A0 - A3)",
        "3. Safeguard Ablation Study",
    ])

    tables_dir = PROJECT_ROOT / "results" / "tables" / "phase5_calibration"
    figures_dir = PROJECT_ROOT / "results" / "figures" / "phase5_calibration"

    with tab_run:
        st.subheader("Online Calibration Trajectory Replay")
        c1, c2, c3 = st.columns(3)
        with c1:
            alpha = st.slider("Adaptation Rate (alpha)", min_value=0.01, max_value=0.20, value=0.04, step=0.01)
        with c2:
            tau = st.slider("Confidence Gate Threshold", min_value=0.80, max_value=0.99, value=0.95, step=0.01)
        with c3:
            enable_freeze = st.checkbox("Enable Freeze-on-Suspicion Logic", value=True)

        feature_storage = FeatureDatasetStorage("flexpulse_calibration_v2")
        try:
            ood_df = feature_storage.load_partition("ood_test")
            available_runs = sorted(ood_df["run_id"].unique())
            selected_run = st.selectbox("Select Simulation Run to Replay", available_runs, index=0)

            run_df = ood_df[ood_df["run_id"] == selected_run].reset_index(drop=True)
            rf_norm_path = PROJECT_ROOT / "models" / "phase5" / "a2_a3_normalized_random_forest.joblib"

            if rf_norm_path.exists():
                import joblib
                rf_model = joblib.load(rf_norm_path)
                cfg = AdaptiveCalibrationConfig(alpha=alpha, tau_conf=tau, enable_freeze_logic=enable_freeze)
                calibrator = ConfidenceGatedAdaptiveCalibrator(cfg)
                preds, probs, diag = calibrator.process_run(run_df, rf_model)
                summary = diag.compute_summary_metrics()

                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Total Windows", summary["total_windows"])
                with m2:
                    st.metric("Accepted Updates", f"{summary['accepted_updates']} ({summary['acceptance_rate']*100:.1f}%)")
                with m3:
                    st.metric("Frozen Time", f"{summary['time_spent_frozen_s']:.1f} s")
                with m4:
                    st.metric("Fault Contamination Count", summary["inappropriate_updates_during_fault"])

                rec_df = pd.DataFrame([r.to_dict() for r in diag.records])

                fig, ax = plt.subplots(figsize=(10, 4.5))
                ax.plot(rec_df["timestamp_end_s"], rec_df["raw_v"], color="#7f7f7f", lw=1.2, label="Observed Voltage Mean")
                ax.plot(rec_df["timestamp_end_s"], rec_df["active_baseline_v"], color="#1f77b4", lw=2.2, label="Active Baseline (EMA)")

                accepted_mask = rec_df["update_accepted"]
                ax.scatter(rec_df.loc[accepted_mask, "timestamp_end_s"], rec_df.loc[accepted_mask, "raw_v"], color="#2ca02c", s=18, label="Accepted Update", zorder=4)
                ax.scatter(rec_df.loc[~accepted_mask, "timestamp_end_s"], rec_df.loc[~accepted_mask, "raw_v"], color="#d62728", marker="x", s=22, label="Rejected Update", zorder=4)

                fault_mask = rec_df["ground_truth_fault"] == 1
                if np.any(fault_mask):
                    ax.fill_between(rec_df["timestamp_end_s"], min(rec_df["raw_v"]) - 0.05, max(rec_df["raw_v"]) + 0.05, where=fault_mask, color="#d62728", alpha=0.15, label="True Fault Interval")

                ax.set_title(f"Online Baseline Tracking Replay: {selected_run} [SIMULATION-BASED EXPERIMENT]")
                ax.set_xlabel("Time [s]")
                ax.set_ylabel("Voltage [V]")
                ax.grid(True, alpha=0.3)
                ax.legend(loc="lower right")
                st.pyplot(fig)
                plt.close(fig)

                st.markdown("**Per-Window Diagnostic History Table:**")
                st.dataframe(rec_df[["window_index", "timestamp_end_s", "raw_v", "active_baseline_v", "confidence", "state", "update_accepted", "rejection_reason", "prediction", "ground_truth_fault"]], use_container_width=True)
            else:
                st.info("Train Phase 5 calibration models to view live run replay.")
        except Exception as e:
            st.info(f"Calibration v2 dataset pending: {e}")

    with tab_benchmark:
        st.subheader("Four-Way Paradigm Comparison (A0 - A3)")
        st.markdown(
            "Comparing **A0 (Fixed Threshold)**, **A1 (Static ML)**, **A2 (Initial Normalization)**, and **A3 (Confidence-Gated Adaptive)** "
            "across nominal in-distribution and shifted out-of-distribution robustness test sets."
        )
        p6 = figures_dir / "fig6_method_comparison_id_ood.png"
        if p6.exists():
            st.image(str(p6), caption="Figure 6: ID Test vs. OOD Robustness Test F1-Score (SIMULATION-BASED EXPERIMENT)")

        ood_table = tables_dir / "method_comparison_ood_test.csv"
        if ood_table.exists():
            st.markdown("**Out-of-Distribution Robustness Test Results (Untouched Held-Out Partition):**")
            st.dataframe(pd.read_csv(ood_table), use_container_width=True)

        id_table = tables_dir / "method_comparison_id_test.csv"
        if id_table.exists():
            st.markdown("**In-Distribution (ID) Test Results:**")
            st.dataframe(pd.read_csv(id_table), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            p7 = figures_dir / "fig7_false_alarm_vs_recall_tradeoff.png"
            if p7.exists():
                st.image(str(p7), caption="Figure 7: Event Recall vs. False Alarm Rate (SIMULATION-BASED EXPERIMENT)")
        with c2:
            p8 = figures_dir / "fig8_causal_detection_delays.png"
            if p8.exists():
                st.image(str(p8), caption="Figure 8: Causal Event Detection Delays (SIMULATION-BASED EXPERIMENT)")

    with tab_ablation:
        st.subheader("Safeguard Ablation and Failure Mode Analysis")
        p10 = figures_dir / "fig10_ablation_study.png"
        if p10.exists():
            st.image(str(p10), caption="Figure 10: Ablation Study of Individual Safeguards (SIMULATION-BASED EXPERIMENT)")

        ab_table = tables_dir / "ablation_study.csv"
        if ab_table.exists():
            st.markdown("**Ablation Performance Table:**")
            st.dataframe(pd.read_csv(ab_table), use_container_width=True)

        st.subheader("Calibration Failure Mode: Gradual Fault Contamination")
        st.markdown(
            "When adaptive updates lack confidence gating or freeze safeguards, the baseline incorrectly tracks "
            "slowly developing degradation, absorbing the fault and causing catastrophic false negatives."
        )
        p9 = figures_dir / "fig9_contamination_failure_modes.png"
        if p9.exists():
            st.image(str(p9), caption="Figure 9: Contamination Failure Mode (Naive vs. Protected Adaptation)")

        contam_table = tables_dir / "contamination_stress_test.csv"
        if contam_table.exists():
            st.markdown("**Contamination Stress Test Table:**")
            st.dataframe(pd.read_csv(contam_table), use_container_width=True)


def page_evaluation() -> None:
    st.title("📊 Model Benchmarking & Empirical Evaluation")
    render_synthetic_notice()
    st.markdown(
        "Comparative evaluation across In-Distribution (ID) and Out-Of-Distribution (OOD) test sets. "
        "Evaluates window classification, event recall, false alarms per minute, detection delays, and scenario breakdowns."
    )

    tables_dir = PROJECT_ROOT / "results" / "tables" / "phase4_benchmark"
    figures_dir = PROJECT_ROOT / "results" / "figures" / "phase4_benchmark"
    manifest_path = PROJECT_ROOT / "results" / "experiments" / "phase4_benchmark" / "phase4_benchmark_manifest.json"

    if not manifest_path.exists() or not tables_dir.exists():
        st.info("No benchmark results found. Train models in the Model Training page first.")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Best Model", manifest.get("best_model_name", "").replace("_", " ").title())
    with col2:
        st.metric("Validation F1", f"{manifest.get('best_model_validation_f1', 0.0):.4f}")
    with col3:
        st.metric("Feature Count", manifest.get("feature_count", 26))
    with col4:
        st.metric("Random Seed", manifest.get("random_seed", 42))

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "1. ID Test Benchmark",
        "2. Generalization (ID vs OOD)",
        "3. ROC & PR Curves",
        "4. Confusion Matrices",
        "5. Scenario Error Breakdown",
        "6. Event Detections & Delays",
        "7. Sensitivity & Bootstrap CIs",
    ])

    with tab1:
        st.subheader("In-Distribution (ID) Test Performance")
        id_csv = tables_dir / "model_comparison_id_test.csv"
        if id_csv.exists():
            st.dataframe(pd.read_csv(id_csv), use_container_width=True)
        fig1 = figures_dir / "fig1_model_comparison_bars.png"
        if fig1.exists():
            st.image(str(fig1), caption="Figure 1: Model Comparison Bars on ID Test (SIMULATED DATA)")

    with tab2:
        st.subheader("Distribution Shift Robustness: ID vs. OOD Test Set")
        st.markdown(
            "Under severe out-of-distribution shifts (elevated ambient temperature, sensor bias offset, sensor drift), "
            "static models experience severe degradation (precision collapse or complete recall loss). "
            "This empirically demonstrates the necessity of adaptive self-calibration."
        )
        fig5 = figures_dir / "fig5_id_vs_ood_comparison.png"
        if fig5.exists():
            st.image(str(fig5), caption="Figure 5: Nominal ID vs. Shifted OOD F1-Score (SIMULATED DATA)")
        ood_csv = tables_dir / "model_comparison_ood_test.csv"
        if ood_csv.exists():
            st.markdown("**OOD Test Set Metrics:**")
            st.dataframe(pd.read_csv(ood_csv), use_container_width=True)

    with tab3:
        st.subheader("Precision-Recall and Receiver Operating Characteristic (ROC) Curves")
        c1, c2 = st.columns(2)
        with c1:
            fig3 = figures_dir / "fig3_precision_recall_curves.png"
            if fig3.exists():
                st.image(str(fig3), caption="Figure 3: Precision-Recall Curves (SIMULATED DATA)")
        with c2:
            fig4 = figures_dir / "fig4_roc_curves.png"
            if fig4.exists():
                st.image(str(fig4), caption="Figure 4: ROC Curves (SIMULATED DATA)")

    with tab4:
        st.subheader("ID Test Confusion Matrices")
        fig2 = figures_dir / "fig2_confusion_matrices.png"
        if fig2.exists():
            st.image(str(fig2), caption="Figure 2: Confusion Matrices for All 6 Models (SIMULATED DATA)")

    with tab5:
        st.subheader("Scenario-Specific Error Analysis")
        fig6 = figures_dir / "fig6_scenario_error_analysis.png"
        if fig6.exists():
            st.image(str(fig6), caption="Figure 6: Best Model Scenario Recall vs. False Positive Rate (SIMULATED DATA)")
        sc_csv = tables_dir / "scenario_breakdown_id_test.csv"
        if sc_csv.exists():
            st.markdown("**ID Test Scenario Metrics Table:**")
            st.dataframe(pd.read_csv(sc_csv), use_container_width=True)

    with tab6:
        st.subheader("Event-Level Detection, False Alarms, and Detection Delay")
        fig8 = figures_dir / "fig8_detection_delay_distribution.png"
        if fig8.exists():
            st.image(str(fig8), caption="Figure 8: Detection Delay Distributions for Injected Fault Events (SIMULATED DATA)")
        ev_csv = tables_dir / "event_level_detections_id_test.csv"
        if ev_csv.exists():
            st.markdown("**Event-Level Detection Metrics (ID Test):**")
            st.dataframe(pd.read_csv(ev_csv), use_container_width=True)

    with tab7:
        st.subheader("Window-Label Target Sensitivity and Bootstrap Confidence Intervals")
        c1, c2 = st.columns(2)
        with c1:
            fig7 = figures_dir / "fig7_window_label_sensitivity.png"
            if fig7.exists():
                st.image(str(fig7), caption="Figure 7: Target Fault Fraction Sensitivity (SIMULATED DATA)")
        with c2:
            sens_csv = tables_dir / "label_sensitivity_id_test.csv"
            if sens_csv.exists():
                st.markdown("**Label Sensitivity Table:**")
                st.dataframe(pd.read_csv(sens_csv), use_container_width=True)

        boot_csv = tables_dir / "bootstrap_ci_id_test.csv"
        if boot_csv.exists():
            st.markdown("**Run-Level Bootstrap 95% Confidence Intervals (200 Resamples):**")
            st.dataframe(pd.read_csv(boot_csv), use_container_width=True)


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
