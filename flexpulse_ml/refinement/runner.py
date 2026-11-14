"""
Phase 6 scientific validation and refinement experiment runner.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from flexpulse_ml.calibration.adaptive import (
    AdaptiveCalibrationConfig,
    ConfidenceGatedAdaptiveCalibrator,
)
from flexpulse_ml.calibration.baseline import InitialBaselineEstimator
from flexpulse_ml.config import PROJECT_ROOT
from flexpulse_ml.data.manifest import compute_file_sha256
from flexpulse_ml.data.storage import DatasetStorage
from flexpulse_ml.features.dataset import FeatureDatasetStorage
from flexpulse_ml.features.extractor import FEATURE_NAMES
from flexpulse_ml.refinement.audit import (
    FalseAlarmInvestigator,
    FeatureNormalizationAudit,
    WindowLevelA2vsA3Diagnostic,
)
from flexpulse_ml.refinement.confidence_eval import ConfidenceCalibrationEvaluator
from flexpulse_ml.refinement.persistence import TemporalPersistenceFilter
from flexpulse_ml.refinement.plotting import (
    plot_a2_vs_a3_baseline_trajectories,
    plot_a2_vs_a3_prediction_differences,
    plot_adaptive_benefit_or_failure_regimes,
    plot_confidence_reliability_diagram,
    plot_detection_delay_tradeoffs,
    plot_false_alarm_analysis,
    plot_feature_normalization_audit,
    plot_paired_comparison_confidence_intervals,
    plot_scenario_performance_audit,
    plot_update_acceptance_rejection_audit,
)
from flexpulse_ml.refinement.regimes import RegimeEvaluationMatrix


@dataclass
class Phase6RefinementSummary:
    experiment_id: str
    timestamp_utc: str
    dataset_id: str
    artifact_paths: Dict[str, str] = field(default_factory=dict)
    checksums_sha256: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Phase6RefinementRunner:
    """Coordinates execution of Phase 6 audits, persistence tests, E1-E10 regimes, and figure generation."""

    def __init__(
        self,
        dataset_id: str = "flexpulse_calibration_v2",
        random_seed: int = 42,
        n_bootstraps: int = 200,
    ) -> None:
        self.dataset_id = dataset_id
        self.random_seed = random_seed
        self.n_bootstraps = n_bootstraps

        self.tables_dir = PROJECT_ROOT / "results" / "tables" / "phase6_refinement"
        self.figures_dir = PROJECT_ROOT / "results" / "figures" / "phase6_refinement"
        self.results_dir = PROJECT_ROOT / "results" / "experiments" / "phase6_refinement"

        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.feature_storage = FeatureDatasetStorage(dataset_id)
        self.raw_storage = DatasetStorage(dataset_id)
        self.baseline_estimator = InitialBaselineEstimator(initial_duration_s=5.0)

        self.models = self._load_phase5_models()

    def _load_phase5_models(self) -> Dict[str, Any]:
        m_dir = PROJECT_ROOT / "models" / "phase5"
        rf_raw = joblib.load(m_dir / "a1_static_random_forest.joblib")
        rf_norm = joblib.load(m_dir / "a2_a3_normalized_random_forest.joblib")
        fixed_th = joblib.load(m_dir / "a0_fixed_threshold.joblib")
        return {
            "rf_raw": rf_raw,
            "rf_norm": rf_norm,
            "fixed_threshold": fixed_th,
        }

    def run(self) -> Phase6RefinementSummary:
        start_time = datetime.now(timezone.utc).isoformat()
        exp_id = f"phase6_refinement_{self.dataset_id}"

        dev_df = self.feature_storage.load_partition(DatasetStorage.SPLIT_VAL)
        ood_test_df = self.feature_storage.load_partition(DatasetStorage.SPLIT_OOD_TEST)

        feat_audit_records = FeatureNormalizationAudit.audit_all_features()
        feat_audit_df = pd.DataFrame([r.to_dict() for r in feat_audit_records])

        diagnostic_engine = WindowLevelA2vsA3Diagnostic()
        a2_a3_dev_diag = diagnostic_engine.evaluate_partition(dev_df, self.models["rf_norm"])
        a2_a3_ood_diag = diagnostic_engine.evaluate_partition(ood_test_df, self.models["rf_norm"])
        diag_comparison_df = pd.DataFrame([
            {"split": "validation_dev", **a2_a3_dev_diag.to_dict()},
            {"split": "ood_test", **a2_a3_ood_diag.to_dict()},
        ])

        norm_runs: List[pd.DataFrame] = []
        for r_id in ood_test_df["run_id"].unique():
            r_df = ood_test_df[ood_test_df["run_id"] == r_id].reset_index(drop=True)
            b = self.baseline_estimator.fit_run(r_df)
            norm_runs.append(self.baseline_estimator.normalize_features(r_df, b))
        ood_norm_df = pd.concat(norm_runs, ignore_index=True)
        X_ood_norm = ood_norm_df[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
        a2_ood_preds = self.models["rf_norm"].predict(X_ood_norm)

        fa_report = FalseAlarmInvestigator.audit_partition_false_alarms(
            df=ood_test_df,
            preds=a2_ood_preds,
            raw_storage=self.raw_storage,
            split_name="ood_test",
        )
        fa_df = pd.DataFrame([{
            "total_runs_evaluated": fa_report.total_runs_evaluated,
            "runs_with_false_alarms": fa_report.runs_with_false_alarms,
            "total_healthy_windows": fa_report.total_healthy_windows,
            "false_positive_windows": fa_report.false_positive_windows,
            "false_positive_window_rate": fa_report.false_positive_window_rate,
            "total_false_alarm_events": fa_report.total_false_alarm_events,
            "false_alarms_per_simulated_minute": fa_report.false_alarms_per_simulated_minute,
            "mean_false_alarm_duration_s": fa_report.mean_false_alarm_duration_s,
            "median_false_alarm_duration_s": fa_report.median_false_alarm_duration_s,
            "max_false_alarm_duration_s": fa_report.max_false_alarm_duration_s,
            "fragmented_false_alarms_fraction": fa_report.fragmented_false_alarms_fraction,
        }])
        scenario_fa_df = pd.DataFrame([
            {
                "scenario": sc,
                "false_positive_windows": fa_report.scenario_false_positive_counts.get(sc, 0),
                "false_alarm_events": fa_report.scenario_false_alarm_events.get(sc, 0),
            }
            for sc in fa_report.scenario_false_positive_counts
        ])

        norm_dev_runs: List[pd.DataFrame] = []
        for r_id in dev_df["run_id"].unique():
            r_df = dev_df[dev_df["run_id"] == r_id].reset_index(drop=True)
            b = self.baseline_estimator.fit_run(r_df)
            norm_dev_runs.append(self.baseline_estimator.normalize_features(r_df, b))
        dev_norm_df = pd.concat(norm_dev_runs, ignore_index=True)
        X_dev_norm = dev_norm_df[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
        a2_dev_preds = self.models["rf_norm"].predict(X_dev_norm)

        persist_results: List[Dict[str, Any]] = []
        base_median_delay = 0.1572
        for n_w in [1, 2, 3]:
            p_filter = TemporalPersistenceFilter(persistence_windows=n_w)
            res_dev = p_filter.evaluate_on_partition(
                df=dev_df,
                raw_preds=a2_dev_preds,
                raw_storage=self.raw_storage,
                split_name="validation",
                baseline_median_delay_s=base_median_delay,
            )
            res_ood = p_filter.evaluate_on_partition(
                df=ood_test_df,
                raw_preds=a2_ood_preds,
                raw_storage=self.raw_storage,
                split_name="ood_test",
                baseline_median_delay_s=base_median_delay,
            )
            persist_results.append(res_dev.to_dict())
            persist_results.append(res_ood.to_dict())
        persist_df = pd.DataFrame(persist_results)

        calib_evaluator = ConfidenceCalibrationEvaluator()
        y_dev_healthy = (dev_df["fault_label"].to_numpy(dtype=np.int32) == 0).astype(np.int32)
        p_dev_healthy = self.models["rf_norm"].predict_proba(X_dev_norm)[:, 0]
        calib_report = calib_evaluator.evaluate_calibration(y_dev_healthy, p_dev_healthy)
        calib_bins_df = pd.DataFrame([b.to_dict() for b in calib_report.reliability_bins])
        threshold_sens_df = pd.DataFrame(calib_report.threshold_sensitivity)

        regime_eval = RegimeEvaluationMatrix(random_seed=self.random_seed)
        regime_rows = regime_eval.evaluate_regimes(self.models, runs_per_regime=4)
        regime_df = pd.DataFrame([r.to_dict() for r in regime_rows])

        paired_ci_df = self._compute_paired_bootstrap_ci(ood_test_df)

        tables_created = self._export_tables(
            feat_audit_df=feat_audit_df,
            diag_comparison_df=diag_comparison_df,
            fa_df=fa_df,
            scenario_fa_df=scenario_fa_df,
            persist_df=persist_df,
            calib_bins_df=calib_bins_df,
            threshold_sens_df=threshold_sens_df,
            regime_df=regime_df,
            paired_ci_df=paired_ci_df,
        )

        figures_created = self._export_figures(
            feat_audit_records=[r.to_dict() for r in feat_audit_records],
            ood_test_df=ood_test_df,
            a2_ood_preds=a2_ood_preds,
            rejection_dict=a2_a3_ood_diag.rejection_breakdown,
            accepted_count=a2_a3_ood_diag.updates_accepted,
            scenario_fp_dict=fa_report.scenario_false_positive_counts,
            persist_results=[r for r in persist_results if r["split"] == "ood_test"],
            calib_report=calib_report,
            regime_rows=[r.to_dict() for r in regime_rows],
            paired_ci_df=paired_ci_df,
        )

        manifest_data = {
            "experiment_id": exp_id,
            "timestamp_utc": start_time,
            "dataset_id": self.dataset_id,
            "random_seed": self.random_seed,
            "system_architecture": f"{platform.system()}_{platform.machine()}",
            "python_version": platform.python_version(),
            "tables": {k: str(v) for k, v in tables_created.items()},
            "figures": {k: str(v) for k, v in figures_created.items()},
        }
        manifest_path = self.results_dir / "phase6_refinement_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        all_artifacts = dict(tables_created)
        all_artifacts.update(figures_created)
        all_artifacts["manifest"] = manifest_path

        checksums = {k: compute_file_sha256(Path(p)) for k, p in all_artifacts.items() if Path(p).exists()}
        checksums_path = self.results_dir / "phase6_refinement_checksums.json"
        with open(checksums_path, "w", encoding="utf-8") as f:
            json.dump(checksums, f, indent=2)

        return Phase6RefinementSummary(
            experiment_id=exp_id,
            timestamp_utc=start_time,
            dataset_id=self.dataset_id,
            artifact_paths={k: str(v) for k, v in all_artifacts.items()},
            checksums_sha256=checksums,
        )

    def _compute_paired_bootstrap_ci(self, ood_df: pd.DataFrame) -> pd.DataFrame:
        runs = ood_df["run_id"].unique()
        n_runs = len(runs)
        rng = np.random.default_rng(self.random_seed)

        calibrator_a3 = ConfidenceGatedAdaptiveCalibrator()

        run_preds_a2: Dict[str, np.ndarray] = {}
        run_preds_a3: Dict[str, np.ndarray] = {}
        run_y: Dict[str, np.ndarray] = {}

        for r_id in runs:
            r_df = ood_df[ood_df["run_id"] == r_id].reset_index(drop=True)
            b = self.baseline_estimator.fit_run(r_df)
            r_norm = self.baseline_estimator.normalize_features(r_df, b)
            X = r_norm[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)

            p2 = self.models["rf_norm"].predict(X)
            p3, _, _ = calibrator_a3.process_run(r_df, self.models["rf_norm"])

            run_preds_a2[r_id] = p2
            run_preds_a3[r_id] = p3
            run_y[r_id] = r_df["fault_label"].to_numpy(dtype=np.int32)

        delta_f1_samples: List[float] = []
        delta_bal_samples: List[float] = []
        delta_recall_samples: List[float] = []
        delta_prec_samples: List[float] = []

        for _ in range(self.n_bootstraps):
            sampled_runs = rng.choice(runs, size=n_runs, replace=True)
            y_s = np.concatenate([run_y[r] for r in sampled_runs])
            p2_s = np.concatenate([run_preds_a2[r] for r in sampled_runs])
            p3_s = np.concatenate([run_preds_a3[r] for r in sampled_runs])

            if len(np.unique(y_s)) > 1:
                f1_2 = float(f1_score(y_s, p2_s, zero_division=0))
                f1_3 = float(f1_score(y_s, p3_s, zero_division=0))
                delta_f1_samples.append(f1_3 - f1_2)

                bal_2 = float(balanced_accuracy_score(y_s, p2_s))
                bal_3 = float(balanced_accuracy_score(y_s, p3_s))
                delta_bal_samples.append(bal_3 - bal_2)

                rec_2 = float(recall_score(y_s, p2_s, zero_division=0))
                rec_3 = float(recall_score(y_s, p3_s, zero_division=0))
                delta_recall_samples.append(rec_3 - rec_2)

                prec_2 = float(precision_score(y_s, p2_s, zero_division=0))
                prec_3 = float(precision_score(y_s, p3_s, zero_division=0))
                delta_prec_samples.append(prec_3 - prec_2)

        ci_rows = [
            {
                "metric_name": "Delta F1 (A3 - A2)",
                "mean_delta": round(float(np.mean(delta_f1_samples)), 4) if delta_f1_samples else 0.0,
                "ci_lower": round(float(np.percentile(delta_f1_samples, 2.5)), 4) if delta_f1_samples else 0.0,
                "ci_upper": round(float(np.percentile(delta_f1_samples, 97.5)), 4) if delta_f1_samples else 0.0,
            },
            {
                "metric_name": "Delta Balanced Acc (A3 - A2)",
                "mean_delta": round(float(np.mean(delta_bal_samples)), 4) if delta_bal_samples else 0.0,
                "ci_lower": round(float(np.percentile(delta_bal_samples, 2.5)), 4) if delta_bal_samples else 0.0,
                "ci_upper": round(float(np.percentile(delta_bal_samples, 97.5)), 4) if delta_bal_samples else 0.0,
            },
            {
                "metric_name": "Delta Recall (A3 - A2)",
                "mean_delta": round(float(np.mean(delta_recall_samples)), 4) if delta_recall_samples else 0.0,
                "ci_lower": round(float(np.percentile(delta_recall_samples, 2.5)), 4) if delta_recall_samples else 0.0,
                "ci_upper": round(float(np.percentile(delta_recall_samples, 97.5)), 4) if delta_recall_samples else 0.0,
            },
            {
                "metric_name": "Delta Precision (A3 - A2)",
                "mean_delta": round(float(np.mean(delta_prec_samples)), 4) if delta_prec_samples else 0.0,
                "ci_lower": round(float(np.percentile(delta_prec_samples, 2.5)), 4) if delta_prec_samples else 0.0,
                "ci_upper": round(float(np.percentile(delta_prec_samples, 97.5)), 4) if delta_prec_samples else 0.0,
            },
        ]
        return pd.DataFrame(ci_rows)

    def _export_tables(
        self,
        feat_audit_df: pd.DataFrame,
        diag_comparison_df: pd.DataFrame,
        fa_df: pd.DataFrame,
        scenario_fa_df: pd.DataFrame,
        persist_df: pd.DataFrame,
        calib_bins_df: pd.DataFrame,
        threshold_sens_df: pd.DataFrame,
        regime_df: pd.DataFrame,
        paired_ci_df: pd.DataFrame,
    ) -> Dict[str, Path]:
        tables: Dict[str, Path] = {}

        def write_tb(df: pd.DataFrame, base_name: str) -> None:
            csv_p = self.tables_dir / f"{base_name}.csv"
            md_p = self.tables_dir / f"{base_name}.md"
            df.to_csv(csv_p, index=False)
            try:
                df.to_markdown(md_p, index=False)
            except Exception:
                md_p.write_text(df.to_string(index=False), encoding="utf-8")
            tables[f"{base_name}_csv"] = csv_p
            tables[f"{base_name}_md"] = md_p

        write_tb(feat_audit_df, "feature_normalization_audit")
        write_tb(diag_comparison_df, "a2_vs_a3_window_diagnostic")
        write_tb(fa_df, "false_alarm_investigation_summary")
        write_tb(scenario_fa_df, "false_alarm_scenario_breakdown")
        write_tb(persist_df, "temporal_persistence_tradeoff")
        write_tb(calib_bins_df, "confidence_reliability_bins")
        write_tb(threshold_sens_df, "confidence_threshold_sensitivity")
        write_tb(regime_df, "controlled_regimes_e1_e10")
        write_tb(paired_ci_df, "paired_bootstrap_ci_a3_minus_a2")

        return tables

    def _export_figures(
        self,
        feat_audit_records: List[Dict[str, Any]],
        ood_test_df: pd.DataFrame,
        a2_ood_preds: np.ndarray,
        rejection_dict: Dict[str, int],
        accepted_count: int,
        scenario_fp_dict: Dict[str, int],
        persist_results: List[Dict[str, Any]],
        calib_report: Any,
        regime_rows: List[Dict[str, Any]],
        paired_ci_df: pd.DataFrame,
    ) -> Dict[str, Path]:
        figures: Dict[str, Path] = {}

        p1 = plot_feature_normalization_audit(
            feat_audit_records,
            output_path=self.figures_dir / "fig1_feature_normalization_audit.png",
        )
        figures["feature_normalization_audit"] = p1

        single_run_id = ood_test_df["run_id"].unique()[0]
        single_df = ood_test_df[ood_test_df["run_id"] == single_run_id].reset_index(drop=True)
        init_b = self.baseline_estimator.fit_run(single_df)
        calibrator = ConfidenceGatedAdaptiveCalibrator()
        p_a3, _, diag = calibrator.process_run(single_df, self.models["rf_norm"])

        t_axis = single_df["window_end_s"].to_numpy()
        v_obs = single_df["voltage_mean_v"].to_numpy()
        a3_base_v = np.array([r.active_baseline_v for r in diag.records])

        p2 = plot_a2_vs_a3_baseline_trajectories(
            time_array=t_axis,
            observed_v=v_obs,
            a2_base_v=init_b.center_v,
            a3_base_v=a3_base_v,
            output_path=self.figures_dir / "fig2_a2_vs_a3_baseline_trajectories.png",
        )
        figures["a2_vs_a3_baseline_trajectories"] = p2

        run_mask = (ood_test_df["run_id"].to_numpy() == single_run_id)
        p_a2_single = a2_ood_preds[run_mask]
        gt_single = single_df["fault_label"].to_numpy(dtype=int)

        p3 = plot_a2_vs_a3_prediction_differences(
            time_array=t_axis,
            y_true=gt_single,
            a2_preds=p_a2_single,
            a3_preds=p_a3,
            output_path=self.figures_dir / "fig3_a2_vs_a3_prediction_differences.png",
        )
        figures["a2_vs_a3_prediction_differences"] = p3

        p4 = plot_update_acceptance_rejection_audit(
            rejection_dict=rejection_dict,
            accepted_count=accepted_count,
            output_path=self.figures_dir / "fig4_update_acceptance_rejection_audit.png",
        )
        figures["update_acceptance_rejection_audit"] = p4

        p5_df_path = PROJECT_ROOT / "results" / "tables" / "phase5_calibration" / "scenario_breakdown_ood_test.csv"
        if p5_df_path.exists():
            sc_df = pd.read_csv(p5_df_path)
            p5 = plot_scenario_performance_audit(
                scenario_df=sc_df,
                output_path=self.figures_dir / "fig5_scenario_performance_audit.png",
            )
            figures["scenario_performance_audit"] = p5

        p6 = plot_false_alarm_analysis(
            scenario_fp_dict=scenario_fp_dict,
            output_path=self.figures_dir / "fig6_false_alarm_analysis.png",
        )
        figures["false_alarm_analysis"] = p6

        p7 = plot_detection_delay_tradeoffs(
            results=persist_results,
            output_path=self.figures_dir / "fig7_detection_delay_tradeoffs.png",
        )
        figures["detection_delay_tradeoffs"] = p7

        p8 = plot_confidence_reliability_diagram(
            bins=[b.to_dict() for b in calib_report.reliability_bins],
            ece=calib_report.expected_calibration_error,
            output_path=self.figures_dir / "fig8_confidence_reliability_diagram.png",
        )
        figures["confidence_reliability_diagram"] = p8

        p9 = plot_adaptive_benefit_or_failure_regimes(
            regime_rows=regime_rows,
            output_path=self.figures_dir / "fig9_adaptive_benefit_or_failure_regimes.png",
        )
        figures["adaptive_benefit_or_failure_regimes"] = p9

        p10 = plot_paired_comparison_confidence_intervals(
            ci_rows=paired_ci_df.to_dict(orient="records"),
            output_path=self.figures_dir / "fig10_paired_comparison_confidence_intervals.png",
        )
        figures["paired_comparison_confidence_intervals"] = p10

        return figures


def run_phase6_refinement(
    dataset_id: str = "flexpulse_calibration_v2",
    random_seed: int = 42,
    n_bootstraps: int = 200,
) -> Phase6RefinementSummary:
    runner = Phase6RefinementRunner(
        dataset_id=dataset_id,
        random_seed=random_seed,
        n_bootstraps=n_bootstraps,
    )
    return runner.run()


if __name__ == "__main__":
    run_phase6_refinement()
