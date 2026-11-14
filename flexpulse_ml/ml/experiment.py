"""
Benchmarking experiment orchestrator for FlexPulse-ML Phase 4.
Trains, evaluates, and compares all classification models across partitions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from flexpulse_ml.config import PROJECT_ROOT
from flexpulse_ml.data.manifest import compute_file_sha256
from flexpulse_ml.data.storage import DatasetStorage
from flexpulse_ml.features.dataset import FeatureDatasetStorage
from flexpulse_ml.features.extractor import FEATURE_NAMES
from flexpulse_ml.ml.evaluate import (
    EventLevelMetrics,
    WindowClassificationMetrics,
    bootstrap_run_level_metrics,
    calculate_window_metrics,
    evaluate_event_level_detections,
    evaluate_scenario_breakdown,
    evaluate_window_label_sensitivity,
)
from flexpulse_ml.ml.models import SUPPORTED_MODELS
from flexpulse_ml.ml.persistence import ModelArtifactStorage
from flexpulse_ml.ml.plotting import (
    plot_confusion_matrices,
    plot_detection_delay_distribution,
    plot_id_vs_ood_comparison,
    plot_model_comparison_bars,
    plot_precision_recall_curves,
    plot_roc_curves,
    plot_scenario_error_analysis,
    plot_window_label_sensitivity,
)
from flexpulse_ml.ml.train import ModelTrainer, TrainedModelRecord


@dataclass
class BenchmarkExecutionSummary:
    experiment_id: str
    timestamp_utc: str
    dataset_id: str
    random_seed: int
    system_architecture: str
    python_version: str
    models_evaluated: List[str]
    best_model_name: str
    best_model_validation_f1: float
    feature_count: int
    artifact_paths: Dict[str, str] = field(default_factory=dict)
    checksums_sha256: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Phase4BenchmarkRunner:
    """End-to-end benchmark coordinator for Phase 4."""

    def __init__(
        self,
        dataset_id: str = "flexpulse_research_v1",
        random_seed: int = 42,
        models_to_train: Optional[List[str]] = None,
        n_bootstraps: int = 200,
        output_dir: Optional[Path] = None,
    ) -> None:
        self.dataset_id = dataset_id
        self.random_seed = random_seed
        self.models_to_train = models_to_train or list(SUPPORTED_MODELS)
        self.n_bootstraps = n_bootstraps

        self.results_dir = output_dir or (PROJECT_ROOT / "results" / "experiments" / "phase4_benchmark")
        self.tables_dir = PROJECT_ROOT / "results" / "tables" / "phase4_benchmark"
        self.figures_dir = PROJECT_ROOT / "results" / "figures" / "phase4_benchmark"
        self.models_dir = PROJECT_ROOT / "models" / "phase4"

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.feature_storage = FeatureDatasetStorage(dataset_id)
        self.raw_storage = DatasetStorage(dataset_id)
        self.artifact_storage = ModelArtifactStorage(
            models_base_dir=self.models_dir,
            results_base_dir=self.results_dir,
            tables_base_dir=self.tables_dir,
        )

    def run(self) -> BenchmarkExecutionSummary:
        start_time = datetime.now(timezone.utc).isoformat()
        experiment_id = f"phase4_benchmark_{self.dataset_id}"

        train_df = self.feature_storage.load_partition(DatasetStorage.SPLIT_TRAIN)
        val_df = self.feature_storage.load_partition(DatasetStorage.SPLIT_VAL)
        id_test_df = self.feature_storage.load_partition(DatasetStorage.SPLIT_ID_TEST)
        ood_test_df = self.feature_storage.load_partition(DatasetStorage.SPLIT_OOD_TEST)

        trainer = ModelTrainer(
            random_seed=self.random_seed,
            models_to_train=self.models_to_train,
        )

        trained_records = trainer.train_all(train_df, val_df)

        best_model_name = ""
        best_score = -1.0
        best_bal_acc = -1.0

        for m_name, rec in trained_records.items():
            if (rec.validation_f1 > best_score) or (
                np.isclose(rec.validation_f1, best_score) and rec.validation_balanced_accuracy > best_bal_acc
            ):
                best_score = rec.validation_f1
                best_bal_acc = rec.validation_balanced_accuracy
                best_model_name = m_name

        splits_data = {
            "validation": (val_df, *trainer.prepare_data(val_df)),
            "id_test": (id_test_df, *trainer.prepare_data(id_test_df)),
            "ood_test": (ood_test_df, *trainer.prepare_data(ood_test_df)),
        }

        comparison_rows: List[Dict[str, Any]] = []
        confusion_id_dict: Dict[str, Dict[str, int]] = {}
        pr_curves_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}
        roc_curves_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}
        delays_dict: Dict[str, List[float]] = {}

        scenario_breakdowns: Dict[str, Dict[str, Dict[str, Any]]] = {}
        event_metrics_all: Dict[str, Dict[str, EventLevelMetrics]] = {}
        bootstrap_ci_all: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
        sensitivity_all: Dict[str, Dict[str, Dict[str, Any]]] = {}

        for m_name, rec in trained_records.items():
            model = rec.model_instance

            meta = rec.to_dict()
            meta["random_seed"] = self.random_seed
            meta["trained_timestamp"] = start_time
            self.artifact_storage.save_model(m_name, model, meta)

            if m_name == best_model_name:
                self.artifact_storage.save_model("best_model", model, meta)

            for split_name, (split_df, X_split, y_split) in splits_data.items():
                t_pred_start = time.perf_counter()
                y_pred = model.predict(X_split)
                t_pred = time.perf_counter() - t_pred_start

                y_prob: Optional[np.ndarray] = None
                if hasattr(model, "predict_proba"):
                    try:
                        probs = model.predict_proba(X_split)
                        if probs.ndim == 2 and probs.shape[1] >= 2:
                            y_prob = probs[:, 1]
                    except Exception:
                        y_prob = None

                metrics = calculate_window_metrics(y_split, y_pred, y_prob)

                row = {
                    "model": m_name,
                    "split": split_name,
                    "accuracy": round(metrics.accuracy, 4),
                    "balanced_accuracy": round(metrics.balanced_accuracy, 4),
                    "precision": round(metrics.precision, 4),
                    "recall": round(metrics.recall, 4),
                    "specificity": round(metrics.specificity, 4),
                    "false_negative_rate": round(metrics.false_negative_rate, 4),
                    "f1": round(metrics.f1, 4),
                    "roc_auc": round(metrics.roc_auc, 4) if metrics.roc_auc is not None else np.nan,
                    "average_precision": round(metrics.average_precision, 4) if metrics.average_precision is not None else np.nan,
                    "train_duration_s": round(rec.train_duration_s, 4),
                    "predict_duration_s": round(t_pred, 4),
                    "tn": metrics.confusion_matrix["tn"],
                    "fp": metrics.confusion_matrix["fp"],
                    "fn": metrics.confusion_matrix["fn"],
                    "tp": metrics.confusion_matrix["tp"],
                }
                comparison_rows.append(row)

                if split_name == "id_test":
                    confusion_id_dict[m_name] = metrics.confusion_matrix
                    if y_prob is not None:
                        try:
                            rec_arr, prec_arr, _ = precision_recall_curve(y_split, y_prob)
                            ap = float(average_precision_score(y_split, y_prob))
                            pr_curves_data[m_name] = (rec_arr, prec_arr, ap)

                            fpr_arr, tpr_arr, _ = roc_curve(y_split, y_prob)
                            auc = float(roc_auc_score(y_split, y_prob))
                            roc_curves_data[m_name] = (fpr_arr, tpr_arr, auc)
                        except Exception:
                            pass

                if split_name in ("id_test", "ood_test"):
                    sc_break = evaluate_scenario_breakdown(split_df, y_pred)
                    if split_name not in scenario_breakdowns:
                        scenario_breakdowns[split_name] = {}
                    scenario_breakdowns[split_name][m_name] = sc_break

                    boot_ci = bootstrap_run_level_metrics(
                        split_df,
                        y_pred,
                        n_bootstraps=self.n_bootstraps,
                        random_seed=self.random_seed,
                    )
                    if split_name not in bootstrap_ci_all:
                        bootstrap_ci_all[split_name] = {}
                    bootstrap_ci_all[split_name][m_name] = boot_ci

                    ev_metrics = evaluate_event_level_detections(
                        split_df,
                        y_pred,
                        self.raw_storage,
                        split=split_name,
                    )
                    if split_name not in event_metrics_all:
                        event_metrics_all[split_name] = {}
                    event_metrics_all[split_name][m_name] = ev_metrics

                    if split_name == "id_test":
                        delays_dict[m_name] = ev_metrics.detection_delays

                    if split_name == "id_test":
                        sens = evaluate_window_label_sensitivity(split_df, y_pred)
                        sensitivity_all[m_name] = sens

        comparison_df = pd.DataFrame(comparison_rows)

        tables_created = self._export_tables(
            comparison_df=comparison_df,
            scenario_breakdowns=scenario_breakdowns,
            event_metrics_all=event_metrics_all,
            bootstrap_ci_all=bootstrap_ci_all,
            sensitivity_all=sensitivity_all,
        )

        figures_created = self._export_figures(
            comparison_df=comparison_df,
            confusion_id_dict=confusion_id_dict,
            pr_curves_data=pr_curves_data,
            roc_curves_data=roc_curves_data,
            delays_dict=delays_dict,
            scenario_breakdowns=scenario_breakdowns,
            sensitivity_all=sensitivity_all,
            best_model_name=best_model_name,
        )

        manifest_data = {
            "experiment_id": experiment_id,
            "timestamp_utc": start_time,
            "dataset_id": self.dataset_id,
            "random_seed": self.random_seed,
            "system_architecture": f"{platform.system()}_{platform.machine()}",
            "python_version": platform.python_version(),
            "models_evaluated": self.models_to_train,
            "best_model_name": best_model_name,
            "best_model_validation_f1": float(best_score),
            "best_model_hyperparameters": trained_records[best_model_name].best_hyperparameters,
            "feature_allowlist": list(FEATURE_NAMES),
            "feature_count": len(FEATURE_NAMES),
            "synthetic_data": True,
            "models_metadata": {m: rec.to_dict() for m, rec in trained_records.items()},
            "evaluation_metrics": comparison_rows,
            "tables": {k: str(v) for k, v in tables_created.items()},
            "figures": {k: str(v) for k, v in figures_created.items()},
        }

        manifest_path = self.results_dir / "phase4_benchmark_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        all_artifacts = dict(tables_created)
        all_artifacts.update(figures_created)
        all_artifacts["manifest"] = manifest_path

        checksums = {k: compute_file_sha256(Path(p)) for k, p in all_artifacts.items() if Path(p).exists()}
        checksums_path = self.results_dir / "phase4_benchmark_checksums.json"
        with open(checksums_path, "w", encoding="utf-8") as f:
            json.dump(checksums, f, indent=2)

        return BenchmarkExecutionSummary(
            experiment_id=experiment_id,
            timestamp_utc=start_time,
            dataset_id=self.dataset_id,
            random_seed=self.random_seed,
            system_architecture=f"{platform.system()}_{platform.machine()}",
            python_version=platform.python_version(),
            models_evaluated=self.models_to_train,
            best_model_name=best_model_name,
            best_model_validation_f1=float(best_score),
            feature_count=len(FEATURE_NAMES),
            artifact_paths={k: str(v) for k, v in all_artifacts.items()},
            checksums_sha256=checksums,
        )

    def _write_markdown(self, df: pd.DataFrame, path: Path) -> None:
        try:
            df.to_markdown(path, index=False)
        except Exception:
            path.write_text(df.to_string(index=False), encoding="utf-8")

    def _export_tables(
        self,
        comparison_df: pd.DataFrame,
        scenario_breakdowns: Dict[str, Dict[str, Dict[str, Any]]],
        event_metrics_all: Dict[str, Dict[str, EventLevelMetrics]],
        bootstrap_ci_all: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
        sensitivity_all: Dict[str, Dict[str, Dict[str, Any]]],
    ) -> Dict[str, Path]:
        tables: Dict[str, Path] = {}

        main_csv = self.tables_dir / "phase4_benchmark_model_comparison.csv"
        main_md = self.tables_dir / "phase4_benchmark_model_comparison.md"
        comparison_df.to_csv(main_csv, index=False)
        self._write_markdown(comparison_df, main_md)
        tables["comparison_csv"] = main_csv
        tables["comparison_md"] = main_md

        for sp in ["validation", "id_test", "ood_test"]:
            sub_df = comparison_df[comparison_df["split"] == sp].reset_index(drop=True)
            sp_csv = self.tables_dir / f"model_comparison_{sp}.csv"
            sp_md = self.tables_dir / f"model_comparison_{sp}.md"
            sub_df.to_csv(sp_csv, index=False)
            self._write_markdown(sub_df, sp_md)
            tables[f"{sp}_csv"] = sp_csv
            tables[f"{sp}_md"] = sp_md

        for sp in ["id_test", "ood_test"]:
            if sp in scenario_breakdowns:
                sc_rows = []
                for mod_name, sc_dict in scenario_breakdowns[sp].items():
                    for sc_name, m_data in sc_dict.items():
                        sc_rows.append({
                            "model": mod_name,
                            "scenario": sc_name,
                            "accuracy": round(m_data["accuracy"], 4),
                            "balanced_accuracy": round(m_data["balanced_accuracy"], 4),
                            "precision": round(m_data["precision"], 4),
                            "recall": round(m_data["recall"], 4),
                            "specificity": round(m_data["specificity"], 4),
                            "f1": round(m_data["f1"], 4),
                            "false_positive_rate": round(m_data["false_positive_rate"], 4),
                            "n_samples": m_data["n_samples"],
                            "n_fault": m_data["n_fault"],
                        })
                sc_df = pd.DataFrame(sc_rows)
                sc_csv = self.tables_dir / f"scenario_breakdown_{sp}.csv"
                sc_md = self.tables_dir / f"scenario_breakdown_{sp}.md"
                sc_df.to_csv(sc_csv, index=False)
                self._write_markdown(sc_df, sc_md)
                tables[f"scenario_{sp}_csv"] = sc_csv
                tables[f"scenario_{sp}_md"] = sc_md

        for sp in ["id_test", "ood_test"]:
            if sp in event_metrics_all:
                ev_rows = []
                for mod_name, ev_obj in event_metrics_all[sp].items():
                    ev_rows.append({
                        "model": mod_name,
                        "total_injected_events": ev_obj.total_injected_events,
                        "observable_injected_events": ev_obj.observable_injected_events,
                        "detected_observable_events": ev_obj.detected_observable_events,
                        "missed_observable_events": ev_obj.missed_observable_events,
                        "event_recall": round(ev_obj.event_recall, 4),
                        "total_predicted_events": ev_obj.total_predicted_events,
                        "false_alarm_events": ev_obj.false_alarm_events,
                        "false_alarms_per_minute": round(ev_obj.false_alarms_per_minute, 4),
                        "mean_detection_delay_s": round(ev_obj.mean_detection_delay_s, 4) if ev_obj.mean_detection_delay_s is not None else np.nan,
                        "median_detection_delay_s": round(ev_obj.median_detection_delay_s, 4) if ev_obj.median_detection_delay_s is not None else np.nan,
                    })
                ev_df = pd.DataFrame(ev_rows)
                ev_csv = self.tables_dir / f"event_level_detections_{sp}.csv"
                ev_md = self.tables_dir / f"event_level_detections_{sp}.md"
                ev_df.to_csv(ev_csv, index=False)
                self._write_markdown(ev_df, ev_md)
                tables[f"event_{sp}_csv"] = ev_csv
                tables[f"event_{sp}_md"] = ev_md

        for sp in ["id_test", "ood_test"]:
            if sp in bootstrap_ci_all:
                boot_rows = []
                for mod_name, b_dict in bootstrap_ci_all[sp].items():
                    row = {"model": mod_name}
                    for metric_name, ci_vals in b_dict.items():
                        row[f"{metric_name}_mean"] = round(ci_vals.get("mean", 0.0), 4)
                        row[f"{metric_name}_ci_lower"] = round(ci_vals.get("ci_lower", 0.0), 4)
                        row[f"{metric_name}_ci_upper"] = round(ci_vals.get("ci_upper", 0.0), 4)
                    boot_rows.append(row)
                boot_df = pd.DataFrame(boot_rows)
                boot_csv = self.tables_dir / f"bootstrap_ci_{sp}.csv"
                boot_md = self.tables_dir / f"bootstrap_ci_{sp}.md"
                boot_df.to_csv(boot_csv, index=False)
                self._write_markdown(boot_df, boot_md)
                tables[f"boot_{sp}_csv"] = boot_csv
                tables[f"boot_{sp}_md"] = boot_md

        sens_rows = []
        for mod_name, s_dict in sensitivity_all.items():
            for th_key, s_data in s_dict.items():
                sens_rows.append({
                    "model": mod_name,
                    "target_threshold": th_key,
                    "accuracy": round(s_data["accuracy"], 4),
                    "balanced_accuracy": round(s_data["balanced_accuracy"], 4),
                    "precision": round(s_data["precision"], 4),
                    "recall": round(s_data["recall"], 4),
                    "specificity": round(s_data["specificity"], 4),
                    "f1": round(s_data["f1"], 4),
                })
        sens_df = pd.DataFrame(sens_rows)
        sens_csv = self.tables_dir / "label_sensitivity_id_test.csv"
        sens_md = self.tables_dir / "label_sensitivity_id_test.md"
        sens_df.to_csv(sens_csv, index=False)
        self._write_markdown(sens_df, sens_md)
        tables["sensitivity_csv"] = sens_csv
        tables["sensitivity_md"] = sens_md

        return tables

    def _export_figures(
        self,
        comparison_df: pd.DataFrame,
        confusion_id_dict: Dict[str, Dict[str, int]],
        pr_curves_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
        roc_curves_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
        delays_dict: Dict[str, List[float]],
        scenario_breakdowns: Dict[str, Dict[str, Dict[str, Any]]],
        sensitivity_all: Dict[str, Dict[str, Dict[str, Any]]],
        best_model_name: str,
    ) -> Dict[str, Path]:
        figures: Dict[str, Path] = {}

        p1 = plot_model_comparison_bars(
            comparison_df,
            output_path=self.figures_dir / "fig1_model_comparison_bars.png",
        )
        figures["model_comparison_bars"] = p1

        p2 = plot_confusion_matrices(
            confusion_id_dict,
            output_path=self.figures_dir / "fig2_confusion_matrices.png",
        )
        figures["confusion_matrices"] = p2

        p3 = plot_precision_recall_curves(
            pr_curves_data,
            output_path=self.figures_dir / "fig3_precision_recall_curves.png",
        )
        figures["precision_recall_curves"] = p3

        p4 = plot_roc_curves(
            roc_curves_data,
            output_path=self.figures_dir / "fig4_roc_curves.png",
        )
        figures["roc_curves"] = p4

        p5 = plot_id_vs_ood_comparison(
            comparison_df,
            output_path=self.figures_dir / "fig5_id_vs_ood_comparison.png",
        )
        figures["id_vs_ood_comparison"] = p5

        if "id_test" in scenario_breakdowns and best_model_name in scenario_breakdowns["id_test"]:
            best_sc_metrics = scenario_breakdowns["id_test"][best_model_name]
            p6 = plot_scenario_error_analysis(
                best_sc_metrics,
                output_path=self.figures_dir / "fig6_scenario_error_analysis.png",
            )
            figures["scenario_error_analysis"] = p6

        if best_model_name in sensitivity_all:
            best_sens = sensitivity_all[best_model_name]
            p7 = plot_window_label_sensitivity(
                best_sens,
                output_path=self.figures_dir / "fig7_window_label_sensitivity.png",
            )
            figures["window_label_sensitivity"] = p7

        p8 = plot_detection_delay_distribution(
            delays_dict,
            output_path=self.figures_dir / "fig8_detection_delay_distribution.png",
        )
        figures["detection_delay_distribution"] = p8

        return figures


def run_phase4_benchmark(
    dataset_id: str = "flexpulse_research_v1",
    random_seed: int = 42,
    models_to_train: Optional[List[str]] = None,
    n_bootstraps: int = 200,
) -> BenchmarkExecutionSummary:
    runner = Phase4BenchmarkRunner(
        dataset_id=dataset_id,
        random_seed=random_seed,
        models_to_train=models_to_train,
        n_bootstraps=n_bootstraps,
    )
    return runner.run()


if __name__ == "__main__":
    run_phase4_benchmark()
