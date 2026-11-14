"""
Phase 5 benchmark experiment orchestrator: evaluates A0-A3, E1-E10 regimes, and ablations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from flexpulse_ml.calibration.adaptive import (
    AdaptiveCalibrationConfig,
    ConfidenceGatedAdaptiveCalibrator,
)
from flexpulse_ml.calibration.baseline import (
    BaselineStatistics,
    InitialBaselineEstimator,
)
from flexpulse_ml.calibration.diagnostics import CalibrationDiagnostics
from flexpulse_ml.calibration.plotting import (
    plot_ablation_study,
    plot_baseline_trajectories,
    plot_causal_detection_delays,
    plot_contamination_failure_modes,
    plot_false_alarm_vs_recall_tradeoff,
    plot_ground_truth_overlay,
    plot_method_comparison_id_ood,
    plot_model_predictions_timeline,
    plot_raw_telemetry_timeseries,
    plot_update_acceptance_rejection,
)
from flexpulse_ml.config import PROJECT_ROOT
from flexpulse_ml.data.manifest import compute_file_sha256
from flexpulse_ml.data.storage import DatasetStorage
from flexpulse_ml.features.dataset import FeatureDatasetStorage
from flexpulse_ml.features.extractor import FEATURE_NAMES
from flexpulse_ml.ml.baselines import FixedThresholdBaseline


@dataclass
class Phase5BenchmarkSummary:
    experiment_id: str
    timestamp_utc: str
    dataset_id: str
    methods_evaluated: List[str]
    artifact_paths: Dict[str, str] = field(default_factory=dict)
    checksums_sha256: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Phase5BenchmarkRunner:
    """Coordinates execution of Phase 5 comparative benchmarks, ablations, and figures."""

    METHOD_A0 = "A0_Fixed_Threshold"
    METHOD_A1 = "A1_Static_ML"
    METHOD_A2 = "A2_Initial_Calibration_Only"
    METHOD_A3 = "A3_Confidence_Gated_Adaptive"

    ALL_METHODS = [METHOD_A0, METHOD_A1, METHOD_A2, METHOD_A3]

    def __init__(
        self,
        dataset_id: str = "flexpulse_calibration_v2",
        random_seed: int = 42,
        n_bootstraps: int = 200,
    ) -> None:
        self.dataset_id = dataset_id
        self.random_seed = random_seed
        self.n_bootstraps = n_bootstraps

        self.tables_dir = PROJECT_ROOT / "results" / "tables" / "phase5_calibration"
        self.figures_dir = PROJECT_ROOT / "results" / "figures" / "phase5_calibration"
        self.models_dir = PROJECT_ROOT / "models" / "phase5"
        self.results_dir = PROJECT_ROOT / "results" / "experiments" / "phase5_calibration"

        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.feature_storage = FeatureDatasetStorage(dataset_id)
        self.raw_storage = DatasetStorage(dataset_id)
        self.baseline_estimator = InitialBaselineEstimator(initial_duration_s=5.0)

    def run(self) -> Phase5BenchmarkSummary:
        start_time = datetime.now(timezone.utc).isoformat()
        experiment_id = f"phase5_calibration_{self.dataset_id}"

        train_df = self.feature_storage.load_partition(DatasetStorage.SPLIT_TRAIN)
        dev_df = self.feature_storage.load_partition(DatasetStorage.SPLIT_VAL)
        id_test_df = self.feature_storage.load_partition(DatasetStorage.SPLIT_ID_TEST)
        ood_test_df = self.feature_storage.load_partition(DatasetStorage.SPLIT_OOD_TEST)

        trained_models = self._train_models(train_df)

        comparison_rows: List[Dict[str, Any]] = []
        event_rows: List[Dict[str, Any]] = []
        delays_dict: Dict[str, List[float]] = {}
        metrics_tradeoff_dict: Dict[str, Dict[str, float]] = {}

        splits = {
            "validation_dev": dev_df,
            "id_test": id_test_df,
            "ood_test": ood_test_df,
        }

        eval_records_by_split: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, Optional[CalibrationDiagnostics]]]] = {}

        for sp_name, sp_df in splits.items():
            eval_records_by_split[sp_name] = {}
            for m_name in self.ALL_METHODS:
                preds, probs, diags = self._predict_method(m_name, sp_df, trained_models)
                eval_records_by_split[sp_name][m_name] = (preds, probs, diags)

                y_true = sp_df["fault_label"].to_numpy(dtype=np.int32)
                row_comp = self._calc_window_metrics(m_name, sp_name, y_true, preds, probs)
                comparison_rows.append(row_comp)

                if sp_name in ("id_test", "ood_test"):
                    ev_row, d_list = self._calc_event_metrics(m_name, sp_name, sp_df, preds)
                    event_rows.append(ev_row)
                    if sp_name == "ood_test":
                        delays_dict[m_name] = d_list
                        metrics_tradeoff_dict[m_name] = {
                            "event_recall": ev_row["event_recall"],
                            "false_alarms_per_minute": ev_row["false_alarms_per_minute"],
                        }

        ablation_df = self._run_ablation_study(ood_test_df, trained_models)
        alpha_df = self._run_alpha_sensitivity_study(ood_test_df, trained_models["rf_norm"])
        contam_df = self._run_contamination_study(ood_test_df, trained_models["rf_norm"])
        bootstrap_df = self._run_bootstrap_ci(ood_test_df, eval_records_by_split["ood_test"])
        scenario_df = self._run_scenario_breakdown(ood_test_df, eval_records_by_split["ood_test"])

        tables_created = self._export_tables(
            comparison_rows=comparison_rows,
            event_rows=event_rows,
            ablation_df=ablation_df,
            alpha_df=alpha_df,
            contam_df=contam_df,
            bootstrap_df=bootstrap_df,
            scenario_df=scenario_df,
        )

        figures_created = self._export_figures(
            id_test_df=id_test_df,
            ood_test_df=ood_test_df,
            comparison_rows=comparison_rows,
            metrics_tradeoff_dict=metrics_tradeoff_dict,
            delays_dict=delays_dict,
            ablation_df=ablation_df,
            trained_models=trained_models,
        )

        manifest_data = {
            "experiment_id": experiment_id,
            "timestamp_utc": start_time,
            "dataset_id": self.dataset_id,
            "random_seed": self.random_seed,
            "system_architecture": f"{platform.system()}_{platform.machine()}",
            "python_version": platform.python_version(),
            "methods_evaluated": self.ALL_METHODS,
            "calibration_config": AdaptiveCalibrationConfig().to_dict(),
            "tables": {k: str(v) for k, v in tables_created.items()},
            "figures": {k: str(v) for k, v in figures_created.items()},
        }

        manifest_path = self.results_dir / "phase5_calibration_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        all_artifacts = dict(tables_created)
        all_artifacts.update(figures_created)
        all_artifacts["manifest"] = manifest_path

        checksums = {k: compute_file_sha256(Path(p)) for k, p in all_artifacts.items() if Path(p).exists()}
        checksums_path = self.results_dir / "phase5_calibration_checksums.json"
        with open(checksums_path, "w", encoding="utf-8") as f:
            json.dump(checksums, f, indent=2)

        return Phase5BenchmarkSummary(
            experiment_id=experiment_id,
            timestamp_utc=start_time,
            dataset_id=self.dataset_id,
            methods_evaluated=self.ALL_METHODS,
            artifact_paths={k: str(v) for k, v in all_artifacts.items()},
            checksums_sha256=checksums,
        )

    def _train_models(self, train_df: pd.DataFrame) -> Dict[str, Any]:
        X_train_raw = train_df[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
        y_train = train_df["fault_label"].to_numpy(dtype=np.int32)

        baseline_threshold = FixedThresholdBaseline(
            voltage_drop_std_multiplier=2.5,
            transient_jump_std_multiplier=2.5,
        )
        baseline_threshold.fit(X_train_raw, y_train)

        rf_raw = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced",
            random_state=self.random_seed,
            n_jobs=-1,
        )
        rf_raw.fit(X_train_raw, y_train)

        norm_train_runs: List[pd.DataFrame] = []
        for r_id in train_df["run_id"].unique():
            r_df = train_df[train_df["run_id"] == r_id]
            base = self.baseline_estimator.fit_run(r_df)
            norm_r_df = self.baseline_estimator.normalize_features(r_df, base)
            norm_train_runs.append(norm_r_df)
        train_norm_df = pd.concat(norm_train_runs, ignore_index=True)

        X_train_norm = train_norm_df[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)

        rf_norm = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced",
            random_state=self.random_seed,
            n_jobs=-1,
        )
        rf_norm.fit(X_train_norm, y_train)

        joblib.dump(baseline_threshold, self.models_dir / "a0_fixed_threshold.joblib")
        joblib.dump(rf_raw, self.models_dir / "a1_static_random_forest.joblib")
        joblib.dump(rf_norm, self.models_dir / "a2_a3_normalized_random_forest.joblib")

        return {
            "fixed_threshold": baseline_threshold,
            "rf_raw": rf_raw,
            "rf_norm": rf_norm,
        }

    def _predict_method(
        self,
        method_name: str,
        df: pd.DataFrame,
        models: Dict[str, Any],
        calibrator_config: Optional[AdaptiveCalibrationConfig] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[CalibrationDiagnostics]]:
        if method_name == self.METHOD_A0:
            model = models["fixed_threshold"]
            X = df[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
            preds = model.predict(X)
            probs = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else np.zeros(len(df))
            return preds, probs, None

        elif method_name == self.METHOD_A1:
            model = models["rf_raw"]
            X = df[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
            preds = model.predict(X)
            probs = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else np.zeros(len(df))
            return preds, probs, None

        elif method_name == self.METHOD_A2:
            model = models["rf_norm"]
            norm_runs: List[pd.DataFrame] = []
            for r_id in df["run_id"].unique():
                r_df = df[df["run_id"] == r_id]
                base = self.baseline_estimator.fit_run(r_df)
                norm_runs.append(self.baseline_estimator.normalize_features(r_df, base))
            norm_df = pd.concat(norm_runs, ignore_index=True)
            X = norm_df[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
            preds = model.predict(X)
            probs = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else np.zeros(len(df))
            return preds, probs, None

        elif method_name == self.METHOD_A3:
            model = models["rf_norm"]
            cfg = calibrator_config or AdaptiveCalibrationConfig()
            calibrator = ConfidenceGatedAdaptiveCalibrator(cfg)

            all_preds: List[int] = []
            all_probs: List[float] = []
            combined_diag = CalibrationDiagnostics()

            for r_id in df["run_id"].unique():
                r_df = df[df["run_id"] == r_id].reset_index(drop=True)
                p_arr, prob_arr, diag = calibrator.process_run(r_df, model)
                all_preds.extend(p_arr)
                all_probs.extend(prob_arr)
                combined_diag.records.extend(diag.records)

            return np.array(all_preds, dtype=np.int32), np.array(all_probs, dtype=np.float64), combined_diag

        else:
            raise ValueError(f"Unknown method '{method_name}'")

    def _calc_window_metrics(
        self,
        method: str,
        split: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        acc = float(accuracy_score(y_true, y_pred))
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

        roc_auc = np.nan
        pr_auc = np.nan
        if y_prob is not None and len(np.unique(y_true)) > 1:
            try:
                roc_auc = float(roc_auc_score(y_true, y_prob))
                pr_auc = float(average_precision_score(y_true, y_prob))
            except Exception:
                pass

        return {
            "method": method,
            "split": split,
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "specificity": round(spec, 4),
            "false_negative_rate": round(fnr, 4),
            "f1": round(f1, 4),
            "roc_auc": round(roc_auc, 4) if not np.isnan(roc_auc) else np.nan,
            "average_precision": round(pr_auc, 4) if not np.isnan(pr_auc) else np.nan,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        }

    def _calc_event_metrics(
        self,
        method: str,
        split: str,
        df: pd.DataFrame,
        y_pred: np.ndarray,
    ) -> Tuple[Dict[str, Any], List[float]]:
        runs = df["run_id"].unique()
        total_injected = 0
        observable_injected = 0
        detected_observable = 0
        total_predicted = 0
        false_alarm_events = 0
        delays_causal: List[float] = []
        total_healthy_duration_s = 0.0

        for r_id in runs:
            try:
                events = self.raw_storage.load_events(r_id, split)
            except Exception:
                events = []

            r_mask = (df["run_id"] == r_id).to_numpy()
            r_df = df[r_mask].reset_index(drop=True)
            r_preds = y_pred[r_mask]

            injected_intervals = []
            for ev in events:
                total_injected += 1
                st = float(ev.get("start_time_s", ev.get("t_start_s", 0.0)))
                en = float(ev.get("end_time_s", ev.get("t_end_s", 0.0)))
                if bool(ev.get("sampled_event", True)):
                    observable_injected += 1
                    injected_intervals.append({"st": st, "en": en, "detected": False})

            pred_intervals: List[Tuple[float, float]] = []
            curr_st = None
            curr_en = None
            for idx, p_val in enumerate(r_preds):
                if p_val == 1:
                    w_st = float(r_df.loc[idx, "window_start_s"])
                    w_en = float(r_df.loc[idx, "window_end_s"])
                    if curr_st is None:
                        curr_st = w_st
                        curr_en = w_en
                    elif w_st <= (curr_en + 0.5):
                        curr_en = max(curr_en, w_en)
                    else:
                        pred_intervals.append((curr_st, curr_en))
                        curr_st = w_st
                        curr_en = w_en
            if curr_st is not None and curr_en is not None:
                pred_intervals.append((curr_st, curr_en))

            total_predicted += len(pred_intervals)

            for p_st, p_en in pred_intervals:
                matched = False
                for inj in injected_intervals:
                    overlap = max(0.0, min(p_en, inj["en"]) - max(p_st, inj["st"]))
                    if overlap > 0 or abs(p_st - inj["st"]) <= 0.5:
                        matched = True
                        if not inj["detected"]:
                            inj["detected"] = True
                            first_det_en = min(
                                [
                                    float(r_df.loc[i, "window_end_s"])
                                    for i, pv in enumerate(r_preds)
                                    if pv == 1
                                    and float(r_df.loc[i, "window_start_s"]) >= p_st
                                    and float(r_df.loc[i, "window_end_s"]) <= p_en
                                    and max(0.0, min(float(r_df.loc[i, "window_end_s"]), inj["en"]) - max(float(r_df.loc[i, "window_start_s"]), inj["st"])) > 0
                                ]
                                or [p_en]
                            )
                            delays_causal.append(max(0.0, first_det_en - inj["st"]))
                if not matched:
                    false_alarm_events += 1

            for inj in injected_intervals:
                if inj["detected"]:
                    detected_observable += 1

            total_healthy_duration_s += max(0.0, 60.0 - (len(injected_intervals) * 5.0))

        ev_recall = float(detected_observable / observable_injected) if observable_injected > 0 else 1.0
        fa_mins = total_healthy_duration_s / 60.0
        fa_rate = float(false_alarm_events / fa_mins) if fa_mins > 0 else 0.0

        ev_dict = {
            "method": method,
            "split": split,
            "total_injected_events": total_injected,
            "observable_injected_events": observable_injected,
            "detected_observable_events": detected_observable,
            "missed_observable_events": observable_injected - detected_observable,
            "event_recall": round(ev_recall, 4),
            "total_predicted_events": total_predicted,
            "false_alarm_events": false_alarm_events,
            "false_alarms_per_minute": round(fa_rate, 4),
            "mean_causal_detection_delay_s": round(float(np.mean(delays_causal)), 4) if delays_causal else np.nan,
            "median_causal_detection_delay_s": round(float(np.median(delays_causal)), 4) if delays_causal else np.nan,
        }
        return ev_dict, delays_causal

    def _run_ablation_study(self, ood_df: pd.DataFrame, models: Dict[str, Any]) -> pd.DataFrame:
        configs = [
            ("1_Static_ML", False, False, False, 0.0),
            ("2_Initial_Calibration_Only", False, False, False, 0.0),
            ("3_Adaptive_No_Confidence_Gate", False, True, True, 0.04),
            ("4_Adaptive_No_Stability_Gate", True, False, True, 0.04),
            ("5_Adaptive_No_Freeze_Logic", True, True, False, 0.04),
            ("6_Complete_Gated_Adaptive", True, True, True, 0.04),
        ]

        rows: List[Dict[str, Any]] = []
        y_true = ood_df["fault_label"].to_numpy(dtype=np.int32)

        for name, use_conf, use_stab, use_freeze, alpha_val in configs:
            if name == "1_Static_ML":
                preds, probs, _ = self._predict_method(self.METHOD_A1, ood_df, models)
            elif name == "2_Initial_Calibration_Only":
                preds, probs, _ = self._predict_method(self.METHOD_A2, ood_df, models)
            else:
                cfg = AdaptiveCalibrationConfig(
                    alpha=alpha_val,
                    enable_confidence_gate=use_conf,
                    enable_stability_gate=use_stab,
                    enable_freeze_logic=use_freeze,
                )
                preds, probs, diag = self._predict_method(self.METHOD_A3, ood_df, models, calibrator_config=cfg)

            acc = float(accuracy_score(y_true, preds))
            bal_acc = float(balanced_accuracy_score(y_true, preds))
            prec = float(precision_score(y_true, preds, zero_division=0))
            rec = float(recall_score(y_true, preds, zero_division=0))
            f1 = float(f1_score(y_true, preds, zero_division=0))

            rows.append({
                "configuration": name,
                "accuracy": round(acc, 4),
                "balanced_accuracy": round(bal_acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
            })

        return pd.DataFrame(rows)

    def _run_alpha_sensitivity_study(self, ood_df: pd.DataFrame, rf_norm_model: Any) -> pd.DataFrame:
        alphas = [0.01, 0.02, 0.04, 0.08, 0.12, 0.20]
        rows: List[Dict[str, Any]] = []
        y_true = ood_df["fault_label"].to_numpy(dtype=np.int32)

        for a in alphas:
            cfg = AdaptiveCalibrationConfig(alpha=a)
            preds, probs, diag = self._predict_method(
                self.METHOD_A3,
                ood_df,
                {"rf_norm": rf_norm_model},
                calibrator_config=cfg,
            )
            f1 = float(f1_score(y_true, preds, zero_division=0))
            bal_acc = float(balanced_accuracy_score(y_true, preds))
            diag_metrics = diag.compute_summary_metrics() if diag else {}

            rows.append({
                "alpha": a,
                "f1": round(f1, 4),
                "balanced_accuracy": round(bal_acc, 4),
                "accepted_updates": diag_metrics.get("accepted_updates", 0),
                "inappropriate_updates": diag_metrics.get("inappropriate_updates_during_fault", 0),
                "contamination_rate": round(float(diag_metrics.get("contamination_rate", 0.0)), 4),
            })

        return pd.DataFrame(rows)

    def _run_contamination_study(self, ood_df: pd.DataFrame, rf_norm_model: Any) -> pd.DataFrame:
        drift_mask = ood_df["scenario"].str.upper() == "GRADUAL_RESISTANCE_DRIFT"
        sub_df = ood_df[drift_mask].reset_index(drop=True)
        if len(sub_df) == 0:
            sub_df = ood_df

        y_true = sub_df["fault_label"].to_numpy(dtype=np.int32)

        test_modes = [
            ("Protected_Confidence_Gated", True, True, True),
            ("Unprotected_No_Gates", False, False, False),
            ("No_Freeze_Logic", True, True, False),
        ]

        rows: List[Dict[str, Any]] = []
        for mode_name, conf_g, stab_g, freeze_g in test_modes:
            cfg = AdaptiveCalibrationConfig(
                alpha=0.06,
                enable_confidence_gate=conf_g,
                enable_stability_gate=stab_g,
                enable_freeze_logic=freeze_g,
            )
            preds, probs, diag = self._predict_method(
                self.METHOD_A3,
                sub_df,
                {"rf_norm": rf_norm_model},
                calibrator_config=cfg,
            )
            rec = float(recall_score(y_true, preds, zero_division=0))
            diag_metrics = diag.compute_summary_metrics() if diag else {}

            rows.append({
                "mode": mode_name,
                "fault_recall": round(rec, 4),
                "inappropriate_updates": diag_metrics.get("inappropriate_updates_during_fault", 0),
                "contamination_rate": round(float(diag_metrics.get("contamination_rate", 0.0)), 4),
                "time_spent_frozen_s": round(float(diag_metrics.get("time_spent_frozen_s", 0.0)), 2),
            })

        return pd.DataFrame(rows)

    def _run_bootstrap_ci(
        self,
        df: pd.DataFrame,
        eval_dict: Dict[str, Tuple[np.ndarray, np.ndarray, Optional[CalibrationDiagnostics]]],
    ) -> pd.DataFrame:
        runs = df["run_id"].unique()
        n_runs = len(runs)
        rng = np.random.default_rng(self.random_seed)
        y_true_all = df["fault_label"].to_numpy(dtype=np.int32)
        run_indices = {r: np.where(df["run_id"].to_numpy() == r)[0] for r in runs}

        rows: List[Dict[str, Any]] = []

        for m_name in self.ALL_METHODS:
            preds, _, _ = eval_dict[m_name]
            f1_samples: List[float] = []
            bal_samples: List[float] = []

            for _ in range(self.n_bootstraps):
                sampled_runs = rng.choice(runs, size=n_runs, replace=True)
                sample_idx = []
                for sr in sampled_runs:
                    sample_idx.extend(run_indices[sr])
                idx_arr = np.array(sample_idx)
                yt = y_true_all[idx_arr]
                yp = preds[idx_arr]

                if len(np.unique(yt)) > 1:
                    f1_samples.append(float(f1_score(yt, yp, zero_division=0)))
                    bal_samples.append(float(balanced_accuracy_score(yt, yp)))

            rows.append({
                "method": m_name,
                "f1_mean": round(float(np.mean(f1_samples)), 4) if f1_samples else np.nan,
                "f1_ci_lower": round(float(np.percentile(f1_samples, 2.5)), 4) if f1_samples else np.nan,
                "f1_ci_upper": round(float(np.percentile(f1_samples, 97.5)), 4) if f1_samples else np.nan,
                "balanced_acc_mean": round(float(np.mean(bal_samples)), 4) if bal_samples else np.nan,
                "balanced_acc_ci_lower": round(float(np.percentile(bal_samples, 2.5)), 4) if bal_samples else np.nan,
                "balanced_acc_ci_upper": round(float(np.percentile(bal_samples, 97.5)), 4) if bal_samples else np.nan,
            })

        return pd.DataFrame(rows)

    def _run_scenario_breakdown(
        self,
        df: pd.DataFrame,
        eval_dict: Dict[str, Tuple[np.ndarray, np.ndarray, Optional[CalibrationDiagnostics]]],
    ) -> pd.DataFrame:
        scenarios = sorted(df["scenario"].unique())
        rows: List[Dict[str, Any]] = []

        for sc in scenarios:
            mask = df["scenario"].to_numpy() == sc
            y_sc = df["fault_label"].to_numpy(dtype=np.int32)[mask]

            for m_name in self.ALL_METHODS:
                preds_all, _, _ = eval_dict[m_name]
                p_sc = preds_all[mask]

                acc = float(accuracy_score(y_sc, p_sc))
                bal_acc = float(balanced_accuracy_score(y_sc, p_sc))
                rec = float(recall_score(y_sc, p_sc, zero_division=0))
                prec = float(precision_score(y_sc, p_sc, zero_division=0))
                f1 = float(f1_score(y_sc, p_sc, zero_division=0))

                cm = confusion_matrix(y_sc, p_sc, labels=[0, 1])
                tn, fp = int(cm[0, 0]), int(cm[0, 1])
                fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

                rows.append({
                    "scenario": sc,
                    "method": m_name,
                    "accuracy": round(acc, 4),
                    "balanced_accuracy": round(bal_acc, 4),
                    "precision": round(prec, 4),
                    "recall": round(rec, 4),
                    "f1": round(f1, 4),
                    "false_positive_rate": round(fpr, 4),
                    "n_samples": int(len(y_sc)),
                    "n_fault": int(np.sum(y_sc == 1)),
                })

        return pd.DataFrame(rows)

    def _export_tables(
        self,
        comparison_rows: List[Dict[str, Any]],
        event_rows: List[Dict[str, Any]],
        ablation_df: pd.DataFrame,
        alpha_df: pd.DataFrame,
        contam_df: pd.DataFrame,
        bootstrap_df: pd.DataFrame,
        scenario_df: pd.DataFrame,
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

        comp_df = pd.DataFrame(comparison_rows)
        write_tb(comp_df, "phase5_method_comparison_all")

        id_comp = comp_df[comp_df["split"] == "id_test"].reset_index(drop=True)
        ood_comp = comp_df[comp_df["split"] == "ood_test"].reset_index(drop=True)
        write_tb(id_comp, "method_comparison_id_test")
        write_tb(ood_comp, "method_comparison_ood_test")

        ev_df = pd.DataFrame(event_rows)
        id_ev = ev_df[ev_df["split"] == "id_test"].reset_index(drop=True)
        ood_ev = ev_df[ev_df["split"] == "ood_test"].reset_index(drop=True)
        write_tb(id_ev, "event_level_detections_id_test")
        write_tb(ood_ev, "event_level_detections_ood_test")

        write_tb(ablation_df, "ablation_study")
        write_tb(alpha_df, "alpha_sensitivity")
        write_tb(contam_df, "contamination_stress_test")
        write_tb(bootstrap_df, "bootstrap_ci_ood_test")
        write_tb(scenario_df, "scenario_breakdown_ood_test")

        return tables

    def _export_figures(
        self,
        id_test_df: pd.DataFrame,
        ood_test_df: pd.DataFrame,
        comparison_rows: List[Dict[str, Any]],
        metrics_tradeoff_dict: Dict[str, Dict[str, float]],
        delays_dict: Dict[str, List[float]],
        ablation_df: pd.DataFrame,
        trained_models: Dict[str, Any],
    ) -> Dict[str, Path]:
        figures: Dict[str, Path] = {}

        nom_run_id = id_test_df["run_id"].unique()[0]
        shift_run_id = ood_test_df["run_id"].unique()[0]
        nom_run_data = self.raw_storage.load_run("id_test", nom_run_id)
        shift_run_data = self.raw_storage.load_run("ood_test", shift_run_id)

        p1 = plot_raw_telemetry_timeseries(
            nominal_df=nom_run_data["observations"],
            shifted_df=shift_run_data["observations"],
            output_path=self.figures_dir / "fig1_raw_telemetry_timeseries.png",
        )
        figures["raw_telemetry_timeseries"] = p1

        single_run_df = ood_test_df[ood_test_df["run_id"] == shift_run_id].reset_index(drop=True)
        calibrator = ConfidenceGatedAdaptiveCalibrator()
        _, _, diag = calibrator.process_run(single_run_df, trained_models["rf_norm"])
        records_dicts = [r.to_dict() for r in diag.records]

        p2 = plot_baseline_trajectories(
            records_dicts,
            output_path=self.figures_dir / "fig2_baseline_trajectories.png",
        )
        figures["baseline_trajectories"] = p2

        p3 = plot_update_acceptance_rejection(
            records_dicts,
            output_path=self.figures_dir / "fig3_update_acceptance_rejection.png",
        )
        figures["update_acceptance_rejection"] = p3

        p4 = plot_ground_truth_overlay(
            records_dicts,
            output_path=self.figures_dir / "fig4_ground_truth_overlay.png",
        )
        figures["ground_truth_overlay"] = p4

        t_axis = single_run_df["window_end_s"].to_numpy()
        gt_axis = single_run_df["fault_label"].to_numpy()
        timelines = {}
        for m_name in self.ALL_METHODS:
            p_vec, _, _ = self._predict_method(m_name, single_run_df, trained_models)
            timelines[m_name] = p_vec

        p5 = plot_model_predictions_timeline(
            timelines,
            time_array=t_axis,
            ground_truth=gt_axis,
            output_path=self.figures_dir / "fig5_model_predictions_timeline.png",
        )
        figures["model_predictions_timeline"] = p5

        comp_df = pd.DataFrame(comparison_rows)
        p6 = plot_method_comparison_id_ood(
            comp_df,
            output_path=self.figures_dir / "fig6_method_comparison_id_ood.png",
        )
        figures["method_comparison_id_ood"] = p6

        p7 = plot_false_alarm_vs_recall_tradeoff(
            metrics_tradeoff_dict,
            output_path=self.figures_dir / "fig7_false_alarm_vs_recall_tradeoff.png",
        )
        figures["false_alarm_vs_recall_tradeoff"] = p7

        p8 = plot_causal_detection_delays(
            delays_dict,
            output_path=self.figures_dir / "fig8_causal_detection_delays.png",
        )
        figures["causal_detection_delays"] = p8

        calibrator_unprot = ConfidenceGatedAdaptiveCalibrator(
            AdaptiveCalibrationConfig(alpha=0.08, enable_confidence_gate=False, enable_stability_gate=False, enable_freeze_logic=False)
        )
        _, _, diag_unprot = calibrator_unprot.process_run(single_run_df, trained_models["rf_norm"])
        unprot_dicts = [r.to_dict() for r in diag_unprot.records]

        p9 = plot_contamination_failure_modes(
            unprotected_records=unprot_dicts,
            protected_records=records_dicts,
            output_path=self.figures_dir / "fig9_contamination_failure_modes.png",
        )
        figures["contamination_failure_modes"] = p9

        p10 = plot_ablation_study(
            ablation_df,
            output_path=self.figures_dir / "fig10_ablation_study.png",
        )
        figures["ablation_study"] = p10

        return figures


def run_phase5_benchmark(
    dataset_id: str = "flexpulse_calibration_v2",
    random_seed: int = 42,
    n_bootstraps: int = 200,
) -> Phase5BenchmarkSummary:
    runner = Phase5BenchmarkRunner(
        dataset_id=dataset_id,
        random_seed=random_seed,
        n_bootstraps=n_bootstraps,
    )
    return runner.run()


if __name__ == "__main__":
    run_phase5_benchmark()
