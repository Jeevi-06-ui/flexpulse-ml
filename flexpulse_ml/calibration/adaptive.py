"""
Confidence-gated adaptive baseline calibration engine with multi-stage safety gating and freeze recovery.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from flexpulse_ml.calibration.baseline import (
    BaselineStatistics,
    InitialBaselineEstimator,
)
from flexpulse_ml.calibration.confidence import ConfidenceEstimator
from flexpulse_ml.calibration.diagnostics import (
    CalibrationDiagnostics,
    CalibrationState,
    WindowCalibrationRecord,
)
from flexpulse_ml.features.extractor import FEATURE_NAMES


@dataclass
class AdaptiveCalibrationConfig:
    alpha: float = 0.04
    tau_conf: float = 0.95
    max_current_variance: float = 0.015
    max_allowed_delta: float = 0.03
    outlier_multiplier: float = 2.5
    anomaly_cooldown_windows: int = 4
    recovery_stable_windows: int = 5
    initial_duration_s: float = 5.0
    enable_confidence_gate: bool = True
    enable_stability_gate: bool = True
    enable_freeze_logic: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ConfidenceGatedAdaptiveCalibrator:
    """Online sequential baseline tracker enforcing multi-stage gating and freeze dynamics."""

    def __init__(
        self,
        config: Optional[AdaptiveCalibrationConfig] = None,
    ) -> None:
        self.config = config or AdaptiveCalibrationConfig()
        self.baseline_estimator = InitialBaselineEstimator(
            initial_duration_s=self.config.initial_duration_s
        )
        self.confidence_estimator = ConfidenceEstimator(
            default_threshold=self.config.tau_conf
        )

        self._v_indices = [
            list(FEATURE_NAMES).index(f)
            for f in InitialBaselineEstimator.VOLTAGE_LEVEL_FEATURES
            if f in FEATURE_NAMES
        ]
        self._i_indices = [
            list(FEATURE_NAMES).index(f)
            for f in InitialBaselineEstimator.CURRENT_LEVEL_FEATURES
            if f in FEATURE_NAMES
        ]
        self._v_mean_idx = list(FEATURE_NAMES).index("voltage_mean_v")
        self._i_mean_idx = list(FEATURE_NAMES).index("current_mean_a")
        self._i_std_idx = list(FEATURE_NAMES).index("current_std_a")

    def process_run(
        self,
        run_df: pd.DataFrame,
        model: Any,
    ) -> Tuple[np.ndarray, np.ndarray, CalibrationDiagnostics]:
        orig_n_jobs = getattr(model, "n_jobs", None)
        if hasattr(model, "n_jobs") and orig_n_jobs != 1:
            try:
                model.n_jobs = 1
            except Exception:
                pass

        try:
            return self._process_run_core(run_df, model)
        finally:
            if hasattr(model, "n_jobs") and orig_n_jobs != 1:
                try:
                    model.n_jobs = orig_n_jobs
                except Exception:
                    pass

    def _process_run_core(
        self,
        run_df: pd.DataFrame,
        model: Any,
    ) -> Tuple[np.ndarray, np.ndarray, CalibrationDiagnostics]:
        n_windows = len(run_df)
        initial_baseline = self.baseline_estimator.fit_run(run_df)

        active_v = initial_baseline.center_v
        active_i = initial_baseline.center_i
        spread_v = initial_baseline.spread_v

        state = CalibrationState.INITIALIZING
        diagnostics = CalibrationDiagnostics()

        preds = np.zeros(n_windows, dtype=np.int32)
        probs = np.zeros(n_windows, dtype=np.float64)

        consecutive_anomalies = 0
        consecutive_stable = 0

        X_raw = run_df[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
        t_ends = run_df["window_end_s"].to_numpy(dtype=np.float64)
        w_indices = (
            run_df["window_index"].to_numpy(dtype=np.int32)
            if "window_index" in run_df.columns
            else np.arange(n_windows, dtype=np.int32)
        )
        gt_faults = (
            run_df["fault_label"].to_numpy(dtype=np.int32)
            if "fault_label" in run_df.columns
            else np.zeros(n_windows, dtype=np.int32)
        )

        for idx in range(n_windows):
            w_idx = int(w_indices[idx])
            t_end = float(t_ends[idx])
            gt_fault = int(gt_faults[idx])

            raw_row = X_raw[idx]
            raw_mean_v = float(raw_row[self._v_mean_idx])
            raw_mean_i = float(raw_row[self._i_mean_idx])
            curr_std_i = float(raw_row[self._i_std_idx])

            norm_row = raw_row.copy()
            norm_row[self._v_indices] -= active_v
            norm_row[self._i_indices] -= active_i
            X_norm = norm_row.reshape(1, -1)

            if hasattr(model, "predict_proba"):
                try:
                    p_arr = model.predict_proba(X_norm)[0]
                    if len(p_arr) >= 2:
                        conf_healthy = float(p_arr[0])
                        fault_prob = float(p_arr[1])
                        pred_val = int(fault_prob >= 0.5)
                    else:
                        conf_healthy = float(p_arr[0])
                        fault_prob = 1.0 - conf_healthy
                        pred_val = int(fault_prob >= 0.5)
                except Exception:
                    pred_val = int(model.predict(X_norm)[0])
                    conf_healthy = 0.98 if pred_val == 0 else 0.02
                    fault_prob = 1.0 - conf_healthy
            else:
                pred_val = int(model.predict(X_norm)[0])
                conf_healthy = 0.98 if pred_val == 0 else 0.02
                fault_prob = 1.0 - conf_healthy

            preds[idx] = pred_val
            probs[idx] = fault_prob

            norm_v = float(norm_row[self._v_mean_idx])
            norm_i = float(norm_row[self._i_mean_idx])

            is_initializing = t_end <= self.config.initial_duration_s
            update_accepted = False
            rejection_reason = ""

            if is_initializing:
                state = CalibrationState.INITIALIZING
                rejection_reason = "INITIALIZING"
                consecutive_stable += 1
            else:
                if state == CalibrationState.INITIALIZING:
                    state = CalibrationState.NORMAL

                passes_conf = True
                if self.config.enable_confidence_gate:
                    if pred_val == 1 or conf_healthy < self.config.tau_conf:
                        passes_conf = False

                passes_stability = True
                if self.config.enable_stability_gate:
                    if curr_std_i > self.config.max_current_variance:
                        passes_stability = False

                passes_cooldown = True
                if self.config.enable_freeze_logic:
                    if (
                        consecutive_anomalies > 0
                        and consecutive_stable < self.config.anomaly_cooldown_windows
                    ):
                        passes_cooldown = False

                v_residual = abs(raw_mean_v - active_v)
                passes_outlier = v_residual <= (self.config.outlier_multiplier * spread_v)

                proposed_delta_v = self.config.alpha * (raw_mean_v - active_v)
                passes_max_delta = abs(proposed_delta_v) <= self.config.max_allowed_delta

                if state == CalibrationState.FROZEN:
                    if self.config.enable_freeze_logic:
                        if passes_conf and passes_stability and passes_outlier:
                            consecutive_stable += 1
                            if consecutive_stable >= self.config.recovery_stable_windows:
                                state = CalibrationState.NORMAL
                                consecutive_anomalies = 0
                        else:
                            consecutive_stable = 0

                    if state == CalibrationState.FROZEN:
                        update_accepted = False
                        rejection_reason = "SYSTEM_FROZEN"

                if state == CalibrationState.NORMAL:
                    if not passes_conf:
                        update_accepted = False
                        rejection_reason = "CONFIDENCE_GATE_FAILED"
                        consecutive_anomalies += 1
                        consecutive_stable = 0
                        if self.config.enable_freeze_logic and consecutive_anomalies >= 2:
                            state = CalibrationState.FROZEN
                    elif not passes_stability:
                        update_accepted = False
                        rejection_reason = "STABILITY_GATE_FAILED"
                        consecutive_stable = 0
                    elif not passes_cooldown:
                        update_accepted = False
                        rejection_reason = "ANOMALY_COOLDOWN"
                        consecutive_stable += 1
                    elif not passes_outlier:
                        update_accepted = False
                        rejection_reason = "OUTLIER_RESIDUAL"
                        consecutive_stable = 0
                        if self.config.enable_freeze_logic:
                            state = CalibrationState.FROZEN
                    elif not passes_max_delta:
                        update_accepted = False
                        rejection_reason = "MAX_DELTA_EXCEEDED"
                    else:
                        update_accepted = True
                        active_v = (1.0 - self.config.alpha) * active_v + self.config.alpha * raw_mean_v
                        active_i = (1.0 - self.config.alpha) * active_i + self.config.alpha * raw_mean_i
                        consecutive_anomalies = 0
                        consecutive_stable += 1

            record = WindowCalibrationRecord(
                window_index=w_idx,
                timestamp_end_s=t_end,
                raw_v=raw_mean_v,
                raw_i=raw_mean_i,
                active_baseline_v=active_v,
                active_baseline_i=active_i,
                normalized_v=norm_v,
                normalized_i=norm_i,
                prediction=pred_val,
                confidence=conf_healthy,
                state=state,
                update_accepted=update_accepted,
                rejection_reason=rejection_reason,
                ground_truth_fault=gt_fault,
            )
            diagnostics.record_window(record)

        return preds, probs, diagnostics
