"""
Controlled experimental regimes E1 to E10 designed to isolate adaptation vs initial calibration factors.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple
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
from flexpulse_ml.features.extractor import FEATURE_NAMES, ElectricalFeatureExtractor
from flexpulse_ml.features.windowing import SlidingWindowGenerator, WindowConfig
from flexpulse_ml.simulation.electrical_model import (
    SimulationGroundTruth,
    SimulationMetadata,
    SimulationObservations,
    SimulationResult,
    compute_circuit_signals,
)


@dataclass
class RegimeComparisonRow:
    regime_id: str
    regime_name: str
    method: str
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    accepted_updates: int
    inappropriate_updates: int
    contamination_rate: float
    a3_vs_a2_delta_f1: float
    behavior_classification: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ControlledRegimeGenerator:
    """Generates synthetic electrical runs for controlled experimental regimes E1-E10."""

    REGIME_E1 = "E1_STABLE_BASELINE"
    REGIME_E2 = "E2_OFFSET_DRIFT"
    REGIME_E3 = "E3_RESISTANCE_DRIFT"
    REGIME_E4 = "E4_ABRUPT_LOAD_CHANGE"
    REGIME_E5 = "E5_ELEVATED_NOISE"
    REGIME_E6 = "E6_HEAVY_LOAD"
    REGIME_E7 = "E7_SLOW_CREEPING_FAULT"
    REGIME_E8 = "E8_SHORT_INTERMITTENT"
    REGIME_E9 = "E9_EARLY_FAULT_POST_INIT"
    REGIME_E10 = "E10_COMBINED_SHIFTS"

    ALL_REGIMES = [
        REGIME_E1,
        REGIME_E2,
        REGIME_E3,
        REGIME_E4,
        REGIME_E5,
        REGIME_E6,
        REGIME_E7,
        REGIME_E8,
        REGIME_E9,
        REGIME_E10,
    ]

    def __init__(self, sampling_rate_hz: float = 100.0, duration_s: float = 60.0) -> None:
        self.sampling_rate_hz = sampling_rate_hz
        self.duration_s = duration_s
        self.extractor = ElectricalFeatureExtractor(sampling_rate_hz=sampling_rate_hz)
        w_cfg = WindowConfig(sampling_rate_hz=sampling_rate_hz, window_duration_s=0.5, overlap_fraction=0.5)
        self.window_gen = SlidingWindowGenerator(config=w_cfg)

    def generate_regime_run(
        self,
        regime_id: str,
        seed: int,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        rng = np.random.default_rng(seed)
        n_samples = int(self.duration_s * self.sampling_rate_hz)
        t = np.arange(n_samples) / self.sampling_rate_hz

        source_v = np.full(n_samples, 5.0, dtype=np.float64)
        load_r = np.full(n_samples, 4.0, dtype=np.float64)
        base_r = np.full(n_samples, 0.080, dtype=np.float64)
        contact_r = np.zeros(n_samples, dtype=np.float64)
        fault_r = np.zeros(n_samples, dtype=np.float64)

        noise_v_std = 0.005
        noise_i_std = 0.006
        bias_v = np.full(n_samples, 0.001, dtype=np.float64)
        bias_i = np.full(n_samples, 0.001, dtype=np.float64)

        fault_label = np.zeros(n_samples, dtype=np.int32)
        event_meta: List[Dict[str, Any]] = []

        if regime_id == self.REGIME_E1:
            events = [(12.0, 15.0), (30.0, 34.0), (45.0, 48.0)]
            for st, en in events:
                mask = (t >= st) & (t <= en)
                fault_r[mask] = 0.450
                fault_label[mask] = 1
                event_meta.append({"start_time_s": st, "end_time_s": en, "fault_type": "INTERMITTENT"})

        elif regime_id == self.REGIME_E2:
            post_mask = t > 5.0
            drift_rate = 0.002
            bias_v[post_mask] += drift_rate * (t[post_mask] - 5.0)

        elif regime_id == self.REGIME_E3:
            post_mask = t > 5.0
            drift_rate_r = 0.003
            base_r[post_mask] += drift_rate_r * (t[post_mask] - 5.0)

        elif regime_id == self.REGIME_E4:
            step_mask = t >= 25.0
            load_r[step_mask] = 2.0

        elif regime_id == self.REGIME_E5:
            noise_v_std = 0.016
            noise_i_std = 0.022
            events = [(15.0, 18.0), (35.0, 39.0)]
            for st, en in events:
                mask = (t >= st) & (t <= en)
                fault_r[mask] = 0.500
                fault_label[mask] = 1
                event_meta.append({"start_time_s": st, "end_time_s": en, "fault_type": "INTERMITTENT"})

        elif regime_id == self.REGIME_E6:
            load_r[:] = 1.90
            events = [(18.0, 22.0), (40.0, 44.0)]
            for st, en in events:
                mask = (t >= st) & (t <= en)
                fault_r[mask] = 0.350
                fault_label[mask] = 1
                event_meta.append({"start_time_s": st, "end_time_s": en, "fault_type": "INTERMITTENT"})

        elif regime_id == self.REGIME_E7:
            creep_mask = (t >= 15.0) & (t <= 45.0)
            fault_r[creep_mask] = 0.015 * (t[creep_mask] - 15.0)
            fault_label[creep_mask] = 1
            event_meta.append({"start_time_s": 15.0, "end_time_s": 45.0, "fault_type": "SLOW_CREEPING"})

        elif regime_id == self.REGIME_E8:
            short_events = [(10.0, 10.12), (18.5, 18.65), (28.0, 28.15), (39.0, 39.10), (50.0, 50.15)]
            for st, en in short_events:
                mask = (t >= st) & (t <= en)
                fault_r[mask] = 1.200
                fault_label[mask] = 1
                event_meta.append({"start_time_s": st, "end_time_s": en, "fault_type": "SHORT_INTERMITTENT"})

        elif regime_id == self.REGIME_E9:
            events = [(5.25, 7.50), (22.0, 25.0)]
            for st, en in events:
                mask = (t >= st) & (t <= en)
                fault_r[mask] = 0.500
                fault_label[mask] = 1
                event_meta.append({"start_time_s": st, "end_time_s": en, "fault_type": "EARLY_POST_INIT"})

        elif regime_id == self.REGIME_E10:
            post_mask = t > 5.0
            bias_v[post_mask] += 0.001 * (t[post_mask] - 5.0)
            step_mask = t >= 30.0
            load_r[step_mask] = 2.20
            noise_v_std = 0.014
            noise_i_std = 0.018
            events = [(12.0, 15.0), (42.0, 46.0)]
            for st, en in events:
                mask = (t >= st) & (t <= en)
                fault_r[mask] = 0.450
                fault_label[mask] = 1
                event_meta.append({"start_time_s": st, "end_time_s": en, "fault_type": "COMBINED_FAULT"})

        true_path_r = base_r + contact_r + fault_r
        v_noise = rng.normal(0.0, noise_v_std, n_samples)
        i_noise = rng.normal(0.0, noise_i_std, n_samples)

        v_true, i_true, v_meas, i_meas = compute_circuit_signals(
            source_voltage_v=source_v,
            path_resistance_ohm=true_path_r,
            load_resistance_ohm=load_r,
            voltage_offset_v=bias_v,
            current_offset_a=bias_i,
            voltage_noise_v=v_noise,
            current_noise_a=i_noise,
        )

        obs_df = pd.DataFrame({
            "timestamp_s": t,
            "source_voltage_v": source_v,
            "voltage_measured_v": v_meas,
            "current_measured_a": i_meas,
            "load_resistance_ohm": load_r,
        })
        gt_df = pd.DataFrame({
            "timestamp_s": t,
            "true_path_resistance_ohm": true_path_r,
            "fault_label": fault_label,
        })

        run_id = f"regime_{regime_id}_{seed:04d}"
        windows = self.window_gen.slice_run(obs_df, gt_df, run_id=run_id, scenario=regime_id)

        rows = []
        for w in windows:
            feat_dict = self.extractor.extract_from_window(w)
            feat_dict["run_id"] = w.run_id
            feat_dict["window_index"] = w.window_index
            feat_dict["window_start_s"] = w.start_time_s
            feat_dict["window_end_s"] = w.end_time_s
            feat_dict["fault_label"] = w.fault_label
            feat_dict["fault_fraction"] = w.fault_fraction
            feat_dict["scenario"] = w.scenario
            rows.append(feat_dict)

        feat_df = pd.DataFrame(rows)
        return feat_df, {"events": event_meta, "regime_id": regime_id, "seed": seed}


class RegimeEvaluationMatrix:
    """Evaluates A1, A2, A3 and ablations across regimes E1-E10."""

    def __init__(self, random_seed: int = 42) -> None:
        self.random_seed = random_seed
        self.gen = ControlledRegimeGenerator()
        self.base_estimator = InitialBaselineEstimator(initial_duration_s=5.0)

    def evaluate_regimes(
        self,
        models: Dict[str, Any],
        runs_per_regime: int = 4,
    ) -> List[RegimeComparisonRow]:
        rf_raw = models["rf_raw"]
        rf_norm = models["rf_norm"]

        rows: List[RegimeComparisonRow] = []

        calibrator_a3 = ConfidenceGatedAdaptiveCalibrator(
            AdaptiveCalibrationConfig(alpha=0.04, tau_conf=0.85, max_current_variance=0.040)
        )
        calibrator_unprot = ConfidenceGatedAdaptiveCalibrator(
            AdaptiveCalibrationConfig(alpha=0.06, enable_confidence_gate=False, enable_stability_gate=False, enable_freeze_logic=False)
        )

        for reg in ControlledRegimeGenerator.ALL_REGIMES:
            reg_dfs: List[pd.DataFrame] = []
            for s_idx in range(runs_per_regime):
                seed = self.random_seed + s_idx * 100 + 7
                f_df, _ = self.gen.generate_regime_run(reg, seed)
                reg_dfs.append(f_df)

            combined_df = pd.concat(reg_dfs, ignore_index=True)
            y_true = combined_df["fault_label"].to_numpy(dtype=np.int32)

            X_raw = combined_df[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
            preds_a1 = rf_raw.predict(X_raw)

            norm_runs_a2: List[pd.DataFrame] = []
            for r_id in combined_df["run_id"].unique():
                r_df = combined_df[combined_df["run_id"] == r_id].reset_index(drop=True)
                b = self.base_estimator.fit_run(r_df)
                norm_runs_a2.append(self.base_estimator.normalize_features(r_df, b))
            a2_df = pd.concat(norm_runs_a2, ignore_index=True)
            X_a2 = a2_df[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
            preds_a2 = rf_norm.predict(X_a2)

            a3_preds_list: List[int] = []
            a3_accepted = 0
            a3_inappropriate = 0
            for r_id in combined_df["run_id"].unique():
                r_df = combined_df[combined_df["run_id"] == r_id].reset_index(drop=True)
                p_vec, _, diag = calibrator_a3.process_run(r_df, rf_norm)
                a3_preds_list.extend(p_vec)
                m = diag.compute_summary_metrics()
                a3_accepted += m.get("accepted_updates", 0)
                a3_inappropriate += m.get("inappropriate_updates_during_fault", 0)
            preds_a3 = np.array(a3_preds_list, dtype=np.int32)

            unprot_preds_list: List[int] = []
            unprot_accepted = 0
            unprot_inappropriate = 0
            for r_id in combined_df["run_id"].unique():
                r_df = combined_df[combined_df["run_id"] == r_id].reset_index(drop=True)
                p_vec, _, diag = calibrator_unprot.process_run(r_df, rf_norm)
                unprot_preds_list.extend(p_vec)
                m = diag.compute_summary_metrics()
                unprot_accepted += m.get("accepted_updates", 0)
                unprot_inappropriate += m.get("inappropriate_updates_during_fault", 0)
            preds_unprot = np.array(unprot_preds_list, dtype=np.int32)

            def get_metrics(preds: np.ndarray) -> Tuple[float, float, float, float, float]:
                ac = float(accuracy_score(y_true, preds))
                ba = float(balanced_accuracy_score(y_true, preds))
                pr = float(precision_score(y_true, preds, zero_division=0))
                rc = float(recall_score(y_true, preds, zero_division=0))
                f1 = float(f1_score(y_true, preds, zero_division=0))
                return ac, ba, pr, rc, f1

            ac1, ba1, pr1, rc1, f11 = get_metrics(preds_a1)
            ac2, ba2, pr2, rc2, f12 = get_metrics(preds_a2)
            ac3, ba3, pr3, rc3, f13 = get_metrics(preds_a3)
            acu, bau, pru, rcu, f1u = get_metrics(preds_unprot)

            delta_f1_a3 = round(f13 - f12, 4)
            if abs(delta_f1_a3) <= 0.005:
                behavior = "A3_MATCHES_A2"
            elif delta_f1_a3 > 0.005:
                behavior = "A3_IMPROVES_OVER_A2"
            else:
                behavior = "A3_WORSE_THAN_A2"

            total_fault_w = int(np.sum(y_true == 1))
            contam_rate_a3 = float(a3_inappropriate / total_fault_w) if total_fault_w > 0 else 0.0
            contam_rate_unprot = float(unprot_inappropriate / total_fault_w) if total_fault_w > 0 else 0.0

            rows.append(
                RegimeComparisonRow(
                    regime_id=reg,
                    regime_name=reg.replace("_", " "),
                    method="A1_Static_ML",
                    accuracy=round(ac1, 4),
                    balanced_accuracy=round(ba1, 4),
                    precision=round(pr1, 4),
                    recall=round(rc1, 4),
                    f1=round(f11, 4),
                    accepted_updates=0,
                    inappropriate_updates=0,
                    contamination_rate=0.0,
                    a3_vs_a2_delta_f1=0.0,
                    behavior_classification="N/A",
                )
            )
            rows.append(
                RegimeComparisonRow(
                    regime_id=reg,
                    regime_name=reg.replace("_", " "),
                    method="A2_Initial_Calibration",
                    accuracy=round(ac2, 4),
                    balanced_accuracy=round(ba2, 4),
                    precision=round(pr2, 4),
                    recall=round(rc2, 4),
                    f1=round(f12, 4),
                    accepted_updates=0,
                    inappropriate_updates=0,
                    contamination_rate=0.0,
                    a3_vs_a2_delta_f1=0.0,
                    behavior_classification="REFERENCE",
                )
            )
            rows.append(
                RegimeComparisonRow(
                    regime_id=reg,
                    regime_name=reg.replace("_", " "),
                    method="A3_Gated_Adaptive",
                    accuracy=round(ac3, 4),
                    balanced_accuracy=round(ba3, 4),
                    precision=round(pr3, 4),
                    recall=round(rc3, 4),
                    f1=round(f13, 4),
                    accepted_updates=a3_accepted,
                    inappropriate_updates=a3_inappropriate,
                    contamination_rate=round(contam_rate_a3, 4),
                    a3_vs_a2_delta_f1=delta_f1_a3,
                    behavior_classification=behavior,
                )
            )
            rows.append(
                RegimeComparisonRow(
                    regime_id=reg,
                    regime_name=reg.replace("_", " "),
                    method="A3_Unprotected_No_Gates",
                    accuracy=round(acu, 4),
                    balanced_accuracy=round(bau, 4),
                    precision=round(pru, 4),
                    recall=round(rcu, 4),
                    f1=round(f1u, 4),
                    accepted_updates=unprot_accepted,
                    inappropriate_updates=unprot_inappropriate,
                    contamination_rate=round(contam_rate_unprot, 4),
                    a3_vs_a2_delta_f1=round(f1u - f12, 4),
                    behavior_classification="UNPROTECTED_BASELINE",
                )
            )

        return rows
