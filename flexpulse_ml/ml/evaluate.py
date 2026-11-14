"""
Evaluation engine for window-level classification, bootstrapping, scenario breakdowns, and event matching.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
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

from flexpulse_ml.data.storage import DatasetStorage


@dataclass
class WindowClassificationMetrics:
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    specificity: float
    false_negative_rate: float
    f1: float
    roc_auc: Optional[float]
    average_precision: Optional[float]
    confusion_matrix: Dict[str, int]
    n_samples: int
    n_healthy: int
    n_fault: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EventLevelMetrics:
    total_injected_events: int
    observable_injected_events: int
    unsampled_injected_events: int
    detected_observable_events: int
    missed_observable_events: int
    event_recall: float
    total_predicted_events: int
    false_alarm_events: int
    false_alarms_per_minute: float
    mean_detection_delay_s: Optional[float]
    median_detection_delay_s: Optional[float]
    detection_delays: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_window_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> WindowClassificationMetrics:
    n_total = len(y_true)
    n_fault = int(np.sum(y_true == 1))
    n_healthy = int(np.sum(y_true == 0))

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    roc_auc = None
    pr_auc = None
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
            pr_auc = float(average_precision_score(y_true, y_prob))
        except Exception:
            roc_auc = None
            pr_auc = None

    return WindowClassificationMetrics(
        accuracy=acc,
        balanced_accuracy=bal_acc,
        precision=prec,
        recall=rec,
        specificity=spec,
        false_negative_rate=fnr,
        f1=f1,
        roc_auc=roc_auc,
        average_precision=pr_auc,
        confusion_matrix={"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        n_samples=n_total,
        n_healthy=n_healthy,
        n_fault=n_fault,
    )


def bootstrap_run_level_metrics(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    n_bootstraps: int = 200,
    random_seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    runs = df["run_id"].unique()
    n_runs = len(runs)
    if n_runs < 2:
        return {}

    rng = np.random.default_rng(random_seed)
    f1_list: List[float] = []
    rec_list: List[float] = []
    prec_list: List[float] = []
    bal_acc_list: List[float] = []

    y_true_all = df["fault_label"].to_numpy()
    run_idx_map = {r: np.where(df["run_id"].to_numpy() == r)[0] for r in runs}

    for _ in range(n_bootstraps):
        sampled_runs = rng.choice(runs, size=n_runs, replace=True)
        sampled_indices: List[int] = []
        for r in sampled_runs:
            sampled_indices.extend(run_idx_map[r])

        idx_arr = np.array(sampled_indices)
        y_t = y_true_all[idx_arr]
        y_p = y_pred[idx_arr]

        if len(np.unique(y_t)) > 1:
            f1_list.append(float(f1_score(y_t, y_p, zero_division=0)))
            rec_list.append(float(recall_score(y_t, y_p, zero_division=0)))
            prec_list.append(float(precision_score(y_t, y_p, zero_division=0)))
            bal_acc_list.append(float(balanced_accuracy_score(y_t, y_p)))

    def ci(arr: List[float]) -> Dict[str, float]:
        if not arr:
            return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
        return {
            "mean": float(np.mean(arr)),
            "ci_lower": float(np.percentile(arr, 2.5)),
            "ci_upper": float(np.percentile(arr, 97.5)),
        }

    return {
        "f1": ci(f1_list),
        "recall": ci(rec_list),
        "precision": ci(prec_list),
        "balanced_accuracy": ci(bal_acc_list),
    }


def evaluate_scenario_breakdown(
    df: pd.DataFrame,
    y_pred: np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for sc in df["scenario"].unique():
        mask = df["scenario"].to_numpy() == sc
        y_sc = df["fault_label"].to_numpy()[mask]
        pred_sc = y_pred[mask]
        metrics = calculate_window_metrics(y_sc, pred_sc)
        m_dict = metrics.to_dict()
        tn = m_dict["confusion_matrix"]["tn"]
        fp = m_dict["confusion_matrix"]["fp"]
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        m_dict["false_positive_rate"] = fpr
        results[str(sc)] = m_dict
    return results


def evaluate_window_label_sensitivity(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    thresholds: Optional[List[float]] = None,
) -> Dict[str, Dict[str, Any]]:
    th_list = thresholds or [0.0, 0.10, 0.20, 0.50]
    results: Dict[str, Dict[str, Any]] = {}
    fractions = df["fault_fraction"].to_numpy()

    for th in th_list:
        alt_target = (fractions > th).astype(int)
        key = f"fault_fraction_gt_{int(th * 100)}pct" if th > 0 else "any_fault_sample"
        metrics = calculate_window_metrics(alt_target, y_pred)
        results[key] = metrics.to_dict()

    return results


def evaluate_event_level_detections(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    dataset_storage: DatasetStorage,
    split: str,
    merge_gap_s: float = 0.5,
) -> EventLevelMetrics:
    runs = df["run_id"].unique()
    total_injected = 0
    observable_injected = 0
    unsampled_injected = 0
    detected_observable = 0
    total_predicted = 0
    false_alarm_events = 0
    detection_delays: List[float] = []
    total_healthy_duration_s = 0.0

    for r_id in runs:
        try:
            run_events = dataset_storage.load_events(r_id, split)
        except Exception:
            run_events = []

        try:
            meta = dataset_storage.load_metadata(r_id, split)
            sim_dur = float(meta.get("simulation_duration_s", 60.0))
        except Exception:
            sim_dur = 60.0

        r_mask = df["run_id"].to_numpy() == r_id
        r_df = df[r_mask].reset_index(drop=True)
        r_preds = y_pred[r_mask]

        injected_intervals: List[Dict[str, Any]] = []
        for ev in run_events:
            total_injected += 1
            st_t = float(ev.get("start_time_s", ev.get("t_start_s", 0.0)))
            en_t = float(ev.get("end_time_s", ev.get("t_end_s", 0.0)))
            is_sampled = bool(ev.get("sampled_event", True))

            if is_sampled:
                observable_injected += 1
                injected_intervals.append({"st": st_t, "en": en_t, "detected": False})
            else:
                unsampled_injected += 1

        pred_intervals: List[Tuple[float, float]] = []
        curr_st: Optional[float] = None
        curr_en: Optional[float] = None

        for idx, pred_val in enumerate(r_preds):
            if pred_val == 1:
                w_st = float(r_df.loc[idx, "window_start_s"])
                w_en = float(r_df.loc[idx, "window_end_s"])
                if curr_st is None:
                    curr_st = w_st
                    curr_en = w_en
                elif w_st <= (curr_en + merge_gap_s):
                    curr_en = max(curr_en, w_en)
                else:
                    pred_intervals.append((curr_st, curr_en))
                    curr_st = w_st
                    curr_en = w_en

        if curr_st is not None and curr_en is not None:
            pred_intervals.append((curr_st, curr_en))

        total_predicted += len(pred_intervals)

        for p_st, p_en in pred_intervals:
            matched_any = False
            for inj in injected_intervals:
                overlap = max(0.0, min(p_en, inj["en"]) - max(p_st, inj["st"]))
                if overlap > 0 or abs(p_st - inj["st"]) <= merge_gap_s:
                    matched_any = True
                    if not inj["detected"]:
                        inj["detected"] = True
                        delay = max(0.0, p_st - inj["st"])
                        detection_delays.append(delay)
            if not matched_any:
                false_alarm_events += 1

        for inj in injected_intervals:
            if inj["detected"]:
                detected_observable += 1

        total_healthy_duration_s += max(0.0, sim_dur - (len(injected_intervals) * 5.0))

    missed_observable = observable_injected - detected_observable
    ev_recall = float(detected_observable / observable_injected) if observable_injected > 0 else 1.0

    fa_minutes = total_healthy_duration_s / 60.0
    fa_per_min = float(false_alarm_events / fa_minutes) if fa_minutes > 0 else 0.0

    mean_delay = float(np.mean(detection_delays)) if detection_delays else None
    med_delay = float(np.median(detection_delays)) if detection_delays else None

    return EventLevelMetrics(
        total_injected_events=total_injected,
        observable_injected_events=observable_injected,
        unsampled_injected_events=unsampled_injected,
        detected_observable_events=detected_observable,
        missed_observable_events=missed_observable,
        event_recall=ev_recall,
        total_predicted_events=total_predicted,
        false_alarm_events=false_alarm_events,
        false_alarms_per_minute=fa_per_min,
        mean_detection_delay_s=mean_delay,
        median_detection_delay_s=med_delay,
        detection_delays=detection_delays,
    )
