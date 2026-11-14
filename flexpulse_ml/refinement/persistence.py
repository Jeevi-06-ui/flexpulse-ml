"""
Candidate temporal persistence filter: evaluates false alarm reduction vs causal detection latency trade-offs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass
class PersistenceEvaluationResult:
    filter_name: str
    persistence_windows: int
    split: str
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    specificity: float
    f1: float
    total_injected_events: int
    detected_events: int
    missed_events: int
    event_recall: float
    total_predicted_events: int
    false_alarm_events: int
    false_alarms_per_minute: float
    mean_causal_delay_s: float
    median_causal_delay_s: float
    added_detection_delay_s: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TemporalPersistenceFilter:
    """Applies causal k-consecutive-window persistence gating to binary predictions."""

    def __init__(self, persistence_windows: int = 2) -> None:
        self.persistence_windows = max(1, persistence_windows)

    def filter_predictions(self, raw_preds: np.ndarray) -> np.ndarray:
        n = len(raw_preds)
        if self.persistence_windows <= 1 or n == 0:
            return raw_preds.copy()

        filtered = np.zeros(n, dtype=np.int32)
        consecutive_ones = 0

        for i in range(n):
            if raw_preds[i] == 1:
                consecutive_ones += 1
                if consecutive_ones >= self.persistence_windows:
                    filtered[i] = 1
            else:
                consecutive_ones = 0
                filtered[i] = 0

        return filtered

    def evaluate_on_partition(
        self,
        df: pd.DataFrame,
        raw_preds: np.ndarray,
        raw_storage: Any,
        split_name: str,
        baseline_median_delay_s: Optional[float] = None,
    ) -> PersistenceEvaluationResult:
        runs = df["run_id"].unique()
        y_true = df["fault_label"].to_numpy(dtype=np.int32)

        filtered_preds_list: List[int] = []
        for r_id in runs:
            r_mask = (df["run_id"].to_numpy() == r_id)
            r_raw = raw_preds[r_mask]
            r_filt = self.filter_predictions(r_raw)
            filtered_preds_list.extend(r_filt)

        filtered_preds = np.array(filtered_preds_list, dtype=np.int32)

        acc = float(accuracy_score(y_true, filtered_preds))
        bal_acc = float(balanced_accuracy_score(y_true, filtered_preds))
        prec = float(precision_score(y_true, filtered_preds, zero_division=0))
        rec = float(recall_score(y_true, filtered_preds, zero_division=0))
        f1 = float(f1_score(y_true, filtered_preds, zero_division=0))

        cm = confusion_matrix(y_true, filtered_preds, labels=[0, 1])
        tn, fp = int(cm[0, 0]), int(cm[0, 1])
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        total_injected = 0
        observable_injected = 0
        detected_observable = 0
        total_predicted = 0
        false_alarm_events = 0
        delays_causal: List[float] = []
        total_healthy_duration_s = 0.0

        for r_id in runs:
            r_mask = (df["run_id"].to_numpy() == r_id)
            r_df = df[r_mask].reset_index(drop=True)
            r_preds = filtered_preds[r_mask]

            try:
                events = raw_storage.load_events(r_id, split_name)
            except Exception:
                events = []

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

        mean_delay = float(np.mean(delays_causal)) if delays_causal else 0.0
        median_delay = float(np.median(delays_causal)) if delays_causal else 0.0
        base_delay = baseline_median_delay_s if baseline_median_delay_s is not None else median_delay
        added_delay = max(0.0, median_delay - base_delay)

        name = f"Persistence_N{self.persistence_windows}" if self.persistence_windows > 1 else "Raw_Predictions"

        return PersistenceEvaluationResult(
            filter_name=name,
            persistence_windows=self.persistence_windows,
            split=split_name,
            accuracy=round(acc, 4),
            balanced_accuracy=round(bal_acc, 4),
            precision=round(prec, 4),
            recall=round(rec, 4),
            specificity=round(spec, 4),
            f1=round(f1, 4),
            total_injected_events=total_injected,
            detected_events=detected_observable,
            missed_events=observable_injected - detected_observable,
            event_recall=round(ev_recall, 4),
            total_predicted_events=total_predicted,
            false_alarm_events=false_alarm_events,
            false_alarms_per_minute=round(fa_rate, 4),
            mean_causal_delay_s=round(mean_delay, 4),
            median_causal_delay_s=round(median_delay, 4),
            added_detection_delay_s=round(added_delay, 4),
        )
