"""
Diagnostics, tracking records, and failure-mode instrumentation for self-calibration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


class CalibrationState:
    INITIALIZING = "INITIALIZING"
    NORMAL = "NORMAL"
    FROZEN = "FROZEN"
    ABSTAIN = "ABSTAIN"
    ANOMALY = "ANOMALY"


@dataclass
class WindowCalibrationRecord:
    window_index: int
    timestamp_end_s: float
    raw_v: float
    raw_i: float
    active_baseline_v: float
    active_baseline_i: float
    normalized_v: float
    normalized_i: float
    prediction: int
    confidence: float
    state: str
    update_accepted: bool
    rejection_reason: str
    ground_truth_fault: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CalibrationDiagnostics:
    """Maintains timeline of calibration decisions, states, and retrospective contamination checks."""

    def __init__(self) -> None:
        self.records: List[WindowCalibrationRecord] = []

    def record_window(self, record: WindowCalibrationRecord) -> None:
        self.records.append(record)

    def compute_summary_metrics(self) -> Dict[str, Any]:
        if not self.records:
            return {
                "total_windows": 0,
                "accepted_updates": 0,
                "rejected_updates": 0,
                "acceptance_rate": 0.0,
                "frozen_windows": 0,
                "time_spent_frozen_s": 0.0,
                "inappropriate_updates_during_fault": 0,
                "contamination_rate": 0.0,
                "mean_confidence": 0.0,
            }

        total_windows = len(self.records)
        accepted = sum(1 for r in self.records if r.update_accepted)
        rejected = total_windows - accepted
        frozen = sum(1 for r in self.records if r.state == CalibrationState.FROZEN)

        fault_records = [r for r in self.records if r.ground_truth_fault == 1]
        inappropriate = sum(1 for r in fault_records if r.update_accepted)
        contam_rate = float(inappropriate / len(fault_records)) if fault_records else 0.0

        mean_conf = float(np.mean([r.confidence for r in self.records]))
        frozen_time_s = float(frozen * 0.25)

        return {
            "total_windows": total_windows,
            "accepted_updates": accepted,
            "rejected_updates": rejected,
            "acceptance_rate": float(accepted / total_windows) if total_windows > 0 else 0.0,
            "frozen_windows": frozen,
            "time_spent_frozen_s": frozen_time_s,
            "inappropriate_updates_during_fault": inappropriate,
            "contamination_rate": contam_rate,
            "mean_confidence": mean_conf,
        }
