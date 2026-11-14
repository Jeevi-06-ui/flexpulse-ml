"""
Fixed electrical-feature threshold detector baseline using observable voltage and current.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from flexpulse_ml.features.extractor import FEATURE_NAMES


class FixedThresholdBaseline(BaseEstimator, ClassifierMixin):
    """Interpretable rule-based detector operating on observed voltage drop and transient jumps."""

    def __init__(
        self,
        voltage_drop_std_multiplier: float = 3.0,
        transient_jump_std_multiplier: float = 3.0,
        voltage_mean_feature: str = "voltage_mean_v",
        transient_jump_feature: str = "voltage_max_abs_diff_v",
    ) -> None:
        self.voltage_drop_std_multiplier = voltage_drop_std_multiplier
        self.transient_jump_std_multiplier = transient_jump_std_multiplier
        self.voltage_mean_feature = voltage_mean_feature
        self.transient_jump_feature = transient_jump_feature

        self.voltage_drop_threshold_: float = 0.0
        self.transient_jump_threshold_: float = 0.0
        self.v_mean_idx_: int = -1
        self.jump_idx_: int = -1
        self.classes_: np.ndarray = np.array([0, 1], dtype=int)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> FixedThresholdBaseline:
        names = feature_names or list(FEATURE_NAMES)
        if self.voltage_mean_feature in names:
            self.v_mean_idx_ = names.index(self.voltage_mean_feature)
        else:
            self.v_mean_idx_ = 0

        if self.transient_jump_feature in names:
            self.jump_idx_ = names.index(self.transient_jump_feature)
        else:
            self.jump_idx_ = min(12, X.shape[1] - 1)

        healthy_mask = y == 0
        if np.any(healthy_mask):
            v_healthy = X[healthy_mask, self.v_mean_idx_]
            jump_healthy = X[healthy_mask, self.jump_idx_]
        else:
            v_healthy = X[:, self.v_mean_idx_]
            jump_healthy = X[:, self.jump_idx_]

        v_mean_ref = float(np.mean(v_healthy))
        v_std_ref = float(np.std(v_healthy, ddof=1)) if len(v_healthy) > 1 else 0.005
        jump_mean_ref = float(np.mean(jump_healthy))
        jump_std_ref = float(np.std(jump_healthy, ddof=1)) if len(jump_healthy) > 1 else 0.005

        self.voltage_drop_threshold_ = v_mean_ref - (self.voltage_drop_std_multiplier * max(v_std_ref, 0.001))
        self.transient_jump_threshold_ = jump_mean_ref + (self.transient_jump_std_multiplier * max(jump_std_ref, 0.001))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        v_mean = X[:, self.v_mean_idx_]
        jump = X[:, self.jump_idx_]

        is_voltage_drop = v_mean < self.voltage_drop_threshold_
        is_transient_jump = jump > self.transient_jump_threshold_

        pred = (is_voltage_drop | is_transient_jump).astype(int)
        return pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        v_mean = X[:, self.v_mean_idx_]
        jump = X[:, self.jump_idx_]

        drop_score = np.maximum(0.0, (self.voltage_drop_threshold_ - v_mean) / 0.05)
        jump_score = np.maximum(0.0, (jump - self.transient_jump_threshold_) / 0.05)
        raw_score = np.maximum(drop_score, jump_score)

        prob_fault = 1.0 / (1.0 + np.exp(-2.0 * (raw_score - 0.5)))
        prob_fault = np.clip(prob_fault, 0.0, 1.0)
        prob_healthy = 1.0 - prob_fault
        return np.column_stack([prob_healthy, prob_fault])
