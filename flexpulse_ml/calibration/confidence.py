"""
Confidence estimation and gating for candidate baseline updates.
"""

from __future__ import annotations

from typing import Any, Optional
import numpy as np


class ConfidenceEstimator:
    """Evaluates classifier prediction confidence for healthy power-path status."""

    def __init__(self, default_threshold: float = 0.95) -> None:
        self.default_threshold = default_threshold

    def estimate_healthy_confidence(
        self,
        model: Any,
        X: np.ndarray,
    ) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(X)
                if probs.ndim == 2 and probs.shape[1] >= 2:
                    return probs[:, 0].astype(np.float64)
            except Exception:
                pass

        if hasattr(model, "decision_function"):
            try:
                df_scores = model.decision_function(X)
                probs = 1.0 / (1.0 + np.exp(df_scores))
                return probs.astype(np.float64)
            except Exception:
                pass

        preds = model.predict(X)
        heuristic_conf = np.where(preds == 0, 0.98, 0.02).astype(np.float64)
        return heuristic_conf

    def is_confident_healthy(
        self,
        confidence: float,
        threshold: Optional[float] = None,
    ) -> bool:
        th = threshold if threshold is not None else self.default_threshold
        return float(confidence) >= th
