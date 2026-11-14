"""
Statistical and electrical feature extraction pipeline for time-series windows.
"""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np

from flexpulse_ml.features.windowing import Window


FEATURE_UNITS: Dict[str, str] = {
    "voltage_mean_v": "V",
    "voltage_median_v": "V",
    "voltage_std_v": "V",
    "voltage_min_v": "V",
    "voltage_max_v": "V",
    "voltage_range_v": "V",
    "voltage_p10_v": "V",
    "voltage_p90_v": "V",
    "voltage_mad_v": "V",
    "voltage_trend_slope_v_per_s": "V/s",
    "voltage_mean_abs_diff_v": "V",
    "voltage_std_diff_v": "V",
    "voltage_max_abs_diff_v": "V",
    "current_mean_a": "A",
    "current_median_a": "A",
    "current_std_a": "A",
    "current_min_a": "A",
    "current_max_a": "A",
    "current_range_a": "A",
    "current_p10_a": "A",
    "current_p90_a": "A",
    "current_mad_a": "A",
    "current_trend_slope_a_per_s": "A/s",
    "current_mean_abs_diff_a": "A",
    "current_std_diff_a": "A",
    "current_max_abs_diff_a": "A",
}

FEATURE_NAMES: List[str] = list(FEATURE_UNITS.keys())


class ElectricalFeatureExtractor:
    """Calculates statistical moments and dynamic features on voltage and current."""

    def __init__(self, sampling_rate_hz: float = 100.0) -> None:
        self.sampling_rate_hz = sampling_rate_hz

    def compute_signal_features(
        self,
        signal: np.ndarray,
        signal_prefix: str,
        unit: str,
    ) -> Dict[str, float]:
        n = len(signal)
        if n == 0:
            return {
                f"{signal_prefix}_mean_{unit}": 0.0,
                f"{signal_prefix}_median_{unit}": 0.0,
                f"{signal_prefix}_std_{unit}": 0.0,
                f"{signal_prefix}_min_{unit}": 0.0,
                f"{signal_prefix}_max_{unit}": 0.0,
                f"{signal_prefix}_range_{unit}": 0.0,
                f"{signal_prefix}_p10_{unit}": 0.0,
                f"{signal_prefix}_p90_{unit}": 0.0,
                f"{signal_prefix}_mad_{unit}": 0.0,
                f"{signal_prefix}_trend_slope_{unit}_per_s": 0.0,
                f"{signal_prefix}_mean_abs_diff_{unit}": 0.0,
                f"{signal_prefix}_std_diff_{unit}": 0.0,
                f"{signal_prefix}_max_abs_diff_{unit}": 0.0,
            }

        s_mean = float(np.mean(signal))
        s_median = float(np.median(signal))
        s_std = float(np.std(signal, ddof=1)) if n > 1 else 0.0
        s_min = float(np.min(signal))
        s_max = float(np.max(signal))
        s_range = float(s_max - s_min)
        s_p10 = float(np.percentile(signal, 10))
        s_p90 = float(np.percentile(signal, 90))
        s_mad = float(np.median(np.abs(signal - s_median)))

        if n > 1:
            dt = 1.0 / self.sampling_rate_hz
            t_rel = np.arange(n) * dt
            t_centered = t_rel - np.mean(t_rel)
            s_centered = signal - s_mean
            denom = float(np.sum(t_centered**2))
            if denom > 1e-12:
                s_slope = float(np.sum(t_centered * s_centered) / denom)
            else:
                s_slope = 0.0
        else:
            s_slope = 0.0

        diff = np.diff(signal)
        if len(diff) > 0:
            abs_diff = np.abs(diff)
            mean_diff = float(np.mean(abs_diff))
            max_diff = float(np.max(abs_diff))
            std_diff = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
        else:
            mean_diff = 0.0
            max_diff = 0.0
            std_diff = 0.0

        return {
            f"{signal_prefix}_mean_{unit}": s_mean,
            f"{signal_prefix}_median_{unit}": s_median,
            f"{signal_prefix}_std_{unit}": s_std,
            f"{signal_prefix}_min_{unit}": s_min,
            f"{signal_prefix}_max_{unit}": s_max,
            f"{signal_prefix}_range_{unit}": s_range,
            f"{signal_prefix}_p10_{unit}": s_p10,
            f"{signal_prefix}_p90_{unit}": s_p90,
            f"{signal_prefix}_mad_{unit}": s_mad,
            f"{signal_prefix}_trend_slope_{unit}_per_s": s_slope,
            f"{signal_prefix}_mean_abs_diff_{unit}": mean_diff,
            f"{signal_prefix}_std_diff_{unit}": std_diff,
            f"{signal_prefix}_max_abs_diff_{unit}": max_diff,
        }

    def extract_from_window(self, window: Window) -> Dict[str, Any]:
        features: Dict[str, Any] = {}
        v_features = self.compute_signal_features(window.voltage, "voltage", "v")
        i_features = self.compute_signal_features(window.current, "current", "a")

        features.update(v_features)
        features.update(i_features)
        return features
