"""
Initial known-healthy baseline estimation and feature normalization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class BaselineStatistics:
    center_v: float
    center_i: float
    spread_v: float
    spread_i: float
    apparent_resistance_ohm: float
    initialization_duration_s: float
    valid_window_count: int
    data_quality_status: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class InitialBaselineEstimator:
    """Estimates robust baseline reference from the initial known-healthy interval."""

    VOLTAGE_LEVEL_FEATURES = [
        "voltage_mean_v",
        "voltage_median_v",
        "voltage_min_v",
        "voltage_max_v",
        "voltage_p10_v",
        "voltage_p90_v",
    ]

    CURRENT_LEVEL_FEATURES = [
        "current_mean_a",
        "current_median_a",
        "current_min_a",
        "current_max_a",
        "current_p10_a",
        "current_p90_a",
    ]

    def __init__(
        self,
        initial_duration_s: float = 5.0,
        min_windows: int = 8,
        max_acceptable_spread_v: float = 0.050,
    ) -> None:
        self.initial_duration_s = initial_duration_s
        self.min_windows = min_windows
        self.max_acceptable_spread_v = max_acceptable_spread_v

    def fit_run(self, run_df: pd.DataFrame) -> BaselineStatistics:
        init_mask = run_df["window_end_s"].to_numpy() <= self.initial_duration_s
        init_df = run_df[init_mask].reset_index(drop=True)
        n_windows = len(init_df)

        if n_windows < self.min_windows:
            v_fallback = float(run_df["voltage_mean_v"].iloc[0]) if len(run_df) > 0 else 5.0
            i_fallback = float(run_df["current_mean_a"].iloc[0]) if len(run_df) > 0 else 1.0
            return BaselineStatistics(
                center_v=v_fallback,
                center_i=i_fallback,
                spread_v=0.010,
                spread_i=0.010,
                apparent_resistance_ohm=0.100,
                initialization_duration_s=self.initial_duration_s,
                valid_window_count=n_windows,
                data_quality_status="INSUFFICIENT_DATA",
            )

        v_vals = init_df["voltage_mean_v"].to_numpy(dtype=np.float64)
        i_vals = init_df["current_mean_a"].to_numpy(dtype=np.float64)

        center_v = float(np.median(v_vals))
        center_i = float(np.median(i_vals))

        spread_v = float(np.median(np.abs(v_vals - center_v))) * 1.4826
        spread_i = float(np.median(np.abs(i_vals - center_i))) * 1.4826
        spread_v = max(spread_v, 0.002)
        spread_i = max(spread_i, 0.002)

        nominal_vs = 5.0
        v_drop = max(0.0, nominal_vs - center_v)
        apparent_r = float(v_drop / max(center_i, 0.05))

        status = "VALID"
        if spread_v > self.max_acceptable_spread_v:
            status = "HIGH_NOISE"

        return BaselineStatistics(
            center_v=center_v,
            center_i=center_i,
            spread_v=spread_v,
            spread_i=spread_i,
            apparent_resistance_ohm=apparent_r,
            initialization_duration_s=self.initial_duration_s,
            valid_window_count=n_windows,
            data_quality_status=status,
        )

    def normalize_features(
        self,
        df: pd.DataFrame,
        baseline: BaselineStatistics,
    ) -> pd.DataFrame:
        norm_df = df.copy()

        for f_name in self.VOLTAGE_LEVEL_FEATURES:
            if f_name in norm_df.columns:
                norm_df[f_name] = norm_df[f_name] - baseline.center_v

        for f_name in self.CURRENT_LEVEL_FEATURES:
            if f_name in norm_df.columns:
                norm_df[f_name] = norm_df[f_name] - baseline.center_i

        return norm_df
