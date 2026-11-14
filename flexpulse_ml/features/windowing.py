"""
Sliding window segmentation and ground-truth alignment for telemetry time-series.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from flexpulse_ml.config import get_app_config


@dataclass(frozen=True)
class WindowConfig:
    sampling_rate_hz: float = 100.0
    window_duration_s: float = 0.5
    overlap_fraction: float = 0.5
    min_samples_per_window: Optional[int] = None

    @property
    def window_size_samples(self) -> int:
        return int(round(self.sampling_rate_hz * self.window_duration_s))

    @property
    def stride_samples(self) -> int:
        stride = int(round(self.window_size_samples * (1.0 - self.overlap_fraction)))
        return max(1, stride)

    @property
    def effective_min_samples(self) -> int:
        if self.min_samples_per_window is not None:
            return min(self.min_samples_per_window, self.window_size_samples)
        return max(1, self.window_size_samples // 2)


@dataclass
class Window:
    run_id: str
    window_index: int
    start_idx: int
    end_idx: int
    start_time_s: float
    end_time_s: float
    voltage: np.ndarray
    current: np.ndarray
    fault_label: int
    fault_fraction: float
    scenario: str


class SlidingWindowGenerator:
    """Slices aligned observation and ground-truth arrays into overlapping windows."""

    def __init__(self, config: Optional[WindowConfig] = None) -> None:
        if config is not None:
            self.config = config
        else:
            app_cfg = get_app_config()
            f_cfg = app_cfg.features
            s_cfg = app_cfg.simulation
            self.config = WindowConfig(
                sampling_rate_hz=float(s_cfg.get("sampling_rate_hz", 100.0)),
                window_duration_s=float(f_cfg.get("window_length_seconds", 0.5)),
                overlap_fraction=float(f_cfg.get("overlap_fraction", 0.5)),
                min_samples_per_window=int(f_cfg.get("min_samples_per_window", 25)),
            )

    def slice_run(
        self,
        observations_df: pd.DataFrame,
        ground_truth_df: pd.DataFrame,
        run_id: str,
        scenario: str,
    ) -> List[Window]:
        n_obs = len(observations_df)
        n_gt = len(ground_truth_df)
        if n_obs != n_gt:
            raise ValueError(f"Length mismatch between observations ({n_obs}) and ground truth ({n_gt}) in run {run_id}.")

        w_size = self.config.window_size_samples
        stride = self.config.stride_samples
        min_samples = self.config.effective_min_samples

        t = observations_df["timestamp_s"].to_numpy()
        v = observations_df["voltage_measured_v"].to_numpy()
        i = observations_df["current_measured_a"].to_numpy()
        gt_labels = ground_truth_df["fault_label"].to_numpy()

        windows: List[Window] = []
        w_idx = 0

        for start_idx in range(0, n_obs, stride):
            end_idx = start_idx + w_size
            if end_idx > n_obs:
                break

            actual_len = end_idx - start_idx
            if actual_len < min_samples:
                continue

            v_win = v[start_idx:end_idx]
            i_win = i[start_idx:end_idx]
            gt_win = gt_labels[start_idx:end_idx]

            fault_count = int(np.sum(gt_win == 1))
            fault_fraction = float(fault_count / actual_len)
            window_fault_label = 1 if fault_count > 0 else 0

            start_t = float(t[start_idx])
            end_t = float(t[end_idx - 1])

            windows.append(
                Window(
                    run_id=run_id,
                    window_index=w_idx,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    start_time_s=start_t,
                    end_time_s=end_t,
                    voltage=v_win,
                    current=i_win,
                    fault_label=window_fault_label,
                    fault_fraction=fault_fraction,
                    scenario=scenario,
                )
            )
            w_idx += 1

        return windows
