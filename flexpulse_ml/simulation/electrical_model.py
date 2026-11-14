"""
Mathematical resistive power-path electrical model and simulation data structures.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimulationObservations:
    """Observed telemetry signals available to the machine learning pipeline."""
    timestamp_s: np.ndarray
    source_voltage_v: np.ndarray
    voltage_measured_v: np.ndarray
    current_measured_a: np.ndarray
    load_resistance_ohm: np.ndarray

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp_s": self.timestamp_s,
                "source_voltage_v": self.source_voltage_v,
                "voltage_measured_v": self.voltage_measured_v,
                "current_measured_a": self.current_measured_a,
                "load_resistance_ohm": self.load_resistance_ohm,
            }
        )


@dataclass(frozen=True)
class SimulationGroundTruth:
    """Hidden physical truth variables used exclusively for evaluation."""
    timestamp_s: np.ndarray
    true_path_resistance_ohm: np.ndarray
    fault_label: np.ndarray
    fault_type: np.ndarray
    event_id: np.ndarray

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp_s": self.timestamp_s,
                "true_path_resistance_ohm": self.true_path_resistance_ohm,
                "fault_label": self.fault_label,
                "fault_type": self.fault_type,
                "event_id": self.event_id,
            }
        )


@dataclass(frozen=True)
class SimulationMetadata:
    """Provenance and configuration record for a simulation execution."""
    run_id: str
    scenario: str
    random_seed: int
    simulation_duration_s: float
    sampling_rate_hz: float
    configuration_used: Dict[str, Any] = field(default_factory=dict)
    synthetic_data: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SimulationResult:
    """Composite container holding separate observations, ground truth and metadata."""
    observations: SimulationObservations
    ground_truth: SimulationGroundTruth
    metadata: SimulationMetadata

    @property
    def sample_count(self) -> int:
        return len(self.observations.timestamp_s)


def validate_physical_parameters(
    source_voltage_v: np.ndarray,
    path_resistance_ohm: np.ndarray,
    load_resistance_ohm: np.ndarray,
) -> None:
    """Verify that electrical simulation inputs comply with physical boundaries."""
    if np.any(np.isnan(source_voltage_v)) or np.any(np.isinf(source_voltage_v)):
        raise ValueError("Source voltage contains NaN or infinite values.")
    if np.any(np.isnan(path_resistance_ohm)) or np.any(np.isinf(path_resistance_ohm)):
        raise ValueError("Path resistance contains NaN or infinite values.")
    if np.any(np.isnan(load_resistance_ohm)) or np.any(np.isinf(load_resistance_ohm)):
        raise ValueError("Load resistance contains NaN or infinite values.")

    if np.any(source_voltage_v <= 0.0):
        raise ValueError("Source voltage must be strictly positive.")
    if np.any(path_resistance_ohm < 0.0):
        raise ValueError("Path resistance cannot be negative.")
    if np.any(load_resistance_ohm <= 0.0):
        raise ValueError("Load resistance must be strictly positive to prevent division by zero.")


def compute_circuit_signals(
    source_voltage_v: np.ndarray,
    path_resistance_ohm: np.ndarray,
    load_resistance_ohm: np.ndarray,
    voltage_offset_v: np.ndarray,
    current_offset_a: np.ndarray,
    voltage_noise_v: np.ndarray,
    current_noise_a: np.ndarray,
    open_circuit_resistance_limit_ohm: float = 100000.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Execute vectorized electrical calculations for the resistive equivalent circuit."""
    validate_physical_parameters(source_voltage_v, path_resistance_ohm, load_resistance_ohm)

    capped_path_resistance = np.minimum(path_resistance_ohm, open_circuit_resistance_limit_ohm)
    total_circuit_resistance = load_resistance_ohm + capped_path_resistance

    true_current = source_voltage_v / total_circuit_resistance
    true_load_voltage = true_current * load_resistance_ohm

    measured_voltage = true_load_voltage + voltage_offset_v + voltage_noise_v
    measured_current = true_current + current_offset_a + current_noise_a

    return true_load_voltage, true_current, measured_voltage, measured_current
