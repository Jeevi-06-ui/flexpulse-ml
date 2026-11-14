"""
Simulation orchestrator and fault injection engine for USB-C-inspired power paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import uuid
from typing import Any, Dict, List, Optional
import numpy as np

from flexpulse_ml.config import AppConfig, get_app_config
from flexpulse_ml.simulation.electrical_model import (
    SimulationGroundTruth,
    SimulationMetadata,
    SimulationObservations,
    SimulationResult,
    compute_circuit_signals,
)


@dataclass(frozen=True)
class SimulationParams:
    """Configurable parameter set for a simulation execution."""
    sampling_rate_hz: float = 100.0
    duration_seconds: float = 60.0
    calibration_duration_seconds: float = 5.0
    source_voltage_nominal_v: float = 5.0
    baseline_path_resistance_nominal_ohms: float = 0.080
    load_resistance_nominal_ohms: float = 3.333
    measurement_noise_voltage_std_v: float = 0.005
    measurement_noise_current_std_a: float = 0.008
    voltage_offset_v: float = 0.001
    current_offset_a: float = 0.001
    voltage_offset_drift_v_per_sec: float = 0.0001
    current_offset_drift_a_per_sec: float = 0.00005
    random_seed: int = 42

    def validate(self) -> None:
        if self.sampling_rate_hz <= 0.0:
            raise ValueError("Sampling rate must be strictly positive.")
        if self.duration_seconds <= 0.0:
            raise ValueError("Duration must be strictly positive.")
        if self.calibration_duration_seconds < 0.0 or self.calibration_duration_seconds >= self.duration_seconds:
            raise ValueError("Calibration duration must be non-negative and less than total simulation duration.")
        if self.source_voltage_nominal_v <= 0.0:
            raise ValueError("Nominal source voltage must be positive.")
        if self.baseline_path_resistance_nominal_ohms < 0.0:
            raise ValueError("Baseline path resistance cannot be negative.")
        if self.load_resistance_nominal_ohms <= 0.0:
            raise ValueError("Nominal load resistance must be strictly positive.")
        if self.measurement_noise_voltage_std_v < 0.0 or self.measurement_noise_current_std_a < 0.0:
            raise ValueError("Measurement noise standard deviation cannot be negative.")


class PowerPathSimulator:
    """Generates synthetic electrical telemetry and ground truth across configured scenarios."""

    SCENARIO_HEALTHY = "HEALTHY"
    SCENARIO_INCREASED_RESISTANCE = "INCREASED_RESISTANCE"
    SCENARIO_INTERMITTENT_CONTACT = "INTERMITTENT_CONTACT"
    SCENARIO_GRADUAL_RESISTANCE_DRIFT = "GRADUAL_RESISTANCE_DRIFT"
    SCENARIO_MEASUREMENT_OFFSET_DRIFT = "MEASUREMENT_OFFSET_DRIFT"
    SCENARIO_OPERATING_CONDITION_SHIFT = "OPERATING_CONDITION_SHIFT"

    ALL_SCENARIOS = (
        SCENARIO_HEALTHY,
        SCENARIO_INCREASED_RESISTANCE,
        SCENARIO_INTERMITTENT_CONTACT,
        SCENARIO_GRADUAL_RESISTANCE_DRIFT,
        SCENARIO_MEASUREMENT_OFFSET_DRIFT,
        SCENARIO_OPERATING_CONDITION_SHIFT,
    )

    def __init__(self, default_params: Optional[SimulationParams] = None) -> None:
        self._default_params = default_params or self._load_params_from_app_config()
        self._default_params.validate()

    @classmethod
    def _load_params_from_app_config(cls) -> SimulationParams:
        cfg = get_app_config()
        sim_cfg = cfg.simulation
        return SimulationParams(
            sampling_rate_hz=sim_cfg.sampling_rate_hz,
            duration_seconds=sim_cfg.duration_seconds,
            calibration_duration_seconds=sim_cfg.calibration_duration_seconds,
            source_voltage_nominal_v=sim_cfg.source_voltage_nominal_v,
            baseline_path_resistance_nominal_ohms=sim_cfg.baseline_path_resistance_nominal_ohms,
            load_resistance_nominal_ohms=sim_cfg.load_resistance_nominal_ohms,
            measurement_noise_voltage_std_v=sim_cfg.measurement_noise_voltage_std_v,
            measurement_noise_current_std_a=sim_cfg.measurement_noise_current_std_a,
            voltage_offset_v=sim_cfg.voltage_offset_v,
            current_offset_a=sim_cfg.current_offset_a,
            voltage_offset_drift_v_per_sec=sim_cfg.voltage_offset_drift_v_per_sec,
            current_offset_drift_a_per_sec=sim_cfg.current_offset_drift_a_per_sec,
            random_seed=cfg.random_seed,
        )

    def run_simulation(
        self,
        scenario: str,
        params: Optional[SimulationParams] = None,
        run_id: Optional[str] = None,
        seed: Optional[int] = None,
        **scenario_overrides: Any,
    ) -> SimulationResult:
        scenario_upper = scenario.upper()
        if scenario_upper not in self.ALL_SCENARIOS:
            raise ValueError(f"Unknown scenario '{scenario}'. Supported: {self.ALL_SCENARIOS}")

        p = params or self._default_params
        p.validate()

        effective_seed = p.random_seed if seed is None else int(seed)
        effective_run_id = run_id or f"RUN_{scenario_upper}_{uuid.uuid4().hex[:8]}"
        rng = np.random.default_rng(effective_seed)

        sample_count = int(np.round(p.duration_seconds * p.sampling_rate_hz))
        time_array = np.linspace(0.0, p.duration_seconds, sample_count, endpoint=False)

        source_voltage = np.full(sample_count, p.source_voltage_nominal_v)
        load_resistance = np.full(sample_count, p.load_resistance_nominal_ohms)
        true_path_resistance = np.full(sample_count, p.baseline_path_resistance_nominal_ohms)

        voltage_offset = p.voltage_offset_v + (p.voltage_offset_drift_v_per_sec * time_array)
        current_offset = p.current_offset_a + (p.current_offset_drift_a_per_sec * time_array)

        voltage_noise_std = p.measurement_noise_voltage_std_v
        current_noise_std = p.measurement_noise_current_std_a

        fault_label = np.zeros(sample_count, dtype=np.int32)
        fault_type = np.full(sample_count, "HEALTHY", dtype=object)
        event_id = np.zeros(sample_count, dtype=np.int32)

        meta_info: Dict[str, Any] = {
            "scenario": scenario_upper,
            "seed": effective_seed,
            "sample_count": sample_count,
            "overrides": scenario_overrides,
        }

        if scenario_upper == self.SCENARIO_HEALTHY:
            load_variation_steps = [15.0, 30.0, 45.0]
            for step_time in load_variation_steps:
                mask = time_array >= step_time
                load_resistance[mask] = p.load_resistance_nominal_ohms * (
                    1.0 + 0.15 * np.sin(time_array[mask])
                )

        elif scenario_upper == self.SCENARIO_INCREASED_RESISTANCE:
            start_time = float(scenario_overrides.get("fault_start_time_s", p.calibration_duration_seconds + 5.0))
            duration = float(scenario_overrides.get("fault_duration_s", 20.0))
            extra_r = float(scenario_overrides.get("fault_resistance_ohms", 0.250))
            end_time = min(p.duration_seconds, start_time + duration)

            if start_time < 0 or start_time >= p.duration_seconds or end_time <= start_time:
                raise ValueError("Invalid fault event start time or duration.")

            event_mask = (time_array >= start_time) & (time_array < end_time)
            true_path_resistance[event_mask] += extra_r
            fault_label[event_mask] = 1
            fault_type[event_mask] = self.SCENARIO_INCREASED_RESISTANCE
            event_id[event_mask] = 1

            meta_info["event_start_time_s"] = start_time
            meta_info["event_end_time_s"] = end_time
            meta_info["fault_resistance_ohms"] = extra_r

        elif scenario_upper == self.SCENARIO_INTERMITTENT_CONTACT:
            event_count = int(scenario_overrides.get("event_count", 4))
            event_duration = float(scenario_overrides.get("event_duration_s", 0.15))
            severity_r = float(scenario_overrides.get("fault_resistance_ohms", 50.0))
            event_start_times = scenario_overrides.get("event_start_times_s")

            if event_start_times is None:
                cal_end = p.calibration_duration_seconds + 2.0
                usable_duration = p.duration_seconds - cal_end - 2.0
                spacing = usable_duration / max(1, event_count)
                event_start_times = [cal_end + i * spacing for i in range(event_count)]

            meta_events: List[Dict[str, float]] = []
            for ev_idx, ev_start in enumerate(event_start_times, start=1):
                ev_end = min(p.duration_seconds, ev_start + event_duration)
                if ev_start < 0 or ev_start >= p.duration_seconds or ev_end <= ev_start:
                    continue
                ev_mask = (time_array >= ev_start) & (time_array < ev_end)
                true_path_resistance[ev_mask] += severity_r
                fault_label[ev_mask] = 1
                fault_type[ev_mask] = self.SCENARIO_INTERMITTENT_CONTACT
                event_id[ev_mask] = ev_idx
                meta_events.append({"event_id": ev_idx, "start_s": ev_start, "end_s": ev_end})

            meta_info["events"] = meta_events
            meta_info["fault_resistance_ohms"] = severity_r

        elif scenario_upper == self.SCENARIO_GRADUAL_RESISTANCE_DRIFT:
            drift_start = float(scenario_overrides.get("drift_start_s", p.calibration_duration_seconds + 2.0))
            drift_end = float(scenario_overrides.get("drift_end_s", p.duration_seconds - 5.0))
            drift_rate = float(scenario_overrides.get("drift_rate_ohms_per_sec", 0.005))

            if drift_start < 0 or drift_start >= p.duration_seconds or drift_end <= drift_start:
                raise ValueError("Invalid gradual drift start or end time.")

            drift_mask = (time_array >= drift_start) & (time_array < drift_end)
            post_drift_mask = time_array >= drift_end

            elapsed = time_array[drift_mask] - drift_start
            true_path_resistance[drift_mask] += drift_rate * elapsed
            fault_label[drift_mask] = 1
            fault_type[drift_mask] = self.SCENARIO_GRADUAL_RESISTANCE_DRIFT
            event_id[drift_mask] = 1

            if np.any(post_drift_mask):
                max_drift = drift_rate * (drift_end - drift_start)
                true_path_resistance[post_drift_mask] += max_drift
                fault_label[post_drift_mask] = 1
                fault_type[post_drift_mask] = self.SCENARIO_GRADUAL_RESISTANCE_DRIFT
                event_id[post_drift_mask] = 1

            meta_info["drift_start_s"] = drift_start
            meta_info["drift_end_s"] = drift_end
            meta_info["drift_rate_ohms_per_sec"] = drift_rate

        elif scenario_upper == self.SCENARIO_MEASUREMENT_OFFSET_DRIFT:
            v_drift_rate = float(scenario_overrides.get("voltage_offset_drift_v_per_sec", 0.002))
            i_drift_rate = float(scenario_overrides.get("current_offset_drift_a_per_sec", 0.001))
            voltage_offset = p.voltage_offset_v + (v_drift_rate * time_array)
            current_offset = p.current_offset_a + (i_drift_rate * time_array)

            meta_info["voltage_drift_rate_v_per_sec"] = v_drift_rate
            meta_info["current_drift_rate_a_per_sec"] = i_drift_rate

        elif scenario_upper == self.SCENARIO_OPERATING_CONDITION_SHIFT:
            shift_time = float(scenario_overrides.get("shift_time_s", p.calibration_duration_seconds + 10.0))
            new_load = float(scenario_overrides.get("shifted_load_ohms", p.load_resistance_nominal_ohms * 0.5))
            noise_factor = float(scenario_overrides.get("noise_multiplier", 3.0))

            shift_mask = time_array >= shift_time
            load_resistance[shift_mask] = new_load

            noise_vector_v = np.where(shift_mask, voltage_noise_std * noise_factor, voltage_noise_std)
            noise_vector_i = np.where(shift_mask, current_noise_std * noise_factor, current_noise_std)

            voltage_noise = rng.normal(0.0, noise_vector_v)
            current_noise = rng.normal(0.0, noise_vector_i)

            meta_info["shift_time_s"] = shift_time
            meta_info["shifted_load_ohms"] = new_load
            meta_info["noise_multiplier"] = noise_factor
        else:
            raise ValueError(f"Unhandled scenario: {scenario_upper}")

        if scenario_upper != self.SCENARIO_OPERATING_CONDITION_SHIFT:
            voltage_noise = rng.normal(0.0, voltage_noise_std, size=sample_count)
            current_noise = rng.normal(0.0, current_noise_std, size=sample_count)

        true_v_load, true_i, measured_v, measured_i = compute_circuit_signals(
            source_voltage_v=source_voltage,
            path_resistance_ohm=true_path_resistance,
            load_resistance_ohm=load_resistance,
            voltage_offset_v=voltage_offset,
            current_offset_a=current_offset,
            voltage_noise_v=voltage_noise,
            current_noise_a=current_noise,
        )

        obs = SimulationObservations(
            timestamp_s=time_array,
            source_voltage_v=source_voltage,
            voltage_measured_v=measured_v,
            current_measured_a=measured_i,
            load_resistance_ohm=load_resistance,
        )

        gt = SimulationGroundTruth(
            timestamp_s=time_array,
            true_path_resistance_ohm=true_path_resistance,
            fault_label=fault_label,
            fault_type=fault_type,
            event_id=event_id,
        )

        meta = SimulationMetadata(
            run_id=effective_run_id,
            scenario=scenario_upper,
            random_seed=effective_seed,
            simulation_duration_s=p.duration_seconds,
            sampling_rate_hz=p.sampling_rate_hz,
            configuration_used=meta_info,
            synthetic_data=True,
        )

        return SimulationResult(observations=obs, ground_truth=gt, metadata=meta)
