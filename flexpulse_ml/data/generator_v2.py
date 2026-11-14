"""
Versioned synthetic dataset generator for FlexPulse-ML Phase 5.
Produces flexpulse_calibration_v2 with guaranteed 5-second healthy initialization,
randomized event timing, and decoupled development and robustness test partitions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from flexpulse_ml.data.manifest import DatasetManifest
from flexpulse_ml.data.storage import DatasetStorage
from flexpulse_ml.seed import SeedManager
from flexpulse_ml.simulation.fault_injector import (
    PowerPathSimulator,
    SimulationParams,
)


@dataclass
class CalibrationDatasetConfig:
    dataset_id: str = "flexpulse_calibration_v2"
    root_seed: int = 2026
    train_runs_per_scenario: int = 6
    dev_runs_per_scenario: int = 6
    id_test_runs_per_scenario: int = 3
    ood_test_runs_per_scenario: int = 6
    guaranteed_healthy_initial_duration_s: float = 5.0
    simulation_duration_s: float = 60.0
    sampling_rate_hz: float = 100.0


class Phase5DatasetGenerator:
    """Generates synthetic dataset with strict healthy initialization and randomized envelopes."""

    def __init__(
        self,
        config: Optional[CalibrationDatasetConfig] = None,
        base_dir: Optional[Path] = None,
    ) -> None:
        self.config = config or CalibrationDatasetConfig()
        self.storage = DatasetStorage(self.config.dataset_id, base_dir=base_dir)
        self.seed_manager = SeedManager(self.config.root_seed)
        self.simulator = PowerPathSimulator()

    def generate(self) -> DatasetManifest:
        self.storage.initialize_structure(allow_overwrite=True)

        splits_dict: Dict[str, List[str]] = {
            DatasetStorage.SPLIT_TRAIN: [],
            DatasetStorage.SPLIT_VAL: [],
            DatasetStorage.SPLIT_ID_TEST: [],
            DatasetStorage.SPLIT_OOD_TEST: [],
        }

        global_run_idx = 0

        for scenario in PowerPathSimulator.ALL_SCENARIOS:
            for i in range(self.config.train_runs_per_scenario):
                global_run_idx += 1
                seed = self.seed_manager.spawn_run_seed(global_run_idx)
                run_id = f"run_train_{scenario.lower()}_{i+1:03d}"
                overrides = self._sample_scenario_parameters(scenario, seed, is_nominal=True)
                params = self._create_nominal_params(seed)

                result = self.simulator.run_simulation(
                    scenario=scenario,
                    params=params,
                    run_id=run_id,
                    seed=seed,
                    **overrides,
                )
                events = self._extract_events(result, overrides)
                self.storage.save_run(
                    split=DatasetStorage.SPLIT_TRAIN,
                    result=result,
                    events=events,
                    condition_profile="nominal",
                )
                splits_dict[DatasetStorage.SPLIT_TRAIN].append(run_id)

        for scenario in PowerPathSimulator.ALL_SCENARIOS:
            for i in range(self.config.dev_runs_per_scenario):
                global_run_idx += 1
                seed = self.seed_manager.spawn_run_seed(global_run_idx)
                run_id = f"run_dev_{scenario.lower()}_{i+1:03d}"
                overrides = self._sample_scenario_parameters(scenario, seed, is_nominal=False)
                params = self._create_shifted_params(seed, shift_level="moderate")

                result = self.simulator.run_simulation(
                    scenario=scenario,
                    params=params,
                    run_id=run_id,
                    seed=seed,
                    **overrides,
                )
                events = self._extract_events(result, overrides)
                self.storage.save_run(
                    split=DatasetStorage.SPLIT_VAL,
                    result=result,
                    events=events,
                    condition_profile="development_shift",
                )
                splits_dict[DatasetStorage.SPLIT_VAL].append(run_id)

        for scenario in PowerPathSimulator.ALL_SCENARIOS:
            for i in range(self.config.id_test_runs_per_scenario):
                global_run_idx += 1
                seed = self.seed_manager.spawn_run_seed(global_run_idx)
                run_id = f"run_idtest_{scenario.lower()}_{i+1:03d}"
                overrides = self._sample_scenario_parameters(scenario, seed, is_nominal=True)
                params = self._create_nominal_params(seed)

                result = self.simulator.run_simulation(
                    scenario=scenario,
                    params=params,
                    run_id=run_id,
                    seed=seed,
                    **overrides,
                )
                events = self._extract_events(result, overrides)
                self.storage.save_run(
                    split=DatasetStorage.SPLIT_ID_TEST,
                    result=result,
                    events=events,
                    condition_profile="nominal_held_out",
                )
                splits_dict[DatasetStorage.SPLIT_ID_TEST].append(run_id)

        for scenario in PowerPathSimulator.ALL_SCENARIOS:
            for i in range(self.config.ood_test_runs_per_scenario):
                global_run_idx += 1
                seed = self.seed_manager.spawn_run_seed(global_run_idx)
                run_id = f"run_oodtest_{scenario.lower()}_{i+1:03d}"
                overrides = self._sample_scenario_parameters(scenario, seed, is_nominal=False)
                params = self._create_shifted_params(seed, shift_level="severe")

                result = self.simulator.run_simulation(
                    scenario=scenario,
                    params=params,
                    run_id=run_id,
                    seed=seed,
                    **overrides,
                )
                events = self._extract_events(result, overrides)
                self.storage.save_run(
                    split=DatasetStorage.SPLIT_OOD_TEST,
                    result=result,
                    events=events,
                    condition_profile="final_held_out_robustness_test",
                )
                splits_dict[DatasetStorage.SPLIT_OOD_TEST].append(run_id)

        for k in splits_dict:
            splits_dict[k] = sorted(splits_dict[k])
        self.storage.save_splits(splits_dict)

        manifest = self._build_manifest(splits_dict)
        manifest.file_checksums_sha256 = self.storage.compute_all_checksums()
        self.storage.save_manifest(manifest)

        return manifest

    def _create_nominal_params(self, seed: int) -> SimulationParams:
        rng = np.random.default_rng(seed)
        return SimulationParams(
            sampling_rate_hz=self.config.sampling_rate_hz,
            duration_seconds=self.config.simulation_duration_s,
            calibration_duration_seconds=self.config.guaranteed_healthy_initial_duration_s,
            source_voltage_nominal_v=5.0 + float(rng.uniform(-0.02, 0.02)),
            baseline_path_resistance_nominal_ohms=0.080 + float(rng.uniform(-0.005, 0.005)),
            load_resistance_nominal_ohms=3.333 + float(rng.uniform(-0.1, 0.1)),
            measurement_noise_voltage_std_v=0.005,
            measurement_noise_current_std_a=0.008,
            voltage_offset_v=0.001,
            current_offset_a=0.001,
            voltage_offset_drift_v_per_sec=0.0001,
            current_offset_drift_a_per_sec=0.00005,
            random_seed=seed,
        )

    def _create_shifted_params(self, seed: int, shift_level: str) -> SimulationParams:
        rng = np.random.default_rng(seed)
        if shift_level == "moderate":
            r0 = float(rng.uniform(0.100, 0.150))
            r_load = float(rng.uniform(2.2, 4.0))
            v_s = float(rng.uniform(4.90, 5.15))
            noise_v = float(rng.uniform(0.007, 0.012))
            noise_i = float(rng.uniform(0.010, 0.018))
            v_drift = float(rng.uniform(0.0001, 0.0003))
            i_drift = float(rng.uniform(0.00005, 0.00015))
        else:
            r0 = float(rng.uniform(0.140, 0.220))
            r_load = float(rng.uniform(1.6, 2.5))
            v_s = float(rng.uniform(4.85, 5.25))
            noise_v = float(rng.uniform(0.012, 0.020))
            noise_i = float(rng.uniform(0.020, 0.035))
            v_drift = float(rng.uniform(0.0002, 0.0005))
            i_drift = float(rng.uniform(0.0001, 0.00025))

        return SimulationParams(
            sampling_rate_hz=self.config.sampling_rate_hz,
            duration_seconds=self.config.simulation_duration_s,
            calibration_duration_seconds=self.config.guaranteed_healthy_initial_duration_s,
            source_voltage_nominal_v=v_s,
            baseline_path_resistance_nominal_ohms=r0,
            load_resistance_nominal_ohms=r_load,
            measurement_noise_voltage_std_v=noise_v,
            measurement_noise_current_std_a=noise_i,
            voltage_offset_v=float(rng.uniform(-0.005, 0.005)),
            current_offset_a=float(rng.uniform(-0.005, 0.005)),
            voltage_offset_drift_v_per_sec=v_drift,
            current_offset_drift_a_per_sec=i_drift,
            random_seed=seed,
        )

    def _sample_scenario_parameters(
        self,
        scenario: str,
        seed: int,
        is_nominal: bool,
    ) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        min_start = self.config.guaranteed_healthy_initial_duration_s + 1.5

        if scenario == PowerPathSimulator.SCENARIO_INCREASED_RESISTANCE:
            start_s = float(rng.uniform(min_start, 25.0))
            duration_s = float(rng.uniform(8.0, 25.0))
            res_val = float(rng.uniform(0.150, 0.350)) if is_nominal else float(rng.uniform(0.100, 0.500))
            return {
                "fault_start_time_s": start_s,
                "fault_duration_s": duration_s,
                "fault_resistance_ohms": res_val,
            }

        elif scenario == PowerPathSimulator.SCENARIO_INTERMITTENT_CONTACT:
            ev_count = int(rng.integers(3, 7))
            duration_s = float(rng.uniform(0.10, 0.30))
            res_val = float(rng.uniform(30.0, 80.0))
            usable_dur = self.config.simulation_duration_s - min_start - 3.0
            step = usable_dur / ev_count
            start_times = [float(min_start + k * step + rng.uniform(0.2, step * 0.7)) for k in range(ev_count)]
            return {
                "event_count": ev_count,
                "event_duration_s": duration_s,
                "fault_resistance_ohms": res_val,
                "event_start_times_s": start_times,
            }

        elif scenario == PowerPathSimulator.SCENARIO_GRADUAL_RESISTANCE_DRIFT:
            drift_start = float(rng.uniform(min_start, 15.0))
            drift_end = float(rng.uniform(40.0, 56.0))
            drift_rate = float(rng.uniform(0.003, 0.008)) if is_nominal else float(rng.uniform(0.002, 0.012))
            return {
                "drift_start_s": drift_start,
                "drift_end_s": drift_end,
                "drift_rate_ohms_per_sec": drift_rate,
            }

        elif scenario == PowerPathSimulator.SCENARIO_MEASUREMENT_OFFSET_DRIFT:
            v_drift = float(rng.uniform(0.001, 0.003)) if is_nominal else float(rng.uniform(0.0015, 0.005))
            i_drift = float(rng.uniform(0.0005, 0.0015)) if is_nominal else float(rng.uniform(0.0008, 0.0025))
            return {
                "voltage_offset_drift_v_per_sec": v_drift,
                "current_offset_drift_a_per_sec": i_drift,
            }

        elif scenario == PowerPathSimulator.SCENARIO_OPERATING_CONDITION_SHIFT:
            shift_time = float(rng.uniform(min_start, 20.0))
            new_load = float(rng.uniform(1.5, 2.5))
            noise_mult = float(rng.uniform(1.5, 3.5))
            return {
                "shift_time_s": shift_time,
                "shifted_load_ohms": new_load,
                "noise_multiplier": noise_mult,
            }

        else:
            return {}

    def _extract_events(
        self,
        result: Any,
        overrides: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        scenario = result.metadata.scenario
        t = result.observations.timestamp_s
        events_list: List[Dict[str, Any]] = []

        if scenario == PowerPathSimulator.SCENARIO_INCREASED_RESISTANCE:
            st = float(overrides.get("fault_start_time_s", 10.0))
            dur = float(overrides.get("fault_duration_s", 20.0))
            en = min(self.config.simulation_duration_s, st + dur)
            res_val = float(overrides.get("fault_resistance_ohms", 0.250))
            is_sampled = bool(np.any((t >= st) & (t < en)))
            events_list.append({
                "event_id": 1,
                "scenario": scenario,
                "start_time_s": st,
                "end_time_s": en,
                "t_start_s": st,
                "t_end_s": en,
                "fault_type": scenario,
                "configured_severity_ohm": res_val,
                "sampled_event": is_sampled,
            })

        elif scenario == PowerPathSimulator.SCENARIO_INTERMITTENT_CONTACT:
            dur = float(overrides.get("event_duration_s", 0.15))
            sev = float(overrides.get("fault_resistance_ohms", 50.0))
            sts = overrides.get("event_start_times_s", [])
            for idx, st_val in enumerate(sts, start=1):
                en_val = min(self.config.simulation_duration_s, st_val + dur)
                is_sampled = bool(np.any((t >= st_val) & (t < en_val)))
                events_list.append({
                    "event_id": idx,
                    "scenario": scenario,
                    "start_time_s": st_val,
                    "end_time_s": en_val,
                    "t_start_s": st_val,
                    "t_end_s": en_val,
                    "fault_type": scenario,
                    "configured_severity_ohm": sev,
                    "sampled_event": is_sampled,
                })

        elif scenario == PowerPathSimulator.SCENARIO_GRADUAL_RESISTANCE_DRIFT:
            st = float(overrides.get("drift_start_s", 7.0))
            en = float(overrides.get("drift_end_s", 55.0))
            rate = float(overrides.get("drift_rate_ohms_per_sec", 0.005))
            is_sampled = bool(np.any(t >= st))
            events_list.append({
                "event_id": 1,
                "scenario": scenario,
                "start_time_s": st,
                "end_time_s": en,
                "t_start_s": st,
                "t_end_s": en,
                "fault_type": scenario,
                "configured_severity_ohm": rate * (en - st),
                "sampled_event": is_sampled,
            })

        return events_list

    def _build_manifest(self, splits_dict: Dict[str, List[str]]) -> DatasetManifest:
        total_runs = 0
        total_samples = 0
        total_fault = 0
        total_healthy = 0
        runs_per_scenario: Dict[str, int] = {sc: 0 for sc in PowerPathSimulator.ALL_SCENARIOS}
        samples_per_scenario: Dict[str, int] = {sc: 0 for sc in PowerPathSimulator.ALL_SCENARIOS}
        sampling_rates: Dict[str, int] = {}
        durations: Dict[str, int] = {}
        condition_profiles: Dict[str, int] = {}
        param_ranges = {
            "source_voltage_v": {"min": float("inf"), "max": float("-inf")},
            "load_resistance_ohm": {"min": float("inf"), "max": float("-inf")},
        }

        for split, r_ids in splits_dict.items():
            for r_id in r_ids:
                total_runs += 1
                run_data = self.storage.load_run(split, r_id)
                obs_df = run_data["observations"]
                gt_df = run_data["ground_truth"]
                cond_df = run_data["conditions"]
                meta = run_data["metadata"]

                sc = meta.get("scenario", "UNKNOWN")
                runs_per_scenario[sc] = runs_per_scenario.get(sc, 0) + 1
                n_s = len(obs_df)
                total_samples += n_s
                samples_per_scenario[sc] = samples_per_scenario.get(sc, 0) + n_s

                f_c = int(np.sum(gt_df["fault_label"] == 1))
                h_c = int(np.sum(gt_df["fault_label"] == 0))
                total_fault += f_c
                total_healthy += h_c

                sr_key = str(meta.get("sampling_frequency_hz", 100))
                sampling_rates[sr_key] = sampling_rates.get(sr_key, 0) + 1
                dur_key = str(meta.get("simulation_duration_s", 60))
                durations[dur_key] = durations.get(dur_key, 0) + 1
                cp_key = str(meta.get("condition_profile", "nominal"))
                condition_profiles[cp_key] = condition_profiles.get(cp_key, 0) + 1

                v_s = cond_df["configured_source_voltage_v"]
                r_l = cond_df["configured_load_resistance_ohm"]
                param_ranges["source_voltage_v"]["min"] = min(param_ranges["source_voltage_v"]["min"], float(v_s.min()))
                param_ranges["source_voltage_v"]["max"] = max(param_ranges["source_voltage_v"]["max"], float(v_s.max()))
                param_ranges["load_resistance_ohm"]["min"] = min(param_ranges["load_resistance_ohm"]["min"], float(r_l.min()))
                param_ranges["load_resistance_ohm"]["max"] = max(param_ranges["load_resistance_ohm"]["max"], float(r_l.max()))

        return DatasetManifest(
            dataset_id=self.config.dataset_id,
            dataset_schema_version="2.0.0",
            root_random_seed=self.config.root_seed,
            total_simulation_runs=total_runs,
            total_samples=total_samples,
            total_fault_samples=total_fault,
            total_healthy_samples=total_healthy,
            run_counts_per_scenario=runs_per_scenario,
            sample_counts_per_scenario=samples_per_scenario,
            binary_class_distribution={
                "healthy_ratio": float(total_healthy / total_samples) if total_samples > 0 else 0.0,
                "fault_ratio": float(total_fault / total_samples) if total_samples > 0 else 0.0,
            },
            sampling_rate_distribution=sampling_rates,
            simulation_duration_distribution=durations,
            condition_profiles=condition_profiles,
            parameter_ranges=param_ranges,
            split_assignments=splits_dict,
            synthetic_data=True,
        )


def generate_calibration_dataset(
    config: Optional[CalibrationDatasetConfig] = None,
) -> DatasetManifest:
    generator = Phase5DatasetGenerator(config)
    return generator.generate()
