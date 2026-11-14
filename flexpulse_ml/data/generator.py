"""
Synthetic dataset generator orchestrating independent simulation runs, splits and metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from flexpulse_ml.config import get_app_config
from flexpulse_ml.data.manifest import DatasetManifest
from flexpulse_ml.data.storage import DatasetStorage
from flexpulse_ml.seed import SeedManager
from flexpulse_ml.simulation.electrical_model import SimulationResult
from flexpulse_ml.simulation.fault_injector import (
    PowerPathSimulator,
    SimulationParams,
)


@dataclass
class DatasetGeneratorConfig:
    dataset_id: str = "flexpulse_dataset_001"
    root_seed: int = 42
    mode: str = "SCENARIO_BALANCED"
    runs_per_scenario: int = 20
    custom_scenario_counts: Dict[str, int] = field(default_factory=dict)
    ood_runs_per_scenario: int = 5
    train_runs_per_scenario: Optional[int] = None
    val_runs_per_scenario: Optional[int] = None
    id_test_runs_per_scenario: Optional[int] = None
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    allow_overwrite: bool = True


class SyntheticDatasetGenerator:
    """Orchestrates independent simulation runs and run-level partitioned storage."""

    MODE_BALANCED = "SCENARIO_BALANCED"
    MODE_IMBALANCED = "SCENARIO_IMBALANCED"

    def __init__(
        self,
        config: Optional[DatasetGeneratorConfig] = None,
        storage_base_dir: Optional[Path] = None,
    ) -> None:
        self.config = config or DatasetGeneratorConfig()
        self.storage = DatasetStorage(dataset_id=self.config.dataset_id, base_dir=storage_base_dir)
        self.seed_manager = SeedManager(master_seed=self.config.root_seed)
        self.simulator = PowerPathSimulator()

    def _determine_scenario_counts(self) -> Dict[str, int]:
        all_scenarios = PowerPathSimulator.ALL_SCENARIOS
        if self.config.mode.upper() == self.MODE_IMBALANCED and self.config.custom_scenario_counts:
            counts = {}
            for sc in all_scenarios:
                counts[sc] = int(self.config.custom_scenario_counts.get(sc, 0))
            return counts
        return {sc: int(self.config.runs_per_scenario) for sc in all_scenarios}

    def _partition_run_ids(
        self,
        scenario_run_map: Dict[str, List[str]],
    ) -> Dict[str, List[str]]:
        train_runs: List[str] = []
        val_runs: List[str] = []
        test_runs: List[str] = []

        cfg = self.config
        explicit_mode = (
            cfg.train_runs_per_scenario is not None
            and cfg.val_runs_per_scenario is not None
            and cfg.id_test_runs_per_scenario is not None
        )

        for scenario, run_ids in scenario_run_map.items():
            total = len(run_ids)
            if total == 0:
                continue

            if explicit_mode:
                n_train = int(cfg.train_runs_per_scenario)
                n_val = int(cfg.val_runs_per_scenario)
                n_test = int(cfg.id_test_runs_per_scenario)
                if n_train + n_val + n_test > total:
                    raise ValueError(
                        f"Explicit partition counts ({n_train} + {n_val} + {n_test} = {n_train + n_val + n_test}) "
                        f"exceed total available runs ({total}) for scenario {scenario}."
                    )
            elif total == 1:
                n_train, n_val, n_test = 1, 0, 0
            elif total == 2:
                n_train, n_val, n_test = 1, 1, 0
            elif total == 3:
                n_train, n_val, n_test = 1, 1, 1
            else:
                n_val = max(1, int(round(total * cfg.val_ratio)))
                n_test = max(1, int(round(total * cfg.test_ratio)))
                n_train = total - n_val - n_test

            train_runs.extend(run_ids[:n_train])
            val_runs.extend(run_ids[n_train : n_train + n_val])
            test_runs.extend(run_ids[n_train + n_val : n_train + n_val + n_test])

        return {
            DatasetStorage.SPLIT_TRAIN: sorted(train_runs),
            DatasetStorage.SPLIT_VAL: sorted(val_runs),
            DatasetStorage.SPLIT_ID_TEST: sorted(test_runs),
            DatasetStorage.SPLIT_OOD_TEST: [],
        }

    def _extract_and_check_events(
        self,
        result: SimulationResult,
    ) -> List[Dict[str, Any]]:
        scenario = result.metadata.scenario
        t = result.observations.timestamp_s
        cfg_used = result.metadata.configuration_used

        events_list: List[Dict[str, Any]] = []

        if scenario == PowerPathSimulator.SCENARIO_INCREASED_RESISTANCE:
            start_s = float(cfg_used.get("event_start_time_s", 10.0))
            end_s = float(cfg_used.get("event_end_time_s", 30.0))
            extra_r = float(cfg_used.get("fault_resistance_ohms", 0.250))
            is_sampled = bool(np.any((t >= start_s) & (t < end_s)))
            events_list.append(
                {
                    "event_id": 1,
                    "scenario": scenario,
                    "start_time_s": start_s,
                    "end_time_s": end_s,
                    "t_start_s": start_s,
                    "t_end_s": end_s,
                    "fault_type": scenario,
                    "configured_severity_ohm": extra_r,
                    "sampled_event": is_sampled,
                }
            )

        elif scenario == PowerPathSimulator.SCENARIO_INTERMITTENT_CONTACT:
            evs = cfg_used.get("events", [])
            sev_r = float(cfg_used.get("fault_resistance_ohms", 50.0))
            for ev in evs:
                st = float(ev.get("start_s", 0.0))
                en = float(ev.get("end_s", 0.0))
                eid = int(ev.get("event_id", 1))
                is_sampled = bool(np.any((t >= st) & (t < en)))
                events_list.append(
                    {
                        "event_id": eid,
                        "scenario": scenario,
                        "start_time_s": st,
                        "end_time_s": en,
                        "t_start_s": st,
                        "t_end_s": en,
                        "fault_type": scenario,
                        "configured_severity_ohm": sev_r,
                        "sampled_event": is_sampled,
                    }
                )

        elif scenario == PowerPathSimulator.SCENARIO_GRADUAL_RESISTANCE_DRIFT:
            st = float(cfg_used.get("drift_start_s", 7.0))
            en = float(cfg_used.get("drift_end_s", 55.0))
            rate = float(cfg_used.get("drift_rate_ohms_per_sec", 0.005))
            is_sampled = bool(np.any((t >= st) & (t < en)))
            events_list.append(
                {
                    "event_id": 1,
                    "scenario": scenario,
                    "start_time_s": st,
                    "end_time_s": en,
                    "t_start_s": st,
                    "t_end_s": en,
                    "fault_type": scenario,
                    "configured_severity_ohm": rate * max(0.0, en - st),
                    "sampled_event": is_sampled,
                }
            )

        return events_list

    def generate(self) -> DatasetManifest:
        self.storage.initialize_structure(allow_overwrite=self.config.allow_overwrite)

        scenario_counts = self._determine_scenario_counts()
        global_run_idx = 0

        nominal_run_results: Dict[str, Tuple[SimulationResult, List[Dict[str, Any]]]] = {}
        scenario_run_map: Dict[str, List[str]] = {sc: [] for sc in PowerPathSimulator.ALL_SCENARIOS}

        for scenario, count in scenario_counts.items():
            for i in range(count):
                global_run_idx += 1
                run_seed = self.seed_manager.spawn_run_seed(global_run_idx)
                run_id = f"run_nom_{scenario.lower()}_{i+1:03d}"

                result = self.simulator.run_simulation(
                    scenario=scenario,
                    run_id=run_id,
                    seed=run_seed,
                )
                events = self._extract_and_check_events(result)
                nominal_run_results[run_id] = (result, events)
                scenario_run_map[scenario].append(run_id)

        splits_dict = self._partition_run_ids(scenario_run_map)

        for split_name in [DatasetStorage.SPLIT_TRAIN, DatasetStorage.SPLIT_VAL, DatasetStorage.SPLIT_ID_TEST]:
            for r_id in splits_dict[split_name]:
                res, evs = nominal_run_results[r_id]
                self.storage.save_run(
                    split=split_name,
                    result=res,
                    events=evs,
                    condition_profile="nominal",
                )

        ood_run_ids: List[str] = []
        ood_counts = self.config.ood_runs_per_scenario

        ood_params = SimulationParams(
            source_voltage_nominal_v=5.2,
            baseline_path_resistance_nominal_ohms=0.180,
            load_resistance_nominal_ohms=1.80,
            measurement_noise_voltage_std_v=0.015,
            measurement_noise_current_std_a=0.025,
            voltage_offset_drift_v_per_sec=0.0004,
            current_offset_drift_a_per_sec=0.0002,
        )

        for scenario in PowerPathSimulator.ALL_SCENARIOS:
            for j in range(ood_counts):
                global_run_idx += 1
                run_seed = self.seed_manager.spawn_run_seed(global_run_idx)
                run_id = f"run_ood_{scenario.lower()}_{j+1:03d}"

                result = self.simulator.run_simulation(
                    scenario=scenario,
                    params=ood_params,
                    run_id=run_id,
                    seed=run_seed,
                )
                events = self._extract_and_check_events(result)
                self.storage.save_run(
                    split=DatasetStorage.SPLIT_OOD_TEST,
                    result=result,
                    events=events,
                    condition_profile="out_of_distribution_shift",
                )
                ood_run_ids.append(run_id)

        splits_dict[DatasetStorage.SPLIT_OOD_TEST] = sorted(ood_run_ids)
        self.storage.save_splits(splits_dict)

        manifest = self._build_manifest(splits_dict)
        manifest.file_checksums_sha256 = self.storage.compute_all_checksums()
        self.storage.save_manifest(manifest)

        return manifest

    def _build_manifest(self, splits_dict: Dict[str, List[str]]) -> DatasetManifest:
        total_runs = 0
        total_samples = 0
        total_fault = 0
        total_healthy = 0
        total_events = 0
        unsampled_events = 0

        runs_per_scenario: Dict[str, int] = {sc: 0 for sc in PowerPathSimulator.ALL_SCENARIOS}
        samples_per_scenario: Dict[str, int] = {sc: 0 for sc in PowerPathSimulator.ALL_SCENARIOS}
        sampling_rates: Dict[str, int] = {}
        durations: Dict[str, int] = {}
        condition_profiles: Dict[str, int] = {}
        parameter_ranges: Dict[str, Any] = {
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
                events = run_data["events"]

                sc = meta.get("scenario", "UNKNOWN")
                runs_per_scenario[sc] = runs_per_scenario.get(sc, 0) + 1

                n_s = len(obs_df)
                total_samples += n_s
                samples_per_scenario[sc] = samples_per_scenario.get(sc, 0) + n_s

                f_count = int(np.sum(gt_df["fault_label"] == 1))
                h_count = int(np.sum(gt_df["fault_label"] == 0))
                total_fault += f_count
                total_healthy += h_count

                sr_key = str(meta.get("sampling_frequency_hz", 100))
                sampling_rates[sr_key] = sampling_rates.get(sr_key, 0) + 1

                dur_key = str(meta.get("simulation_duration_s", 60))
                durations[dur_key] = durations.get(dur_key, 0) + 1

                cp_key = str(meta.get("condition_profile", "nominal"))
                condition_profiles[cp_key] = condition_profiles.get(cp_key, 0) + 1

                v_s = cond_df["configured_source_voltage_v"]
                r_l = cond_df["configured_load_resistance_ohm"]
                parameter_ranges["source_voltage_v"]["min"] = min(
                    parameter_ranges["source_voltage_v"]["min"], float(v_s.min())
                )
                parameter_ranges["source_voltage_v"]["max"] = max(
                    parameter_ranges["source_voltage_v"]["max"], float(v_s.max())
                )
                parameter_ranges["load_resistance_ohm"]["min"] = min(
                    parameter_ranges["load_resistance_ohm"]["min"], float(r_l.min())
                )
                parameter_ranges["load_resistance_ohm"]["max"] = max(
                    parameter_ranges["load_resistance_ohm"]["max"], float(r_l.max())
                )

                for ev in events:
                    total_events += 1
                    if not ev.get("sampled_event", True):
                        unsampled_events += 1

        binary_dist = {
            "healthy_ratio": float(total_healthy / total_samples) if total_samples > 0 else 0.0,
            "fault_ratio": float(total_fault / total_samples) if total_samples > 0 else 0.0,
        }

        return DatasetManifest(
            dataset_id=self.config.dataset_id,
            dataset_schema_version="1.0.0",
            simulator_version="0.2.0",
            root_random_seed=self.config.root_seed,
            total_simulation_runs=total_runs,
            total_samples=total_samples,
            total_fault_samples=total_fault,
            total_healthy_samples=total_healthy,
            run_counts_per_scenario=runs_per_scenario,
            sample_counts_per_scenario=samples_per_scenario,
            binary_class_distribution=binary_dist,
            fault_event_counts=total_events,
            unsampled_fault_event_counts=unsampled_events,
            sampling_rate_distribution=sampling_rates,
            simulation_duration_distribution=durations,
            condition_profiles=condition_profiles,
            parameter_ranges=parameter_ranges,
            split_assignments=splits_dict,
            synthetic_data=True,
        )


def generate_synthetic_dataset(
    dataset_id: str = "flexpulse_dataset_001",
    runs_per_scenario: int = 20,
    ood_runs_per_scenario: int = 5,
    train_runs_per_scenario: Optional[int] = None,
    val_runs_per_scenario: Optional[int] = None,
    id_test_runs_per_scenario: Optional[int] = None,
    root_seed: int = 42,
    mode: str = "SCENARIO_BALANCED",
    custom_scenario_counts: Optional[Dict[str, int]] = None,
    allow_overwrite: bool = True,
    base_dir: Optional[Path] = None,
) -> DatasetManifest:
    cfg = DatasetGeneratorConfig(
        dataset_id=dataset_id,
        root_seed=root_seed,
        mode=mode,
        runs_per_scenario=runs_per_scenario,
        custom_scenario_counts=custom_scenario_counts or {},
        ood_runs_per_scenario=ood_runs_per_scenario,
        train_runs_per_scenario=train_runs_per_scenario,
        val_runs_per_scenario=val_runs_per_scenario,
        id_test_runs_per_scenario=id_test_runs_per_scenario,
        allow_overwrite=allow_overwrite,
    )
    generator = SyntheticDatasetGenerator(config=cfg, storage_base_dir=base_dir)
    return generator.generate()
