from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from flexpulse_ml.data.manifest import DatasetManifest, compute_file_sha256
from flexpulse_ml.data.storage import DatasetStorage
from flexpulse_ml.seed import SeedManager
from flexpulse_ml.simulation.electrical_model import (
    SimulationGroundTruth,
    SimulationMetadata,
    SimulationObservations,
    SimulationResult,
    compute_circuit_signals,
)
from flexpulse_ml.simulation.fault_injector import PowerPathSimulator


@dataclass
class ConfirmatoryDatasetConfig:
    dataset_id: str = "flexpulse_final_test_v5"
    root_seed: int = 20261202
    runs_per_scenario_ood: int = 5
    runs_per_scenario_id: int = 2
    runs_per_scenario_nominal: int = 1
    simulation_duration_s: float = 60.0
    sampling_rate_hz: float = 100.0
    guaranteed_healthy_initial_duration_s: float = 5.0


class ConfirmatoryDatasetGenerator:

    SCENARIO_S1_HEALTHY = "HEALTHY"
    SCENARIO_S2_INCREASED_RESISTANCE = "INCREASED_RESISTANCE"
    SCENARIO_S3_INTERMITTENT_CONTACT = "INTERMITTENT_CONTACT"
    SCENARIO_S4_GRADUAL_RESISTANCE_DRIFT = "GRADUAL_RESISTANCE_DRIFT"
    SCENARIO_S5_MEASUREMENT_OFFSET_DRIFT = "MEASUREMENT_OFFSET_DRIFT"
    SCENARIO_S6_MEASUREMENT_NOISE_SHIFT = "MEASUREMENT_NOISE_SHIFT"
    SCENARIO_S7_LOAD_OPERATING_SHIFT = "LOAD_OPERATING_SHIFT"
    SCENARIO_S8_OPERATING_CONDITION_SHIFT = "OPERATING_CONDITION_SHIFT"
    SCENARIO_S9_SLOW_DEVELOPING_FAULT = "SLOW_DEVELOPING_FAULT"
    SCENARIO_S10_COMBINED_SHIFTS = "COMBINED_OPERATING_SHIFTS"

    ID_SCENARIOS = [
        SCENARIO_S1_HEALTHY,
        SCENARIO_S2_INCREASED_RESISTANCE,
        SCENARIO_S3_INTERMITTENT_CONTACT,
        SCENARIO_S4_GRADUAL_RESISTANCE_DRIFT,
        SCENARIO_S5_MEASUREMENT_OFFSET_DRIFT,
        SCENARIO_S6_MEASUREMENT_NOISE_SHIFT,
        SCENARIO_S7_LOAD_OPERATING_SHIFT,
        SCENARIO_S8_OPERATING_CONDITION_SHIFT,
        SCENARIO_S9_SLOW_DEVELOPING_FAULT,
        SCENARIO_S10_COMBINED_SHIFTS,
    ]

    OOD_SCENARIO_MAP = [
        ("HEALTHY_NOMINAL", SCENARIO_S1_HEALTHY),
        ("INCREASED_PATH_RESISTANCE", SCENARIO_S2_INCREASED_RESISTANCE),
        ("INTERMITTENT_CONTACT_FAULTS", SCENARIO_S3_INTERMITTENT_CONTACT),
        ("GRADUAL_RESISTANCE_DRIFT", SCENARIO_S4_GRADUAL_RESISTANCE_DRIFT),
        ("MEASUREMENT_OFFSET_DRIFT", SCENARIO_S5_MEASUREMENT_OFFSET_DRIFT),
        ("MEASUREMENT_NOISE_SHIFT", SCENARIO_S6_MEASUREMENT_NOISE_SHIFT),
        ("LOAD_OPERATING_SHIFT", SCENARIO_S7_LOAD_OPERATING_SHIFT),
        ("ABRUPT_LOAD_CHANGE", SCENARIO_S8_OPERATING_CONDITION_SHIFT),
        ("SLOW_DEVELOPING_FAULT", SCENARIO_S9_SLOW_DEVELOPING_FAULT),
        ("COMBINED_OPERATING_SHIFTS", SCENARIO_S10_COMBINED_SHIFTS),
    ]

    def __init__(
        self,
        config: Optional[ConfirmatoryDatasetConfig] = None,
        base_dir: Optional[Path] = None,
    ) -> None:
        self.config = config or ConfirmatoryDatasetConfig()
        self.storage = DatasetStorage(self.config.dataset_id, base_dir=base_dir)
        self.seed_manager = SeedManager(self.config.root_seed)

    def generate(self) -> DatasetManifest:
        self.storage.initialize_structure(allow_overwrite=True)

        splits_dict: Dict[str, List[str]] = {
            DatasetStorage.SPLIT_TRAIN: [],
            DatasetStorage.SPLIT_VAL: [],
            DatasetStorage.SPLIT_ID_TEST: [],
            DatasetStorage.SPLIT_OOD_TEST: [],
        }

        global_run_idx = 0

        for sc in PowerPathSimulator.ALL_SCENARIOS:
            for i in range(self.config.runs_per_scenario_nominal):
                global_run_idx += 1
                seed = self.seed_manager.spawn_run_seed(global_run_idx)
                run_id = f"run_train_{sc.lower()}_{i+1:03d}"
                result, events = self._simulate_run(sc, seed, is_ood=False, run_id=run_id, public_scenario_name=sc)
                self.storage.save_run(
                    split=DatasetStorage.SPLIT_TRAIN,
                    result=result,
                    events=events,
                    condition_profile="nominal",
                )
                splits_dict[DatasetStorage.SPLIT_TRAIN].append(run_id)

        for sc in PowerPathSimulator.ALL_SCENARIOS:
            for i in range(self.config.runs_per_scenario_nominal):
                global_run_idx += 1
                seed = self.seed_manager.spawn_run_seed(global_run_idx)
                run_id = f"run_val_{sc.lower()}_{i+1:03d}"
                result, events = self._simulate_run(sc, seed, is_ood=False, run_id=run_id, public_scenario_name=sc)
                self.storage.save_run(
                    split=DatasetStorage.SPLIT_VAL,
                    result=result,
                    events=events,
                    condition_profile="nominal",
                )
                splits_dict[DatasetStorage.SPLIT_VAL].append(run_id)

        for sc in self.ID_SCENARIOS:
            for i in range(self.config.runs_per_scenario_id):
                global_run_idx += 1
                seed = self.seed_manager.spawn_run_seed(global_run_idx)
                run_id = f"run_idtest_{sc.lower()}_{i+1:03d}"
                result, events = self._simulate_run(sc, seed, is_ood=False, run_id=run_id, public_scenario_name=sc)
                self.storage.save_run(
                    split=DatasetStorage.SPLIT_ID_TEST,
                    result=result,
                    events=events,
                    condition_profile="nominal",
                )
                splits_dict[DatasetStorage.SPLIT_ID_TEST].append(run_id)

        for public_name, sc_type in self.OOD_SCENARIO_MAP:
            for i in range(self.config.runs_per_scenario_ood):
                global_run_idx += 1
                seed = self.seed_manager.spawn_run_seed(global_run_idx)
                run_id = f"run_ood_{public_name.lower()}_{i+1:03d}"
                result, events = self._simulate_run(sc_type, seed, is_ood=True, run_id=run_id, public_scenario_name=public_name)
                self.storage.save_run(
                    split=DatasetStorage.SPLIT_OOD_TEST,
                    result=result,
                    events=events,
                    condition_profile="shifted",
                )
                splits_dict[DatasetStorage.SPLIT_OOD_TEST].append(run_id)

        manifest = self._build_manifest(splits_dict)
        return manifest

    def _simulate_run(
        self,
        scenario: str,
        seed: int,
        is_ood: bool,
        run_id: str,
        public_scenario_name: str,
    ) -> Tuple[SimulationResult, List[Dict[str, Any]]]:
        rng = np.random.default_rng(seed)
        n_samples = int(self.config.simulation_duration_s * self.config.sampling_rate_hz)
        t = np.arange(n_samples) / self.config.sampling_rate_hz

        source_v = np.full(n_samples, 5.0, dtype=np.float64)
        load_r = np.full(n_samples, 4.0, dtype=np.float64)
        base_r = np.full(n_samples, 0.080, dtype=np.float64)
        contact_r = np.zeros(n_samples, dtype=np.float64)
        fault_r = np.zeros(n_samples, dtype=np.float64)
        fault_labels = np.zeros(n_samples, dtype=np.int32)
        event_ids = np.full(n_samples, "NONE", dtype=object)
        fault_types = np.full(n_samples, "NONE", dtype=object)

        noise_v_std = 0.005
        noise_i_std = 0.006
        bias_v = np.full(n_samples, 0.001, dtype=np.float64)
        bias_i = np.full(n_samples, 0.001, dtype=np.float64)

        if is_ood:
            noise_v_std = 0.015
            noise_i_std = 0.022
            base_r[:] = 0.140
            source_v[:] = 5.10
            bias_v[:] = 0.008

        events_meta: List[Dict[str, Any]] = []

        def add_event(ev_id: str, f_type: str, st: float, en: float, r_val: float):
            mask = (t >= st) & (t <= en)
            fault_r[mask] = np.maximum(fault_r[mask], r_val)
            fault_labels[mask] = 1
            event_ids[mask] = ev_id
            fault_types[mask] = f_type
            events_meta.append({
                "event_id": ev_id,
                "fault_type": f_type,
                "start_time_s": st,
                "end_time_s": en,
                "duration_s": round(en - st, 4),
                "resistance_increase_ohm": round(r_val, 4),
                "sampled_event": True,
            })

        if scenario == self.SCENARIO_S1_HEALTHY:
            pass

        elif scenario == self.SCENARIO_S2_INCREASED_RESISTANCE:
            base_r[:] += 0.120 if is_ood else 0.060
            ev_st = float(rng.uniform(10.0, 16.0))
            ev_dur = float(rng.uniform(12.0, 20.0))
            add_event("EV_IR_01", "INCREASED_RESISTANCE", ev_st, ev_st + ev_dur, float(rng.uniform(0.30, 0.55)))

        elif scenario == self.SCENARIO_S3_INTERMITTENT_CONTACT:
            n_events = int(rng.integers(3, 5))
            st_cursor = 8.0
            for k in range(n_events):
                st = float(rng.uniform(st_cursor, st_cursor + 6.0))
                dur = float(rng.uniform(0.30, 1.80))
                add_event(f"EV_IC_{k+1:02d}", "INTERMITTENT_CONTACT", st, st + dur, float(rng.uniform(0.40, 0.85)))
                st_cursor = st + dur + 4.0

        elif scenario == self.SCENARIO_S4_GRADUAL_RESISTANCE_DRIFT:
            post_mask = t >= 5.0
            drift_rate = float(rng.uniform(0.002, 0.004))
            dur = float(self.config.simulation_duration_s - 5.0)
            fault_r[post_mask] = drift_rate * (t[post_mask] - 5.0)
            fault_labels[post_mask] = 1
            event_ids[post_mask] = "EV_GD_01"
            fault_types[post_mask] = "GRADUAL_RESISTANCE_DRIFT"
            events_meta.append({
                "event_id": "EV_GD_01",
                "fault_type": "GRADUAL_RESISTANCE_DRIFT",
                "start_time_s": 5.0,
                "end_time_s": float(self.config.simulation_duration_s),
                "duration_s": round(dur, 4),
                "resistance_increase_ohm": round(drift_rate * dur, 4),
                "sampled_event": True,
            })

        elif scenario == self.SCENARIO_S5_MEASUREMENT_OFFSET_DRIFT:
            post_mask = t > 5.0
            drift_rate_v = float(rng.uniform(0.0015, 0.003))
            bias_v[post_mask] += drift_rate_v * (t[post_mask] - 5.0)

        elif scenario == self.SCENARIO_S6_MEASUREMENT_NOISE_SHIFT:
            noise_v_std = 0.018 if is_ood else 0.012
            noise_i_std = 0.026 if is_ood else 0.018
            st1 = float(rng.uniform(12.0, 18.0))
            add_event("EV_NS_01", "INTERMITTENT_CONTACT", st1, st1 + 3.0, 0.450)
            st2 = float(rng.uniform(34.0, 42.0))
            add_event("EV_NS_02", "INTERMITTENT_CONTACT", st2, st2 + 4.0, 0.500)

        elif scenario == self.SCENARIO_S7_LOAD_OPERATING_SHIFT:
            load_r[:] = 1.90 if is_ood else 2.50
            st1 = float(rng.uniform(14.0, 20.0))
            add_event("EV_LS_01", "INTERMITTENT_CONTACT", st1, st1 + 4.0, 0.400)
            st2 = float(rng.uniform(38.0, 44.0))
            add_event("EV_LS_02", "INTERMITTENT_CONTACT", st2, st2 + 4.0, 0.450)

        elif scenario == self.SCENARIO_S8_OPERATING_CONDITION_SHIFT:
            jump_t = float(rng.uniform(22.0, 28.0))
            step_mask = t >= jump_t
            load_r[step_mask] = 2.0 if is_ood else 2.8

        elif scenario == self.SCENARIO_S9_SLOW_DEVELOPING_FAULT:
            st = float(rng.uniform(12.0, 16.0))
            dur = float(rng.uniform(22.0, 28.0))
            en = st + dur
            creep_mask = (t >= st) & (t <= en)
            rate = float(rng.uniform(0.014, 0.020))
            fault_r[creep_mask] = rate * (t[creep_mask] - st)
            fault_labels[creep_mask] = 1
            event_ids[creep_mask] = "EV_SD_01"
            fault_types[creep_mask] = "SLOW_CREEPING"
            events_meta.append({
                "event_id": "EV_SD_01",
                "fault_type": "SLOW_CREEPING",
                "start_time_s": st,
                "end_time_s": en,
                "duration_s": round(dur, 4),
                "resistance_increase_ohm": round(rate * dur, 4),
                "sampled_event": True,
            })

        elif scenario == self.SCENARIO_S10_COMBINED_SHIFTS:
            post_mask = t > 5.0
            bias_v[post_mask] += 0.001 * (t[post_mask] - 5.0)
            step_mask = t >= 30.0
            load_r[step_mask] = 2.20
            noise_v_std = 0.016
            noise_i_std = 0.024
            st1 = float(rng.uniform(12.0, 16.0))
            add_event("EV_CB_01", "COMBINED_FAULT", st1, st1 + 3.5, 0.500)
            st2 = float(rng.uniform(42.0, 46.0))
            add_event("EV_CB_02", "COMBINED_FAULT", st2, st2 + 4.0, 0.550)

        true_path_r = base_r + contact_r + fault_r
        v_noise = rng.normal(0.0, noise_v_std, n_samples)
        i_noise = rng.normal(0.0, noise_i_std, n_samples)

        v_true, i_true, v_meas, i_meas = compute_circuit_signals(
            source_voltage_v=source_v,
            path_resistance_ohm=true_path_r,
            load_resistance_ohm=load_r,
            voltage_offset_v=bias_v,
            current_offset_a=bias_i,
            voltage_noise_v=v_noise,
            current_noise_a=i_noise,
        )

        obs = SimulationObservations(
            timestamp_s=t,
            source_voltage_v=source_v,
            voltage_measured_v=v_meas,
            current_measured_a=i_meas,
            load_resistance_ohm=load_r,
        )
        gt = SimulationGroundTruth(
            timestamp_s=t,
            true_path_resistance_ohm=true_path_r,
            fault_label=fault_labels,
            fault_type=fault_types,
            event_id=event_ids,
        )
        meta = SimulationMetadata(
            run_id=run_id,
            scenario=public_scenario_name,
            random_seed=seed,
            simulation_duration_s=self.config.simulation_duration_s,
            sampling_rate_hz=self.config.sampling_rate_hz,
            configuration_used={
                "is_ood": is_ood,
                "base_resistance": float(np.mean(base_r)),
                "noise_v_std": noise_v_std,
                "noise_i_std": noise_i_std,
            },
            synthetic_data=True,
        )
        return SimulationResult(observations=obs, ground_truth=gt, metadata=meta), events_meta

    def _build_manifest(self, splits_dict: Dict[str, List[str]]) -> DatasetManifest:
        all_runs: List[str] = []
        for r_list in splits_dict.values():
            all_runs.extend(r_list)

        splits_path = self.storage.splits_path
        with open(splits_path, "w", encoding="utf-8") as f:
            json.dump(splits_dict, f, indent=2)

        file_manifest: Dict[str, str] = {}
        file_manifest["splits.json"] = compute_file_sha256(splits_path)

        total_samples = 0
        total_fault_samples = 0
        scenario_counts: Dict[str, int] = {}

        for sp_name, r_ids in splits_dict.items():
            for r_id in r_ids:
                r_dir = self.storage.get_run_dir(sp_name, r_id)
                for f_name in ["observations.csv", "conditions.csv", "ground_truth.csv", "events.json", "metadata.json"]:
                    f_p = r_dir / f_name
                    if f_p.exists():
                        rel = str(f_p.relative_to(self.storage.base_dir))
                        file_manifest[rel] = compute_file_sha256(f_p)

                gt_df = pd.read_csv(r_dir / "ground_truth.csv")
                meta_p = r_dir / "metadata.json"
                with open(meta_p, "r", encoding="utf-8") as fp:
                    m_data = json.load(fp)

                sc = m_data.get("scenario", "UNKNOWN")
                scenario_counts[sc] = scenario_counts.get(sc, 0) + 1

                n_s = len(gt_df)
                total_samples += n_s
                total_fault_samples += int(np.sum(gt_df["fault_label"] == 1))

        manifest = DatasetManifest(
            dataset_id=self.config.dataset_id,
            dataset_schema_version="2.0.0",
            simulator_version="1.0.0",
            generation_timestamp=datetime.now(timezone.utc).isoformat(),
            root_random_seed=self.config.root_seed,
            total_simulation_runs=len(all_runs),
            total_samples=total_samples,
            total_fault_samples=total_fault_samples,
            total_healthy_samples=total_samples - total_fault_samples,
            run_counts_per_scenario=scenario_counts,
            sample_counts_per_scenario={k: v * int(self.config.simulation_duration_s * self.config.sampling_rate_hz) for k, v in scenario_counts.items()},
            binary_class_distribution={
                "healthy_ratio": float((total_samples - total_fault_samples) / total_samples) if total_samples > 0 else 0.0,
                "fault_ratio": float(total_fault_samples / total_samples) if total_samples > 0 else 0.0,
            },
            fault_event_counts=len(all_runs),
            unsampled_fault_event_counts=0,
            sampling_rate_distribution={str(int(self.config.sampling_rate_hz)): len(all_runs)},
            simulation_duration_distribution={str(int(self.config.simulation_duration_s)): len(all_runs)},
            condition_profiles={"nominal": len(splits_dict.get(DatasetStorage.SPLIT_ID_TEST, [])), "shifted": len(splits_dict.get(DatasetStorage.SPLIT_OOD_TEST, []))},
            parameter_ranges={
                "source_voltage_v": {"min": 5.0, "max": 5.10},
                "load_resistance_ohm": {"min": 1.90, "max": 4.0},
            },
            split_assignments=splits_dict,
            synthetic_data=True,
            file_checksums_sha256=file_manifest,
        )

        with open(self.storage.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)

        return manifest


def generate_confirmatory_dataset(root_seed: int = 20261202) -> DatasetManifest:
    cfg = ConfirmatoryDatasetConfig(root_seed=root_seed)
    gen = ConfirmatoryDatasetGenerator(cfg)
    return gen.generate()


if __name__ == "__main__":
    generate_confirmatory_dataset()
