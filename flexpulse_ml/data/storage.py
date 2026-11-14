"""
Data persistence, run directory layout and filesystem management for synthetic datasets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from flexpulse_ml.config import PROJECT_ROOT
from flexpulse_ml.data.manifest import DatasetManifest, compute_file_sha256
from flexpulse_ml.simulation.electrical_model import SimulationResult

SYNTHETIC_DATA_ROOT = PROJECT_ROOT / "data" / "synthetic"


class DatasetStorage:
    """Manages reading and writing run directories and manifests."""

    SPLIT_TRAIN = "train"
    SPLIT_VAL = "validation"
    SPLIT_ID_TEST = "id_test"
    SPLIT_OOD_TEST = "ood_test"

    ALL_SPLITS = (SPLIT_TRAIN, SPLIT_VAL, SPLIT_ID_TEST, SPLIT_OOD_TEST)

    def __init__(self, dataset_id: str, base_dir: Optional[Path] = None) -> None:
        self.dataset_id = dataset_id
        if base_dir is not None:
            base_path = Path(base_dir)
            if base_path.name == dataset_id:
                self.base_dir = base_path
            else:
                self.base_dir = base_path / dataset_id
        else:
            self.base_dir = SYNTHETIC_DATA_ROOT / dataset_id

    @property
    def root_path(self) -> Path:
        return self.base_dir

    @property
    def dataset_dir(self) -> Path:
        return self.base_dir

    @property
    def manifest_path(self) -> Path:
        return self.base_dir / "manifest.json"

    @property
    def splits_path(self) -> Path:
        return self.base_dir / "splits.json"

    def exists(self) -> bool:
        return self.base_dir.exists() and self.manifest_path.exists()

    def initialize_structure(self, allow_overwrite: bool = False) -> None:
        if self.exists() and not allow_overwrite:
            raise FileExistsError(
                f"Dataset directory '{self.base_dir}' already exists. Use allow_overwrite=True to overwrite."
            )
        for split in self.ALL_SPLITS:
            split_run_dir = self.base_dir / split / "runs"
            split_run_dir.mkdir(parents=True, exist_ok=True)

    def get_run_dir(self, split: str, run_id: str) -> Path:
        return self.base_dir / split / "runs" / run_id

    def save_run(
        self,
        split: str,
        result: SimulationResult,
        events: List[Dict[str, Any]],
        condition_profile: str = "nominal",
    ) -> Path:
        if split not in self.ALL_SPLITS:
            raise ValueError(f"Invalid split '{split}'. Must be one of {self.ALL_SPLITS}")

        run_id = result.metadata.run_id
        target_dir = self.get_run_dir(split, run_id)
        target_dir.mkdir(parents=True, exist_ok=True)

        obs = result.observations
        observations_df = pd.DataFrame(
            {
                "timestamp_s": obs.timestamp_s,
                "voltage_measured_v": obs.voltage_measured_v,
                "current_measured_a": obs.current_measured_a,
                "source_voltage_v": obs.source_voltage_v,
            }
        )
        observations_df.to_csv(target_dir / "observations.csv", index=False)

        conditions_df = pd.DataFrame(
            {
                "timestamp_s": obs.timestamp_s,
                "configured_source_voltage_v": obs.source_voltage_v,
                "configured_load_resistance_ohm": obs.load_resistance_ohm,
            }
        )
        conditions_df.to_csv(target_dir / "conditions.csv", index=False)

        gt = result.ground_truth
        ground_truth_df = pd.DataFrame(
            {
                "timestamp_s": gt.timestamp_s,
                "true_path_resistance_ohm": gt.true_path_resistance_ohm,
                "fault_label": gt.fault_label,
                "fault_type": gt.fault_type,
                "event_id": gt.event_id,
            }
        )
        ground_truth_df.to_csv(target_dir / "ground_truth.csv", index=False)

        with open(target_dir / "events.json", "w", encoding="utf-8") as file_handle:
            json.dump(events, file_handle, indent=2)

        meta_payload = {
            "run_id": run_id,
            "scenario": result.metadata.scenario,
            "random_seed": result.metadata.random_seed,
            "simulation_configuration": result.metadata.configuration_used,
            "sampling_frequency_hz": result.metadata.sampling_rate_hz,
            "simulation_duration_s": result.metadata.simulation_duration_s,
            "condition_profile": condition_profile,
            "simulator_version": "0.2.0",
            "dataset_identifier": self.dataset_id,
            "synthetic_data": True,
            "split": split,
            "is_out_of_distribution": (split == self.SPLIT_OOD_TEST),
            "noise_multiplier": 3.0 if (split == self.SPLIT_OOD_TEST) else float(result.metadata.configuration_used.get("noise_multiplier", 1.0)),
        }
        with open(target_dir / "metadata.json", "w", encoding="utf-8") as file_handle:
            json.dump(meta_payload, file_handle, indent=2)

        return target_dir

    def save_splits(self, splits_dict: Dict[str, List[str]]) -> Path:
        target_path = self.splits_path
        with open(target_path, "w", encoding="utf-8") as file_handle:
            json.dump(splits_dict, file_handle, indent=2)
        return target_path

    def save_manifest(self, manifest: DatasetManifest) -> Path:
        target_path = self.manifest_path
        manifest.save_json(target_path)
        return target_path

    def load_manifest(self) -> DatasetManifest:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {self.manifest_path}")
        return DatasetManifest.load_json(self.manifest_path)

    def load_splits(self) -> Dict[str, List[str]]:
        if not self.splits_path.exists():
            raise FileNotFoundError(f"Splits file not found at {self.splits_path}")
        with open(self.splits_path, "r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)
        return payload if isinstance(payload, dict) else {}

    def load_run(self, split: str, run_id: str) -> Dict[str, Any]:
        run_dir = self.get_run_dir(split, run_id)
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        obs_df = pd.read_csv(run_dir / "observations.csv")
        cond_df = pd.read_csv(run_dir / "conditions.csv")
        gt_df = pd.read_csv(run_dir / "ground_truth.csv")
        with open(run_dir / "events.json", "r", encoding="utf-8") as fh:
            events_data = json.load(fh)
        with open(run_dir / "metadata.json", "r", encoding="utf-8") as fh:
            meta_data = json.load(fh)

        return {
            "observations": obs_df,
            "conditions": cond_df,
            "ground_truth": gt_df,
            "events": events_data,
            "metadata": meta_data,
        }

    def list_runs(self, split: Optional[str] = None) -> Dict[str, List[str]]:
        results: Dict[str, List[str]] = {}
        target_splits = [split] if split else list(self.ALL_SPLITS)
        for sp in target_splits:
            sp_dir = self.base_dir / sp / "runs"
            if sp_dir.exists():
                results[sp] = sorted([p.name for p in sp_dir.iterdir() if p.is_dir()])
            else:
                results[sp] = []
        return results

    def list_runs_in_split(self, split: str) -> List[str]:
        return self.list_runs(split).get(split, [])

    def list_all_runs(self) -> List[str]:
        all_runs: List[str] = []
        for runs in self.list_runs().values():
            all_runs.extend(runs)
        return sorted(all_runs)

    def get_run_split(self, run_id: str) -> Optional[str]:
        for sp in self.ALL_SPLITS:
            run_dir = self.get_run_dir(sp, run_id)
            if run_dir.exists():
                return sp
        return None

    def load_observations(self, run_id: str, split: Optional[str] = None) -> pd.DataFrame:
        sp = split or self.get_run_split(run_id)
        if not sp:
            raise FileNotFoundError(f"Run {run_id} not found in any split")
        return pd.read_csv(self.get_run_dir(sp, run_id) / "observations.csv")

    def load_conditions(self, run_id: str, split: Optional[str] = None) -> pd.DataFrame:
        sp = split or self.get_run_split(run_id)
        if not sp:
            raise FileNotFoundError(f"Run {run_id} not found in any split")
        return pd.read_csv(self.get_run_dir(sp, run_id) / "conditions.csv")

    def load_ground_truth(self, run_id: str, split: Optional[str] = None) -> pd.DataFrame:
        sp = split or self.get_run_split(run_id)
        if not sp:
            raise FileNotFoundError(f"Run {run_id} not found in any split")
        return pd.read_csv(self.get_run_dir(sp, run_id) / "ground_truth.csv")

    def load_events(self, run_id: str, split: Optional[str] = None) -> List[Dict[str, Any]]:
        sp = split or self.get_run_split(run_id)
        if not sp:
            raise FileNotFoundError(f"Run {run_id} not found in any split")
        with open(self.get_run_dir(sp, run_id) / "events.json", "r", encoding="utf-8") as fh:
            return json.load(fh)

    def load_metadata(self, run_id: str, split: Optional[str] = None) -> Dict[str, Any]:
        sp = split or self.get_run_split(run_id)
        if not sp:
            raise FileNotFoundError(f"Run {run_id} not found in any split")
        with open(self.get_run_dir(sp, run_id) / "metadata.json", "r", encoding="utf-8") as fh:
            return json.load(fh)

    def compute_all_checksums(self) -> Dict[str, str]:
        checksums: Dict[str, str] = {}
        for file_path in sorted(self.base_dir.rglob("*")):
            if file_path.is_file() and file_path.name != "manifest.json":
                rel_path = str(file_path.relative_to(self.base_dir))
                checksums[rel_path] = compute_file_sha256(file_path)
        return checksums
