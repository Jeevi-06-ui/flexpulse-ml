"""
Dataset manifest schema, metadata modeling and checksum calculation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def compute_file_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as file_handle:
        while chunk := file_handle.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass
class DatasetManifest:
    dataset_id: str
    dataset_schema_version: str = "1.0.0"
    simulator_version: str = "0.2.0"
    generation_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    root_random_seed: int = 42
    total_simulation_runs: int = 0
    total_samples: int = 0
    total_fault_samples: int = 0
    total_healthy_samples: int = 0
    run_counts_per_scenario: Dict[str, int] = field(default_factory=dict)
    sample_counts_per_scenario: Dict[str, int] = field(default_factory=dict)
    binary_class_distribution: Dict[str, float] = field(default_factory=dict)
    fault_event_counts: int = 0
    unsampled_fault_event_counts: int = 0
    sampling_rate_distribution: Dict[str, int] = field(default_factory=dict)
    simulation_duration_distribution: Dict[str, int] = field(default_factory=dict)
    condition_profiles: Dict[str, Any] = field(default_factory=dict)
    parameter_ranges: Dict[str, Any] = field(default_factory=dict)
    split_assignments: Dict[str, List[str]] = field(default_factory=dict)
    synthetic_data: bool = True
    file_checksums_sha256: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, target_path: Path) -> None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as file_handle:
            json.dump(self.to_dict(), file_handle, indent=2)

    @classmethod
    def load_json(cls, source_path: Path) -> DatasetManifest:
        with open(source_path, "r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)
        return cls(**payload)
