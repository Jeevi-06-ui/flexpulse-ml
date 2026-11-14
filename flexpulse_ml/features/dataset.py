"""
Feature dataset generation orchestrating sliding windows, feature extraction and manifest serialization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from flexpulse_ml.config import PROJECT_ROOT
from flexpulse_ml.data.manifest import compute_file_sha256
from flexpulse_ml.data.storage import DatasetStorage
from flexpulse_ml.features.extractor import (
    ElectricalFeatureExtractor,
    FEATURE_NAMES,
    FEATURE_UNITS,
)
from flexpulse_ml.features.windowing import SlidingWindowGenerator, WindowConfig

PROCESSED_FEATURES_ROOT = PROJECT_ROOT / "data" / "processed" / "features"


@dataclass
class FeatureManifest:
    source_dataset_id: str
    feature_schema_version: str = "1.0.0"
    sampling_frequency_hz: float = 100.0
    window_duration_s: float = 0.5
    window_size_samples: int = 50
    overlap_fraction: float = 0.5
    stride_samples: int = 25
    total_windows: int = 0
    total_healthy_windows: int = 0
    total_fault_windows: int = 0
    feature_names: List[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    feature_units: Dict[str, str] = field(default_factory=lambda: dict(FEATURE_UNITS))
    partition_window_counts: Dict[str, int] = field(default_factory=dict)
    partition_class_distributions: Dict[str, Dict[str, float]] = field(default_factory=dict)
    partition_scenario_distributions: Dict[str, Dict[str, int]] = field(default_factory=dict)
    synthetic_data: bool = True
    generation_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    file_checksums_sha256: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, target_path: Path) -> None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as file_handle:
            json.dump(self.to_dict(), file_handle, indent=2)

    @classmethod
    def load_json(cls, source_path: Path) -> FeatureManifest:
        with open(source_path, "r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)
        return cls(**payload)


class FeatureDatasetStorage:
    """Manages reading and writing tabular feature matrices and manifests."""

    SPLIT_FILES = {
        DatasetStorage.SPLIT_TRAIN: "train_features.csv",
        DatasetStorage.SPLIT_VAL: "validation_features.csv",
        DatasetStorage.SPLIT_ID_TEST: "id_test_features.csv",
        DatasetStorage.SPLIT_OOD_TEST: "ood_test_features.csv",
    }

    def __init__(self, dataset_id: str, base_dir: Optional[Path] = None) -> None:
        self.dataset_id = dataset_id
        if base_dir is not None:
            base_p = Path(base_dir)
            if base_p.name == dataset_id:
                self.base_dir = base_p
            else:
                self.base_dir = base_p / dataset_id
        else:
            self.base_dir = PROCESSED_FEATURES_ROOT / dataset_id

    @property
    def root_path(self) -> Path:
        return self.base_dir

    @property
    def manifest_path(self) -> Path:
        return self.base_dir / "feature_manifest.json"

    def get_partition_path(self, split: str) -> Path:
        filename = self.SPLIT_FILES.get(split)
        if not filename:
            raise ValueError(f"Unknown split '{split}'. Must be one of {list(self.SPLIT_FILES.keys())}")
        return self.base_dir / filename

    def save_partition(self, split: str, df: pd.DataFrame) -> Path:
        target_path = self.get_partition_path(split)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(target_path, index=False)
        return target_path

    def load_partition(self, split: str) -> pd.DataFrame:
        target_path = self.get_partition_path(split)
        if not target_path.exists():
            raise FileNotFoundError(f"Feature partition file missing: {target_path}")
        return pd.read_csv(target_path)

    def save_manifest(self, manifest: FeatureManifest) -> Path:
        manifest.save_json(self.manifest_path)
        return self.manifest_path

    def load_manifest(self) -> FeatureManifest:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Feature manifest missing at {self.manifest_path}")
        return FeatureManifest.load_json(self.manifest_path)

    def compute_all_checksums(self) -> Dict[str, str]:
        checksums: Dict[str, str] = {}
        for file_path in sorted(self.base_dir.rglob("*")):
            if file_path.is_file() and file_path.name != "feature_manifest.json":
                rel_path = str(file_path.relative_to(self.base_dir))
                checksums[rel_path] = compute_file_sha256(file_path)
        return checksums


class FeatureDatasetGenerator:
    """Orchestrates sliding-window generation, feature extraction, and tabular dataset creation."""

    METADATA_COLUMNS = [
        "run_id",
        "window_index",
        "window_start_s",
        "window_end_s",
        "scenario",
        "fault_fraction",
    ]

    TARGET_COLUMN = "fault_label"

    def __init__(
        self,
        dataset_id: str,
        window_config: Optional[WindowConfig] = None,
        source_base_dir: Optional[Path] = None,
        output_base_dir: Optional[Path] = None,
    ) -> None:
        self.dataset_id = dataset_id
        self.source_storage = DatasetStorage(dataset_id=dataset_id, base_dir=source_base_dir)
        self.output_storage = FeatureDatasetStorage(dataset_id=dataset_id, base_dir=output_base_dir)

        self.window_generator = SlidingWindowGenerator(config=window_config)
        self.extractor = ElectricalFeatureExtractor(sampling_rate_hz=self.window_generator.config.sampling_rate_hz)

    def generate(self) -> FeatureManifest:
        splits_dict = self.source_storage.load_splits()
        w_cfg = self.window_generator.config

        partition_counts: Dict[str, int] = {}
        partition_classes: Dict[str, Dict[str, float]] = {}
        partition_scenarios: Dict[str, Dict[str, int]] = {}

        total_windows = 0
        total_healthy = 0
        total_fault = 0

        for split in DatasetStorage.ALL_SPLITS:
            run_ids = splits_dict.get(split, [])
            rows: List[Dict[str, Any]] = []
            scenario_counts: Dict[str, int] = {}
            split_fault = 0
            split_healthy = 0

            for r_id in run_ids:
                run_data = self.source_storage.load_run(split, r_id)
                obs_df = run_data["observations"]
                gt_df = run_data["ground_truth"]
                meta = run_data["metadata"]
                scenario = meta.get("scenario", "UNKNOWN")

                windows = self.window_generator.slice_run(
                    observations_df=obs_df,
                    ground_truth_df=gt_df,
                    run_id=r_id,
                    scenario=scenario,
                )

                for w in windows:
                    row_data: Dict[str, Any] = {
                        "run_id": w.run_id,
                        "window_index": w.window_index,
                        "window_start_s": w.start_time_s,
                        "window_end_s": w.end_time_s,
                        "scenario": w.scenario,
                        "fault_fraction": w.fault_fraction,
                    }

                    features = self.extractor.extract_from_window(w)
                    row_data.update(features)
                    row_data[self.TARGET_COLUMN] = w.fault_label
                    rows.append(row_data)

                    scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
                    if w.fault_label == 1:
                        split_fault += 1
                    else:
                        split_healthy += 1

            column_order = self.METADATA_COLUMNS + list(FEATURE_NAMES) + [self.TARGET_COLUMN]
            if rows:
                partition_df = pd.DataFrame(rows)[column_order]
            else:
                partition_df = pd.DataFrame(columns=column_order)

            self.output_storage.save_partition(split, partition_df)

            n_win = len(rows)
            partition_counts[split] = n_win
            partition_scenarios[split] = scenario_counts
            partition_classes[split] = {
                "healthy_ratio": float(split_healthy / n_win) if n_win > 0 else 0.0,
                "fault_ratio": float(split_fault / n_win) if n_win > 0 else 0.0,
            }

            total_windows += n_win
            total_fault += split_fault
            total_healthy += split_healthy

        manifest = FeatureManifest(
            source_dataset_id=self.dataset_id,
            feature_schema_version="1.0.0",
            sampling_frequency_hz=w_cfg.sampling_rate_hz,
            window_duration_s=w_cfg.window_duration_s,
            window_size_samples=w_cfg.window_size_samples,
            overlap_fraction=w_cfg.overlap_fraction,
            stride_samples=w_cfg.stride_samples,
            total_windows=total_windows,
            total_healthy_windows=total_healthy,
            total_fault_windows=total_fault,
            feature_names=list(FEATURE_NAMES),
            feature_units=dict(FEATURE_UNITS),
            partition_window_counts=partition_counts,
            partition_class_distributions=partition_classes,
            partition_scenario_distributions=partition_scenarios,
            synthetic_data=True,
        )

        manifest.file_checksums_sha256 = self.output_storage.compute_all_checksums()
        self.output_storage.save_manifest(manifest)
        return manifest


def generate_feature_dataset(
    dataset_id: str,
    window_duration_s: float = 0.5,
    overlap_fraction: float = 0.5,
    source_base_dir: Optional[Path] = None,
    output_base_dir: Optional[Path] = None,
) -> FeatureManifest:
    cfg = WindowConfig(
        sampling_rate_hz=100.0,
        window_duration_s=window_duration_s,
        overlap_fraction=overlap_fraction,
    )
    generator = FeatureDatasetGenerator(
        dataset_id=dataset_id,
        window_config=cfg,
        source_base_dir=source_base_dir,
        output_base_dir=output_base_dir,
    )
    return generator.generate()
