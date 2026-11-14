"""
Feature dataset validation engine verifying numerical integrity, schema consistency, and leakage prevention.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import numpy as np
import pandas as pd

from flexpulse_ml.data.manifest import compute_file_sha256
from flexpulse_ml.data.storage import DatasetStorage
from flexpulse_ml.features.dataset import (
    FeatureDatasetGenerator,
    FeatureDatasetStorage,
    FeatureManifest,
)
from flexpulse_ml.features.extractor import FEATURE_NAMES


@dataclass
class FeatureValidationCheck:
    name: str
    status: str
    message: str


@dataclass
class FeatureValidationReport:
    dataset_id: str
    status: str
    is_valid: bool
    checks: List[FeatureValidationCheck] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def print_summary(self) -> None:
        print("=" * 68)
        print(f" Feature Dataset Validation Report: {self.dataset_id}")
        print(f" Overall Status: [{self.status}]")
        print("=" * 68)
        for check in self.checks:
            icon = "✓" if check.status == "PASS" else ("!" if check.status == "WARNING" else "✗")
            print(f" [{icon}] {check.name:<40} : {check.status} - {check.message}")
        if self.errors:
            print("\nCritical Errors:")
            for err in self.errors:
                print(f"  - {err}")
        if self.warnings:
            print("\nWarnings:")
            for wrn in self.warnings:
                print(f"  - {wrn}")
        print("=" * 68)


class FeatureValidator:
    """Verifies numerical hygiene, target labeling, and split isolation for feature datasets."""

    FORBIDDEN_COLUMNS = {
        "true_path_resistance_ohm",
        "fault_type",
        "event_id",
        "fault_resistance_ohms",
        "severity",
        "v_load_true_v",
        "i_true_a",
        "voltage_bias_v",
        "current_bias_a",
    }

    def __init__(self, dataset_path_or_storage: Any) -> None:
        if isinstance(dataset_path_or_storage, FeatureDatasetStorage):
            self.storage = dataset_path_or_storage
        else:
            p = Path(dataset_path_or_storage)
            self.storage = FeatureDatasetStorage(dataset_id=p.name, base_dir=p)

    def validate(self) -> FeatureValidationReport:
        checks: List[FeatureValidationCheck] = []
        errors: List[str] = []
        warnings: List[str] = []

        root = self.storage.root_path
        if not root.exists():
            return FeatureValidationReport(
                dataset_id=self.storage.dataset_id,
                status="FAIL",
                is_valid=False,
                errors=[f"Feature directory does not exist: {root}"],
            )

        manifest_file = self.storage.manifest_path
        manifest: Optional[FeatureManifest] = None
        if not manifest_file.exists():
            errors.append(f"Missing feature_manifest.json at {manifest_file}")
            checks.append(FeatureValidationCheck("feature_manifest_file_exists", "FAIL", "feature_manifest.json missing."))
        else:
            checks.append(FeatureValidationCheck("feature_manifest_file_exists", "PASS", "feature_manifest.json found."))
            try:
                manifest = self.storage.load_manifest()
            except Exception as ex:
                errors.append(f"Failed to load feature_manifest.json: {ex}")

        partition_dfs: Dict[str, pd.DataFrame] = {}
        all_partitions_exist = True
        for sp in DatasetStorage.ALL_SPLITS:
            p_path = self.storage.get_partition_path(sp)
            if not p_path.exists():
                errors.append(f"Partition file missing: {p_path.name}")
                all_partitions_exist = False
            else:
                try:
                    partition_dfs[sp] = pd.read_csv(p_path)
                except Exception as ex:
                    errors.append(f"Failed to read {p_path.name}: {ex}")
                    all_partitions_exist = False

        checks.append(
            FeatureValidationCheck(
                "all_partitions_exist",
                "PASS" if all_partitions_exist else "FAIL",
                "All 4 feature CSV files exist.",
            )
        )

        first_columns: Optional[List[str]] = None
        consistent_columns = True
        for sp, df in partition_dfs.items():
            cols = list(df.columns)
            if first_columns is None:
                first_columns = cols
            elif cols != first_columns:
                errors.append(f"Column mismatch in partition {sp}.")
                consistent_columns = False

        expected_cols = (
            FeatureDatasetGenerator.METADATA_COLUMNS
            + list(FEATURE_NAMES)
            + [FeatureDatasetGenerator.TARGET_COLUMN]
        )
        if first_columns != expected_cols:
            errors.append(f"Feature schema columns do not match expected standard.")
            consistent_columns = False

        checks.append(
            FeatureValidationCheck(
                "consistent_columns",
                "PASS" if consistent_columns else "FAIL",
                "Feature columns consistent across all partitions.",
            )
        )

        no_nan_or_inf = True
        binary_targets_valid = True
        fault_fraction_valid = True
        boundaries_valid = True
        no_leakage = True

        partition_runs: Dict[str, Set[str]] = {}
        total_windows_counted = 0
        total_healthy_counted = 0
        total_fault_counted = 0

        for sp, df in partition_dfs.items():
            n_rows = len(df)
            total_windows_counted += n_rows
            if n_rows == 0:
                errors.append(f"Partition {sp} has 0 feature rows.")
                continue

            r_ids = set(df["run_id"].unique())
            partition_runs[sp] = r_ids

            feature_cols = [c for c in df.columns if c in FEATURE_NAMES]
            feat_vals = df[feature_cols].select_dtypes(include=[np.number])

            if feat_vals.isna().any().any():
                errors.append(f"NaN values found in features for partition {sp}.")
                no_nan_or_inf = False
            if np.isinf(feat_vals).any().any():
                errors.append(f"Infinite values found in features for partition {sp}.")
                no_nan_or_inf = False

            forbidden_present = self.FORBIDDEN_COLUMNS.intersection(set(df.columns))
            if forbidden_present:
                errors.append(f"Forbidden ground-truth columns leaked into partition {sp}: {forbidden_present}")
                no_leakage = False

            targets = df[FeatureDatasetGenerator.TARGET_COLUMN].to_numpy()
            unique_targets = set(np.unique(targets))
            if not unique_targets.issubset({0, 1}):
                errors.append(f"Invalid target labels in partition {sp}: {unique_targets}")
                binary_targets_valid = False

            fault_counts = int(np.sum(targets == 1))
            healthy_counts = int(np.sum(targets == 0))
            total_fault_counted += fault_counts
            total_healthy_counted += healthy_counts

            fractions = df["fault_fraction"].to_numpy()
            if np.any((fractions < 0.0) | (fractions > 1.0)):
                errors.append(f"Fault fraction out of [0, 1] range in partition {sp}.")
                fault_fraction_valid = False

            faulty_mask = targets == 1
            if np.any(fractions[faulty_mask] <= 0.0):
                errors.append(f"Inconsistent fault label: y=1 with fault_fraction=0 in partition {sp}.")
                fault_fraction_valid = False
            if np.any(fractions[~faulty_mask] > 0.0):
                errors.append(f"Inconsistent fault label: y=0 with fault_fraction>0 in partition {sp}.")
                fault_fraction_valid = False

            durations = df["window_end_s"].to_numpy() - df["window_start_s"].to_numpy()
            if manifest:
                target_dur = manifest.window_duration_s
                if np.any(durations <= 0.0) or np.any(np.abs(durations - target_dur) > (1.5 / manifest.sampling_frequency_hz)):
                    errors.append(f"Window duration mismatch in partition {sp}.")
                    boundaries_valid = False

        checks.append(
            FeatureValidationCheck(
                "no_nan_or_inf",
                "PASS" if no_nan_or_inf else "FAIL",
                "All feature values are finite and non-NaN.",
            )
        )
        checks.append(
            FeatureValidationCheck(
                "binary_targets_valid",
                "PASS" if binary_targets_valid else "FAIL",
                "Target labels are strictly binary {0, 1}.",
            )
        )
        checks.append(
            FeatureValidationCheck(
                "fault_fraction_valid",
                "PASS" if fault_fraction_valid else "FAIL",
                "Fault fractions match binary target labels.",
            )
        )
        checks.append(
            FeatureValidationCheck(
                "correct_window_count_and_boundaries",
                "PASS" if boundaries_valid else "FAIL",
                "Window durations align with sampling parameters.",
            )
        )
        checks.append(
            FeatureValidationCheck(
                "no_ground_truth_leakage",
                "PASS" if no_leakage else "FAIL",
                "Zero hidden ground-truth columns in feature table.",
            )
        )

        no_overlap = True
        splits_list = list(partition_runs.keys())
        for i in range(len(splits_list)):
            for j in range(i + 1, len(splits_list)):
                sp_a = splits_list[i]
                sp_b = splits_list[j]
                inter = partition_runs[sp_a].intersection(partition_runs[sp_b])
                if inter:
                    errors.append(f"Run overlap between {sp_a} and {sp_b}: {inter}")
                    no_overlap = False

        checks.append(
            FeatureValidationCheck(
                "no_run_overlap",
                "PASS" if no_overlap else "FAIL",
                "No run IDs overlap across partitions.",
            )
        )

        provenance_ok = manifest.synthetic_data if manifest else False
        checks.append(
            FeatureValidationCheck(
                "synthetic_provenance",
                "PASS" if provenance_ok else "FAIL",
                "Synthetic data provenance confirmed.",
            )
        )

        manifest_match = True
        if manifest:
            if manifest.total_windows != total_windows_counted:
                errors.append(f"Manifest total_windows ({manifest.total_windows}) != actual ({total_windows_counted})")
                manifest_match = False
            if manifest.total_fault_windows != total_fault_counted:
                errors.append(f"Manifest total_fault_windows ({manifest.total_fault_windows}) != actual ({total_fault_counted})")
                manifest_match = False
            if manifest.total_healthy_windows != total_healthy_counted:
                errors.append(f"Manifest total_healthy_windows ({manifest.total_healthy_windows}) != actual ({total_healthy_counted})")
                manifest_match = False

        checks.append(
            FeatureValidationCheck(
                "feature_manifest_consistency",
                "PASS" if manifest_match else "FAIL",
                "Manifest window counts and classes match CSV data.",
            )
        )

        checksums_ok = True
        if manifest and manifest.file_checksums_sha256:
            for rel_str, exp_hash in manifest.file_checksums_sha256.items():
                p_file = root / rel_str
                if not p_file.exists():
                    errors.append(f"Missing file for hash check: {rel_str}")
                    checksums_ok = False
                    continue
                act_hash = compute_file_sha256(p_file)
                if act_hash != exp_hash:
                    errors.append(f"Checksum mismatch for {rel_str}: computed {act_hash} != manifest {exp_hash}")
                    checksums_ok = False

            checks.append(
                FeatureValidationCheck(
                    "checksum_integrity",
                    "PASS" if checksums_ok else "FAIL",
                    "Cryptographic SHA-256 hashes verified.",
                )
            )
        elif manifest:
            checks.append(FeatureValidationCheck("checksum_integrity", "WARNING", "No checksums in feature manifest."))
        else:
            checks.append(FeatureValidationCheck("checksum_integrity", "FAIL", "Manifest missing for checksums."))

        overall_valid = len(errors) == 0
        overall_status = "PASS" if overall_valid else "FAIL"
        if overall_valid and len(warnings) > 0:
            overall_status = "WARNING"

        summary = {
            "total_windows": total_windows_counted,
            "total_healthy_windows": total_healthy_counted,
            "total_fault_windows": total_fault_counted,
            "fault_ratio": float(total_fault_counted / total_windows_counted) if total_windows_counted > 0 else 0.0,
            "feature_count": len(FEATURE_NAMES),
        }

        return FeatureValidationReport(
            dataset_id=self.storage.dataset_id,
            status=overall_status,
            is_valid=overall_valid,
            checks=checks,
            errors=errors,
            warnings=warnings,
            summary=summary,
        )


def validate_feature_dataset(feature_dataset_path: Path) -> FeatureValidationReport:
    validator = FeatureValidator(feature_dataset_path)
    return validator.validate()
