"""
Dataset validation engine verifying integrity, schema separation, and event alignment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import numpy as np
import pandas as pd

from flexpulse_ml.data.manifest import DatasetManifest, compute_file_sha256
from flexpulse_ml.data.storage import DatasetStorage
from flexpulse_ml.simulation.fault_injector import PowerPathSimulator


@dataclass
class ValidationCheck:
    name: str
    status: str
    message: str


@dataclass
class DatasetValidationReport:
    dataset_id: str
    status: str
    is_valid: bool
    checks: List[ValidationCheck] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def print_summary(self) -> None:
        print("=" * 68)
        print(f" Dataset Validation Report: {self.dataset_id}")
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


class DatasetValidator:
    """Verifies that a synthetic dataset satisfies all scientific integrity criteria."""

    REQUIRED_RUN_FILES = (
        "observations.csv",
        "conditions.csv",
        "ground_truth.csv",
        "events.json",
        "metadata.json",
    )

    REQUIRED_OBSERVATION_COLUMNS = {
        "timestamp_s",
        "voltage_measured_v",
        "current_measured_a",
        "source_voltage_v",
    }

    REQUIRED_CONDITION_COLUMNS = {
        "timestamp_s",
        "configured_source_voltage_v",
        "configured_load_resistance_ohm",
    }

    REQUIRED_GROUND_TRUTH_COLUMNS = {
        "timestamp_s",
        "true_path_resistance_ohm",
        "fault_label",
        "fault_type",
        "event_id",
    }

    REQUIRED_METADATA_FIELDS = {
        "run_id",
        "scenario",
        "random_seed",
        "sampling_frequency_hz",
        "simulation_duration_s",
        "condition_profile",
        "dataset_identifier",
        "synthetic_data",
    }

    FORBIDDEN_OBSERVATION_COLUMNS = {
        "true_path_resistance_ohm",
        "fault_label",
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
        if isinstance(dataset_path_or_storage, DatasetStorage):
            self.storage = dataset_path_or_storage
        else:
            p = Path(dataset_path_or_storage)
            self.storage = DatasetStorage(dataset_id=p.name, base_dir=p)

    def validate(self) -> DatasetValidationReport:
        checks: List[ValidationCheck] = []
        errors: List[str] = []
        warnings: List[str] = []

        root = self.storage.root_path
        if not root.exists():
            return DatasetValidationReport(
                dataset_id=self.storage.dataset_id,
                status="FAIL",
                is_valid=False,
                errors=[f"Dataset directory does not exist: {root}"],
            )

        manifest_file = root / "manifest.json"
        splits_file = root / "splits.json"

        manifest_exists = manifest_file.exists()
        if manifest_exists:
            checks.append(ValidationCheck("manifest_file_exists", "PASS", "manifest.json found at root."))
        else:
            errors.append("Missing manifest.json at dataset root.")
            checks.append(ValidationCheck("manifest_file_exists", "FAIL", "manifest.json missing at root."))

        splits_exists = splits_file.exists()
        if splits_exists:
            checks.append(ValidationCheck("splits_file_exists", "PASS", "splits.json found at root."))
        else:
            errors.append("Missing splits.json at dataset root.")
            checks.append(ValidationCheck("splits_file_exists", "FAIL", "splits.json missing at root."))

        splits: Dict[str, List[str]] = {}
        if splits_exists:
            try:
                splits = self.storage.load_splits()
            except Exception as ex:
                errors.append(f"Failed to parse splits.json: {ex}")

        all_splits_present = all(sp in splits for sp in DatasetStorage.ALL_SPLITS)
        if all_splits_present:
            checks.append(ValidationCheck("all_splits_present", "PASS", "All required splits defined."))
        else:
            errors.append(f"Missing required splits in splits.json: {set(DatasetStorage.ALL_SPLITS) - set(splits.keys())}")
            checks.append(ValidationCheck("all_splits_present", "FAIL", "Missing required split keys."))

        empty_partitions = [sp for sp in DatasetStorage.ALL_SPLITS if len(splits.get(sp, [])) == 0]
        if empty_partitions:
            errors.append(f"Empty partitions detected: {empty_partitions}. Every required partition must have at least one run.")
            checks.append(ValidationCheck("all_partitions_non_empty", "FAIL", f"Empty partitions: {empty_partitions}."))
        else:
            checks.append(ValidationCheck("all_partitions_non_empty", "PASS", "All required partitions contain at least one run."))

        all_run_ids: List[str] = []
        split_sets: Dict[str, Set[str]] = {}
        for sp in DatasetStorage.ALL_SPLITS:
            r_list = splits.get(sp, [])
            split_sets[sp] = set(r_list)
            all_run_ids.extend(r_list)

        unique_run_ids = set(all_run_ids)
        if len(unique_run_ids) == len(all_run_ids):
            checks.append(ValidationCheck("unique_run_ids_splits", "PASS", f"All {len(unique_run_ids)} run IDs unique."))
        else:
            errors.append("Duplicate run IDs detected in splits.json.")
            checks.append(ValidationCheck("unique_run_ids_splits", "FAIL", "Duplicate run IDs across splits."))

        split_overlap = False
        nominal_runs = (
            split_sets.get(DatasetStorage.SPLIT_TRAIN, set())
            | split_sets.get(DatasetStorage.SPLIT_VAL, set())
            | split_sets.get(DatasetStorage.SPLIT_ID_TEST, set())
        )
        ood_runs = split_sets.get(DatasetStorage.SPLIT_OOD_TEST, set())

        if nominal_runs.intersection(ood_runs):
            errors.append("Nominal splits and OOD test split share overlapping run IDs.")
            split_overlap = True
        if split_sets.get(DatasetStorage.SPLIT_TRAIN, set()).intersection(split_sets.get(DatasetStorage.SPLIT_VAL, set())):
            errors.append("Train and validation splits share overlapping run IDs.")
            split_overlap = True
        if split_sets.get(DatasetStorage.SPLIT_TRAIN, set()).intersection(split_sets.get(DatasetStorage.SPLIT_ID_TEST, set())):
            errors.append("Train and ID test splits share overlapping run IDs.")
            split_overlap = True
        if split_sets.get(DatasetStorage.SPLIT_VAL, set()).intersection(split_sets.get(DatasetStorage.SPLIT_ID_TEST, set())):
            errors.append("Validation and ID test splits share overlapping run IDs.")
            split_overlap = True

        if not split_overlap:
            checks.append(ValidationCheck("no_run_overlap_between_splits", "PASS", "Splits are strictly disjoint."))
        else:
            checks.append(ValidationCheck("no_run_overlap_between_splits", "FAIL", "Data leakage: splits overlap."))

        seen_seeds: Set[int] = set()
        duplicate_seeds = 0
        file_structure_ok = True
        obs_schema_ok = True
        cond_schema_ok = True
        gt_schema_ok = True
        meta_schema_ok = True
        leakage_ok = True
        no_nan_or_inf_ok = True
        binary_labels_ok = True
        drift_label_ok = True
        timestamp_aligned_ok = True
        event_consistency_ok = True
        provenance_ok = True
        ood_isolation_ok = True

        total_samples_counted = 0
        total_fault_counted = 0
        total_healthy_counted = 0
        total_events_counted = 0
        unsampled_events_counted = 0

        scenarios_in_split: Dict[str, Dict[str, int]] = {
            DatasetStorage.SPLIT_TRAIN: {},
            DatasetStorage.SPLIT_VAL: {},
            DatasetStorage.SPLIT_ID_TEST: {},
            DatasetStorage.SPLIT_OOD_TEST: {},
        }

        for sp, run_list in splits.items():
            for r_id in run_list:
                run_dir = self.storage.get_run_dir(sp, r_id)
                if not run_dir.exists():
                    errors.append(f"Run directory missing: {run_dir}")
                    file_structure_ok = False
                    continue

                for req_f in self.REQUIRED_RUN_FILES:
                    if not (run_dir / req_f).exists():
                        errors.append(f"Run {r_id}: {req_f} missing (required file: {req_f})")
                        file_structure_ok = False

                if not file_structure_ok:
                    continue

                try:
                    obs_df = pd.read_csv(run_dir / "observations.csv")
                    cond_df = pd.read_csv(run_dir / "conditions.csv")
                    gt_df = pd.read_csv(run_dir / "ground_truth.csv")
                    with open(run_dir / "events.json", "r", encoding="utf-8") as fh:
                        events_data = json.load(fh)
                    with open(run_dir / "metadata.json", "r", encoding="utf-8") as fh:
                        meta_data = json.load(fh)
                except Exception as read_ex:
                    errors.append(f"Error reading run {r_id} data: {read_ex}")
                    continue

                seed_val = int(meta_data.get("random_seed", -1))
                if seed_val in seen_seeds:
                    duplicate_seeds += 1
                seen_seeds.add(seed_val)

                if not meta_data.get("synthetic_data", False):
                    errors.append(f"Run {r_id} missing synthetic_data=True provenance flag.")
                    provenance_ok = False

                scenario_name = str(meta_data.get("scenario", "")).upper()
                if sp in scenarios_in_split:
                    scenarios_in_split[sp][scenario_name] = scenarios_in_split[sp].get(scenario_name, 0) + 1

                is_ood = bool(meta_data.get("is_out_of_distribution", False)) or r_id.startswith("run_ood_")
                if sp == DatasetStorage.SPLIT_OOD_TEST:
                    if not is_ood:
                        errors.append(f"Run {r_id} in ood_test partition is not marked out of distribution.")
                        ood_isolation_ok = False
                else:
                    if is_ood:
                        errors.append(f"OOD run {r_id} found inside nominal partition '{sp}'.")
                        ood_isolation_ok = False

                if not self.REQUIRED_OBSERVATION_COLUMNS.issubset(set(obs_df.columns)):
                    errors.append(f"observations.csv schema mismatch in {r_id}.")
                    obs_schema_ok = False

                if not self.REQUIRED_CONDITION_COLUMNS.issubset(set(cond_df.columns)):
                    errors.append(f"conditions.csv schema mismatch in {r_id}.")
                    cond_schema_ok = False

                if not self.REQUIRED_GROUND_TRUTH_COLUMNS.issubset(set(gt_df.columns)):
                    errors.append(f"ground_truth.csv schema mismatch in {r_id}.")
                    gt_schema_ok = False

                if not self.REQUIRED_METADATA_FIELDS.issubset(set(meta_data.keys())):
                    errors.append(f"metadata.json schema mismatch in {r_id}.")
                    meta_schema_ok = False

                forbidden_found = self.FORBIDDEN_OBSERVATION_COLUMNS.intersection(set(obs_df.columns))
                if forbidden_found:
                    errors.append(f"Data Leakage in {r_id}: observations.csv contains ground-truth: {forbidden_found}")
                    leakage_ok = False

                n_obs = len(obs_df)
                n_gt = len(gt_df)
                n_cond = len(cond_df)

                if n_obs != n_gt or n_obs != n_cond:
                    errors.append(f"Length mismatch in {r_id}: obs={n_obs}, gt={n_gt}, cond={n_cond}")
                    timestamp_aligned_ok = False

                t_obs = obs_df["timestamp_s"].to_numpy()
                t_gt = gt_df["timestamp_s"].to_numpy()

                if not np.allclose(t_obs, t_gt, atol=1e-9):
                    errors.append(f"Timestamp mismatch between observations and ground truth in {r_id}.")
                    timestamp_aligned_ok = False

                if np.any(np.diff(t_obs) <= 0.0):
                    errors.append(f"Non-monotonic timestamps in {r_id}.")
                    timestamp_aligned_ok = False

                if obs_df.isna().any().any() or gt_df.isna().any().any():
                    errors.append(f"NaN values found in run {r_id}.")
                    no_nan_or_inf_ok = False

                if np.isinf(obs_df.select_dtypes(include=[np.number])).any().any():
                    errors.append(f"Infinite values found in observations for {r_id}.")
                    no_nan_or_inf_ok = False

                labels = gt_df["fault_label"].to_numpy()
                unique_labels = set(np.unique(labels))
                if not unique_labels.issubset({0, 1}):
                    errors.append(f"Invalid fault labels in {r_id}: {unique_labels}")
                    binary_labels_ok = False

                if scenario_name in ("MEASUREMENT_OFFSET_DRIFT", "OPERATING_CONDITION_SHIFT"):
                    if np.any(labels != 0):
                        errors.append(f"Scenario {scenario_name} in {r_id} must have fault_label == 0 throughout.")
                        drift_label_ok = False

                total_samples_counted += n_obs
                total_fault_counted += int(np.sum(labels == 1))
                total_healthy_counted += int(np.sum(labels == 0))

                for ev in events_data:
                    total_events_counted += 1
                    st_time = float(ev.get("start_time_s", ev.get("t_start_s", 0.0)))
                    en_time = float(ev.get("end_time_s", ev.get("t_end_s", 0.0)))
                    is_sampled = bool(ev.get("sampled_event", True))

                    if not is_sampled:
                        unsampled_events_counted += 1

                    ev_points = (t_obs >= st_time) & (t_obs < en_time)
                    if is_sampled and np.any(ev_points):
                        if not np.all(labels[ev_points] == 1):
                            errors.append(f"Fault event window mismatch in {r_id} for event {ev.get('event_id')}")
                            event_consistency_ok = False

        nominal_scenarios_ok = True
        missing_nominal: Dict[str, Set[str]] = {}
        for sp in [DatasetStorage.SPLIT_TRAIN, DatasetStorage.SPLIT_VAL, DatasetStorage.SPLIT_ID_TEST]:
            found_scs = set(scenarios_in_split.get(sp, {}).keys())
            missing = set(PowerPathSimulator.ALL_SCENARIOS) - found_scs
            if missing:
                missing_nominal[sp] = missing
                nominal_scenarios_ok = False

        if nominal_scenarios_ok:
            checks.append(ValidationCheck("nominal_scenarios_in_all_splits", "PASS", "All nominal scenarios represented in train, val, and id_test."))
        else:
            errors.append(f"Nominal scenarios missing from partitions: {missing_nominal}")
            checks.append(ValidationCheck("nominal_scenarios_in_all_splits", "FAIL", f"Missing scenarios: {missing_nominal}."))

        allocation_consistent = True
        for sp in [DatasetStorage.SPLIT_TRAIN, DatasetStorage.SPLIT_VAL, DatasetStorage.SPLIT_ID_TEST]:
            counts = list(scenarios_in_split.get(sp, {}).values())
            if len(counts) > 0 and (len(set(counts)) > 1 or counts[0] <= 0):
                allocation_consistent = False

        if allocation_consistent:
            checks.append(ValidationCheck("scenario_run_allocation_valid", "PASS", "Scenario run counts are balanced across nominal splits."))
        else:
            warnings.append("Scenario run counts are not perfectly uniform across nominal splits.")
            checks.append(ValidationCheck("scenario_run_allocation_valid", "WARNING", "Non-uniform scenario counts."))

        checks.append(ValidationCheck("ood_partition_isolation", "PASS" if ood_isolation_ok else "FAIL", "OOD runs strictly quarantined."))

        if duplicate_seeds == 0:
            checks.append(ValidationCheck("unique_run_seeds", "PASS", f"All {len(seen_seeds)} run seeds unique."))
        else:
            errors.append(f"Detected {duplicate_seeds} duplicate run seeds across simulation runs.")
            checks.append(ValidationCheck("unique_run_seeds", "FAIL", f"{duplicate_seeds} duplicate seeds."))

        checks.append(ValidationCheck("file_structure_integrity", "PASS" if file_structure_ok else "FAIL", "All run files present."))
        checks.append(ValidationCheck("observations_schema", "PASS" if obs_schema_ok else "FAIL", "observations.csv schema valid."))
        checks.append(ValidationCheck("conditions_schema", "PASS" if cond_schema_ok else "FAIL", "conditions.csv schema valid."))
        checks.append(ValidationCheck("ground_truth_schema", "PASS" if gt_schema_ok else "FAIL", "ground_truth.csv schema valid."))
        checks.append(ValidationCheck("metadata_schema", "PASS" if meta_schema_ok else "FAIL", "metadata.json schema valid."))
        checks.append(ValidationCheck("no_ground_truth_leakage", "PASS" if leakage_ok else "FAIL", "Zero ground-truth in observations."))
        checks.append(ValidationCheck("no_nan_or_inf", "PASS" if no_nan_or_inf_ok else "FAIL", "Signals finite without NaN or Inf."))
        checks.append(ValidationCheck("binary_labels_valid", "PASS" if binary_labels_ok else "FAIL", "Labels in {0, 1}."))
        checks.append(ValidationCheck("sensor_drift_label_integrity", "PASS" if drift_label_ok else "FAIL", "Drift and shift have label 0."))
        checks.append(ValidationCheck("timestamp_monotonic_and_aligned", "PASS" if timestamp_aligned_ok else "FAIL", "Timestamps monotonic and aligned."))
        checks.append(ValidationCheck("event_sampling_consistency", "PASS" if event_consistency_ok else "FAIL", "Events match ground truth labels."))
        checks.append(ValidationCheck("synthetic_provenance_flag", "PASS" if provenance_ok else "FAIL", "Runs tagged synthetic_data=True."))

        manifest: Optional[DatasetManifest] = None
        if manifest_exists:
            try:
                manifest = self.storage.load_manifest()
            except Exception as ex:
                errors.append(f"Failed to parse manifest.json: {ex}")

        checksum_ok = True
        if manifest and manifest.file_checksums_sha256:
            for rel_p_str, expected_hash in manifest.file_checksums_sha256.items():
                p_file = root / rel_p_str
                if not p_file.exists():
                    errors.append(f"File missing during checksum verification: {rel_p_str}")
                    checksum_ok = False
                    continue
                actual_hash = compute_file_sha256(p_file)
                if actual_hash != expected_hash:
                    errors.append(f"Checksum mismatch for {rel_p_str}: computed {actual_hash} != manifest {expected_hash}")
                    checksum_ok = False
            checks.append(ValidationCheck("checksum_integrity", "PASS" if checksum_ok else "FAIL", "SHA-256 checksums verified."))
        elif manifest:
            checks.append(ValidationCheck("checksum_integrity", "WARNING", "No checksums present in manifest."))
        else:
            checks.append(ValidationCheck("checksum_integrity", "FAIL", "Manifest missing for checksum verification."))

        manifest_match_ok = True
        if manifest:
            if manifest.total_simulation_runs != len(all_run_ids):
                errors.append(f"Manifest total_runs ({manifest.total_simulation_runs}) != actual runs ({len(all_run_ids)})")
                manifest_match_ok = False
            if manifest.total_samples != total_samples_counted:
                errors.append(f"Manifest total_samples ({manifest.total_samples}) != actual samples ({total_samples_counted})")
                manifest_match_ok = False
            if manifest.total_fault_samples != total_fault_counted:
                errors.append(f"Manifest total_fault_samples ({manifest.total_fault_samples}) != actual ({total_fault_counted})")
                manifest_match_ok = False
            checks.append(ValidationCheck("manifest_counts_consistency", "PASS" if manifest_match_ok else "FAIL", "Manifest counts verified against files."))
        else:
            checks.append(ValidationCheck("manifest_counts_consistency", "FAIL", "Manifest missing for count verification."))

        overall_valid = len(errors) == 0
        overall_status = "PASS" if overall_valid else "FAIL"
        if overall_valid and len(warnings) > 0:
            overall_status = "WARNING"

        summary = {
            "total_runs": len(all_run_ids),
            "total_samples": total_samples_counted,
            "total_fault_samples": total_fault_counted,
            "total_healthy_samples": total_healthy_counted,
            "fault_ratio": float(total_fault_counted / total_samples_counted) if total_samples_counted > 0 else 0.0,
            "total_events": total_events_counted,
            "unsampled_events": unsampled_events_counted,
        }

        return DatasetValidationReport(
            dataset_id=self.storage.dataset_id,
            status=overall_status,
            is_valid=overall_valid,
            checks=checks,
            errors=errors,
            warnings=warnings,
            summary=summary,
        )


def validate_synthetic_dataset(dataset_path: Path) -> DatasetValidationReport:
    validator = DatasetValidator(dataset_path)
    return validator.validate()
