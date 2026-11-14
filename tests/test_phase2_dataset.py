"""
Automated unit and integration tests for Phase 2: Synthetic Dataset Generator and Validator.
"""

from pathlib import Path
import json
import pandas as pd
import pytest

from flexpulse_ml.data.generator import (
    DatasetGeneratorConfig,
    SyntheticDatasetGenerator,
    generate_synthetic_dataset,
)
from flexpulse_ml.data.manifest import DatasetManifest, compute_file_sha256
from flexpulse_ml.data.storage import DatasetStorage
from flexpulse_ml.data.validator import DatasetValidator, validate_synthetic_dataset
from flexpulse_ml.simulation.fault_injector import PowerPathSimulator


@pytest.fixture
def temp_dataset_dir(tmp_path: Path) -> Path:
    return tmp_path / "data_storage"


def test_generation_all_six_scenarios(temp_dataset_dir: Path):
    manifest = generate_synthetic_dataset(
        dataset_id="test_all_scenarios",
        runs_per_scenario=1,
        ood_runs_per_scenario=1,
        root_seed=101,
        base_dir=temp_dataset_dir,
    )

    assert manifest.total_simulation_runs == 12
    assert len(manifest.run_counts_per_scenario) == 6
    for scenario in PowerPathSimulator.ALL_SCENARIOS:
        assert manifest.run_counts_per_scenario[scenario] == 2

    storage = DatasetStorage(dataset_id="test_all_scenarios", base_dir=temp_dataset_dir)
    assert storage.manifest_path.exists()
    assert storage.splits_path.exists()

    all_runs = storage.list_all_runs()
    assert len(all_runs) == 12
    for run_id in all_runs:
        split = storage.get_run_split(run_id)
        assert split is not None
        obs_df = storage.load_observations(run_id, split)
        gt_df = storage.load_ground_truth(run_id, split)
        cond_df = storage.load_conditions(run_id, split)
        events = storage.load_events(run_id, split)
        meta = storage.load_metadata(run_id, split)

        assert not obs_df.empty
        assert not gt_df.empty
        assert not cond_df.empty
        assert isinstance(events, list)
        assert meta["run_id"] == run_id
        assert meta["synthetic_data"] is True


def test_unique_run_ids_and_seeds(temp_dataset_dir: Path):
    manifest = generate_synthetic_dataset(
        dataset_id="test_uniqueness",
        runs_per_scenario=2,
        ood_runs_per_scenario=1,
        root_seed=202,
        base_dir=temp_dataset_dir,
    )

    storage = DatasetStorage(dataset_id="test_uniqueness", base_dir=temp_dataset_dir)
    all_runs = storage.list_all_runs()
    assert len(all_runs) == 18
    assert len(set(all_runs)) == 18

    observed_seeds = set()
    for run_id in all_runs:
        split = storage.get_run_split(run_id)
        meta = storage.load_metadata(run_id, split)
        observed_seeds.add(meta["random_seed"])

    assert len(observed_seeds) == 18


def test_exact_reproducibility_identical_root_seed(tmp_path: Path):
    dir_a = tmp_path / "run_a"
    dir_b = tmp_path / "run_b"

    manifest_a = generate_synthetic_dataset(
        dataset_id="dataset_repro",
        runs_per_scenario=1,
        ood_runs_per_scenario=1,
        root_seed=999,
        base_dir=dir_a,
    )
    manifest_b = generate_synthetic_dataset(
        dataset_id="dataset_repro",
        runs_per_scenario=1,
        ood_runs_per_scenario=1,
        root_seed=999,
        base_dir=dir_b,
    )

    assert manifest_a.file_checksums_sha256 == manifest_b.file_checksums_sha256

    storage_a = DatasetStorage(dataset_id="dataset_repro", base_dir=dir_a)
    storage_b = DatasetStorage(dataset_id="dataset_repro", base_dir=dir_b)

    for run_id in storage_a.list_all_runs():
        split = storage_a.get_run_split(run_id)
        df_a = storage_a.load_observations(run_id, split)
        df_b = storage_b.load_observations(run_id, split)
        pd.testing.assert_frame_equal(df_a, df_b)


def test_noise_divergence_different_seeds(tmp_path: Path):
    dir_1 = tmp_path / "seed_1"
    dir_2 = tmp_path / "seed_2"

    generate_synthetic_dataset(
        dataset_id="dataset_seed",
        runs_per_scenario=1,
        ood_runs_per_scenario=0,
        root_seed=111,
        base_dir=dir_1,
    )
    generate_synthetic_dataset(
        dataset_id="dataset_seed",
        runs_per_scenario=1,
        ood_runs_per_scenario=0,
        root_seed=888,
        base_dir=dir_2,
    )

    storage_1 = DatasetStorage(dataset_id="dataset_seed", base_dir=dir_1)
    storage_2 = DatasetStorage(dataset_id="dataset_seed", base_dir=dir_2)

    run_id = storage_1.list_all_runs()[0]
    split = storage_1.get_run_split(run_id)

    obs_1 = storage_1.load_observations(run_id, split)
    obs_2 = storage_2.load_observations(run_id, split)

    assert not obs_1["voltage_measured_v"].equals(obs_2["voltage_measured_v"])


def test_balanced_vs_imbalanced_modes(temp_dataset_dir: Path):
    gen_balanced = SyntheticDatasetGenerator(
        config=DatasetGeneratorConfig(
            dataset_id="balanced_ds",
            mode=SyntheticDatasetGenerator.MODE_BALANCED,
            runs_per_scenario=2,
            ood_runs_per_scenario=0,
        ),
        storage_base_dir=temp_dataset_dir,
    )
    manifest_bal = gen_balanced.generate()
    for sc in PowerPathSimulator.ALL_SCENARIOS:
        assert manifest_bal.run_counts_per_scenario[sc] == 2

    custom_counts = {
        "HEALTHY": 5,
        "INCREASED_RESISTANCE": 3,
        "INTERMITTENT_CONTACT": 2,
        "GRADUAL_RESISTANCE_DRIFT": 1,
        "MEASUREMENT_OFFSET_DRIFT": 1,
        "OPERATING_CONDITION_SHIFT": 1,
    }
    gen_imbalanced = SyntheticDatasetGenerator(
        config=DatasetGeneratorConfig(
            dataset_id="imbalanced_ds",
            mode=SyntheticDatasetGenerator.MODE_IMBALANCED,
            custom_scenario_counts=custom_counts,
            ood_runs_per_scenario=0,
        ),
        storage_base_dir=temp_dataset_dir,
    )
    manifest_imbal = gen_imbalanced.generate()
    for sc, count in custom_counts.items():
        assert manifest_imbal.run_counts_per_scenario[sc] == count


def test_ground_truth_segregation(temp_dataset_dir: Path):
    generate_synthetic_dataset(
        dataset_id="segregation_ds",
        runs_per_scenario=1,
        ood_runs_per_scenario=0,
        root_seed=42,
        base_dir=temp_dataset_dir,
    )

    storage = DatasetStorage(dataset_id="segregation_ds", base_dir=temp_dataset_dir)
    run_id = storage.list_all_runs()[0]
    split = storage.get_run_split(run_id)

    obs_df = storage.load_observations(run_id, split)
    expected_observable = {
        "timestamp_s",
        "voltage_measured_v",
        "current_measured_a",
        "source_voltage_v",
    }
    assert set(obs_df.columns) == expected_observable

    forbidden_columns = {
        "true_path_resistance_ohm",
        "fault_label",
        "fault_type",
        "event_id",
        "v_load_true_v",
        "i_true_a",
        "voltage_bias_v",
        "current_bias_a",
    }
    for col in forbidden_columns:
        assert col not in obs_df.columns

    gt_df = storage.load_ground_truth(run_id, split)
    for col in ["fault_label", "fault_type", "event_id"]:
        assert col in gt_df.columns


def test_split_proportions_and_zero_leakage(temp_dataset_dir: Path):
    generate_synthetic_dataset(
        dataset_id="split_leakage_ds",
        runs_per_scenario=20,
        ood_runs_per_scenario=5,
        root_seed=777,
        base_dir=temp_dataset_dir,
    )

    storage = DatasetStorage(dataset_id="split_leakage_ds", base_dir=temp_dataset_dir)
    train_runs = set(storage.list_runs_in_split(DatasetStorage.SPLIT_TRAIN))
    val_runs = set(storage.list_runs_in_split(DatasetStorage.SPLIT_VAL))
    id_test_runs = set(storage.list_runs_in_split(DatasetStorage.SPLIT_ID_TEST))
    ood_runs = set(storage.list_runs_in_split(DatasetStorage.SPLIT_OOD_TEST))

    assert len(train_runs) == 84
    assert len(val_runs) == 18
    assert len(id_test_runs) == 18
    assert len(ood_runs) == 30
    assert len(train_runs) + len(val_runs) + len(id_test_runs) == 120

    assert len(train_runs.intersection(val_runs)) == 0
    assert len(train_runs.intersection(id_test_runs)) == 0
    assert len(val_runs.intersection(id_test_runs)) == 0
    assert len(train_runs.intersection(ood_runs)) == 0
    assert len(val_runs.intersection(ood_runs)) == 0
    assert len(id_test_runs.intersection(ood_runs)) == 0


def test_ood_parameter_shift_isolation(temp_dataset_dir: Path):
    generate_synthetic_dataset(
        dataset_id="ood_check_ds",
        runs_per_scenario=1,
        ood_runs_per_scenario=1,
        root_seed=505,
        base_dir=temp_dataset_dir,
    )

    storage = DatasetStorage(dataset_id="ood_check_ds", base_dir=temp_dataset_dir)
    ood_runs = storage.list_runs_in_split(DatasetStorage.SPLIT_OOD_TEST)
    assert len(ood_runs) == 6

    for r_id in ood_runs:
        meta = storage.load_metadata(r_id, DatasetStorage.SPLIT_OOD_TEST)
        assert meta["split"] == DatasetStorage.SPLIT_OOD_TEST
        assert meta["is_out_of_distribution"] is True
        assert meta["noise_multiplier"] >= 2.0


def test_event_sampling_validation(temp_dataset_dir: Path):
    generate_synthetic_dataset(
        dataset_id="event_sampling_ds",
        runs_per_scenario=2,
        ood_runs_per_scenario=0,
        root_seed=303,
        base_dir=temp_dataset_dir,
    )

    storage = DatasetStorage(dataset_id="event_sampling_ds", base_dir=temp_dataset_dir)
    all_runs = storage.list_all_runs()
    events_found = False
    for r_id in all_runs:
        split = storage.get_run_split(r_id)
        events = storage.load_events(r_id, split)
        if events:
            events_found = True
            for ev in events:
                assert "sampled_event" in ev
                assert isinstance(ev["sampled_event"], bool)
                assert "event_id" in ev
                assert "t_start_s" in ev
                assert "t_end_s" in ev
    assert events_found is True


def test_dataset_validator_success(temp_dataset_dir: Path):
    generate_synthetic_dataset(
        dataset_id="valid_ds",
        runs_per_scenario=3,
        ood_runs_per_scenario=1,
        root_seed=4242,
        base_dir=temp_dataset_dir,
    )

    dataset_path = temp_dataset_dir / "valid_ds"
    report = validate_synthetic_dataset(dataset_path)

    assert report.is_valid is True
    assert report.status == "PASS"
    assert len(report.errors) == 0
    assert len(report.checks) >= 15


def test_dataset_validator_fails_if_id_test_empty(temp_dataset_dir: Path):
    generate_synthetic_dataset(
        dataset_id="no_id_test_ds",
        runs_per_scenario=2,
        ood_runs_per_scenario=1,
        root_seed=123,
        base_dir=temp_dataset_dir,
    )

    dataset_path = temp_dataset_dir / "no_id_test_ds"
    report = validate_synthetic_dataset(dataset_path)

    assert report.is_valid is False
    assert report.status == "FAIL"
    assert any("Empty partitions" in err or "id_test" in err for err in report.errors)


def test_dataset_validator_detects_missing_file(temp_dataset_dir: Path):
    generate_synthetic_dataset(
        dataset_id="missing_file_ds",
        runs_per_scenario=3,
        ood_runs_per_scenario=1,
        root_seed=123,
        base_dir=temp_dataset_dir,
    )

    dataset_path = temp_dataset_dir / "missing_file_ds"
    storage = DatasetStorage(dataset_id="missing_file_ds", base_dir=temp_dataset_dir)
    first_run = storage.list_all_runs()[0]
    split = storage.get_run_split(first_run)
    obs_file = storage.dataset_dir / split / "runs" / first_run / "observations.csv"
    obs_file.unlink()

    report = validate_synthetic_dataset(dataset_path)
    assert report.is_valid is False
    assert report.status == "FAIL"
    assert any("observations.csv missing" in err for err in report.errors)


def test_dataset_validator_detects_data_leakage(temp_dataset_dir: Path):
    generate_synthetic_dataset(
        dataset_id="leakage_ds",
        runs_per_scenario=3,
        ood_runs_per_scenario=1,
        root_seed=456,
        base_dir=temp_dataset_dir,
    )

    dataset_path = temp_dataset_dir / "leakage_ds"
    storage = DatasetStorage(dataset_id="leakage_ds", base_dir=temp_dataset_dir)
    first_run = storage.list_all_runs()[0]
    split = storage.get_run_split(first_run)
    obs_file = storage.dataset_dir / split / "runs" / first_run / "observations.csv"

    df = pd.read_csv(obs_file)
    df["fault_label"] = 1
    df.to_csv(obs_file, index=False)

    report = validate_synthetic_dataset(dataset_path)
    assert report.is_valid is False
    assert report.status == "FAIL"
    assert any("Data Leakage" in err or "Checksum mismatch" in err for err in report.errors)


def test_dataset_validator_detects_checksum_tampering(temp_dataset_dir: Path):
    generate_synthetic_dataset(
        dataset_id="tampered_ds",
        runs_per_scenario=3,
        ood_runs_per_scenario=1,
        root_seed=789,
        base_dir=temp_dataset_dir,
    )

    dataset_path = temp_dataset_dir / "tampered_ds"
    storage = DatasetStorage(dataset_id="tampered_ds", base_dir=temp_dataset_dir)
    first_run = storage.list_all_runs()[0]
    split = storage.get_run_split(first_run)
    cond_file = storage.dataset_dir / split / "runs" / first_run / "conditions.csv"

    with open(cond_file, "a", encoding="utf-8") as f:
        f.write("\n9999.0,0.0,0.0,0.0\n")

    report = validate_synthetic_dataset(dataset_path)
    assert report.is_valid is False
    assert report.status == "FAIL"
    assert any("Checksum mismatch" in err for err in report.errors)


def test_manifest_schema_and_serialization(tmp_path: Path):
    manifest = DatasetManifest(
        dataset_id="manifest_test",
        total_simulation_runs=10,
        total_samples=1000,
        binary_class_distribution={"0": 0.8, "1": 0.2},
    )

    manifest_file = tmp_path / "manifest.json"
    manifest.save_json(manifest_file)
    assert manifest_file.exists()

    loaded = DatasetManifest.load_json(manifest_file)
    assert loaded.dataset_id == "manifest_test"
    assert loaded.total_simulation_runs == 10
    assert loaded.total_samples == 1000
    assert loaded.binary_class_distribution == {"0": 0.8, "1": 0.2}
    assert loaded.synthetic_data is True
