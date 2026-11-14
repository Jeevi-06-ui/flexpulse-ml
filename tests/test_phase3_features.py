"""
Automated unit and integration tests for Phase 3: Electrical Feature Engineering Pipeline.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

from flexpulse_ml.data.generator import generate_synthetic_dataset
from flexpulse_ml.features.dataset import (
    FeatureDatasetGenerator,
    FeatureDatasetStorage,
    FeatureManifest,
    generate_feature_dataset,
)
from flexpulse_ml.features.extractor import (
    ElectricalFeatureExtractor,
    FEATURE_NAMES,
    FEATURE_UNITS,
)
from flexpulse_ml.features.validator import (
    FeatureValidator,
    validate_feature_dataset,
)
from flexpulse_ml.features.windowing import (
    SlidingWindowGenerator,
    Window,
    WindowConfig,
)


@pytest.fixture
def temp_features_dir(tmp_path: Path) -> Path:
    return tmp_path / "processed_features"


@pytest.fixture
def temp_raw_dataset_dir(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "raw_data"
    generate_synthetic_dataset(
        dataset_id="test_ds_phase3",
        runs_per_scenario=3,
        ood_runs_per_scenario=1,
        root_seed=123,
        base_dir=raw_dir,
    )
    return raw_dir


def test_window_config_stride_and_samples():
    cfg = WindowConfig(sampling_rate_hz=100.0, window_duration_s=0.5, overlap_fraction=0.5)
    assert cfg.window_size_samples == 50
    assert cfg.stride_samples == 25

    cfg_80 = WindowConfig(sampling_rate_hz=200.0, window_duration_s=1.0, overlap_fraction=0.8)
    assert cfg_80.window_size_samples == 200
    assert cfg_80.stride_samples == 40


def test_sliding_window_generator():
    cfg = WindowConfig(sampling_rate_hz=100.0, window_duration_s=0.5, overlap_fraction=0.5)
    gen = SlidingWindowGenerator(config=cfg)

    t = np.linspace(0.0, 1.0, 100, endpoint=False)
    v = np.full(100, 5.0)
    i = np.full(100, 1.5)
    gt = np.zeros(100, dtype=int)
    gt[20:30] = 1

    obs_df = pd.DataFrame({"timestamp_s": t, "voltage_measured_v": v, "current_measured_a": i})
    gt_df = pd.DataFrame({"timestamp_s": t, "fault_label": gt})

    windows = gen.slice_run(obs_df, gt_df, run_id="run_001", scenario="HEALTHY")
    assert len(windows) == 3

    assert windows[0].start_idx == 0
    assert windows[0].end_idx == 50
    assert windows[0].fault_label == 1
    assert np.isclose(windows[0].fault_fraction, 10 / 50)

    assert windows[1].start_idx == 25
    assert windows[1].end_idx == 75
    assert windows[1].fault_label == 1
    assert np.isclose(windows[1].fault_fraction, 5 / 50)

    assert windows[2].start_idx == 50
    assert windows[2].end_idx == 100
    assert windows[2].fault_label == 0
    assert windows[2].fault_fraction == 0.0


def test_window_label_and_fault_fraction():
    cfg = WindowConfig(sampling_rate_hz=10.0, window_duration_s=1.0, overlap_fraction=0.0)
    gen = SlidingWindowGenerator(config=cfg)

    t = np.arange(20) * 0.1
    v = np.ones(20)
    i = np.ones(20)
    gt = np.zeros(20, dtype=int)
    gt[5] = 1

    obs_df = pd.DataFrame({"timestamp_s": t, "voltage_measured_v": v, "current_measured_a": i})
    gt_df = pd.DataFrame({"timestamp_s": t, "fault_label": gt})

    windows = gen.slice_run(obs_df, gt_df, run_id="run_test", scenario="INTERMITTENT")
    assert len(windows) == 2
    assert windows[0].fault_label == 1
    assert np.isclose(windows[0].fault_fraction, 1 / 10)
    assert windows[1].fault_label == 0
    assert windows[1].fault_fraction == 0.0


def test_feature_extractor_known_signals():
    extractor = ElectricalFeatureExtractor(sampling_rate_hz=100.0)

    const_v = np.full(50, 4.80)
    feats = extractor.compute_signal_features(const_v, "voltage", "v")
    assert np.isclose(feats["voltage_mean_v"], 4.80)
    assert np.isclose(feats["voltage_median_v"], 4.80)
    assert np.isclose(feats["voltage_std_v"], 0.0)
    assert np.isclose(feats["voltage_range_v"], 0.0)
    assert np.isclose(feats["voltage_mad_v"], 0.0)
    assert np.isclose(feats["voltage_trend_slope_v_per_s"], 0.0)
    assert np.isclose(feats["voltage_mean_abs_diff_v"], 0.0)
    assert np.isclose(feats["voltage_max_abs_diff_v"], 0.0)

    t = np.arange(50) * 0.01
    ramp_v = 1.0 + 2.0 * t
    ramp_feats = extractor.compute_signal_features(ramp_v, "voltage", "v")
    assert np.isclose(ramp_feats["voltage_trend_slope_v_per_s"], 2.0, atol=1e-6)
    assert np.isclose(ramp_feats["voltage_min_v"], 1.0)
    assert np.isclose(ramp_feats["voltage_max_v"], 1.0 + 2.0 * 0.49)
    assert np.isclose(ramp_feats["voltage_mean_abs_diff_v"], 0.02, atol=1e-6)

    step_i = np.zeros(50)
    step_i[25:] = 1.5
    step_feats = extractor.compute_signal_features(step_i, "current", "a")
    assert np.isclose(step_feats["current_max_abs_diff_a"], 1.5)
    assert np.isclose(step_feats["current_range_a"], 1.5)


def test_feature_names_and_units_consistency():
    assert len(FEATURE_NAMES) == 26
    assert len(FEATURE_UNITS) == 26
    v_cols = [c for c in FEATURE_NAMES if c.startswith("voltage_")]
    i_cols = [c for c in FEATURE_NAMES if c.startswith("current_")]
    assert len(v_cols) == 13
    assert len(i_cols) == 13
    for name, unit in FEATURE_UNITS.items():
        assert unit in ["V", "V/s", "A", "A/s"]


def test_feature_dataset_generation(temp_raw_dataset_dir: Path, temp_features_dir: Path):
    manifest = generate_feature_dataset(
        dataset_id="test_ds_phase3",
        window_duration_s=0.5,
        overlap_fraction=0.5,
        source_base_dir=temp_raw_dataset_dir,
        output_base_dir=temp_features_dir,
    )

    assert manifest.source_dataset_id == "test_ds_phase3"
    assert manifest.total_windows > 0
    assert len(manifest.partition_window_counts) == 4
    for sp in ["train", "validation", "id_test", "ood_test"]:
        assert manifest.partition_window_counts[sp] > 0

    storage = FeatureDatasetStorage(dataset_id="test_ds_phase3", base_dir=temp_features_dir)
    assert storage.manifest_path.exists()

    for sp in ["train", "validation", "id_test", "ood_test"]:
        df = storage.load_partition(sp)
        assert not df.empty
        assert "run_id" in df.columns
        assert "fault_label" in df.columns
        assert "fault_fraction" in df.columns
        for f_name in FEATURE_NAMES:
            assert f_name in df.columns


def test_feature_validator_success(temp_raw_dataset_dir: Path, temp_features_dir: Path):
    generate_feature_dataset(
        dataset_id="test_ds_phase3",
        window_duration_s=0.5,
        overlap_fraction=0.5,
        source_base_dir=temp_raw_dataset_dir,
        output_base_dir=temp_features_dir,
    )

    report = validate_feature_dataset(temp_features_dir / "test_ds_phase3")
    assert report.is_valid is True
    assert report.status == "PASS"
    assert len(report.errors) == 0
    assert len(report.checks) >= 10


def test_feature_validator_detects_nan(temp_raw_dataset_dir: Path, temp_features_dir: Path):
    generate_feature_dataset(
        dataset_id="test_ds_phase3",
        window_duration_s=0.5,
        overlap_fraction=0.5,
        source_base_dir=temp_raw_dataset_dir,
        output_base_dir=temp_features_dir,
    )

    storage = FeatureDatasetStorage(dataset_id="test_ds_phase3", base_dir=temp_features_dir)
    train_path = storage.get_partition_path("train")
    df = pd.read_csv(train_path)
    df.loc[0, "voltage_mean_v"] = np.nan
    df.to_csv(train_path, index=False)

    report = validate_feature_dataset(temp_features_dir / "test_ds_phase3")
    assert report.is_valid is False
    assert report.status == "FAIL"
    assert any("NaN" in err or "Checksum mismatch" in err for err in report.errors)


def test_feature_validator_detects_leakage(temp_raw_dataset_dir: Path, temp_features_dir: Path):
    generate_feature_dataset(
        dataset_id="test_ds_phase3",
        window_duration_s=0.5,
        overlap_fraction=0.5,
        source_base_dir=temp_raw_dataset_dir,
        output_base_dir=temp_features_dir,
    )

    storage = FeatureDatasetStorage(dataset_id="test_ds_phase3", base_dir=temp_features_dir)
    val_path = storage.get_partition_path("validation")
    df = pd.read_csv(val_path)
    df["true_path_resistance_ohm"] = 0.08
    df.to_csv(val_path, index=False)

    report = validate_feature_dataset(temp_features_dir / "test_ds_phase3")
    assert report.is_valid is False
    assert report.status == "FAIL"
    assert any("Forbidden" in err or "Checksum mismatch" in err for err in report.errors)


def test_feature_validator_detects_corrupt_hash(temp_raw_dataset_dir: Path, temp_features_dir: Path):
    generate_feature_dataset(
        dataset_id="test_ds_phase3",
        window_duration_s=0.5,
        overlap_fraction=0.5,
        source_base_dir=temp_raw_dataset_dir,
        output_base_dir=temp_features_dir,
    )

    storage = FeatureDatasetStorage(dataset_id="test_ds_phase3", base_dir=temp_features_dir)
    test_path = storage.get_partition_path("id_test")
    with open(test_path, "a", encoding="utf-8") as f:
        f.write("\ncorrupt_data,0,0,0,0,0\n")

    report = validate_feature_dataset(temp_features_dir / "test_ds_phase3")
    assert report.is_valid is False
    assert report.status == "FAIL"
    assert any("Checksum mismatch" in err for err in report.errors)


def test_deterministic_feature_extraction():
    extractor = ElectricalFeatureExtractor(sampling_rate_hz=100.0)
    w1 = Window(
        run_id="r1",
        window_index=0,
        start_idx=0,
        end_idx=50,
        start_time_s=0.0,
        end_time_s=0.5,
        voltage=np.array([4.9, 5.0, 5.1, 4.8]),
        current=np.array([1.2, 1.25, 1.15, 1.3]),
        fault_label=0,
        fault_fraction=0.0,
        scenario="HEALTHY",
    )
    w2 = Window(
        run_id="r1",
        window_index=0,
        start_idx=0,
        end_idx=50,
        start_time_s=0.0,
        end_time_s=0.5,
        voltage=np.array([4.9, 5.0, 5.1, 4.8]),
        current=np.array([1.2, 1.25, 1.15, 1.3]),
        fault_label=0,
        fault_fraction=0.0,
        scenario="HEALTHY",
    )
    f1 = extractor.extract_from_window(w1)
    f2 = extractor.extract_from_window(w2)
    assert f1 == f2
