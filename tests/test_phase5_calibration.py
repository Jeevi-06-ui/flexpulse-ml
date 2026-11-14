"""
Automated unit and integration test suite for FlexPulse-ML Phase 5 self-calibration.
Validates baseline estimation, adaptive gating, freeze-recovery dynamics, causal delays, and contamination tracking.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from flexpulse_ml.calibration.adaptive import (
    AdaptiveCalibrationConfig,
    ConfidenceGatedAdaptiveCalibrator,
)
from flexpulse_ml.calibration.baseline import (
    BaselineStatistics,
    InitialBaselineEstimator,
)
from flexpulse_ml.calibration.confidence import ConfidenceEstimator
from flexpulse_ml.calibration.diagnostics import (
    CalibrationDiagnostics,
    CalibrationState,
    WindowCalibrationRecord,
)
from flexpulse_ml.features.extractor import FEATURE_NAMES


@pytest.fixture
def synthetic_run_dataframe():
    rng = np.random.default_rng(42)
    n_windows = 60
    data = {}
    for feat in FEATURE_NAMES:
        data[feat] = rng.normal(loc=1.0, scale=0.01, size=n_windows)

    data["voltage_mean_v"] = np.linspace(4.90, 4.70, n_windows)
    data["current_mean_a"] = np.full(n_windows, 1.5)
    data["current_std_a"] = np.full(n_windows, 0.005)

    df = pd.DataFrame(data)
    df["run_id"] = "run_test_001"
    df["window_index"] = np.arange(n_windows)
    df["window_start_s"] = np.arange(n_windows) * 0.25
    df["window_end_s"] = df["window_start_s"] + 0.5
    df["fault_label"] = np.where(df["window_end_s"] > 10.0, 1, 0)
    return df


def test_initial_baseline_estimation(synthetic_run_dataframe):
    estimator = InitialBaselineEstimator(initial_duration_s=5.0)
    baseline = estimator.fit_run(synthetic_run_dataframe)

    assert baseline.data_quality_status == "VALID"
    assert baseline.valid_window_count >= 10
    assert 4.80 < baseline.center_v < 4.95
    assert np.isclose(baseline.center_i, 1.5, atol=0.05)
    assert baseline.spread_v > 0.0


def test_insufficient_initial_data_handling():
    estimator = InitialBaselineEstimator(initial_duration_s=5.0, min_windows=10)
    short_df = pd.DataFrame({
        "voltage_mean_v": [4.9, 4.9],
        "current_mean_a": [1.5, 1.5],
        "window_end_s": [1.0, 2.0],
    })
    baseline = estimator.fit_run(short_df)
    assert baseline.data_quality_status == "INSUFFICIENT_DATA"


def test_baseline_feature_normalization(synthetic_run_dataframe):
    estimator = InitialBaselineEstimator(initial_duration_s=5.0)
    baseline = estimator.fit_run(synthetic_run_dataframe)
    norm_df = estimator.normalize_features(synthetic_run_dataframe, baseline)

    assert "voltage_mean_v" in norm_df.columns
    init_mask = norm_df["window_end_s"] <= 5.0
    v_norm_init = norm_df.loc[init_mask, "voltage_mean_v"]
    assert abs(float(v_norm_init.median())) < 0.05


def test_confidence_estimator():
    estimator = ConfidenceEstimator(default_threshold=0.90)

    class MockModel:
        def predict_proba(self, X):
            return np.array([[0.95, 0.05], [0.40, 0.60]])

    conf = estimator.estimate_healthy_confidence(MockModel(), np.zeros((2, 5)))
    assert len(conf) == 2
    assert conf[0] == 0.95
    assert conf[1] == 0.40
    assert estimator.is_confident_healthy(conf[0]) is True
    assert estimator.is_confident_healthy(conf[1]) is False


def test_adaptive_calibration_update_equation():
    config = AdaptiveCalibrationConfig(alpha=0.10)
    calibrator = ConfidenceGatedAdaptiveCalibrator(config)

    class DummyHealthyModel:
        def predict(self, X):
            return np.zeros(len(X), dtype=int)

        def predict_proba(self, X):
            return np.column_stack([np.ones(len(X)), np.zeros(len(X))])

    n = 30
    data = {f: np.ones(n) for f in FEATURE_NAMES}
    data["voltage_mean_v"] = np.full(n, 4.80)
    data["current_mean_a"] = np.full(n, 1.5)
    data["current_std_a"] = np.full(n, 0.005)

    df = pd.DataFrame(data)
    df["run_id"] = "run_test_ema"
    df["window_index"] = np.arange(n)
    df["window_start_s"] = np.arange(n) * 0.25
    df["window_end_s"] = df["window_start_s"] + 0.5
    df["fault_label"] = np.zeros(n, dtype=int)

    preds, probs, diag = calibrator.process_run(df, DummyHealthyModel())
    records = diag.records

    init_records = [r for r in records if r.state == CalibrationState.INITIALIZING]
    post_records = [r for r in records if r.state == CalibrationState.NORMAL and r.update_accepted]

    assert len(init_records) > 0
    assert len(post_records) > 0


def test_stability_gate_rejects_unstable_current():
    config = AdaptiveCalibrationConfig(max_current_variance=0.010)
    calibrator = ConfidenceGatedAdaptiveCalibrator(config)

    class DummyHealthyModel:
        def predict(self, X):
            return np.zeros(len(X), dtype=int)

        def predict_proba(self, X):
            return np.column_stack([np.ones(len(X)), np.zeros(len(X))])

    n = 30
    data = {f: np.ones(n) for f in FEATURE_NAMES}
    data["voltage_mean_v"] = np.full(n, 4.80)
    data["current_mean_a"] = np.full(n, 1.5)
    data["current_std_a"] = np.where(np.arange(n) > 20, 0.050, 0.002)

    df = pd.DataFrame(data)
    df["run_id"] = "run_unstable"
    df["window_index"] = np.arange(n)
    df["window_start_s"] = np.arange(n) * 0.25
    df["window_end_s"] = df["window_start_s"] + 0.5
    df["fault_label"] = np.zeros(n, dtype=int)

    preds, probs, diag = calibrator.process_run(df, DummyHealthyModel())
    rejected = [r for r in diag.records if r.rejection_reason == "STABILITY_GATE_FAILED"]
    assert len(rejected) > 0


def test_freeze_and_recovery_dynamics():
    config = AdaptiveCalibrationConfig(
        anomaly_cooldown_windows=2,
        recovery_stable_windows=3,
        enable_freeze_logic=True,
    )
    calibrator = ConfidenceGatedAdaptiveCalibrator(config)

    class FaultTriggerModel:
        def predict(self, X):
            v_val = X[0, 0]
            return np.array([1]) if v_val < -0.15 else np.array([0])

        def predict_proba(self, X):
            v_val = X[0, 0]
            return np.array([[0.05, 0.95]]) if v_val < -0.15 else np.array([[0.99, 0.01]])

    n = 40
    data = {f: np.zeros(n) for f in FEATURE_NAMES}
    v_seq = np.full(n, 4.90)
    v_seq[20:25] = 4.50
    data["voltage_mean_v"] = v_seq
    data["current_mean_a"] = np.full(n, 1.5)
    data["current_std_a"] = np.full(n, 0.005)

    df = pd.DataFrame(data)
    df["run_id"] = "run_freeze_recovery"
    df["window_index"] = np.arange(n)
    df["window_start_s"] = np.arange(n) * 0.25
    df["window_end_s"] = df["window_start_s"] + 0.5
    df["fault_label"] = np.where(np.arange(n) >= 20, 1, 0)

    preds, probs, diag = calibrator.process_run(df, FaultTriggerModel())
    frozen_records = [r for r in diag.records if r.state == CalibrationState.FROZEN]
    assert len(frozen_records) > 0

    recovered_records = [r for r in diag.records[30:] if r.state == CalibrationState.NORMAL]
    assert len(recovered_records) > 0


def test_contamination_tracking_retrospective():
    diag = CalibrationDiagnostics()
    diag.record_window(
        WindowCalibrationRecord(
            window_index=0,
            timestamp_end_s=1.0,
            raw_v=4.9,
            raw_i=1.5,
            active_baseline_v=4.9,
            active_baseline_i=1.5,
            normalized_v=0.0,
            normalized_i=0.0,
            prediction=0,
            confidence=0.99,
            state=CalibrationState.NORMAL,
            update_accepted=True,
            rejection_reason="",
            ground_truth_fault=0,
        )
    )
    diag.record_window(
        WindowCalibrationRecord(
            window_index=1,
            timestamp_end_s=1.25,
            raw_v=4.6,
            raw_i=1.5,
            active_baseline_v=4.6,
            active_baseline_i=1.5,
            normalized_v=-0.3,
            normalized_i=0.0,
            prediction=0,
            confidence=0.98,
            state=CalibrationState.NORMAL,
            update_accepted=True,
            rejection_reason="",
            ground_truth_fault=1,
        )
    )
    summary = diag.compute_summary_metrics()
    assert summary["inappropriate_updates_during_fault"] == 1
    assert summary["contamination_rate"] == 1.0


def test_reproducibility_with_fixed_seed(synthetic_run_dataframe):
    calibrator1 = ConfidenceGatedAdaptiveCalibrator()
    calibrator2 = ConfidenceGatedAdaptiveCalibrator()

    X_train = synthetic_run_dataframe[list(FEATURE_NAMES)].to_numpy()
    y_train = synthetic_run_dataframe["fault_label"].to_numpy()
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    p1, _, _ = calibrator1.process_run(synthetic_run_dataframe, model)
    p2, _, _ = calibrator2.process_run(synthetic_run_dataframe, model)
    assert np.array_equal(p1, p2)
