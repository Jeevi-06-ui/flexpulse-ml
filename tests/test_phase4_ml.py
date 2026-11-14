"""
Automated unit and integration test suite for FlexPulse-ML Phase 4.
Validates model factories, baselines, trainer, evaluation metrics, persistence, and leakage prevention.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from flexpulse_ml.features.extractor import FEATURE_NAMES
from flexpulse_ml.ml.baselines import FixedThresholdBaseline
from flexpulse_ml.ml.evaluate import (
    bootstrap_run_level_metrics,
    calculate_window_metrics,
    evaluate_scenario_breakdown,
    evaluate_window_label_sensitivity,
)
from flexpulse_ml.ml.models import (
    MODEL_DECISION_TREE,
    MODEL_FIXED_THRESHOLD,
    MODEL_GRADIENT_BOOSTING,
    MODEL_LOGISTIC_REGRESSION,
    MODEL_RANDOM_FOREST,
    MODEL_SVM,
    SUPPORTED_MODELS,
    build_model_pipeline,
    get_default_hyperparameters,
    get_hyperparameter_search_grid,
)
from flexpulse_ml.ml.persistence import ModelArtifactStorage
from flexpulse_ml.ml.train import ModelTrainer


@pytest.fixture
def synthetic_feature_data():
    rng = np.random.default_rng(123)
    n_samples = 120
    data = {}
    for feat in FEATURE_NAMES:
        data[feat] = rng.normal(loc=5.0, scale=0.5, size=n_samples)

    labels = rng.integers(0, 2, size=n_samples)
    scenarios = rng.choice(["scenario_a", "scenario_b", "nominal_normal_operation"], size=n_samples)
    run_ids = [f"run_{i // 20}" for i in range(n_samples)]
    fractions = np.where(labels == 1, rng.uniform(0.1, 1.0, size=n_samples), 0.0)

    df = pd.DataFrame(data)
    df["fault_label"] = labels
    df["scenario"] = scenarios
    df["run_id"] = run_ids
    df["window_index"] = np.arange(n_samples)
    df["window_start_s"] = np.arange(n_samples) * 0.25
    df["window_end_s"] = df["window_start_s"] + 0.5
    df["fault_fraction"] = fractions
    return df


def test_supported_models_count():
    assert len(SUPPORTED_MODELS) == 6
    assert MODEL_FIXED_THRESHOLD in SUPPORTED_MODELS
    assert MODEL_LOGISTIC_REGRESSION in SUPPORTED_MODELS
    assert MODEL_DECISION_TREE in SUPPORTED_MODELS
    assert MODEL_RANDOM_FOREST in SUPPORTED_MODELS
    assert MODEL_SVM in SUPPORTED_MODELS
    assert MODEL_GRADIENT_BOOSTING in SUPPORTED_MODELS


def test_default_hyperparameters_and_search_grids():
    for model_name in SUPPORTED_MODELS:
        defaults = get_default_hyperparameters(model_name)
        assert isinstance(defaults, dict)
        assert len(defaults) > 0

        grid = get_hyperparameter_search_grid(model_name)
        assert isinstance(grid, list)
        assert len(grid) >= 1
        for candidate in grid:
            assert isinstance(candidate, dict)


def test_build_model_pipelines():
    for model_name in SUPPORTED_MODELS:
        pipeline = build_model_pipeline(model_name, random_seed=42)
        assert pipeline is not None
        assert hasattr(pipeline, "fit")
        assert hasattr(pipeline, "predict")


def test_fixed_threshold_baseline():
    rng = np.random.default_rng(42)
    n = 100
    X = rng.normal(loc=5.0, scale=0.01, size=(n, len(FEATURE_NAMES)))
    y = np.zeros(n, dtype=int)

    X[80:, 0] = 3.5
    y[80:] = 1

    baseline = FixedThresholdBaseline(
        voltage_drop_std_multiplier=3.0,
        transient_jump_std_multiplier=3.0,
    )
    baseline.fit(X, y)
    preds = baseline.predict(X)
    assert len(preds) == n
    assert preds[0] == 0
    assert preds[90] == 1

    probs = baseline.predict_proba(X)
    assert probs.shape == (n, 2)
    assert np.allclose(np.sum(probs, axis=1), 1.0)


def test_allowlist_enforcement_and_target_validation(synthetic_feature_data):
    trainer = ModelTrainer(random_seed=42)
    df_missing = synthetic_feature_data.drop(columns=[FEATURE_NAMES[0]])
    with pytest.raises(ValueError, match="Dataset missing approved features"):
        trainer.prepare_data(df_missing)

    df_no_target = synthetic_feature_data.drop(columns=["fault_label"])
    with pytest.raises(ValueError, match="Dataset missing target column"):
        trainer.prepare_data(df_no_target)

    X, y = trainer.prepare_data(synthetic_feature_data)
    assert X.shape == (len(synthetic_feature_data), len(FEATURE_NAMES))
    assert len(y) == len(synthetic_feature_data)


def test_scaling_inside_pipeline_prevents_leakage():
    pipe = build_model_pipeline(MODEL_LOGISTIC_REGRESSION, random_seed=42)
    assert isinstance(pipe, Pipeline)
    assert "scaler" in pipe.named_steps
    assert isinstance(pipe.named_steps["scaler"], StandardScaler)


def test_calculate_window_metrics():
    y_true = np.array([0, 0, 0, 1, 1, 1], dtype=int)
    y_pred = np.array([0, 0, 1, 1, 1, 0], dtype=int)
    y_prob = np.array([0.1, 0.2, 0.7, 0.9, 0.8, 0.3], dtype=float)

    metrics = calculate_window_metrics(y_true, y_pred, y_prob)
    assert metrics.n_samples == 6
    assert metrics.n_healthy == 3
    assert metrics.n_fault == 3
    assert metrics.confusion_matrix == {"tn": 2, "fp": 1, "fn": 1, "tp": 2}
    assert np.isclose(metrics.accuracy, 4 / 6)
    assert np.isclose(metrics.precision, 2 / 3)
    assert np.isclose(metrics.recall, 2 / 3)
    assert np.isclose(metrics.specificity, 2 / 3)
    assert np.isclose(metrics.false_negative_rate, 1 / 3)
    assert metrics.roc_auc is not None
    assert metrics.average_precision is not None


def test_bootstrap_run_level_metrics(synthetic_feature_data):
    df = synthetic_feature_data
    y_pred = df["fault_label"].to_numpy().copy()
    boot_ci = bootstrap_run_level_metrics(df, y_pred, n_bootstraps=50, random_seed=42)
    assert "f1" in boot_ci
    assert "recall" in boot_ci
    assert "precision" in boot_ci
    assert "balanced_accuracy" in boot_ci
    for metric in ["f1", "recall", "precision", "balanced_accuracy"]:
        assert boot_ci[metric]["ci_lower"] <= boot_ci[metric]["ci_upper"]


def test_scenario_breakdown(synthetic_feature_data):
    df = synthetic_feature_data
    y_pred = df["fault_label"].to_numpy().copy()
    breakdown = evaluate_scenario_breakdown(df, y_pred)
    for sc in df["scenario"].unique():
        assert str(sc) in breakdown
        assert "balanced_accuracy" in breakdown[str(sc)]
        assert "false_positive_rate" in breakdown[str(sc)]


def test_window_label_sensitivity(synthetic_feature_data):
    df = synthetic_feature_data
    y_pred = df["fault_label"].to_numpy().copy()
    sens = evaluate_window_label_sensitivity(df, y_pred, thresholds=[0.0, 0.20, 0.50])
    assert "any_fault_sample" in sens
    assert "fault_fraction_gt_20pct" in sens
    assert "fault_fraction_gt_50pct" in sens


def test_model_persistence_roundtrip(synthetic_feature_data):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        storage = ModelArtifactStorage(
            models_base_dir=tmp_path / "models",
            results_base_dir=tmp_path / "results",
            tables_base_dir=tmp_path / "tables",
        )

        trainer = ModelTrainer(random_seed=42, models_to_train=[MODEL_DECISION_TREE])
        X, y = trainer.prepare_data(synthetic_feature_data)

        rec = trainer.tune_and_train_model(MODEL_DECISION_TREE, X, y, X, y)
        model_path, meta_path = storage.save_model(MODEL_DECISION_TREE, rec.model_instance, rec.to_dict())

        assert model_path.exists()
        assert meta_path.exists()

        loaded_model, loaded_meta = storage.load_model(MODEL_DECISION_TREE)
        assert loaded_meta["model_name"] == MODEL_DECISION_TREE

        orig_preds = rec.model_instance.predict(X)
        loaded_preds = loaded_model.predict(X)
        assert np.array_equal(orig_preds, loaded_preds)


def test_determinism_with_fixed_random_seed(synthetic_feature_data):
    trainer1 = ModelTrainer(random_seed=42, models_to_train=[MODEL_RANDOM_FOREST])
    trainer2 = ModelTrainer(random_seed=42, models_to_train=[MODEL_RANDOM_FOREST])

    X, y = trainer1.prepare_data(synthetic_feature_data)

    rec1 = trainer1.tune_and_train_model(MODEL_RANDOM_FOREST, X, y, X, y)
    rec2 = trainer2.tune_and_train_model(MODEL_RANDOM_FOREST, X, y, X, y)

    preds1 = rec1.model_instance.predict(X)
    preds2 = rec2.model_instance.predict(X)
    assert np.array_equal(preds1, preds2)
