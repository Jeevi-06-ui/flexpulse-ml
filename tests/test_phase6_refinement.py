"""
Unit and integration test suite for Phase 6 scientific validation and refinement modules.
"""

from __future__ import annotations

import glob
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from flexpulse_ml.features.extractor import FEATURE_NAMES
from flexpulse_ml.refinement.audit import (
    FeatureNormalizationAudit,
    WindowLevelA2vsA3Diagnostic,
    FalseAlarmInvestigator,
)
from flexpulse_ml.refinement.confidence_eval import (
    ConfidenceCalibrationEvaluator,
    ConfidenceCalibrationReport,
)
from flexpulse_ml.refinement.persistence import TemporalPersistenceFilter
from flexpulse_ml.refinement.regimes import ControlledRegimeGenerator


def test_feature_normalization_audit_completeness():
    records = FeatureNormalizationAudit.audit_all_features()
    assert len(records) == 26
    names = [r.feature_name for r in records]
    assert set(names) == set(FEATURE_NAMES)

    compliance = FeatureNormalizationAudit.audit_implementation_compliance()
    assert compliance["is_mathematically_compliant"] is True
    assert len(compliance["improperly_normalized_features"]) == 0


def test_feature_normalization_mathematical_compliance():
    records = FeatureNormalizationAudit.audit_all_features()
    for r in records:
        assert r.is_dimensionally_valid is True
        if r.category in ("Spread", "Rate", "Difference"):
            assert r.is_level_feature is False
            assert r.baseline_transformation == "None (Unchanged)"
        elif r.category == "Level":
            assert r.is_level_feature is True
            assert "-" in r.baseline_transformation


def test_temporal_persistence_filter_logic():
    raw_preds = np.array([0, 1, 0, 1, 1, 0, 1, 1, 1, 0], dtype=np.int32)

    filter_n1 = TemporalPersistenceFilter(persistence_windows=1)
    filtered_1 = filter_n1.filter_predictions(raw_preds)
    assert np.array_equal(filtered_1, raw_preds)

    filter_n2 = TemporalPersistenceFilter(persistence_windows=2)
    filtered_2 = filter_n2.filter_predictions(raw_preds)
    expected_2 = np.array([0, 0, 0, 0, 1, 0, 0, 1, 1, 0], dtype=np.int32)
    assert np.array_equal(filtered_2, expected_2)

    filter_n3 = TemporalPersistenceFilter(persistence_windows=3)
    filtered_3 = filter_n3.filter_predictions(raw_preds)
    expected_3 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=np.int32)
    assert np.array_equal(filtered_3, expected_3)


def test_confidence_calibration_evaluator():
    evaluator = ConfidenceCalibrationEvaluator(n_bins=10)
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.int32)
    p_conf = np.array([0.9, 0.8, 0.85, 0.7, 0.2, 0.1, 0.3, 0.15], dtype=np.float64)

    report = evaluator.evaluate_calibration(y_true, p_conf)
    assert isinstance(report, ConfidenceCalibrationReport)
    assert 0.0 <= report.brier_score <= 1.0
    assert 0.0 <= report.expected_calibration_error <= 1.0
    assert len(report.reliability_bins) == 10
    assert len(report.threshold_sensitivity) > 0


def test_controlled_regimes_generator():
    generator = ControlledRegimeGenerator(duration_s=10.0)

    for reg_id in [ControlledRegimeGenerator.REGIME_E1, ControlledRegimeGenerator.REGIME_E4, ControlledRegimeGenerator.REGIME_E5]:
        df, meta = generator.generate_regime_run(reg_id, seed=123)
        assert len(df) > 0
        assert "voltage_mean_v" in df.columns
        assert "fault_label" in df.columns
        assert meta["regime_id"] == reg_id


def test_zero_comment_rule_phase6():
    py_files = glob.glob("flexpulse_ml/refinement/*.py") + ["tests/test_phase6_refinement.py"]
    for path_str in py_files:
        with open(path_str, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                assert not line.strip().startswith("#"), f"Comment violation in {path_str}:{line_no}: {line.strip()}"
