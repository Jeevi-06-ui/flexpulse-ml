"""
Automated test suite verifying Phase 7 final scientific evidence package.
"""

from __future__ import annotations

import json
from pathlib import Path
import tokenize
import pandas as pd
import pytest

from flexpulse_ml.config import PROJECT_ROOT
from flexpulse_ml.data.manifest import compute_file_sha256
from flexpulse_ml.data.validator import validate_synthetic_dataset
from flexpulse_ml.features.validator import validate_feature_dataset
from flexpulse_ml.final_evidence.package_builder import FinalResearchPackageBuilder


def test_frozen_protocol_hash():
    proto_path = PROJECT_ROOT / "docs" / "final_evaluation_protocol.md"
    assert proto_path.exists(), "Protocol file missing"
    current_hash = compute_file_sha256(proto_path)
    assert current_hash == FinalResearchPackageBuilder.PROTOCOL_SHA256, (
        f"Protocol file hash mismatch! Expected {FinalResearchPackageBuilder.PROTOCOL_SHA256}, got {current_hash}"
    )


def test_final_dataset_validation():
    ds_path = PROJECT_ROOT / "data" / "synthetic" / "flexpulse_final_test_v4"
    assert ds_path.exists(), "Final test dataset missing"
    report = validate_synthetic_dataset(ds_path)
    assert report.is_valid, f"Dataset validation failed: {report.errors}"


def test_final_feature_dataset_validation():
    feat_path = PROJECT_ROOT / "data" / "processed" / "features" / "flexpulse_final_test_v4"
    assert feat_path.exists(), "Feature dataset missing"
    report = validate_feature_dataset(feat_path)
    assert report.is_valid, f"Feature validation failed: {report.errors}"


def test_all_seven_tables_exist():
    pkg_tables = PROJECT_ROOT / "results" / "final_research_package" / "tables"
    for i in range(1, 8):
        matching = list(pkg_tables.glob(f"table{i}_*.csv"))
        assert len(matching) == 1, f"Table {i} CSV missing in {pkg_tables}"
        md_matching = list(pkg_tables.glob(f"table{i}_*.md"))
        assert len(md_matching) == 1, f"Table {i} Markdown missing in {pkg_tables}"


def test_all_six_figures_exist():
    pkg_figures = PROJECT_ROOT / "results" / "final_research_package" / "figures"
    for i in range(1, 7):
        matching = list(pkg_figures.glob(f"fig{i}_*.png"))
        assert len(matching) == 1, f"Figure {i} PNG missing in {pkg_figures}"
        assert matching[0].stat().st_size > 1000, f"Figure {i} appears empty or corrupted"


def test_table1_metrics_validity():
    t1_path = PROJECT_ROOT / "results" / "final_research_package" / "tables" / "table1_overall_performance.csv"
    assert t1_path.exists()
    df = pd.read_csv(t1_path)
    assert len(df) == 12, "Table 1 should contain 12 rows (6 methods x 2 splits)"
    assert set(df["Partition"].unique()) == {"ID Test", "OOD Test"}
    for _, row in df.iterrows():
        assert 0.0 <= float(row["Accuracy"]) <= 1.0
        assert 0.0 <= float(row["Specificity"]) <= 1.0


def test_table2_bootstrap_equivalence():
    t2_path = PROJECT_ROOT / "results" / "final_research_package" / "tables" / "table2_bootstrap_significance.csv"
    assert t2_path.exists()
    df = pd.read_csv(t2_path)
    m4_m3_row = df[df["Comparison"] == "M4 vs M3"].iloc[0]
    assert float(m4_m3_row["Delta F1 (Mean)"]) == 0.0, "M4 vs M3 must show Delta F1 == 0.0000"


def test_table5_persistence_tradeoff():
    t5_path = PROJECT_ROOT / "results" / "final_research_package" / "tables" / "table5_persistence_ablation.csv"
    assert t5_path.exists()
    df = pd.read_csv(t5_path)
    n1_row = df[df["Persistence Windows (N)"] == 1].iloc[0]
    n2_row = df[df["Persistence Windows (N)"] == 2].iloc[0]
    assert float(n2_row["Relative FA Reduction (%)"]) >= 40.0
    assert float(n2_row["Event Recall (%)"]) >= 90.0
    assert float(n2_row["Event Recall (%)"]) == float(n1_row["Event Recall (%)"])


def test_table6_gating_integrity():
    t6_path = PROJECT_ROOT / "results" / "final_research_package" / "tables" / "table6_gating_ablation.csv"
    assert t6_path.exists()
    df = pd.read_csv(t6_path)
    a3_row = df[df["Adaptation Configuration"] == "Fully Gated Adaptive (A3)"].iloc[0]
    assert float(a3_row["Contamination Rate (%)"]) == 0.0


def test_zero_python_comments():
    files_to_check = [
        PROJECT_ROOT / "flexpulse_ml" / "data" / "generator_v4.py",
        PROJECT_ROOT / "flexpulse_ml" / "final_evidence" / "__init__.py",
        PROJECT_ROOT / "flexpulse_ml" / "final_evidence" / "evaluator.py",
        PROJECT_ROOT / "flexpulse_ml" / "final_evidence" / "figures.py",
        PROJECT_ROOT / "flexpulse_ml" / "final_evidence" / "package_builder.py",
        PROJECT_ROOT / "tests" / "test_phase7_final_evidence.py",
    ]
    for path in files_to_check:
        assert path.exists(), f"File {path} missing"
        with open(path, "rb") as f:
            tokens = list(tokenize.tokenize(f.readline))
            comments = [t for t in tokens if t.type == tokenize.COMMENT]
            assert len(comments) == 0, f"Found comment in {path}: {comments}"
