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


CONFIRMATORY_PROTOCOL_SHA256 = "390921bd91ae6fda118dd74d7d2bf40ca32e727ccbecac3938212e44cee7ec91"


def test_confirmatory_protocol_hash():
    proto_path = PROJECT_ROOT / "docs" / "confirmatory_evaluation_protocol_v5.md"
    assert proto_path.exists(), "Confirmatory protocol missing"
    current_hash = compute_file_sha256(proto_path)
    assert current_hash == CONFIRMATORY_PROTOCOL_SHA256, (
        f"Confirmatory protocol hash mismatch! Expected {CONFIRMATORY_PROTOCOL_SHA256}, got {current_hash}"
    )


def test_confirmatory_dataset_validation():
    ds_path = PROJECT_ROOT / "data" / "synthetic" / "flexpulse_final_test_v5"
    assert ds_path.exists(), "Confirmatory synthetic dataset missing"
    report = validate_synthetic_dataset(ds_path)
    assert report.is_valid, f"Dataset validation failed: {report.errors}"


def test_confirmatory_feature_dataset_validation():
    feat_path = PROJECT_ROOT / "data" / "processed" / "features" / "flexpulse_final_test_v5"
    assert feat_path.exists(), "Confirmatory feature dataset missing"
    report = validate_feature_dataset(feat_path)
    assert report.is_valid, f"Feature validation failed: {report.errors}"


def test_all_seven_confirmatory_tables_exist():
    pkg_tables = PROJECT_ROOT / "results" / "confirmatory_v5" / "tables"
    for i in range(1, 8):
        matching_csv = list(pkg_tables.glob(f"table{i}_*.csv"))
        assert len(matching_csv) == 1, f"Confirmatory Table {i} CSV missing in {pkg_tables}"
        matching_md = list(pkg_tables.glob(f"table{i}_*.md"))
        assert len(matching_md) == 1, f"Confirmatory Table {i} Markdown missing in {pkg_tables}"


def test_all_six_confirmatory_figures_exist():
    pkg_figures = PROJECT_ROOT / "results" / "confirmatory_v5" / "figures"
    for i in range(1, 7):
        matching_png = list(pkg_figures.glob(f"fig{i}_*.png"))
        assert len(matching_png) == 1, f"Confirmatory Figure {i} PNG missing in {pkg_figures}"
        assert matching_png[0].stat().st_size > 1000, f"Confirmatory Figure {i} appears corrupted"


def test_confirmatory_m4_m3_equivalence():
    t2_path = PROJECT_ROOT / "results" / "confirmatory_v5" / "tables" / "table2_bootstrap_significance.csv"
    assert t2_path.exists()
    df = pd.read_csv(t2_path)
    m4_m3_row = df[df["Comparison"] == "M4 vs M3"].iloc[0]
    assert float(m4_m3_row["Delta F1 (Mean)"]) == 0.0, "M4 vs M3 must show Delta F1 == 0.0000"


def test_confirmatory_zero_python_comments():
    files_to_check = [
        PROJECT_ROOT / "flexpulse_ml" / "data" / "generator_v5.py",
        PROJECT_ROOT / "flexpulse_ml" / "final_evidence" / "confirmatory_runner.py",
        PROJECT_ROOT / "tests" / "test_confirmatory_v5.py",
    ]
    for path in files_to_check:
        assert path.exists(), f"File {path} missing"
        with open(path, "rb") as f:
            tokens = list(tokenize.tokenize(f.readline))
            comments = [t for t in tokens if t.type == tokenize.COMMENT]
            assert len(comments) == 0, f"Found comment in {path}: {comments}"
