"""
Integration tests for synthetic demo generator and CSV loader.
"""

from pathlib import Path
import pytest

from flexpulse_ml.acquisition.csv_loader import load_dataset_from_csv
from flexpulse_ml.acquisition.schema import DatasetCategory
from flexpulse_ml.acquisition.synthetic_generator import (
    export_synthetic_dataset_to_csv,
    generate_synthetic_cable_dataset,
)


def test_synthetic_generator_labels_properly():
    """Verify that EVERY synthetic record has category SYNTHETIC_DEMO."""
    records = generate_synthetic_cable_dataset(samples_per_trial=20, random_seed=123)
    assert len(records) == 80 # 4 cables * 20 samples

    for rec in records:
        assert rec.dataset_category == DatasetCategory.SYNTHETIC_DEMO
        assert "DEMO" in rec.cable.cable_id
        assert rec.provenance.source_device == "SYNTHETIC_DEMO_GENERATOR"


def test_csv_export_and_ingestion_roundtrip(tmp_path: Path):
    """Test generating CSV to disk and loading it back cleanly."""
    csv_file = tmp_path / "test_demo_dataset.csv"
    export_synthetic_dataset_to_csv(csv_file, samples_per_trial=15)

    assert csv_file.exists()

    # Load with standard loader (allowing synthetic)
    valid_records, report = load_dataset_from_csv(csv_file, disallow_synthetic=False)
    assert report.total_records == 60
    assert report.valid_records == 60
    assert report.invalid_records == 0
    assert report.synthetic_demo_records == 60
    assert report.real_measurement_records == 0
    assert len(report.unique_cables) == 4


def test_csv_loader_blocks_synthetic_when_demanded(tmp_path: Path):
    """Ensure load_dataset_from_csv rejects synthetic rows when disallow_synthetic=True."""
    csv_file = tmp_path / "test_demo_dataset.csv"
    export_synthetic_dataset_to_csv(csv_file, samples_per_trial=10)

    audit_log = tmp_path / "quarantine.jsonl"
    from flexpulse_ml.acquisition.validator import DataIntegrityValidator
    custom_val = DataIntegrityValidator(audit_file_path=audit_log, disallow_synthetic=True)

    valid_records, report = load_dataset_from_csv(
        csv_file, disallow_synthetic=True, validator=custom_val
    )
    assert report.valid_records == 0
    assert report.invalid_records == 40
    assert "SYNTHETIC_CONTAMINATION" in report.validation_errors_summary

