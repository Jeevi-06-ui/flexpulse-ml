"""
Unit tests for FlexPulse-ML DataIntegrityValidator and scientific rules.
"""

from pathlib import Path
import pytest

from flexpulse_ml.acquisition.schema import (
    AcquisitionQuality,
    CableMeasurementRecord,
    DataProvenance,
    DatasetCategory,
    ElectricalObservation,
    FaultClass,
    FaultOrigin,
    InstrumentCalibration,
    LabelSource,
    PhysicalCableIdentity,
    TestConditions,
    TrialSessionIdentity,
)
from flexpulse_ml.acquisition.validator import DataIntegrityValidator


@pytest.fixture
def base_record() -> CableMeasurementRecord:
    return CableMeasurementRecord(
        cable=PhysicalCableIdentity(cable_id="CABLE-VALID-01"),
        session=TrialSessionIdentity(session_id="SESS-01", trial_id="TR-01", step_index=0),
        conditions=TestConditions(),
        observation=ElectricalObservation(
            v_source_v=5.00,
            v_sink_v=4.88,
            current_a=1.50,
            timestamp_rel_ms=10.0,
        ),
        ground_truth=FaultClass.INTACT,
        label_source=LabelSource.HUMAN_VERIFIED,
        fault_origin=FaultOrigin.GOLDEN_REFERENCE,
        provenance=DataProvenance(source_device="BENCH_FIXTURE"),
        dataset_category=DatasetCategory.REAL_MEASUREMENT,
    )


def test_validator_accepts_valid_record(base_record: CableMeasurementRecord, tmp_path: Path):
    audit_file = tmp_path / "test_audit.jsonl"
    validator = DataIntegrityValidator(audit_file_path=audit_file)
    errors = validator.validate_record(base_record)
    assert errors == []


def test_validator_rejects_missing_cable_id(base_record: CableMeasurementRecord, tmp_path: Path):
    audit_file = tmp_path / "test_audit.jsonl"
    validator = DataIntegrityValidator(audit_file_path=audit_file)

    bad_record = base_record
    # Empty cable id
    bad_record.cable = PhysicalCableIdentity(cable_id="")
    errors = validator.validate_record(bad_record)
    assert any("MISSING_CABLE_ID" in err for err in errors)


def test_validator_rejects_passive_power_violation(base_record: CableMeasurementRecord, tmp_path: Path):
    """Sink voltage cannot physically exceed source voltage in a passive USB cable."""
    audit_file = tmp_path / "test_audit.jsonl"
    validator = DataIntegrityValidator(audit_file_path=audit_file)

    bad_record = base_record
    bad_record.observation = ElectricalObservation(
        v_source_v=4.80,
        v_sink_v=5.20, # Higher than source!
        current_a=1.0,
        timestamp_rel_ms=10.0,
    )
    errors = validator.validate_record(bad_record)
    assert any("PASSIVE_POWER_VIOLATION" in err for err in errors)


def test_validator_rejects_out_of_bounds_current(base_record: CableMeasurementRecord, tmp_path: Path):
    audit_file = tmp_path / "test_audit.jsonl"
    validator = DataIntegrityValidator(audit_file_path=audit_file)

    bad_record = base_record
    bad_record.observation = ElectricalObservation(
        v_source_v=5.0,
        v_sink_v=4.5,
        current_a=15.0, # Beyond max allowable safe limit (6A)
        timestamp_rel_ms=10.0,
    )
    errors = validator.validate_record(bad_record)
    assert any("INVALID_CURRENT" in err for err in errors)


def test_duplicate_record_detection(base_record: CableMeasurementRecord, tmp_path: Path):
    audit_file = tmp_path / "test_audit.jsonl"
    validator = DataIntegrityValidator(audit_file_path=audit_file)

    records = [base_record, base_record] # Duplicate identical record
    valid_recs, report = validator.validate_batch(records)

    assert report.total_records == 2
    assert report.valid_records == 1
    assert report.invalid_records == 1
    assert report.duplicate_records == 1
    assert audit_file.exists()


def test_synthetic_data_rejection_in_real_benchmarks(base_record: CableMeasurementRecord, tmp_path: Path):
    """Ensure disallow_synthetic flag catches SYNTHETIC_DEMO records to protect real research."""
    audit_file = tmp_path / "test_audit.jsonl"
    validator = DataIntegrityValidator(audit_file_path=audit_file, disallow_synthetic=True)

    base_record.dataset_category = DatasetCategory.SYNTHETIC_DEMO
    errors = validator.validate_record(base_record)
    assert any("SYNTHETIC_CONTAMINATION" in err for err in errors)


def test_unverified_label_normalization(base_record: CableMeasurementRecord, tmp_path: Path):
    """Unverified labels must be marked as UNKNOWN to prevent ground-truth contamination."""
    validator = DataIntegrityValidator()
    base_record.label_source = LabelSource.UNVERIFIED
    base_record.ground_truth = FaultClass.RESISTIVE_BEND

    validator.validate_record(base_record)
    assert base_record.ground_truth == FaultClass.UNKNOWN

