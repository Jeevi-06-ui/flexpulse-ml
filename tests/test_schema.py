"""
Unit tests for FlexPulse-ML typed data schema and serialization.
"""

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


@pytest.fixture
def sample_valid_record() -> CableMeasurementRecord:
    cable = PhysicalCableIdentity(
        cable_id="CABLE-TEST-001",
        manufacturer="ANKER",
        cable_type="USB_C_TO_C",
        rated_current_a=3.0,
        rated_length_m=1.0,
        has_emask_chip=False,
    )
    session = TrialSessionIdentity(
        session_id="SESS-20261111-01",
        trial_id="TR-001",
        step_index=0,
    )
    conditions = TestConditions(
        nominal_voltage_v=5.0,
        target_current_a=1.5,
        ambient_temp_c=24.0,
        flex_angle_deg=0.0,
        mechanical_state="STATIC",
    )
    observation = ElectricalObservation(
        v_source_v=5.02,
        v_sink_v=4.90,
        current_a=1.50,
        timestamp_rel_ms=0.0,
        sampling_jitter_us=5.0,
    )
    provenance = DataProvenance(
        source_device="MOCK_TEST_FIXTURE",
        firmware_version="1.0.0",
        operator_id="OPERATOR_01",
        recorded_at_utc="2026-11-11T12:00:00Z",
    )
    return CableMeasurementRecord(
        cable=cable,
        session=session,
        conditions=conditions,
        observation=observation,
        ground_truth=FaultClass.INTACT,
        label_source=LabelSource.HUMAN_VERIFIED,
        fault_origin=FaultOrigin.GOLDEN_REFERENCE,
        provenance=provenance,
        calibration=InstrumentCalibration(),
        quality=AcquisitionQuality(),
        dataset_category=DatasetCategory.REAL_MEASUREMENT,
    )


def test_record_electrical_physics(sample_valid_record: CableMeasurementRecord):
    """Test voltage drop and Ohm's law apparent resistance calculations."""
    obs = sample_valid_record.observation
    # delta V = 5.02 - 4.90 = 0.12 V
    assert pytest.approx(obs.voltage_drop_v, abs=1e-4) == 0.12
    # R = delta V / I = 0.12 / 1.50 = 0.080 Ohms (80 mOhms)
    assert pytest.approx(obs.apparent_resistance_ohms, abs=1e-4) == 0.080


def test_apparent_resistance_zero_current():
    """Ensure apparent resistance does not raise ZeroDivisionError when current is 0."""
    obs = ElectricalObservation(
        v_source_v=5.0,
        v_sink_v=5.0,
        current_a=0.0,
        timestamp_rel_ms=0.0,
    )
    assert obs.apparent_resistance_ohms == 0.0


def test_record_serialization_roundtrip(sample_valid_record: CableMeasurementRecord):
    """Ensure CableMeasurementRecord can be flattened to a dictionary and restored identically."""
    flat = sample_valid_record.to_flat_dict()
    assert isinstance(flat, dict)
    assert flat["cable_id"] == "CABLE-TEST-001"
    assert flat["fault_class"] == FaultClass.INTACT.value
    assert flat["dataset_category"] == DatasetCategory.REAL_MEASUREMENT.value

    # Reconstruct from dict
    restored = CableMeasurementRecord.from_flat_dict(flat)
    assert restored.cable.cable_id == sample_valid_record.cable.cable_id
    assert restored.session.session_id == sample_valid_record.session.session_id
    assert pytest.approx(restored.observation.v_source_v) == sample_valid_record.observation.v_source_v
    assert restored.ground_truth == FaultClass.INTACT
    assert restored.dataset_category == DatasetCategory.REAL_MEASUREMENT


def test_record_sha256_reproducibility(sample_valid_record: CableMeasurementRecord):
    """Verify that cryptographic hash of record contents is deterministic."""
    sha1 = sample_valid_record.compute_sha256()
    sha2 = sample_valid_record.compute_sha256()
    assert isinstance(sha1, str)
    assert len(sha1) == 64
    assert sha1 == sha2

