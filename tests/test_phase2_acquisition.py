"""
Comprehensive tests for Phase 2: Real Data Acquisition, Serial Adapter,
Canonical CSV Importer, Experiment Manager, and Raw Storage.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import pytest

from flexpulse_ml.acquisition.csv_loader import (
    load_canonical_csv,
    load_dataset_from_csv,
)
from flexpulse_ml.acquisition.experiment_manager import ExperimentManager
from flexpulse_ml.acquisition.hardware_interface import HardwareNotConnectedError
from flexpulse_ml.acquisition.raw_storage import RawDataStorage
from flexpulse_ml.acquisition.schema import (
    CANONICAL_COLUMNS,
    CanonicalMeasurementRecord,
    DatasetCategory,
    FaultClass,
    FaultOrigin,
    LabelSource,
    MeasurementQuality,
)
from flexpulse_ml.acquisition.serial_adapter import (
    ESP32JSONSerialAdapter,
    SimulatedESP32Stream,
)
from flexpulse_ml.acquisition.synthetic_generator import (
    export_canonical_synthetic_csv,
    generate_canonical_synthetic_dataset,
)


# ==============================================================================
# A. CSV DATA IMPORTER & SCHEMA TESTS
# ==============================================================================

def test_canonical_schema_has_22_columns():
    """Verify that CANONICAL_COLUMNS defines exactly the 22 required attributes."""
    assert len(CANONICAL_COLUMNS) == 22
    assert "timestamp_utc" in CANONICAL_COLUMNS
    assert "cable_id" in CANONICAL_COLUMNS
    assert "vin_v" in CANONICAL_COLUMNS
    assert "vout_v" in CANONICAL_COLUMNS
    assert "current_a" in CANONICAL_COLUMNS
    assert "load_setpoint_a" in CANONICAL_COLUMNS
    assert "temperature_c" in CANONICAL_COLUMNS
    assert "source_type" in CANONICAL_COLUMNS
    assert "ground_truth_label" in CANONICAL_COLUMNS
    assert "ground_truth_source" in CANONICAL_COLUMNS
    assert "fault_origin" in CANONICAL_COLUMNS
    assert "fault_condition" in CANONICAL_COLUMNS
    assert "fixture_id" in CANONICAL_COLUMNS
    assert "instrument_id" in CANONICAL_COLUMNS
    assert "calibration_id" in CANONICAL_COLUMNS
    assert "action_id" in CANONICAL_COLUMNS
    assert "motion_region" in CANONICAL_COLUMNS
    assert "is_settled" in CANONICAL_COLUMNS
    assert "measurement_quality" in CANONICAL_COLUMNS


def test_valid_canonical_csv_loading(tmp_path: Path):
    """Test loading a fully compliant 22-column canonical CSV dataset."""
    csv_path = tmp_path / "valid_canonical.csv"
    export_canonical_synthetic_csv(csv_path, samples_per_trial=20)
    assert csv_path.exists()

    records, report = load_canonical_csv(csv_path, disallow_synthetic=False)
    assert report.total_rows == 80 # 4 cables * 20 samples
    assert report.valid_rows == 80
    assert report.invalid_rows == 0
    assert len(report.missing_columns) == 0
    assert len(records) == 80
    assert report.is_valid is True
    assert len(report.unique_cables) == 4

    # Check physics calculations on first record
    rec = records[0]
    assert rec.vin_v >= rec.vout_v
    assert rec.vdrop_v == pytest.approx(rec.vin_v - rec.vout_v, abs=1e-5)
    assert rec.resistance_ohms == pytest.approx(rec.vdrop_v / rec.current_a, abs=1e-4)


def test_csv_loader_detects_missing_columns(tmp_path: Path):
    """Test that a CSV lacking required canonical columns is rejected with details."""
    csv_path = tmp_path / "missing_cols.csv"
    # Create CSV missing current_a and temperature_c
    df = pd_df = pytest.importorskip("pandas").DataFrame([
        {
            "timestamp_utc": "2026-11-11T12:00:00Z",
            "cable_id": "CABLE-01",
            "session_id": "SESS-01",
            "trial_id": "TR-01",
            "sample_index": 0,
            "vin_v": 5.0,
            "vout_v": 4.9,
            # current_a missing!
            "source_type": "REAL_MEASUREMENT",
        }
    ])
    df.to_csv(csv_path, index=False)

    records, report = load_canonical_csv(csv_path)
    assert len(report.missing_columns) > 0
    assert "current_a" in report.missing_columns
    assert "temperature_c" in report.missing_columns
    assert report.is_valid is False


def test_csv_loader_rejects_invalid_numeric_values(tmp_path: Path):
    """Test rejection of out-of-bounds voltages and currents."""
    csv_path = tmp_path / "invalid_numerics.csv"
    pd = pytest.importorskip("pandas")
    records = generate_canonical_synthetic_dataset(samples_per_trial=5)
    rows = [r.to_dict() for r in records]

    # Inject physically impossible values:
    # 1. Negative voltage
    rows[0]["vin_v"] = -1.5
    # 2. Over-voltage (>28V)
    rows[1]["vout_v"] = 35.0
    # 3. Passive power violation: Vout > Vin + 0.15V
    rows[2]["vin_v"] = 4.80
    rows[2]["vout_v"] = 5.25
    # 4. Over-current (>6A)
    rows[3]["current_a"] = 12.0

    pd.DataFrame(rows).to_csv(csv_path, index=False)

    valid_recs, report = load_canonical_csv(csv_path, quarantine_invalid=True)
    assert report.invalid_rows >= 4
    assert "OUT_OF_BOUNDS_VIN" in report.error_summary
    assert "OUT_OF_BOUNDS_VOUT" in report.error_summary
    assert "PASSIVE_POWER_VIOLATION" in report.error_summary
    assert "OUT_OF_BOUNDS_CURRENT" in report.error_summary
    assert report.quarantine_path is not None


def test_csv_loader_detects_duplicate_sample_identifiers(tmp_path: Path):
    """Ensure duplicate (cable_id, session_id, trial_id, sample_index) rows are quarantined."""
    csv_path = tmp_path / "duplicates.csv"
    pd = pytest.importorskip("pandas")
    records = generate_canonical_synthetic_dataset(samples_per_trial=3)
    rows = [r.to_dict() for r in records]

    # Duplicate the first row
    rows.append(rows[0].copy())
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    valid_recs, report = load_canonical_csv(csv_path)
    assert report.duplicate_rows == 1
    assert "DUPLICATE_SAMPLE_INDEX" in report.error_summary


def test_real_and_synthetic_demo_separation(tmp_path: Path):
    """Verify that disallow_synthetic strictly quarantines SYNTHETIC_DEMO records."""
    csv_path = tmp_path / "synthetic_demo.csv"
    export_canonical_synthetic_csv(csv_path, samples_per_trial=5)

    # When disallow_synthetic=True, ALL rows must be quarantined
    valid_recs, report = load_canonical_csv(csv_path, disallow_synthetic=True)
    assert report.valid_rows == 0
    assert report.invalid_rows == 20
    assert "SYNTHETIC_CONTAMINATION" in report.error_summary


# ==============================================================================
# B. SERIAL DATA ADAPTER TESTS
# ==============================================================================

def test_serial_adapter_graceful_missing_port():
    """Verify missing hardware port fails gracefully and does not raise unhandled exception."""
    adapter = ESP32JSONSerialAdapter(port="/dev/tty.nonexistent_usb_port_9999")
    assert adapter.check_port_available() is False
    assert adapter.connect() is False
    assert adapter.is_connected() is False


def test_serial_adapter_read_disconnected_raises_specific_error():
    """Reading from disconnected adapter raises HardwareNotConnectedError."""
    adapter = ESP32JSONSerialAdapter(port="/dev/tty.nonexistent_usb_port_9999")
    with pytest.raises(HardwareNotConnectedError):
        stream = adapter.read_stream()
        next(stream)


def test_serial_adapter_handles_malformed_json_and_missing_keys():
    """Verify bad packet detection on simulated telemetry stream."""
    adapter = ESP32JSONSerialAdapter(port="MOCK_TEST_PORT")
    feeder = SimulatedESP32Stream(samples=10, inject_corrupt_lines=True)
    lines = feeder.generate_lines()

    valid_packets = 0
    for line in lines:
        res = adapter.parse_and_validate_line(line)
        if res.is_valid:
            valid_packets += 1

    assert adapter.stats.total_lines_read == 10
    assert adapter.stats.corrupt_json_lines == 1 # "{bad_json_packet..."
    assert adapter.stats.missing_fields_packets == 1 # missing current_ma and vout_mv
    assert valid_packets == 8


def test_serial_adapter_converts_packet_to_canonical_record():
    """Verify that a valid packet is accurately transformed into a 22-column CanonicalMeasurementRecord."""
    adapter = ESP32JSONSerialAdapter()
    raw_packet = {
        "packet_type": "TELEMETRY",
        "sample_idx": 15,
        "vin_mv": 5005,
        "vout_mv": 4875,
        "current_ma": 1490.5,
        "temp_c": 27.2,
        "settled": True,
        "quality": "OK",
    }
    res = adapter.parse_and_validate_line(json.dumps(raw_packet))
    assert res.is_valid is True

    record = adapter.packet_to_canonical_record(
        parsed_dict=res.parsed_dict,
        cable_id="CABLE-TEST-01",
        session_id="SESS-01",
        trial_id="TR-01",
        fixture_id="FIXTURE_ESP32_INA226_01",
        ground_truth_label=FaultClass.INTACT.value,
        ground_truth_source=LabelSource.HUMAN_VERIFIED.value,
    )

    assert record.cable_id == "CABLE-TEST-01"
    assert record.sample_index == 15
    assert record.vin_v == pytest.approx(5.005, abs=1e-4)
    assert record.vout_v == pytest.approx(4.875, abs=1e-4)
    assert record.current_a == pytest.approx(1.4905, abs=1e-4)
    assert record.temperature_c == 27.2
    assert record.is_settled is True
    assert record.ground_truth_label == "INTACT"


# ==============================================================================
# C. RAW DATA STORAGE & MANIFESTS
# ==============================================================================

def test_raw_data_storage_saves_with_checksum(tmp_path: Path):
    """Test saving canonical records with SHA-256 manifest generation."""
    storage = RawDataStorage(
        real_dir=tmp_path / "raw" / "real",
        synthetic_dir=tmp_path / "raw" / "synthetic",
        quarantine_dir=tmp_path / "quarantine",
        manifests_dir=tmp_path / "manifests",
    )

    records = generate_canonical_synthetic_dataset(samples_per_trial=10)
    saved_csv = storage.save_canonical_records(records, is_synthetic=True)
    assert saved_csv.exists()

    # Verify manifest exists
    manifest_files = list((tmp_path / "manifests").glob("*.json"))
    assert len(manifest_files) == 1

    with open(manifest_files[0], "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    assert manifest_data["record_count"] == 40
    assert "sha256_checksum" in manifest_data
    assert manifest_data["source_type"] == "SYNTHETIC_DEMO"


def test_raw_data_storage_blocks_synthetic_in_real_mode(tmp_path: Path):
    """Ensure saving SYNTHETIC_DEMO records into real directory raises ValueError."""
    storage = RawDataStorage(
        real_dir=tmp_path / "raw" / "real",
        synthetic_dir=tmp_path / "raw" / "synthetic",
    )
    records = generate_canonical_synthetic_dataset(samples_per_trial=5)
    # records have source_type = SYNTHETIC_DEMO
    with pytest.raises(ValueError) as excinfo:
        storage.save_canonical_records(records, is_synthetic=False)

    assert "Integrity Violation" in str(excinfo.value)


# ==============================================================================
# D. EXPERIMENT MANAGEMENT SUBSYSTEM
# ==============================================================================

def test_experiment_manager_cable_registration_and_session(tmp_path: Path):
    """Test physical cable registration, session creation, trial manifest recording, and completion."""
    reg_file = tmp_path / "test_cable_registry.json"
    sess_dir = tmp_path / "test_sessions"
    mgr = ExperimentManager(registry_file=reg_file, sessions_dir=sess_dir)

    # 1. Register physical cables
    cable = mgr.register_cable(
        cable_id="CABLE-ANKER-PU-01",
        manufacturer="ANKER",
        cable_type="USB_C_TO_C",
        rated_current_a=3.0,
        rated_length_m=1.8,
        notes="Lab reference cable A",
    )
    assert cable.cable_id == "CABLE-ANKER-PU-01"
    assert mgr.get_cable("CABLE-ANKER-PU-01") is not None
    assert len(mgr.list_cables()) == 1

    # 2. Start session
    session = mgr.start_session(
        session_id="SESS-20261111-TEST",
        operator_id="RESEARCHER_JEEVITH",
        description="Presidency University cable degradation test",
    )
    assert session.status == "ACTIVE"

    # 3. Register trial with independent ground truth label
    trial = mgr.register_trial(
        session_id="SESS-20261111-TEST",
        trial_id="TR-001",
        cable_id="CABLE-ANKER-PU-01",
        action_id="FLEX_SWEEP_0_90",
        ground_truth_label=FaultClass.RESISTIVE_BEND.value,
        ground_truth_source=LabelSource.LAB_BENCH_TESTER.value,
        fault_origin=FaultOrigin.FIXTURE_INJECTED.value,
        fault_condition="SERIES_RESISTIVE_BEND_500M",
    )
    assert trial.trial_id == "TR-001"
    assert trial.ground_truth_label == "RESISTIVE_BEND"

    # 4. Update count and complete session
    mgr.update_trial_observation_count("SESS-20261111-TEST", "TR-001", count=150)
    completed_session = mgr.complete_session("SESS-20261111-TEST")

    assert completed_session.status == "COMPLETED"
    assert completed_session.total_samples == 150
    assert (sess_dir / "SESS-20261111-TEST.json").exists()

