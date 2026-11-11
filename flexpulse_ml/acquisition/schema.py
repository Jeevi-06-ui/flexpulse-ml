"""
Typed Experimental Data Schema for FlexPulse-ML.

Stores distinct concepts for:
- Physical cable identity
- Trial and session identity
- Test conditions
- Raw voltage/current observations
- Ground-truth condition labels
- Source of the ground truth
- Data provenance
- Instrument calibration
- Sensor and acquisition quality
- Fault origin
- Dataset category
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
from typing import Any, Dict, Optional


class DatasetCategory(str, Enum):
    """Identifies the scientific authenticity of the record.

    CRITICAL FOR RESEARCH INTEGRITY:
    SYNTHETIC_DEMO records are strictly isolated and never mixed into real
    experimental benchmarks or final published research metrics.
    """
    REAL_MEASUREMENT = "REAL_MEASUREMENT"
    SYNTHETIC_DEMO = "SYNTHETIC_DEMO"


class FaultOrigin(str, Enum):
    """Origin mechanism of the observed condition."""
    NATURAL_WEAR = "NATURAL_WEAR"
    FIXTURE_INJECTED = "FIXTURE_INJECTED"
    GOLDEN_REFERENCE = "GOLDEN_REFERENCE"
    UNKNOWN = "UNKNOWN"


class FaultClass(str, Enum):
    """Categorical ground truth condition of the USB-C cable."""
    INTACT = "INTACT"
    RESISTIVE_BEND = "RESISTIVE_BEND"
    INTERMITTENT_DISCONNECT = "INTERMITTENT_DISCONNECT"
    VBUS_DEGRADATION = "VBUS_DEGRADATION"
    GND_DEGRADATION = "GND_DEGRADATION"
    UNKNOWN = "UNKNOWN"


class LabelSource(str, Enum):
    """Authority that assigned the ground-truth label."""
    HUMAN_VERIFIED = "HUMAN_VERIFIED"
    LAB_BENCH_TESTER = "LAB_BENCH_TESTER"
    CONTROLLED_FIXTURE_INJECTION = "CONTROLLED_FIXTURE_INJECTION"
    UNVERIFIED = "UNVERIFIED"


@dataclass(frozen=True)
class PhysicalCableIdentity:
    """Hardware identity of the specific physical cable under test.

    Used for Leave-One-Cable-Out cross validation to prevent data leakage.
    """
    cable_id: str                      # Unique ID, e.g. "CABLE-ANKER-01"
    manufacturer: str = "GENERIC"
    cable_type: str = "USB_C_TO_C"     # e.g., USB_C_TO_C, USB_A_TO_C
    rated_current_a: float = 3.0       # Units: Amperes (e.g., 3.0A or 5.0A)
    rated_length_m: float = 1.0        # Units: Meters
    has_emask_chip: bool = False       # E-marker presence indicator


@dataclass(frozen=True)
class TrialSessionIdentity:
    """Identifier for the experimental trial and session execution."""
    session_id: str                    # Unique session identifier, e.g., "SESS-20261111-01"
    trial_id: str                      # Specific trial pass, e.g., "TR-0012"
    step_index: int = 0                # Time-sequence index within this trial


@dataclass(frozen=True)
class TestConditions:
    """Environmental and electrical operating parameters."""
    __test__ = False  # Prevent pytest from collecting this schema dataclass as a test case

    nominal_voltage_v: float = 5.0     # Units: Volts (e.g. 5.0V USB standard)
    target_current_a: float = 1.0      # Units: Amperes
    ambient_temp_c: float = 25.0       # Units: Degrees Celsius
    flex_angle_deg: float = 0.0        # Units: Degrees (e.g. 0=straight, 90=bent)
    mechanical_state: str = "STATIC"   # e.g., "STATIC", "BENDING_SWEEP", "TORSION"


@dataclass(frozen=True)
class ElectricalObservation:
    """Raw high-precision electrical readings."""
    v_source_v: float                  # Source voltage (upstream of cable), Units: Volts
    v_sink_v: float                    # Sink voltage (downstream at load), Units: Volts
    current_a: float                   # Current passing through cable path, Units: Amperes
    timestamp_rel_ms: float            # Relative time since trial start, Units: Milliseconds
    sampling_jitter_us: float = 0.0    # Jitter relative to nominal sampling period, Units: Microseconds

    @property
    def voltage_drop_v(self) -> float:
        """Calculate direct raw voltage drop across the cable."""
        return self.v_source_v - self.v_sink_v

    @property
    def apparent_resistance_ohms(self) -> float:
        """Apparent resistance (Ohm's Law: R = delta_V / I).

        Returns 0.0 if current is near zero to avoid division by zero.
        """
        if abs(self.current_a) < 1e-4:
            return 0.0
        return self.voltage_drop_v / self.current_a


@dataclass(frozen=True)
class DataProvenance:
    """Traceability metadata ensuring all data can be audited for scientific review."""
    source_device: str                 # e.g., "ESP32_INA226_FIXTURE_V1" or "MOCK_HARNESS"
    firmware_version: str = "1.0.0"
    operator_id: str = "LAB_OPERATOR_01"
    git_commit_hash: str = "UNCOMMITTED"
    recorded_at_utc: str = "1970-01-01T00:00:00Z" # ISO 8601 UTC timestamp
    checksum_sha256: str = ""          # Optional payload hash for immutability check


@dataclass(frozen=True)
class InstrumentCalibration:
    """Test fixture calibration offsets.

    The test fixture itself (traces, connectors, shunt) has resistance that must be
    tared out to isolate cable resistance.
    """
    fixture_tare_resistance_ohms: float = 0.040  # Tare resistance of test fixture, Units: Ohms
    v_offset_v: float = 0.0                      # Voltage offset, Units: Volts
    i_offset_a: float = 0.0                      # Current offset, Units: Amperes
    last_calibrated_utc: str = "2026-01-01T00:00:00Z"


@dataclass(frozen=True)
class AcquisitionQuality:
    """Sensor health and acquisition integrity metrics."""
    sample_rate_hz: float = 100.0      # Nominal acquisition rate, Units: Hz
    dropped_samples_count: int = 0     # Number of lost or missed ticks
    is_clipped: bool = False           # Flag if ADC reached saturation limit
    adc_settled: bool = True           # Flag if INA226 conversion period was respected


@dataclass
class CableMeasurementRecord:
    """Master typed experimental record for FlexPulse-ML.

    Encapsulates all physical, electrical, provenance, and experimental dimensions
    into a single strictly-validated scientific structure.
    """
    cable: PhysicalCableIdentity
    session: TrialSessionIdentity
    conditions: TestConditions
    observation: ElectricalObservation
    ground_truth: FaultClass = FaultClass.UNKNOWN
    label_source: LabelSource = LabelSource.UNVERIFIED
    fault_origin: FaultOrigin = FaultOrigin.UNKNOWN
    provenance: DataProvenance = field(default_factory=lambda: DataProvenance(source_device="UNKNOWN"))
    calibration: InstrumentCalibration = field(default_factory=InstrumentCalibration)
    quality: AcquisitionQuality = field(default_factory=AcquisitionQuality)
    dataset_category: DatasetCategory = DatasetCategory.REAL_MEASUREMENT

    def compute_sha256(self) -> str:
        """Compute cryptographic hash of the observation and identity for provenance."""
        payload = f"{self.cable.cable_id}:{self.session.session_id}:{self.session.trial_id}:{self.session.step_index}:{self.observation.timestamp_rel_ms}:{self.observation.v_source_v}:{self.observation.v_sink_v}:{self.observation.current_a}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_flat_dict(self) -> Dict[str, Any]:
        """Flatten nested dataclass into a single dictionary for pandas DataFrame export."""
        return {
            # Cable Identity
            "cable_id": self.cable.cable_id,
            "cable_manufacturer": self.cable.manufacturer,
            "cable_type": self.cable.cable_type,
            "rated_current_a": self.cable.rated_current_a,
            "rated_length_m": self.cable.rated_length_m,
            "has_emask_chip": self.cable.has_emask_chip,
            # Session Identity
            "session_id": self.session.session_id,
            "trial_id": self.session.trial_id,
            "step_index": self.session.step_index,
            # Test Conditions
            "nominal_voltage_v": self.conditions.nominal_voltage_v,
            "target_current_a": self.conditions.target_current_a,
            "ambient_temp_c": self.conditions.ambient_temp_c,
            "flex_angle_deg": self.conditions.flex_angle_deg,
            "mechanical_state": self.conditions.mechanical_state,
            # Electrical Observations
            "v_source_v": self.observation.v_source_v,
            "v_sink_v": self.observation.v_sink_v,
            "current_a": self.observation.current_a,
            "timestamp_rel_ms": self.observation.timestamp_rel_ms,
            "sampling_jitter_us": self.observation.sampling_jitter_us,
            "voltage_drop_v": self.observation.voltage_drop_v,
            "apparent_resistance_ohms": self.observation.apparent_resistance_ohms,
            # Labels & Origin
            "fault_class": self.ground_truth.value,
            "label_source": self.label_source.value,
            "fault_origin": self.fault_origin.value,
            "dataset_category": self.dataset_category.value,
            # Provenance
            "source_device": self.provenance.source_device,
            "firmware_version": self.provenance.firmware_version,
            "operator_id": self.provenance.operator_id,
            "git_commit_hash": self.provenance.git_commit_hash,
            "recorded_at_utc": self.provenance.recorded_at_utc,
            # Calibration & Quality
            "fixture_tare_resistance_ohms": self.calibration.fixture_tare_resistance_ohms,
            "v_offset_v": self.calibration.v_offset_v,
            "i_offset_a": self.calibration.i_offset_a,
            "sample_rate_hz": self.quality.sample_rate_hz,
            "dropped_samples_count": self.quality.dropped_samples_count,
            "is_clipped": self.quality.is_clipped,
            "adc_settled": self.quality.adc_settled,
        }

    def to_json(self) -> str:
        """Serialize full record to structured JSON."""
        return json.dumps(self.to_flat_dict(), indent=2)

    @classmethod
    def from_flat_dict(cls, d: Dict[str, Any]) -> CableMeasurementRecord:
        """Reconstruct a typed CableMeasurementRecord from a flat dictionary."""
        # Normalize enums safely
        try:
            category = DatasetCategory(d.get("dataset_category", DatasetCategory.REAL_MEASUREMENT.value))
        except ValueError:
            category = DatasetCategory.REAL_MEASUREMENT

        try:
            fault = FaultClass(d.get("fault_class", FaultClass.UNKNOWN.value))
        except ValueError:
            fault = FaultClass.UNKNOWN

        try:
            label_src = LabelSource(d.get("label_source", LabelSource.UNVERIFIED.value))
        except ValueError:
            label_src = LabelSource.UNVERIFIED

        try:
            origin = FaultOrigin(d.get("fault_origin", FaultOrigin.UNKNOWN.value))
        except ValueError:
            origin = FaultOrigin.UNKNOWN

        cable = PhysicalCableIdentity(
            cable_id=str(d.get("cable_id", "")),
            manufacturer=str(d.get("cable_manufacturer", "GENERIC")),
            cable_type=str(d.get("cable_type", "USB_C_TO_C")),
            rated_current_a=float(d.get("rated_current_a", 3.0)),
            rated_length_m=float(d.get("rated_length_m", 1.0)),
            has_emask_chip=bool(d.get("has_emask_chip", False)),
        )

        session = TrialSessionIdentity(
            session_id=str(d.get("session_id", "")),
            trial_id=str(d.get("trial_id", "")),
            step_index=int(d.get("step_index", 0)),
        )

        conditions = TestConditions(
            nominal_voltage_v=float(d.get("nominal_voltage_v", 5.0)),
            target_current_a=float(d.get("target_current_a", 1.0)),
            ambient_temp_c=float(d.get("ambient_temp_c", 25.0)),
            flex_angle_deg=float(d.get("flex_angle_deg", 0.0)),
            mechanical_state=str(d.get("mechanical_state", "STATIC")),
        )

        observation = ElectricalObservation(
            v_source_v=float(d.get("v_source_v", 0.0)),
            v_sink_v=float(d.get("v_sink_v", 0.0)),
            current_a=float(d.get("current_a", 0.0)),
            timestamp_rel_ms=float(d.get("timestamp_rel_ms", 0.0)),
            sampling_jitter_us=float(d.get("sampling_jitter_us", 0.0)),
        )

        provenance = DataProvenance(
            source_device=str(d.get("source_device", "UNKNOWN")),
            firmware_version=str(d.get("firmware_version", "1.0.0")),
            operator_id=str(d.get("operator_id", "OPERATOR")),
            git_commit_hash=str(d.get("git_commit_hash", "UNCOMMITTED")),
            recorded_at_utc=str(d.get("recorded_at_utc", "1970-01-01T00:00:00Z")),
        )

        calibration = InstrumentCalibration(
            fixture_tare_resistance_ohms=float(d.get("fixture_tare_resistance_ohms", 0.040)),
            v_offset_v=float(d.get("v_offset_v", 0.0)),
            i_offset_a=float(d.get("i_offset_a", 0.0)),
        )

        quality = AcquisitionQuality(
            sample_rate_hz=float(d.get("sample_rate_hz", 100.0)),
            dropped_samples_count=int(d.get("dropped_samples_count", 0)),
            is_clipped=bool(d.get("is_clipped", False)),
            adc_settled=bool(d.get("adc_settled", True)),
        )

        return cls(
            cable=cable,
            session=session,
            conditions=conditions,
            observation=observation,
            ground_truth=fault,
            label_source=label_src,
            fault_origin=origin,
            provenance=provenance,
            calibration=calibration,
            quality=quality,
            dataset_category=category,
        )
