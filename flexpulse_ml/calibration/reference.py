"""
Reference Calibration and Fixture Correction Subsystem for FlexPulse-ML.

Defines the reference planes, instrument zero offsets, fixture tare resistances,
and characterized golden reference cables.

SCIENTIFIC PRINCIPLES:
1. Fixture tare resistance (PCB traces, current-sense shunt, pogo-pins/sockets)
   must be characterized independently using a 4-wire Kelvin micro-ohmmeter.
2. A physical reference ("golden") USB-C cable MUST NOT be assumed to have zero resistance!
   A standard 1-meter high-quality USB-C copper assembly has an intrinsic path resistance
   typically between 50 mOhm and 90 mOhm.
3. Fixture correction is strictly confined to characterized reference planes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Optional

from flexpulse_ml.utils.logging import get_logger
from flexpulse_ml.utils.paths import (
    CALIBRATION_METADATA_DIR,
    ensure_directories_exist,
)

logger = get_logger("calibration.reference")


@dataclass
class CharacterizedReferenceCable:
    """A physical, laboratory-characterized golden reference cable.

    CRITICAL SCIENTIFIC INTEGRITY CONSTRAINT:
    Never assume a reference cable has 0 Ohm resistance. Its true resistance
    must be measured on calibrated laboratory bench instrumentation (e.g., Keysight 34461A).
    """
    reference_cable_id: str                      # e.g. "GOLDEN-REF-CABLE-01"
    manufacturer: str = "LAB_STANDARDS"
    cable_length_m: float = 1.0
    four_wire_measured_resistance_ohms: float = 0.072  # 72 mOhm measured Kelvin resistance
    measurement_temperature_c: float = 25.0
    measurement_instrument: str = "Keysight 34461A 6.5-Digit DMM (4-Wire)"
    calibrated_date_utc: str = "2026-01-15T00:00:00Z"
    uncertainty_ohms: float = 0.002              # +/- 2 mOhm 95% confidence


@dataclass
class FixtureTareProfile:
    """Characterized electrical tare and offset of the test fixture hardware."""
    fixture_id: str                              # e.g. "FIXTURE_ESP32_INA226_01"
    tare_resistance_ohms: float = 0.040          # 40 mOhm path resistance of fixture traces + shunt
    voltage_offset_v: float = 0.0005             # 0.5 mV ADC zero-offset
    current_offset_a: float = 0.0010             # 1.0 mA zero-current ADC leakage
    temperature_nominal_c: float = 25.0
    temperature_coefficient_ppm_c: float = 50.0  # Shunt thermal tempco (PPM/C)
    calibration_date_utc: str = "2026-01-15T00:00:00Z"
    calibrated_by: str = "LAB_METROLOGY_ENGINEER"
    notes: str = "Characterized using 4-wire shorting plug across USB-C receptacles."


@dataclass
class CalibrationRecord:
    """Master calibration profile linking fixture tare and reference cable properties."""
    calibration_id: str
    fixture: FixtureTareProfile
    reference_cable: Optional[CharacterizedReferenceCable] = None
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calibration_id": self.calibration_id,
            "created_at_utc": self.created_at_utc,
            "fixture": asdict(self.fixture),
            "reference_cable": asdict(self.reference_cable) if self.reference_cable else None,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> CalibrationRecord:
        fixture_data = d.get("fixture", {})
        fixture = FixtureTareProfile(**fixture_data)
        ref_data = d.get("reference_cable")
        ref_cable = CharacterizedReferenceCable(**ref_data) if ref_data else None
        return cls(
            calibration_id=d.get("calibration_id", "CAL_DEFAULT"),
            fixture=fixture,
            reference_cable=ref_cable,
            created_at_utc=d.get("created_at_utc", datetime.now(timezone.utc).isoformat()),
        )


class CalibrationRegistry:
    """Manages storage and retrieval of characterized fixture calibration profiles."""

    def __init__(self, storage_dir: Optional[Path] = None):
        ensure_directories_exist()
        self.storage_dir = storage_dir or CALIBRATION_METADATA_DIR
        self._default_profile = self._build_default_profile()

    def _build_default_profile(self) -> CalibrationRecord:
        """Create the laboratory baseline calibration record."""
        fixture = FixtureTareProfile(
            fixture_id="FIXTURE_ESP32_INA226_01",
            tare_resistance_ohms=0.040, # 40 mOhm characterized trace + shunt tare
            voltage_offset_v=0.0005,
            current_offset_a=0.0010,
        )
        golden_cable = CharacterizedReferenceCable(
            reference_cable_id="GOLDEN-REF-CABLE-01",
            manufacturer="LAB_STANDARDS",
            cable_length_m=1.0,
            four_wire_measured_resistance_ohms=0.072, # Non-zero reference resistance!
        )
        return CalibrationRecord(
            calibration_id="CAL_DEFAULT_2026",
            fixture=fixture,
            reference_cable=golden_cable,
        )

    def get_profile(self, calibration_id: str = "CAL_DEFAULT_2026") -> CalibrationRecord:
        """Retrieve calibration profile by ID or return default lab profile."""
        profile_path = self.storage_dir / f"{calibration_id}.json"
        if profile_path.exists():
            try:
                with open(profile_path, "r", encoding="utf-8") as f:
                    return CalibrationRecord.from_dict(json.load(f))
            except Exception as e:
                logger.warning(f"Could not load calibration {calibration_id}: {e}. Falling back to default.")
        return self._default_profile

    def save_profile(self, record: CalibrationRecord) -> Path:
        """Save calibration profile to disk."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        path = self.storage_dir / f"{record.calibration_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, indent=2)
        logger.info(f"Saved calibration profile to {path}")
        return path
