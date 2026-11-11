"""
PySerial Hardware Data Adapter for FlexPulse-ML.

Receives newline-delimited JSON (NDJSON) telemetry packets from a future ESP32 + INA226
hardware controller over USB-UART.

DOCUMENTED ESP32 TELEMETRY PACKET FORMAT:
Each line transmitted over UART is a standalone JSON object terminated by a newline (\\n):
{
    "packet_type": "TELEMETRY",
    "sample_idx": 42,
    "timestamp_ms": 420.5,
    "vin_mv": 5012,            // Input/upstream voltage in millivolts
    "vout_mv": 4892,           // Output/downstream voltage in millivolts
    "current_ma": 1502.4,      // Measured current in milliamperes
    "temp_c": 26.4,            // Fixture temperature in Celsius
    "settled": true,           // Flag: ADC conversion time elapsed, steady state
    "quality": "OK",           // Sensor quality: "OK", "CLIPPED", "JITTER_HIGH"
    "crc16": "A1F4"            // Optional CRC-16 or simple hex checksum
}

ELECTRICAL SAFETY & CONTROLLER INTERFACE CONSTRAINTS:
1. Strict Read-Only Behavior: This driver NEVER issues commands that alter power
   delivery, switch electronic load relays, or modify power states.
2. The INA226 sensor operates at low-to-medium sampling rates (10 Hz - 1 kHz).
   It cannot capture microsecond RF data line transients.
3. If hardware is absent or unplugged, the adapter fails gracefully without crashing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

from flexpulse_ml.acquisition.hardware_interface import (
    BaseAcquisitionSource,
    HardwareNotConnectedError,
    HardwareSafetyViolationError,
)
from flexpulse_ml.acquisition.schema import (
    CanonicalMeasurementRecord,
    DatasetCategory,
    FaultClass,
    FaultOrigin,
    LabelSource,
    MeasurementQuality,
)
from flexpulse_ml.utils.logging import get_logger

logger = get_logger("acquisition.serial_adapter")


@dataclass
class PacketValidationResult:
    """Audit result for an incoming serial packet."""
    is_valid: bool
    raw_payload: str
    parsed_dict: Optional[Dict[str, Any]] = None
    error_reason: Optional[str] = None


@dataclass
class SerialStreamStatistics:
    """Live statistics for the telemetry acquisition session."""
    total_lines_read: int = 0
    valid_packets: int = 0
    corrupt_json_lines: int = 0
    missing_fields_packets: int = 0
    out_of_bounds_packets: int = 0
    sequence_gaps_detected: int = 0
    last_sample_index: int = -1


class ESP32JSONSerialAdapter(BaseAcquisitionSource):
    """Configurable PySerial adapter for newline-delimited JSON ESP32 telemetry.

    Parameters
    ----------
    port : str
        Serial device path (e.g. '/dev/tty.usbserial-0001' on macOS).
    baud_rate : int
        Baud rate (default 115200 bps).
    timeout_s : float
        Read timeout in seconds.
    """

    MANDATORY_PACKET_KEYS = {"sample_idx", "vin_mv", "vout_mv", "current_ma"}

    def __init__(
        self,
        port: str = "/dev/tty.usbserial-0001",
        baud_rate: int = 115200,
        timeout_s: float = 1.0,
    ):
        self.port = port
        self.baud_rate = baud_rate
        self.timeout_s = timeout_s
        self._serial_handle = None
        self._is_connected = False
        self.stats = SerialStreamStatistics()

    def check_port_available(self) -> bool:
        """Scan available macOS device nodes for matching serial interface."""
        import glob
        available = (
            glob.glob("/dev/tty.usb*")
            + glob.glob("/dev/tty.wch*")
            + glob.glob("/dev/tty.SLAB*")
            + glob.glob("/dev/cu.usb*")
        )
        return self.port in available

    def connect(self) -> bool:
        """Establish serial connection. Fails gracefully if hardware is disconnected."""
        try:
            import serial
        except ImportError:
            raise ImportError("PySerial is required for hardware acquisition. Install pyserial.")

        if not self.check_port_available():
            self._is_connected = False
            logger.info(
                f"ESP32 hardware not detected on port {self.port}. Gracefully operating in offline/demo mode."
            )
            return False

        try:
            self._serial_handle = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=self.timeout_s,
                rtscts=False,
                dsrdtr=False,
            )
            self._is_connected = True
            logger.info(f"Connected to ESP32 on {self.port} at {self.baud_rate} baud.")
            return True
        except Exception as e:
            self._is_connected = False
            logger.warning(f"Could not open serial port {self.port}: {e}")
            return False

    def disconnect(self) -> None:
        """Safely close serial connection."""
        if self._serial_handle and hasattr(self._serial_handle, "is_open") and self._serial_handle.is_open:
            try:
                self._serial_handle.close()
            except Exception as e:
                logger.warning(f"Error while closing serial port: {e}")
        self._is_connected = False
        logger.info(f"Disconnected serial port {self.port}.")

    def is_connected(self) -> bool:
        return bool(self._is_connected and self._serial_handle and self._serial_handle.is_open)

    def parse_and_validate_line(self, line: str) -> PacketValidationResult:
        """Validate a single newline-delimited line from the controller."""
        self.stats.total_lines_read += 1
        clean = line.strip()

        if not clean:
            return PacketValidationResult(is_valid=False, raw_payload=line, error_reason="EMPTY_LINE")

        # 1. Parse JSON syntax
        try:
            data = json.loads(clean)
        except json.JSONDecodeError as e:
            self.stats.corrupt_json_lines += 1
            return PacketValidationResult(
                is_valid=False, raw_payload=clean, error_reason=f"CORRUPT_JSON: {e}"
            )

        if not isinstance(data, dict):
            self.stats.corrupt_json_lines += 1
            return PacketValidationResult(
                is_valid=False, raw_payload=clean, error_reason="INVALID_JSON_TYPE: Root must be a JSON object"
            )

        # 2. Check mandatory keys
        missing = self.MANDATORY_PACKET_KEYS - set(data.keys())
        if missing:
            self.stats.missing_fields_packets += 1
            return PacketValidationResult(
                is_valid=False,
                raw_payload=clean,
                parsed_dict=data,
                error_reason=f"MISSING_KEYS: {sorted(list(missing))}",
            )

        # 3. Numeric range checks
        try:
            vin_v = float(data["vin_mv"]) / 1000.0
            vout_v = float(data["vout_mv"]) / 1000.0
            current_a = float(data["current_ma"]) / 1000.0
            sample_idx = int(data["sample_idx"])
        except (ValueError, TypeError) as e:
            self.stats.out_of_bounds_packets += 1
            return PacketValidationResult(
                is_valid=False, raw_payload=clean, parsed_dict=data, error_reason=f"NON_NUMERIC_VALUES: {e}"
            )

        if vin_v < 0.0 or vin_v > 28.0:
            self.stats.out_of_bounds_packets += 1
            return PacketValidationResult(
                is_valid=False, raw_payload=clean, parsed_dict=data, error_reason=f"OUT_OF_BOUNDS_VIN: {vin_v:.3f}V"
            )

        if vout_v < 0.0 or vout_v > 28.0:
            self.stats.out_of_bounds_packets += 1
            return PacketValidationResult(
                is_valid=False, raw_payload=clean, parsed_dict=data, error_reason=f"OUT_OF_BOUNDS_VOUT: {vout_v:.3f}V"
            )

        if vout_v > (vin_v + 0.15):
            self.stats.out_of_bounds_packets += 1
            return PacketValidationResult(
                is_valid=False,
                raw_payload=clean,
                parsed_dict=data,
                error_reason=f"PASSIVE_POWER_VIOLATION: Vout ({vout_v:.3f}V) > Vin ({vin_v:.3f}V)",
            )

        if current_a < 0.0 or current_a > 6.0:
            self.stats.out_of_bounds_packets += 1
            return PacketValidationResult(
                is_valid=False, raw_payload=clean, parsed_dict=data, error_reason=f"OUT_OF_BOUNDS_CURRENT: {current_a:.3f}A"
            )

        # 4. Sequence Continuity Check
        if self.stats.last_sample_index >= 0:
            expected = self.stats.last_sample_index + 1
            if sample_idx != expected:
                self.stats.sequence_gaps_detected += 1
                logger.warning(f"Telemetry sequence gap: expected sample {expected}, received {sample_idx}")

        self.stats.last_sample_index = sample_idx
        self.stats.valid_packets += 1
        return PacketValidationResult(is_valid=True, raw_payload=clean, parsed_dict=data)

    def packet_to_canonical_record(
        self,
        parsed_dict: Dict[str, Any],
        cable_id: str,
        session_id: str,
        trial_id: str,
        fixture_id: str = "FIXTURE_ESP32_INA226_01",
        instrument_id: str = "INA226_ADDR_0x40",
        calibration_id: str = "CAL_DEFAULT",
        action_id: str = "STATIC_HOLD",
        motion_region: str = "STATIC",
        load_setpoint_a: float = 1.0,
        ground_truth_label: str = FaultClass.UNKNOWN.value,
        ground_truth_source: str = LabelSource.UNVERIFIED.value,
        fault_origin: str = FaultOrigin.UNKNOWN.value,
        fault_condition: str = "NONE",
        source_type: str = DatasetCategory.REAL_MEASUREMENT.value,
    ) -> CanonicalMeasurementRecord:
        """Convert a validated raw packet dictionary into a CanonicalMeasurementRecord."""
        vin_v = float(parsed_dict["vin_mv"]) / 1000.0
        vout_v = float(parsed_dict["vout_mv"]) / 1000.0
        current_a = float(parsed_dict["current_ma"]) / 1000.0
        sample_idx = int(parsed_dict["sample_idx"])
        temp_c = float(parsed_dict.get("temp_c", 25.0))
        settled_val = parsed_dict.get("settled", True)
        is_settled = bool(settled_val) if not isinstance(settled_val, str) else settled_val.lower() == "true"
        quality_str = str(parsed_dict.get("quality", MeasurementQuality.OK.value))

        return CanonicalMeasurementRecord(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            cable_id=cable_id,
            session_id=session_id,
            trial_id=trial_id,
            sample_index=sample_idx,
            vin_v=vin_v,
            vout_v=vout_v,
            current_a=current_a,
            load_setpoint_a=load_setpoint_a,
            temperature_c=temp_c,
            source_type=source_type,
            ground_truth_label=ground_truth_label,
            ground_truth_source=ground_truth_source,
            fault_origin=fault_origin,
            fault_condition=fault_condition,
            fixture_id=fixture_id,
            instrument_id=instrument_id,
            calibration_id=calibration_id,
            action_id=action_id,
            motion_region=motion_region,
            is_settled=is_settled,
            measurement_quality=quality_str,
        )

    def read_stream(
        self, max_samples: Optional[int] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream raw parsed JSON packets from the physical serial port."""
        if not self.is_connected():
            raise HardwareNotConnectedError(
                f"Cannot stream telemetry: Physical hardware is not connected on {self.port}."
            )

        samples_emitted = 0
        while max_samples is None or samples_emitted < max_samples:
            raw_line = self._serial_handle.readline().decode("utf-8", errors="replace").strip()
            if not raw_line:
                continue

            val_res = self.parse_and_validate_line(raw_line)
            if val_res.is_valid and val_res.parsed_dict:
                yield val_res.parsed_dict
                samples_emitted += 1
            else:
                logger.warning(f"Discarding corrupted serial packet: {val_res.error_reason}")


class SimulatedESP32Stream:
    """Deterministic packet feeder for automated testing of the serial adapter.

    Generates lines identical to the physical ESP32 UART stream without requiring hardware.
    """

    def __init__(self, samples: int = 10, inject_corrupt_lines: bool = False):
        self.samples = samples
        self.inject_corrupt_lines = inject_corrupt_lines

    def generate_lines(self) -> List[str]:
        lines = []
        for i in range(self.samples):
            if self.inject_corrupt_lines and i == 2:
                # Inject a malformed JSON line
                lines.append("{bad_json_packet, not valid")
            elif self.inject_corrupt_lines and i == 5:
                # Inject missing key packet
                lines.append(json.dumps({"sample_idx": i, "vin_mv": 5000}))
            else:
                pkt = {
                    "packet_type": "TELEMETRY",
                    "sample_idx": i,
                    "timestamp_ms": i * 10.0,
                    "vin_mv": 5010,
                    "vout_mv": 4890,
                    "current_ma": 1500.0,
                    "temp_c": 25.4,
                    "settled": True,
                    "quality": "OK",
                }
                lines.append(json.dumps(pkt))
        return lines
