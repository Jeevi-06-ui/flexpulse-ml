"""
Hardware Abstraction Layer (HAL) for FlexPulse-ML.

Provides an abstract interface for acquisition sources, and a concrete
implementation for a future ESP32 + INA226 USB-C test fixture.
Handles missing hardware gracefully.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generator, List, Optional
import time

from flexpulse_ml.acquisition.schema import CableMeasurementRecord
from flexpulse_ml.utils.logging import get_logger

logger = get_logger("acquisition.hardware")


class HardwareNotConnectedError(Exception):
    """Raised when the physical test fixture is unavailable."""
    pass


class HardwareSafetyViolationError(Exception):
    """Raised when requested test conditions violate physical safety boundaries."""
    pass


class BaseAcquisitionSource(ABC):
    """Abstract base class for all telemetry input sources."""

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the telemetry source."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Safely release the source."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Return True if the source is active and ready."""
        pass

    @abstractmethod
    def read_stream(self, max_samples: Optional[int] = None) -> Generator[CableMeasurementRecord, None, None]:
        """Stream measurement records from the source."""
        pass


class ESP32SerialAdapter(BaseAcquisitionSource):
    """Serial communication adapter for an ESP32 micro-controller reading an INA226 sensor.

    NOTE ON PHYSICAL CONSTRAINTS:
    The INA226 current/power monitor uses an integrating delta-sigma ADC with typical conversion
    times between 140 us and 8.24 ms. It is suited for steady-state path resistance and low-frequency
    contact flutter, but cannot detect microsecond-level transients or verify USB 2.0/3.x high-speed
    differential compliance.
    """

    def __init__(self, port: str = "/dev/tty.usbserial-0001", baud_rate: int = 115200, timeout_s: float = 2.0):
        self.port = port
        self.baud_rate = baud_rate
        self.timeout_s = timeout_s
        self._serial_handle = None
        self._is_connected = False

    def check_port_available(self) -> bool:
        """Check if the physical serial device node exists on macOS."""
        import glob
        available_ports = glob.glob("/dev/tty.usb*") + glob.glob("/dev/tty.wch*") + glob.glob("/dev/cu.usb*")
        return self.port in available_ports

    def connect(self) -> bool:
        """Attempt to connect to the serial fixture.

        Fails gracefully with HardwareNotConnectedError if the device is not plugged in.
        """
        try:
            import serial
        except ImportError:
            raise ImportError("PySerial is required for hardware acquisition. Install it or use CSV ingestion.")

        if not self.check_port_available():
            self._is_connected = False
            logger.info(f"ESP32 hardware not detected on port {self.port}. Gracefully operating in offline mode.")
            return False

        try:
            self._serial_handle = serial.Serial(self.port, self.baud_rate, timeout=self.timeout_s)
            self._is_connected = True
            logger.info(f"Connected to ESP32 test fixture on {self.port} at {self.baud_rate} baud.")
            return True
        except serial.SerialException as e:
            self._is_connected = False
            logger.warning(f"Could not open serial port {self.port}: {e}")
            return False

    def disconnect(self) -> None:
        """Close the serial port."""
        if self._serial_handle and self._serial_handle.is_open:
            self._serial_handle.close()
        self._is_connected = False

    def is_connected(self) -> bool:
        return self._is_connected and (self._serial_handle is not None) and self._serial_handle.is_open

    def read_stream(self, max_samples: Optional[int] = None) -> Generator[CableMeasurementRecord, None, None]:
        """Yield measurement records from the physical serial fixture.

        Raises HardwareNotConnectedError if called when hardware is disconnected.
        """
        if not self.is_connected():
            raise HardwareNotConnectedError(
                f"Cannot stream telemetry: Physical hardware is not connected on {self.port}. "
                "Connect the USB-C test fixture or use CSV/synthetic ingestion instead."
            )

        samples_read = 0
        while max_samples is None or samples_read < max_samples:
            line = self._serial_handle.readline().decode("utf-8", errors="replace").strip()
            if not line:
                continue
            # Future Phase: parse real hardware packets from JSON/CSV serial stream
            samples_read += 1
            time.sleep(0.01)
