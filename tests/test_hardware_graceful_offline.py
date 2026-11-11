"""
Tests for Hardware Abstraction Layer ensuring graceful operation in offline mode.
"""

import pytest

from flexpulse_ml.acquisition.hardware_interface import (
    ESP32SerialAdapter,
    HardwareNotConnectedError,
)


def test_hardware_offline_graceful_check():
    """Verify that an unavailable serial port does not raise an unhandled crash on connection attempt."""
    adapter = ESP32SerialAdapter(port="/dev/tty.nonexistent_usb_port_9999")
    # check_port_available should return False
    assert adapter.check_port_available() is False

    # connect should return False gracefully, NOT crash
    connected = adapter.connect()
    assert connected is False
    assert adapter.is_connected() is False


def test_hardware_read_without_connection_raises_informative_error():
    """Attempting to read from a disconnected hardware adapter raises HardwareNotConnectedError."""
    adapter = ESP32SerialAdapter(port="/dev/tty.nonexistent_usb_port_9999")
    with pytest.raises(HardwareNotConnectedError) as exc_info:
        # Request stream generator
        stream = adapter.read_stream(max_samples=5)
        next(stream)

    assert "Physical hardware is not connected" in str(exc_info.value)
