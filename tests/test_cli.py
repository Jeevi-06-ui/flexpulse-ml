"""
Integration tests for FlexPulse-ML CLI commands.
"""

from pathlib import Path
import pytest

from flexpulse_ml.cli import (
    cmd_env_check,
    cmd_generate_demo,
    cmd_status,
    cmd_validate_data,
)


def test_cli_env_check():
    """Verify environment check succeeds in the configured workspace."""
    exit_code = cmd_env_check()
    assert exit_code == 0


def test_cli_status():
    """Verify research status check succeeds and displays integrity information."""
    exit_code = cmd_status()
    assert exit_code == 0


def test_cli_demo_and_validation(tmp_path: Path):
    """Verify CLI generation of demo dataset and subsequent validation."""
    demo_file = tmp_path / "cli_demo_test.csv"
    gen_code = cmd_generate_demo(str(demo_file), samples=25)
    assert gen_code == 0
    assert demo_file.exists()

    # Validate generated file
    val_code = cmd_validate_data(str(demo_file), disallow_synthetic=False)
    assert val_code == 0

    # Validate with disallow_synthetic should return error code 1 (since it contains synthetic data)
    val_disallow_code = cmd_validate_data(str(demo_file), disallow_synthetic=True)
    assert val_disallow_code == 1
