"""
Automated unit tests for Phase 0: Project Foundation.
"""

from pathlib import Path
import numpy as np
import pytest

from flexpulse_ml.config import (
    AppConfig,
    DEFAULT_CONFIG_PATH,
    get_app_config,
    load_yaml_config,
)
from flexpulse_ml.logging import configure_logger, get_logger
from flexpulse_ml.seed import SeedManager


def test_default_config_yaml_exists():
    assert DEFAULT_CONFIG_PATH.exists()
    raw = load_yaml_config(DEFAULT_CONFIG_PATH)
    assert isinstance(raw, dict)
    assert raw["project"]["name"] == "FlexPulse-ML"
    assert "simulation" in raw
    assert "scenarios" in raw
    assert "models" in raw
    assert "self_calibration" in raw


def test_get_app_config_properties():
    cfg = get_app_config()
    assert isinstance(cfg, AppConfig)
    assert cfg.project_name == "FlexPulse-ML"
    assert cfg.random_seed == 42
    assert cfg.simulation.sampling_rate_hz == 100.0
    assert cfg.simulation.source_voltage_nominal_v == 5.0
    assert cfg.features.window_length_seconds == 0.5
    assert len(cfg.models.classifiers) == 4
    assert cfg.self_calibration.confidence_threshold == 0.85


def test_seed_manager_reproducibility():
    mgr1 = SeedManager(master_seed=12345)
    rng1 = mgr1.get_rng()
    vals1 = rng1.normal(0, 1, size=10)

    mgr2 = SeedManager(master_seed=12345)
    rng2 = mgr2.get_rng()
    vals2 = rng2.normal(0, 1, size=10)

    assert np.allclose(vals1, vals2)


def test_seed_manager_spawn_run_seeds():
    mgr = SeedManager(master_seed=42)
    seed_0 = mgr.spawn_run_seed(0)
    seed_1 = mgr.spawn_run_seed(1)
    seed_0_again = mgr.spawn_run_seed(0)

    assert seed_0 != seed_1
    assert seed_0 == seed_0_again


def test_logging_configuration():
    logger = configure_logger(name="test_logger", level="DEBUG")
    assert logger.name == "test_logger"
    assert get_logger("test_logger") is logger


def test_streamlit_pages_completeness():
    import flexpulse_ml.app as app

    expected_pages = [
        "1. Overview",
        "2. Simulator Controls",
        "3. Data Explorer",
        "4. Feature Analysis",
        "5. Model Training",
        "6. Self-Calibration",
        "7. Evaluation",
        "8. Export and Reports",
    ]
    assert hasattr(app, "render_synthetic_notice")
    assert hasattr(app, "main")
