"""
FlexPulse-ML: A Simulation-Based Self-Calibrating Machine Learning Framework
for Intermittent Fault Detection in USB-C-Inspired Power Paths.
"""

from flexpulse_ml.config import AppConfig, get_app_config
from flexpulse_ml.logging import configure_logger, get_logger
from flexpulse_ml.seed import SeedManager
from flexpulse_ml.simulation import (
    PowerPathSimulator,
    SimulationParams,
    SimulationResult,
)

__version__ = "0.2.0"
__all__ = [
    "AppConfig",
    "PowerPathSimulator",
    "SeedManager",
    "SimulationParams",
    "SimulationResult",
    "configure_logger",
    "get_app_config",
    "get_logger",
]
