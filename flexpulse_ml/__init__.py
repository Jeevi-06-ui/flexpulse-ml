"""
FlexPulse-ML: A Simulation-Based Self-Calibrating Machine Learning Framework
for Intermittent Fault Detection in USB-C-Inspired Power Paths.
"""

from flexpulse_ml.config import AppConfig, get_app_config
from flexpulse_ml.logging import configure_logger, get_logger
from flexpulse_ml.seed import SeedManager

__version__ = "0.2.0"
__all__ = [
    "AppConfig",
    "SeedManager",
    "configure_logger",
    "get_app_config",
    "get_logger",
]
