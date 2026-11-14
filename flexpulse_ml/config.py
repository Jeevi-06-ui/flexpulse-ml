"""
Configuration management module for FlexPulse-ML.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

from flexpulse_ml.logging import configure_logger

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default_config.yaml"


@dataclass
class SimulationConfig:
    sampling_rate_hz: float = 100.0
    duration_seconds: float = 60.0
    calibration_duration_seconds: float = 5.0
    source_voltage_nominal_v: float = 5.0
    baseline_path_resistance_nominal_ohms: float = 0.080
    load_resistance_nominal_ohms: float = 3.333
    measurement_noise_voltage_std_v: float = 0.005
    measurement_noise_current_std_a: float = 0.008
    voltage_offset_v: float = 0.001
    current_offset_a: float = 0.001
    voltage_offset_drift_v_per_sec: float = 0.0001
    current_offset_drift_a_per_sec: float = 0.00005


@dataclass
class FeatureConfig:
    window_length_seconds: float = 0.5
    overlap_fraction: float = 0.5
    min_samples_per_window: int = 25


@dataclass
class ModelConfig:
    baseline: str = "fixed_threshold"
    classifiers: List[str] = field(
        default_factory=lambda: [
            "logistic_regression",
            "decision_tree",
            "random_forest",
            "support_vector_machine",
        ]
    )
    test_size: float = 0.20
    val_size: float = 0.20
    cross_validation_runs: int = 5


@dataclass
class CalibrationConfig:
    initial_calibration_duration_s: float = 5.0
    confidence_threshold: float = 0.85
    adaptation_rate: float = 0.05
    max_update_delta: float = 0.02
    stability_variance_threshold: float = 0.001


@dataclass
class AppConfig:
    project_name: str = "FlexPulse-ML"
    title: str = "FlexPulse-ML: A Simulation-Based Self-Calibrating Machine Learning Framework"
    version: str = "0.2.0"
    random_seed: int = 42
    paths: Dict[str, Path] = field(default_factory=dict)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    scenarios: Dict[str, Any] = field(default_factory=dict)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    self_calibration: CalibrationConfig = field(default_factory=CalibrationConfig)

    def resolve_path(self, key: str) -> Path:
        rel_or_abs = self.paths.get(key, Path(key))
        if Path(rel_or_abs).is_absolute():
            return Path(rel_or_abs)
        return PROJECT_ROOT / rel_or_abs


def load_yaml_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    path_to_open = config_path or DEFAULT_CONFIG_PATH
    if not path_to_open.exists():
        return {}
    with open(path_to_open, "r", encoding="utf-8") as file_handle:
        loaded = yaml.safe_load(file_handle)
    return loaded if isinstance(loaded, dict) else {}


def get_app_config(config_path: Optional[Path] = None) -> AppConfig:
    raw_dict = load_yaml_config(config_path)

    project_meta = raw_dict.get("project", {})
    seed = int(raw_dict.get("random_seed", 42))

    paths_dict: Dict[str, Path] = {}
    raw_paths = raw_dict.get("paths", {})
    for key, value in raw_paths.items():
        paths_dict[key] = PROJECT_ROOT / value

    sim_dict = raw_dict.get("simulation", {})
    sim_cfg = SimulationConfig(**sim_dict) if sim_dict else SimulationConfig()

    feat_dict = raw_dict.get("features", {})
    feat_cfg = FeatureConfig(**feat_dict) if feat_dict else FeatureConfig()

    mod_dict = raw_dict.get("models", {})
    mod_cfg = ModelConfig(**mod_dict) if mod_dict else ModelConfig()

    cal_dict = raw_dict.get("self_calibration", {})
    cal_cfg = CalibrationConfig(**cal_dict) if cal_dict else CalibrationConfig()

    logging_dict = raw_dict.get("logging", {})
    configure_logger(
        level=logging_dict.get("level", "INFO"),
        log_format=logging_dict.get("format"),
        date_format=logging_dict.get("date_format"),
    )

    return AppConfig(
        project_name=project_meta.get("name", "FlexPulse-ML"),
        title=project_meta.get("title", "FlexPulse-ML"),
        version=project_meta.get("version", "0.2.0"),
        random_seed=seed,
        paths=paths_dict,
        simulation=sim_cfg,
        scenarios=raw_dict.get("scenarios", {}),
        features=feat_cfg,
        models=mod_cfg,
        self_calibration=cal_cfg,
    )
