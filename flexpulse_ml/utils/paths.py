"""
Path definitions and workspace management for FlexPulse-ML.
Ensures uniform directory access across CLI, acquisition, models, and tests.
"""

from pathlib import Path

# Repository Root (parent of the flexpulse_ml package folder)
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PACKAGE_ROOT.parent

# Standard Research Directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_REAL_DIR = RAW_DATA_DIR / "real"
RAW_SYNTHETIC_DIR = RAW_DATA_DIR / "synthetic_demo"

PROCESSED_DATA_DIR = DATA_DIR / "processed"
DEMO_DATA_DIR = DATA_DIR / "demo"
METADATA_DIR = DATA_DIR / "metadata"
QUARANTINE_DIR = METADATA_DIR / "quarantine"
SESSIONS_METADATA_DIR = METADATA_DIR / "sessions"
MANIFESTS_METADATA_DIR = METADATA_DIR / "manifests"
CALIBRATION_METADATA_DIR = METADATA_DIR / "calibration"

MODELS_DIR = PROJECT_ROOT / "models"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
EXPERIMENTS_DIR = RESULTS_DIR / "experiments"

CONFIGS_DIR = PROJECT_ROOT / "configs"
TESTS_DIR = PROJECT_ROOT / "tests"
DOCS_DIR = PROJECT_ROOT / "docs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

AUDIT_LOG_FILE = METADATA_DIR / "audit_invalid_records.jsonl"
RESEARCH_STATUS_FILE = PROJECT_ROOT / "research_status.json"
CABLE_REGISTRY_FILE = METADATA_DIR / "cable_registry.json"


def ensure_directories_exist() -> None:
    """Ensure all core project directories are created."""
    dirs = [
        RAW_DATA_DIR,
        RAW_REAL_DIR,
        RAW_SYNTHETIC_DIR,
        PROCESSED_DATA_DIR,
        DEMO_DATA_DIR,
        METADATA_DIR,
        QUARANTINE_DIR,
        SESSIONS_METADATA_DIR,
        MANIFESTS_METADATA_DIR,
        CALIBRATION_METADATA_DIR,
        MODELS_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        EXPERIMENTS_DIR,
        CONFIGS_DIR,
        TESTS_DIR,
        DOCS_DIR,
        SCRIPTS_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
