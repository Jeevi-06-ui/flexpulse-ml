"""
Data management module providing synthetic dataset generation, storage, and validation.
"""

from flexpulse_ml.data.generator import (
    DatasetGeneratorConfig,
    SyntheticDatasetGenerator,
    generate_synthetic_dataset,
)
from flexpulse_ml.data.manifest import (
    DatasetManifest,
    compute_file_sha256,
)
from flexpulse_ml.data.storage import DatasetStorage
from flexpulse_ml.data.validator import (
    DatasetValidationReport,
    DatasetValidator,
    ValidationCheck,
    validate_synthetic_dataset,
)

__all__ = [
    "DatasetManifest",
    "compute_file_sha256",
    "DatasetStorage",
    "DatasetGeneratorConfig",
    "SyntheticDatasetGenerator",
    "generate_synthetic_dataset",
    "DatasetValidationReport",
    "DatasetValidator",
    "ValidationCheck",
    "validate_synthetic_dataset",
]
