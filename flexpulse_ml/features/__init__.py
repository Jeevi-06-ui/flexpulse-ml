"""
Electrical feature engineering package for sliding-window segmentation, extraction and validation.
"""

from flexpulse_ml.features.dataset import (
    FeatureDatasetGenerator,
    FeatureDatasetStorage,
    FeatureManifest,
    generate_feature_dataset,
)
from flexpulse_ml.features.extractor import (
    ElectricalFeatureExtractor,
    FEATURE_NAMES,
    FEATURE_UNITS,
)
from flexpulse_ml.features.validator import (
    FeatureValidationCheck,
    FeatureValidationReport,
    FeatureValidator,
    validate_feature_dataset,
)
from flexpulse_ml.features.windowing import (
    SlidingWindowGenerator,
    Window,
    WindowConfig,
)

__all__ = [
    "WindowConfig",
    "Window",
    "SlidingWindowGenerator",
    "ElectricalFeatureExtractor",
    "FEATURE_NAMES",
    "FEATURE_UNITS",
    "FeatureManifest",
    "FeatureDatasetStorage",
    "FeatureDatasetGenerator",
    "generate_feature_dataset",
    "FeatureValidationCheck",
    "FeatureValidationReport",
    "FeatureValidator",
    "validate_feature_dataset",
]
