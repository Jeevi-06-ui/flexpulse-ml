"""
Signal Preprocessing and Feature Extraction Subsystem for FlexPulse-ML.
"""

from flexpulse_ml.processing.features import (
    FEATURE_COLUMN_NAMES,
    FEATURE_DICTIONARY,
    detect_distinct_current_levels,
    export_feature_dictionary_docs,
    extract_feature_dataset,
    extract_features_from_conditioned_signal,
    fit_multi_load_resistance,
)
from flexpulse_ml.processing.preprocessing import (
    ConditionedSignal,
    PreprocessingAuditSummary,
    condition_trial_records,
)

__all__ = [
    "ConditionedSignal",
    "FEATURE_COLUMN_NAMES",
    "FEATURE_DICTIONARY",
    "PreprocessingAuditSummary",
    "condition_trial_records",
    "detect_distinct_current_levels",
    "export_feature_dictionary_docs",
    "extract_feature_dataset",
    "extract_features_from_conditioned_signal",
    "fit_multi_load_resistance",
]
