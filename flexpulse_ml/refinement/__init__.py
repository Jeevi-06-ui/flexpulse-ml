"""
Phase 6 scientific validation, normalization audit, and refinement package.
"""

from flexpulse_ml.refinement.audit import (
    FeatureNormalizationAudit,
    WindowLevelA2vsA3Diagnostic,
    FalseAlarmInvestigator,
)
from flexpulse_ml.refinement.confidence_eval import (
    ConfidenceCalibrationEvaluator,
    ConfidenceCalibrationReport,
)
from flexpulse_ml.refinement.persistence import (
    TemporalPersistenceFilter,
    PersistenceEvaluationResult,
)
from flexpulse_ml.refinement.regimes import (
    ControlledRegimeGenerator,
    RegimeEvaluationMatrix,
)

__all__ = [
    "FeatureNormalizationAudit",
    "WindowLevelA2vsA3Diagnostic",
    "FalseAlarmInvestigator",
    "ConfidenceCalibrationEvaluator",
    "ConfidenceCalibrationReport",
    "TemporalPersistenceFilter",
    "PersistenceEvaluationResult",
    "ControlledRegimeGenerator",
    "RegimeEvaluationMatrix",
]
