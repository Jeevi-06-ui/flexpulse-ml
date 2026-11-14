"""
Adaptive baseline self-calibration package for FlexPulse-ML Phase 5.
"""

from flexpulse_ml.calibration.baseline import (
    BaselineStatistics,
    InitialBaselineEstimator,
)
from flexpulse_ml.calibration.confidence import ConfidenceEstimator
from flexpulse_ml.calibration.diagnostics import (
    CalibrationDiagnostics,
    CalibrationState,
    WindowCalibrationRecord,
)
from flexpulse_ml.calibration.adaptive import (
    AdaptiveCalibrationConfig,
    ConfidenceGatedAdaptiveCalibrator,
)
from flexpulse_ml.calibration.experiment import (
    Phase5BenchmarkRunner,
    Phase5BenchmarkSummary,
    run_phase5_benchmark,
)

__all__ = [
    "BaselineStatistics",
    "InitialBaselineEstimator",
    "ConfidenceEstimator",
    "CalibrationDiagnostics",
    "CalibrationState",
    "WindowCalibrationRecord",
    "AdaptiveCalibrationConfig",
    "ConfidenceGatedAdaptiveCalibrator",
    "Phase5BenchmarkRunner",
    "Phase5BenchmarkSummary",
    "run_phase5_benchmark",
]
