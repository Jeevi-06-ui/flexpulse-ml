"""
Machine Learning Classifiers, Benchmarks, and Cross-Validation Package for FlexPulse-ML.
"""

from flexpulse_ml.ml.baselines import FixedThresholdBaseline
from flexpulse_ml.ml.models import (
    MODEL_DECISION_TREE,
    MODEL_FIXED_THRESHOLD,
    MODEL_GRADIENT_BOOSTING,
    MODEL_LOGISTIC_REGRESSION,
    MODEL_RANDOM_FOREST,
    MODEL_SVM,
    SUPPORTED_MODELS,
    build_model_pipeline,
    get_default_hyperparameters,
    get_hyperparameter_search_grid,
)
from flexpulse_ml.ml.evaluate import (
    EventLevelMetrics,
    WindowClassificationMetrics,
    bootstrap_run_level_metrics,
    calculate_window_metrics,
    evaluate_event_level_detections,
    evaluate_scenario_breakdown,
    evaluate_window_label_sensitivity,
)
from flexpulse_ml.ml.train import ModelTrainer, TrainedModelRecord
from flexpulse_ml.ml.persistence import ModelArtifactStorage
from flexpulse_ml.ml.experiment import (
    BenchmarkExecutionSummary,
    Phase4BenchmarkRunner,
    run_phase4_benchmark,
)

__all__ = [
    "FixedThresholdBaseline",
    "MODEL_FIXED_THRESHOLD",
    "MODEL_LOGISTIC_REGRESSION",
    "MODEL_DECISION_TREE",
    "MODEL_RANDOM_FOREST",
    "MODEL_SVM",
    "MODEL_GRADIENT_BOOSTING",
    "SUPPORTED_MODELS",
    "build_model_pipeline",
    "get_default_hyperparameters",
    "get_hyperparameter_search_grid",
    "EventLevelMetrics",
    "WindowClassificationMetrics",
    "bootstrap_run_level_metrics",
    "calculate_window_metrics",
    "evaluate_event_level_detections",
    "evaluate_scenario_breakdown",
    "evaluate_window_label_sensitivity",
    "ModelTrainer",
    "TrainedModelRecord",
    "ModelArtifactStorage",
    "BenchmarkExecutionSummary",
    "Phase4BenchmarkRunner",
    "run_phase4_benchmark",
]
