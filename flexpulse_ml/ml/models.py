"""
Machine learning classifier factories, pipelines, and hyperparameter grids.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from flexpulse_ml.ml.baselines import FixedThresholdBaseline

MODEL_FIXED_THRESHOLD = "fixed_threshold"
MODEL_LOGISTIC_REGRESSION = "logistic_regression"
MODEL_DECISION_TREE = "decision_tree"
MODEL_RANDOM_FOREST = "random_forest"
MODEL_SVM = "support_vector_machine"
MODEL_GRADIENT_BOOSTING = "gradient_boosting"

SUPPORTED_MODELS: List[str] = [
    MODEL_FIXED_THRESHOLD,
    MODEL_LOGISTIC_REGRESSION,
    MODEL_DECISION_TREE,
    MODEL_RANDOM_FOREST,
    MODEL_SVM,
    MODEL_GRADIENT_BOOSTING,
]


def get_default_hyperparameters(model_name: str) -> Dict[str, Any]:
    name = model_name.lower()
    if name == MODEL_FIXED_THRESHOLD:
        return {
            "voltage_drop_std_multiplier": 3.0,
            "transient_jump_std_multiplier": 3.0,
        }
    elif name == MODEL_LOGISTIC_REGRESSION:
        return {
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 1000,
            "class_weight": "balanced",
        }
    elif name == MODEL_DECISION_TREE:
        return {
            "max_depth": 6,
            "min_samples_leaf": 10,
            "criterion": "gini",
            "class_weight": "balanced",
        }
    elif name == MODEL_RANDOM_FOREST:
        return {
            "n_estimators": 100,
            "max_depth": 8,
            "min_samples_leaf": 5,
            "max_features": "sqrt",
            "class_weight": "balanced",
            "n_jobs": -1,
        }
    elif name == MODEL_SVM:
        return {
            "C": 1.0,
            "gamma": "scale",
            "kernel": "rbf",
            "max_iter": 5000,
            "class_weight": "balanced",
            "probability": True,
        }
    elif name == MODEL_GRADIENT_BOOSTING:
        return {
            "max_iter": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "min_samples_leaf": 10,
            "class_weight": "balanced",
        }
    else:
        raise ValueError(f"Unknown model name '{model_name}'. Supported: {SUPPORTED_MODELS}")


def get_hyperparameter_search_grid(model_name: str) -> List[Dict[str, Any]]:
    name = model_name.lower()
    if name == MODEL_FIXED_THRESHOLD:
        return [
            {"voltage_drop_std_multiplier": v, "transient_jump_std_multiplier": j}
            for v in [2.0, 2.5, 3.0, 3.5]
            for j in [2.0, 2.5, 3.0, 3.5]
        ]
    elif name == MODEL_LOGISTIC_REGRESSION:
        return [
            {"C": c, "solver": "lbfgs", "max_iter": 1000, "class_weight": "balanced"}
            for c in [0.01, 0.1, 1.0, 10.0]
        ]
    elif name == MODEL_DECISION_TREE:
        return [
            {"max_depth": d, "min_samples_leaf": l, "criterion": "gini", "class_weight": "balanced"}
            for d in [4, 6, 8, 10]
            for l in [5, 10, 20]
        ]
    elif name == MODEL_RANDOM_FOREST:
        return [
            {"n_estimators": n, "max_depth": d, "min_samples_leaf": l, "class_weight": "balanced", "n_jobs": -1}
            for n in [50, 100]
            for d in [6, 8, 12]
            for l in [5, 10]
        ]
    elif name == MODEL_SVM:
        return [
            {"C": c, "gamma": g, "kernel": "rbf", "max_iter": 5000, "class_weight": "balanced", "probability": True}
            for c in [0.1, 1.0, 5.0]
            for g in ["scale", 0.05, 0.1]
        ]
    elif name == MODEL_GRADIENT_BOOSTING:
        return [
            {"max_iter": n, "max_depth": d, "learning_rate": lr, "class_weight": "balanced"}
            for n in [50, 100]
            for d in [4, 6]
            for lr in [0.05, 0.1]
        ]
    else:
        raise ValueError(f"Unknown model name '{model_name}'. Supported: {SUPPORTED_MODELS}")


def build_model_pipeline(
    model_name: str,
    hyperparameters: Optional[Dict[str, Any]] = None,
    random_seed: int = 42,
) -> Any:
    name = model_name.lower()
    params = dict(get_default_hyperparameters(name))
    if hyperparameters:
        params.update(hyperparameters)

    if name == MODEL_FIXED_THRESHOLD:
        return FixedThresholdBaseline(
            voltage_drop_std_multiplier=float(params.get("voltage_drop_std_multiplier", 3.0)),
            transient_jump_std_multiplier=float(params.get("transient_jump_std_multiplier", 3.0)),
        )

    elif name == MODEL_LOGISTIC_REGRESSION:
        params["random_state"] = random_seed
        clf = LogisticRegression(**params)
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    elif name == MODEL_DECISION_TREE:
        params["random_state"] = random_seed
        return DecisionTreeClassifier(**params)

    elif name == MODEL_RANDOM_FOREST:
        params["random_state"] = random_seed
        return RandomForestClassifier(**params)

    elif name == MODEL_SVM:
        params["random_state"] = random_seed
        clf = SVC(**params)
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    elif name == MODEL_GRADIENT_BOOSTING:
        params["random_state"] = random_seed
        return HistGradientBoostingClassifier(**params)

    else:
        raise ValueError(f"Unsupported model name '{model_name}'. Supported: {SUPPORTED_MODELS}")
