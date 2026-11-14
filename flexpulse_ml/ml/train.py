"""
Training pipeline with strict allowlist selection, validation-based tuning, and freezing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

from flexpulse_ml.features.extractor import FEATURE_NAMES
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


@dataclass
class TrainedModelRecord:
    model_name: str
    model_instance: Any
    best_hyperparameters: Dict[str, Any]
    train_duration_s: float
    predict_duration_s: float
    validation_f1: float
    validation_balanced_accuracy: float
    feature_names: List[str] = field(default_factory=lambda: list(FEATURE_NAMES))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "best_hyperparameters": self.best_hyperparameters,
            "train_duration_s": self.train_duration_s,
            "predict_duration_s": self.predict_duration_s,
            "validation_f1": self.validation_f1,
            "validation_balanced_accuracy": self.validation_balanced_accuracy,
            "feature_names": self.feature_names,
        }


class ModelTrainer:
    """Manages feature extraction allowlist, validation tuning, and model freezing."""

    FEATURE_ALLOWLIST: List[str] = list(FEATURE_NAMES)
    TARGET_COLUMN: str = "fault_label"

    def __init__(
        self,
        random_seed: int = 42,
        models_to_train: Optional[List[str]] = None,
    ) -> None:
        self.random_seed = random_seed
        self.models_to_train = models_to_train or list(SUPPORTED_MODELS)

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        missing_features = [f for f in self.FEATURE_ALLOWLIST if f not in df.columns]
        if missing_features:
            raise ValueError(f"Dataset missing approved features from allowlist: {missing_features}")

        if self.TARGET_COLUMN not in df.columns:
            raise ValueError(f"Dataset missing target column '{self.TARGET_COLUMN}'.")

        X = df[self.FEATURE_ALLOWLIST].to_numpy(dtype=np.float64)
        y = df[self.TARGET_COLUMN].to_numpy(dtype=np.int32)
        return X, y

    def tune_and_train_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> TrainedModelRecord:
        name = model_name.lower()
        param_grid = get_hyperparameter_search_grid(name)

        best_score = -1.0
        best_bal_acc = -1.0
        best_params = get_default_hyperparameters(name)

        for candidate_params in param_grid:
            model = build_model_pipeline(
                model_name=name,
                hyperparameters=candidate_params,
                random_seed=self.random_seed,
            )
            try:
                model.fit(X_train, y_train)
                val_preds = model.predict(X_val)
                score = float(f1_score(y_val, val_preds, zero_division=0))
                bal_acc = float(balanced_accuracy_score(y_val, val_preds))
            except Exception:
                continue

            if (score > best_score) or (np.isclose(score, best_score) and bal_acc > best_bal_acc):
                best_score = score
                best_bal_acc = bal_acc
                best_params = candidate_params

        t_start = time.perf_counter()
        final_model = build_model_pipeline(
            model_name=name,
            hyperparameters=best_params,
            random_seed=self.random_seed,
        )
        final_model.fit(X_train, y_train)
        t_train = time.perf_counter() - t_start

        t_pred_start = time.perf_counter()
        val_preds = final_model.predict(X_val)
        t_pred = time.perf_counter() - t_pred_start

        final_f1 = float(f1_score(y_val, val_preds, zero_division=0))
        final_bal_acc = float(balanced_accuracy_score(y_val, val_preds))

        return TrainedModelRecord(
            model_name=name,
            model_instance=final_model,
            best_hyperparameters=best_params,
            train_duration_s=t_train,
            predict_duration_s=t_pred,
            validation_f1=final_f1,
            validation_balanced_accuracy=final_bal_acc,
            feature_names=list(self.FEATURE_ALLOWLIST),
        )

    def train_all(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> Dict[str, TrainedModelRecord]:
        X_train, y_train = self.prepare_data(train_df)
        X_val, y_val = self.prepare_data(val_df)

        records: Dict[str, TrainedModelRecord] = {}
        for m_name in self.models_to_train:
            rec = self.tune_and_train_model(m_name, X_train, y_train, X_val, y_val)
            records[m_name] = rec

        return records
