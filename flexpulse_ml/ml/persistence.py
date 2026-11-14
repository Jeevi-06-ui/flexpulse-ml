"""
Model artifact serialization, reloading, and experiment result export.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import pandas as pd

from flexpulse_ml.config import PROJECT_ROOT
from flexpulse_ml.data.manifest import compute_file_sha256

MODELS_DIR = PROJECT_ROOT / "models" / "phase4"
RESULTS_DIR = PROJECT_ROOT / "results" / "experiments" / "phase4_benchmark"
TABLES_DIR = PROJECT_ROOT / "results" / "tables" / "phase4_benchmark"


class ModelArtifactStorage:
    """Manages serializing and loading trained scikit-learn models and manifests."""

    def __init__(
        self,
        models_base_dir: Optional[Path] = None,
        results_base_dir: Optional[Path] = None,
        tables_base_dir: Optional[Path] = None,
    ) -> None:
        self.models_dir = models_base_dir or MODELS_DIR
        self.results_dir = results_base_dir or RESULTS_DIR
        self.tables_dir = tables_base_dir or TABLES_DIR

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

    def save_model(
        self,
        model_name: str,
        model_instance: Any,
        metadata: Dict[str, Any],
    ) -> Tuple[Path, Path]:
        clean_name = model_name.lower()
        model_path = self.models_dir / f"{clean_name}.joblib"
        meta_path = self.models_dir / f"{clean_name}_metadata.json"

        joblib.dump(model_instance, model_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return model_path, meta_path

    def load_model(self, model_name: str) -> Tuple[Any, Dict[str, Any]]:
        clean_name = model_name.lower()
        model_path = self.models_dir / f"{clean_name}.joblib"
        meta_path = self.models_dir / f"{clean_name}_metadata.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {model_path}")

        instance = joblib.load(model_path)
        meta = {}
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

        return instance, meta

    def save_experiment_results(
        self,
        experiment_manifest: Dict[str, Any],
        comparison_df: pd.DataFrame,
        filename_prefix: str = "phase4_benchmark",
    ) -> Dict[str, Path]:
        manifest_path = self.results_dir / f"{filename_prefix}_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(experiment_manifest, f, indent=2)

        csv_path = self.tables_dir / f"{filename_prefix}_model_comparison.csv"
        comparison_df.to_csv(csv_path, index=False)

        md_path = self.tables_dir / f"{filename_prefix}_model_comparison.md"
        try:
            comparison_df.to_markdown(md_path, index=False)
        except Exception:
            md_path.write_text(comparison_df.to_string(index=False), encoding="utf-8")

        checksums = {
            "manifest_sha256": compute_file_sha256(manifest_path),
            "table_csv_sha256": compute_file_sha256(csv_path),
            "table_md_sha256": compute_file_sha256(md_path),
        }
        with open(self.results_dir / f"{filename_prefix}_checksums.json", "w", encoding="utf-8") as f:
            json.dump(checksums, f, indent=2)

        return {
            "manifest": manifest_path,
            "comparison_csv": csv_path,
            "comparison_md": md_path,
        }
