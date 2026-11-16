from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from flexpulse_ml.config import PROJECT_ROOT
from flexpulse_ml.final_evidence.evaluator import Phase7Evaluator
from flexpulse_ml.final_evidence.figures import Phase7FigureGenerator


def run_confirmatory_v5_evaluation() -> Dict[str, Any]:
    out_dir = PROJECT_ROOT / "results" / "confirmatory_v5"
    tables_dir = out_dir / "tables"
    metrics_dir = out_dir / "metrics"
    figures_dir = out_dir / "figures"

    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    evaluator = Phase7Evaluator(
        dataset_id="flexpulse_final_test_v5",
        output_dir=out_dir,
        mirror_tables_dir=tables_dir,
        n_bootstraps=1000,
        random_seed=20261202,
    )
    eval_results = evaluator.run_all_evaluations()

    fig_gen = Phase7FigureGenerator(
        output_dir=figures_dir,
        mirror_dir=figures_dir,
        dataset_id="flexpulse_final_test_v5",
    )
    fig_paths = fig_gen.generate_all_figures(
        table1=eval_results["table1"],
        table3=eval_results["table3"],
        table5=eval_results["table5"],
        table6=eval_results["table6"],
        ood_preds=eval_results["ood_predictions"],
    )

    feature_storage = evaluator.feature_storage
    id_df = feature_storage.load_partition("id_test")
    ood_df = feature_storage.load_partition("ood_test")

    y_id = id_df["fault_label"].to_numpy(dtype=np.int32)
    y_ood = ood_df["fault_label"].to_numpy(dtype=np.int32)

    cm_records: Dict[str, Dict[str, Dict[str, int]]] = {"id_test": {}, "ood_test": {}}

    for m_id, preds in eval_results["id_predictions"].items():
        tn, fp, fn, tp = confusion_matrix(y_id, preds, labels=[0, 1]).ravel()
        cm_records["id_test"][m_id] = {
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
        }

    for m_id, preds in eval_results["ood_predictions"].items():
        tn, fp, fn, tp = confusion_matrix(y_ood, preds, labels=[0, 1]).ravel()
        cm_records["ood_test"][m_id] = {
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
        }

    with open(metrics_dir / "confusion_matrices.json", "w", encoding="utf-8") as f:
        json.dump(cm_records, f, indent=2)

    return {
        "evaluation_results": eval_results,
        "figure_paths": fig_paths,
        "confusion_matrices": cm_records,
    }


if __name__ == "__main__":
    run_confirmatory_v5_evaluation()
