"""
Publication-quality research figures for benchmark models, confusion matrices, and robustness.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from flexpulse_ml.config import PROJECT_ROOT

FIGURES_DIR = PROJECT_ROOT / "results" / "figures" / "phase4_benchmark"


def plot_model_comparison_bars(
    comparison_df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = output_path or (FIGURES_DIR / "fig1_model_comparison_bars.png")

    metrics = ["balanced_accuracy", "precision", "recall", "f1"]
    metric_labels = ["Balanced Acc", "Precision", "Recall", "F1-Score"]
    models = comparison_df["model"].unique()

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    width = 0.18

    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]

    for i, (m, label) in enumerate(zip(metrics, metric_labels)):
        vals = [
            float(comparison_df[(comparison_df["model"] == mod) & (comparison_df["split"] == "id_test")][m].values[0])
            for mod in models
        ]
        ax.bar(x + i * width, vals, width, label=label, color=colors[i], alpha=0.9)

    ax.set_title("Model Performance Benchmark on ID Test Set (SIMULATED DATA ONLY)", fontsize=12, fontweight="bold")
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([m.replace("_", " ").title() for m in models], rotation=15, ha="right")
    ax.set_ylabel("Score [0.0 - 1.0]")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="lower right")
    plt.tight_layout()

    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_confusion_matrices(
    confusion_dict: Dict[str, Dict[str, int]],
    output_path: Optional[Path] = None,
) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = output_path or (FIGURES_DIR / "fig2_confusion_matrices.png")

    models = list(confusion_dict.keys())
    n_models = len(models)
    cols = 3
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for idx, m_name in enumerate(models):
        cm = confusion_dict[m_name]
        matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])

        ax = axes[idx]
        cax = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
        ax.set_title(f"{m_name.replace('_', ' ').title()}", fontsize=11, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Healthy", "Pred Fault"])
        ax.set_yticklabels(["True Healthy", "True Fault"])

        for i in range(2):
            for j in range(2):
                val = matrix[i, j]
                text_color = "white" if val > np.max(matrix) / 2 else "black"
                ax.text(j, i, f"{val:,}", ha="center", va="center", color=text_color, fontweight="bold")

    for idx in range(n_models, len(axes)):
        axes[idx].axis("off")

    fig.suptitle("ID Test Confusion Matrices (SIMULATED DATA ONLY)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()

    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_precision_recall_curves(
    pr_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    output_path: Optional[Path] = None,
) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = output_path or (FIGURES_DIR / "fig3_precision_recall_curves.png")

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for idx, (m_name, (rec, prec, ap)) in enumerate(pr_data.items()):
        color = colors[idx % len(colors)]
        label = f"{m_name.replace('_', ' ').title()} (AP = {ap:.3f})"
        ax.plot(rec, prec, color=color, lw=2, label=label)

    ax.set_title("Precision-Recall Curves on ID Test Set (SIMULATED DATA ONLY)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    plt.tight_layout()

    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_roc_curves(
    roc_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    output_path: Optional[Path] = None,
) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = output_path or (FIGURES_DIR / "fig4_roc_curves.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1.5, label="Random Guess (AUC = 0.50)")

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for idx, (m_name, (fpr, tpr, roc_auc)) in enumerate(roc_data.items()):
        color = colors[idx % len(colors)]
        label = f"{m_name.replace('_', ' ').title()} (AUC = {roc_auc:.3f})"
        ax.plot(fpr, tpr, color=color, lw=2, label=label)

    ax.set_title("Receiver Operating Characteristic (ROC) Curves (SIMULATED DATA ONLY)", fontsize=12, fontweight="bold")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    plt.tight_layout()

    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_id_vs_ood_comparison(
    comparison_df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = output_path or (FIGURES_DIR / "fig5_id_vs_ood_comparison.png")

    models = comparison_df["model"].unique()
    id_f1 = [
        float(comparison_df[(comparison_df["model"] == m) & (comparison_df["split"] == "id_test")]["f1"].values[0])
        for m in models
    ]
    ood_f1 = [
        float(comparison_df[(comparison_df["model"] == m) & (comparison_df["split"] == "ood_test")]["f1"].values[0])
        for m in models
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(models))
    width = 0.35

    ax.bar(x - width / 2, id_f1, width, label="ID Test F1 (Nominal)", color="#2b5c8f", alpha=0.9)
    ax.bar(x + width / 2, ood_f1, width, label="OOD Test F1 (Shifted)", color="#d95f02", alpha=0.9)

    ax.set_title("Generalization Robustness: Nominal vs. OOD Test F1-Score (SIMULATED DATA ONLY)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in models], rotation=15, ha="right")
    ax.set_ylabel("F1-Score")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="lower right")
    plt.tight_layout()

    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_scenario_error_analysis(
    scenario_metrics: Dict[str, Dict[str, Any]],
    output_path: Optional[Path] = None,
) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = output_path or (FIGURES_DIR / "fig6_scenario_error_analysis.png")

    scenarios = list(scenario_metrics.keys())
    recalls = [float(scenario_metrics[s].get("recall", 0.0)) for s in scenarios]
    fprs = [float(scenario_metrics[s].get("false_positive_rate", 0.0)) for s in scenarios]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(scenarios))
    width = 0.35

    ax.bar(x - width / 2, recalls, width, label="Fault Recall", color="#2ca02c", alpha=0.85)
    ax.bar(x + width / 2, fprs, width, label="False Alarm Rate (FPR)", color="#d62728", alpha=0.85)

    ax.set_title("Scenario-Specific Fault Recall vs. False Positive Rate (SIMULATED DATA ONLY)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", " ").title() for s in scenarios], rotation=20, ha="right")
    ax.set_ylabel("Rate [0.0 - 1.0]")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    plt.tight_layout()

    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_window_label_sensitivity(
    sensitivity_dict: Dict[str, Dict[str, Any]],
    output_path: Optional[Path] = None,
) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = output_path or (FIGURES_DIR / "fig7_window_label_sensitivity.png")

    labels = list(sensitivity_dict.keys())
    f1_vals = [float(sensitivity_dict[k]["f1"]) for k in labels]
    rec_vals = [float(sensitivity_dict[k]["recall"]) for k in labels]
    prec_vals = [float(sensitivity_dict[k]["precision"]) for k in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(labels, f1_vals, marker="o", lw=2, label="F1-Score", color="#1f77b4")
    ax.plot(labels, rec_vals, marker="s", lw=2, label="Recall", color="#2ca02c")
    ax.plot(labels, prec_vals, marker="^", lw=2, label="Precision", color="#ff7f0e")

    ax.set_title("Window-Label Target Sensitivity Analysis (SIMULATED DATA ONLY)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Metric Score")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_detection_delay_distribution(
    delays_dict: Dict[str, List[float]],
    output_path: Optional[Path] = None,
) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = output_path or (FIGURES_DIR / "fig8_detection_delay_distribution.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    data = []
    labels = []
    for m_name, d_list in delays_dict.items():
        if d_list:
            data.append(d_list)
            labels.append(m_name.replace("_", " ").title())

    if data:
        try:
            ax.boxplot(data, tick_labels=labels, patch_artist=True, boxprops=dict(facecolor="#adc2eb", alpha=0.7))
        except TypeError:
            ax.boxplot(data, labels=labels, patch_artist=True, boxprops=dict(facecolor="#adc2eb", alpha=0.7))
        ax.set_title("Observable Fault Event Detection Delay Distribution (SIMULATED DATA ONLY)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Detection Delay from First Fault Sample [s]")
        ax.grid(True, alpha=0.3, axis="y")
        plt.xticks(rotation=15, ha="right")
    else:
        ax.text(0.5, 0.5, "No observable fault detection delays recorded", ha="center", va="center")

    plt.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out
