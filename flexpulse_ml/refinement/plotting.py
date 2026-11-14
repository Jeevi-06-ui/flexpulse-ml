"""
Publication-quality research figures for Phase 6 scientific validation and refinement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WATERMARK_TEXT = "SIMULATION-BASED EXPERIMENT (SOFTWARE-ONLY MODEL)"


def _apply_watermark(fig: plt.Figure) -> None:
    fig.text(
        0.5,
        0.015,
        WATERMARK_TEXT,
        fontsize=8,
        color="#7f7f7f",
        ha="center",
        va="bottom",
        style="italic",
        weight="bold",
    )


def plot_feature_normalization_audit(
    records: List[Dict[str, Any]],
    output_path: Path,
) -> Path:
    df = pd.DataFrame(records)
    categories = df["category"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    bars = ax.bar(categories.index, categories.values, color=["#4e79a7", "#e15759", "#76b7b2", "#59a14f"], width=0.55)
    ax.set_title("Feature Normalization Audit: Breakdown of 26 Electrical Features", fontsize=11, weight="bold")
    ax.set_ylabel("Number of Features", fontsize=10)
    ax.set_xlabel("Feature Functional Category", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, weight="bold")

    _apply_watermark(fig)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_a2_vs_a3_baseline_trajectories(
    time_array: np.ndarray,
    observed_v: np.ndarray,
    a2_base_v: float,
    a3_base_v: np.ndarray,
    output_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    ax.plot(time_array, observed_v, color="#cccccc", lw=1.2, label="Observed Mean Voltage")
    ax.axhline(a2_base_v, color="#e15759", linestyle="--", lw=2.0, label=f"A2 Fixed Initial Baseline ({a2_base_v:.4f} V)")
    ax.plot(time_array, a3_base_v, color="#1f77b4", lw=2.2, label="A3 Gated Adaptive Baseline (Online EMA)")

    ax.set_title("A2 (Initial Calibration) vs A3 (Adaptive Baseline) Trajectories Across Time", fontsize=11, weight="bold")
    ax.set_xlabel("Time (seconds)", fontsize=10)
    ax.set_ylabel("Voltage (V)", fontsize=10)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4)

    _apply_watermark(fig)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_a2_vs_a3_prediction_differences(
    time_array: np.ndarray,
    y_true: np.ndarray,
    a2_preds: np.ndarray,
    a3_preds: np.ndarray,
    output_path: Path,
) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True, dpi=300)

    axes[0].step(time_array, y_true, color="#2ca02c", where="post", lw=1.8, label="Ground Truth Fault")
    axes[0].set_ylabel("True Fault", fontsize=9, weight="bold")
    axes[0].set_ylim(-0.1, 1.2)
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].step(time_array, a2_preds, color="#e15759", where="post", lw=1.5, label="A2 Initial Calibration")
    axes[1].set_ylabel("A2 Predictions", fontsize=9, weight="bold")
    axes[1].set_ylim(-0.1, 1.2)
    axes[1].grid(True, linestyle="--", alpha=0.4)

    axes[2].step(time_array, a3_preds, color="#1f77b4", where="post", lw=1.5, label="A3 Gated Adaptive")
    axes[2].set_ylabel("A3 Predictions", fontsize=9, weight="bold")
    axes[2].set_xlabel("Time (seconds)", fontsize=10)
    axes[2].set_ylim(-0.1, 1.2)
    axes[2].grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Window-by-Window Comparison: Ground Truth vs A2 and A3 Predictions", fontsize=11, weight="bold")
    _apply_watermark(fig)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_update_acceptance_rejection_audit(
    rejection_dict: Dict[str, int],
    accepted_count: int,
    output_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    labels = ["ACCEPTED"] + list(rejection_dict.keys())
    counts = [accepted_count] + list(rejection_dict.values())
    colors = ["#2ca02c"] + ["#d62728"] * len(rejection_dict)

    bars = ax.barh(labels, counts, color=colors, height=0.55)
    ax.set_title("Audit of Baseline Update Status Across OOD Evaluation Windows", fontsize=11, weight="bold")
    ax.set_xlabel("Number of Windows", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4, axis="x")

    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{int(w)}", xy=(w, bar.get_y() + bar.get_height() / 2), xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=9, weight="bold")

    _apply_watermark(fig)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_scenario_performance_audit(
    scenario_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    scenarios = scenario_df["scenario"].unique()
    x = np.arange(len(scenarios))
    width = 0.20

    methods = ["A0_Fixed_Threshold", "A1_Static_ML", "A2_Initial_Calibration_Only", "A3_Confidence_Gated_Adaptive"]
    colors = ["#7f7f7f", "#e15759", "#4e79a7", "#59a14f"]

    for i, m in enumerate(methods):
        sub = scenario_df[scenario_df["method"] == m].set_index("scenario").reindex(scenarios)
        f1_vals = sub["f1"].fillna(0.0).values
        ax.bar(x + (i - 1.5) * width, f1_vals, width, label=m.replace("_", " "), color=colors[i], alpha=0.9)

    ax.set_title("Scenario-Specific F1 Performance Across Evaluation Regimes", fontsize=11, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in scenarios], fontsize=8)
    ax.set_ylabel("F1-Score", fontsize=10)
    ax.set_ylim(0.0, 1.1)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    _apply_watermark(fig)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_false_alarm_analysis(
    scenario_fp_dict: Dict[str, int],
    output_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    labels = list(scenario_fp_dict.keys())
    counts = list(scenario_fp_dict.values())

    bars = ax.bar([l.replace("_", "\n") for l in labels], counts, color="#f28e2b", width=0.55)
    ax.set_title("Attribution of False Positive Windows Across Scenarios (OOD)", fontsize=11, weight="bold")
    ax.set_ylabel("False Positive Windows", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{int(h)}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, weight="bold")

    _apply_watermark(fig)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_detection_delay_tradeoffs(
    results: List[Dict[str, Any]],
    output_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    names = [r["filter_name"].replace("_", " ") for r in results]
    delays = [r["median_causal_delay_s"] for r in results]
    far = [r["false_alarms_per_minute"] for r in results]

    ax.scatter(delays, far, color="#1f77b4", s=90, zorder=4)
    for i, txt in enumerate(names):
        ax.annotate(txt, xy=(delays[i], far[i]), xytext=(8, 4), textcoords="offset points", fontsize=9, weight="bold")

    ax.set_title("Causal Detection Delay vs False Alarm Rate Trade-Off (Persistence Filtering)", fontsize=11, weight="bold")
    ax.set_xlabel("Median Causal Detection Delay (seconds)", fontsize=10)
    ax.set_ylabel("False Alarms per Simulated Minute", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)

    _apply_watermark(fig)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_confidence_reliability_diagram(
    bins: List[Dict[str, Any]],
    ece: float,
    output_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#7f7f7f", label="Perfect Calibration")

    centers = [b["bin_center"] for b in bins if b["sample_count"] > 0]
    accs = [b["observed_accuracy"] for b in bins if b["sample_count"] > 0]
    counts = [b["sample_count"] for b in bins if b["sample_count"] > 0]

    ax.plot(centers, accs, marker="o", color="#1f77b4", lw=1.8, label=f"Observed Accuracy (ECE={ece:.4f})")
    ax.bar(centers, accs, width=0.08, alpha=0.2, color="#1f77b4")

    ax.set_title("Reliability Diagram: Empirical Probability Calibration of P(healthy)", fontsize=11, weight="bold")
    ax.set_xlabel("Confidence Bin Center P(healthy)", fontsize=10)
    ax.set_ylabel("Empirical Fraction of Healthy Windows", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4)

    _apply_watermark(fig)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_adaptive_benefit_or_failure_regimes(
    regime_rows: List[Dict[str, Any]],
    output_path: Path,
) -> Path:
    df = pd.DataFrame(regime_rows)
    a2_df = df[df["method"] == "A2_Initial_Calibration"].set_index("regime_id")
    a3_df = df[df["method"] == "A3_Gated_Adaptive"].set_index("regime_id")
    regimes = a2_df.index.tolist()

    x = np.arange(len(regimes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    ax.bar(x - width / 2, a2_df["f1"].values, width, label="A2 Initial Calibration Only", color="#4e79a7", alpha=0.9)
    ax.bar(x + width / 2, a3_df["f1"].values, width, label="A3 Gated Adaptive Baseline", color="#59a14f", alpha=0.9)

    ax.set_title("Regime Analysis E1 to E10: Controlled Comparison of Initial Calibration vs Adaptive Updates", fontsize=11, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([r.replace("_", "\n") for r in regimes], fontsize=8)
    ax.set_ylabel("F1-Score", fontsize=10)
    ax.set_ylim(0.0, 1.15)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    for i in range(len(regimes)):
        delta = a3_df.loc[regimes[i], "f1"] - a2_df.loc[regimes[i], "f1"]
        sign = "+" if delta >= 0 else ""
        col = "#2ca02c" if delta > 0.005 else ("#d62728" if delta < -0.005 else "#7f7f7f")
        ax.annotate(f"{sign}{delta:.3f}", xy=(x[i], max(a2_df.loc[regimes[i], "f1"], a3_df.loc[regimes[i], "f1"]) + 0.03), ha="center", fontsize=8, weight="bold", color=col)

    _apply_watermark(fig)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_paired_comparison_confidence_intervals(
    ci_rows: List[Dict[str, Any]],
    output_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    metrics = [r["metric_name"] for r in ci_rows]
    deltas = [r["mean_delta"] for r in ci_rows]
    ci_low = [r["ci_lower"] for r in ci_rows]
    ci_high = [r["ci_upper"] for r in ci_rows]

    y_pos = np.arange(len(metrics))
    xerr = [np.array(deltas) - np.array(ci_low), np.array(ci_high) - np.array(deltas)]

    ax.errorbar(deltas, y_pos, xerr=xerr, fmt="o", color="#1f77b4", ecolor="#4e79a7", elinewidth=2, capsize=4, markersize=7)
    ax.axvline(0.0, color="#d62728", linestyle="--", lw=1.2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics, fontsize=9, weight="bold")
    ax.set_title("Paired Difference Bootstrap 95% Confidence Intervals (A3 minus A2)", fontsize=11, weight="bold")
    ax.set_xlabel("Metric Difference (A3 - A2)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)

    _apply_watermark(fig)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path
