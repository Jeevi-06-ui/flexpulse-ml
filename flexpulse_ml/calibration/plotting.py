"""
Research figures for self-calibration benchmarking, baseline tracking, and ablation analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flexpulse_ml.config import PROJECT_ROOT

FIGURES_DIR = PROJECT_ROOT / "results" / "figures" / "phase5_calibration"


def plot_raw_telemetry_timeseries(
    nominal_df: pd.DataFrame,
    shifted_df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = output_path or (FIGURES_DIR / "fig1_raw_telemetry_timeseries.png")

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    t_nom = nominal_df["timestamp_s"].to_numpy()
    v_nom = nominal_df["voltage_measured_v"].to_numpy()
    i_nom = nominal_df["current_measured_a"].to_numpy()

    t_shift = shifted_df["timestamp_s"].to_numpy()
    v_shift = shifted_df["voltage_measured_v"].to_numpy()
    i_shift = shifted_df["current_measured_a"].to_numpy()

    axes[0].plot(t_nom, v_nom, label="Nominal Condition", color="#1f77b4", lw=1.2, alpha=0.8)
    axes[0].plot(t_shift, v_shift, label="Shifted (High Load / Offset)", color="#d62728", lw=1.2, alpha=0.8)
    axes[0].set_ylabel("Measured Voltage [V]")
    axes[0].set_title("Observable Telemetry Signals (SIMULATION-BASED EXPERIMENT)", fontsize=11, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(t_nom, i_nom, label="Nominal Condition", color="#1f77b4", lw=1.2, alpha=0.8)
    axes[1].plot(t_shift, i_shift, label="Shifted Condition", color="#d62728", lw=1.2, alpha=0.8)
    axes[1].set_ylabel("Measured Current [A]")
    axes[1].set_xlabel("Time [s]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_baseline_trajectories(
    run_records: List[Dict[str, Any]],
    output_path: Optional[Path] = None,
) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = output_path or (FIGURES_DIR / "fig2_baseline_trajectories.png")

    t_arr = np.array([r["timestamp_end_s"] for r in run_records])
    raw_v = np.array([r["raw_v"] for r in run_records])
    base_v = np.array([r["active_baseline_v"] for r in run_records])
    gt_fault = np.array([r["ground_truth_fault"] for r in run_records])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_arr, raw_v, label="Observed Mean Voltage", color="#7f7f7f", lw=1.0, alpha=0.7)
    ax.plot(t_arr, base_v, label="Adaptive Baseline Center (EMA)", color="#1f77b4", lw=2.2)

    fault_mask = gt_fault == 1
    if np.any(fault_mask):
        ax.fill_between(t_arr, 0, 6, where=fault_mask, color="#d62728", alpha=0.15, label="Ground-Truth Fault Interval")

    ax.set_title("Adaptive Baseline Trajectory Tracking (SIMULATION-BASED EXPERIMENT)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Voltage Level [V]")
    ax.set_ylim(min(raw_v) - 0.1, max(raw_v) + 0.1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_update_acceptance_rejection(
    run_records: List[Dict[str, Any]],
    output_path: Optional[Path] = None,
) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = output_path or (FIGURES_DIR / "fig3_update_acceptance_rejection.png")

    t_arr = np.array([r["timestamp_end_s"] for r in run_records])
    raw_v = np.array([r["raw_v"] for r in run_records])
    accepted = np.array([r["update_accepted"] for r in run_records])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_arr, raw_v, color="#cccccc", lw=1.2, label="Observed Mean Voltage")

    ax.scatter(t_arr[accepted], raw_v[accepted], color="#2ca02c", s=18, label="Update Accepted (Gated)", zorder=4)
    ax.scatter(t_arr[~accepted], raw_v[~accepted], color="#d62728", marker="x", s=22, label="Update Rejected (Frozen/Anomalous)", zorder=4)

    ax.set_title("Online Baseline Update Acceptance vs. Rejection (SIMULATION-BASED EXPERIMENT)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Voltage [V]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_ground_truth_overlay(
    run_records: List[Dict[str, Any]],
    output_path: Optional[Path] = None,
) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = output_path or (FIGURES_DIR / "fig4_ground_truth_overlay.png")

    t_arr = np.array([r["timestamp_end_s"] for r in run_records])
    conf = np.array([r["confidence"] for r in run_records])
    pred = np.array([r["prediction"] for r in run_records])
    gt_fault = np.array([r["ground_truth_fault"] for r in run_records])

    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)

    axes[0].plot(t_arr, conf, color="#1f77b4", lw=1.8, label="Healthy Confidence P(healthy)")
    axes[0].axhline(0.95, color="#e15759", linestyle="--", label="Confidence Gate (0.95)")
    axes[0].set_ylabel("Confidence Score")
    axes[0].set_title("Confidence Gating and Fault Detection Overlay (SIMULATION-BASED EXPERIMENT)", fontsize=11, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="lower left")

    axes[1].step(t_arr, pred, color="#ff7f0e", where="post", lw=1.8, label="Adaptive Detector Prediction")
    axes[1].step(t_arr, gt_fault, color="#2ca02c", where="post", lw=1.5, linestyle=":", label="Ground-Truth Label")
    axes[1].set_ylabel("Class Label [0 or 1]")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="center left")

    plt.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_model_predictions_timeline(
    timeline_dict: Dict[str, np.ndarray],
    time_array: np.ndarray,
    ground_truth: np.ndarray,
    output_path: Optional[Path] = None,
) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = output_path or (FIGURES_DIR / "fig5_model_predictions_timeline.png")

    methods = list(timeline_dict.keys())
    fig, axes = plt.subplots(len(methods) + 1, 1, figsize=(10, 2.0 * (len(methods) + 1)), sharex=True)

    axes[0].step(time_array, ground_truth, color="#000000", where="post", lw=1.8)
    axes[0].set_ylabel("Ground Truth")
    axes[0].set_title("Detection Timelines Across Compared Approaches (SIMULATION-BASED EXPERIMENT)", fontsize=11, fontweight="bold")
    axes[0].set_ylim(-0.1, 1.1)
    axes[0].grid(True, alpha=0.3)

    colors = ["#4e79a7", "#e15759", "#76b7b2", "#59a14f"]
    for idx, m_name in enumerate(methods):
        ax = axes[idx + 1]
        preds = timeline_dict[m_name]
        c = colors[idx % len(colors)]
        ax.step(time_array, preds, color=c, where="post", lw=1.6)
        ax.set_ylabel(m_name, fontsize=9)
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time [s]")
    plt.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_method_comparison_id_ood(
    comparison_df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = output_path or (FIGURES_DIR / "fig6_method_comparison_id_ood.png")

    methods = comparison_df["method"].unique()
    x = np.arange(len(methods))
    width = 0.35

    id_f1 = [
        float(comparison_df[(comparison_df["method"] == m) & (comparison_df["split"] == "id_test")]["f1"].values[0])
        for m in methods
    ]
    ood_f1 = [
        float(comparison_df[(comparison_df["method"] == m) & (comparison_df["split"] == "ood_test")]["f1"].values[0])
        for m in methods
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, id_f1, width, label="In-Distribution (ID) Test F1", color="#2b5c8f", alpha=0.9)
    ax.bar(x + width / 2, ood_f1, width, label="Robustness (OOD) Test F1", color="#d95f02", alpha=0.9)

    ax.set_title("ID Test vs. OOD Robustness Test F1-Score (SIMULATION-BASED EXPERIMENT)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("F1-Score")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_false_alarm_vs_recall_tradeoff(
    metrics_by_method: Dict[str, Dict[str, float]],
    output_path: Optional[Path] = None,
) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = output_path or (FIGURES_DIR / "fig7_false_alarm_vs_recall_tradeoff.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4e79a7", "#e15759", "#76b7b2", "#59a14f", "#b07aa1"]

    for idx, (m_name, m_data) in enumerate(metrics_by_method.items()):
        rec = m_data.get("event_recall", 0.0)
        fa_rate = m_data.get("false_alarms_per_minute", 0.0)
        c = colors[idx % len(colors)]
        ax.scatter(fa_rate, rec, color=c, s=120, label=m_name, zorder=4)
        ax.annotate(
            m_name,
            (fa_rate, rec),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_title("Event Recall vs. False Alarm Rate (SIMULATION-BASED EXPERIMENT)", fontsize=11, fontweight="bold")
    ax.set_xlabel("False Alarms per Simulated Minute")
    ax.set_ylabel("Observable Fault Event Recall")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_causal_detection_delays(
    delays_by_method: Dict[str, List[float]],
    output_path: Optional[Path] = None,
) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = output_path or (FIGURES_DIR / "fig8_causal_detection_delays.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    data = []
    labels = []

    for m_name, d_list in delays_by_method.items():
        if d_list:
            data.append(d_list)
            labels.append(m_name)

    if data:
        try:
            ax.boxplot(data, tick_labels=labels, patch_artist=True, boxprops=dict(facecolor="#adc2eb", alpha=0.7))
        except TypeError:
            ax.boxplot(data, labels=labels, patch_artist=True, boxprops=dict(facecolor="#adc2eb", alpha=0.7))
        ax.set_title("Causal Fault Event Detection Delays (SIMULATION-BASED EXPERIMENT)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Detection Delay [s]")
        ax.grid(True, alpha=0.3, axis="y")
    else:
        ax.text(0.5, 0.5, "No detection delays recorded", ha="center", va="center")

    plt.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_contamination_failure_modes(
    unprotected_records: List[Dict[str, Any]],
    protected_records: List[Dict[str, Any]],
    output_path: Optional[Path] = None,
) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = output_path or (FIGURES_DIR / "fig9_contamination_failure_modes.png")

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    t_arr = np.array([r["timestamp_end_s"] for r in unprotected_records])
    raw_v = np.array([r["raw_v"] for r in unprotected_records])
    unprot_base = np.array([r["active_baseline_v"] for r in unprotected_records])
    prot_base = np.array([r["active_baseline_v"] for r in protected_records])
    gt_fault = np.array([r["ground_truth_fault"] for r in unprotected_records])

    axes[0].plot(t_arr, raw_v, color="#cccccc", lw=1.0, label="Observed Voltage")
    axes[0].plot(t_arr, unprot_base, color="#d62728", lw=2.0, label="Unprotected Baseline (Contaminated / Absorbed Fault)")
    fault_mask = gt_fault == 1
    if np.any(fault_mask):
        axes[0].fill_between(t_arr, min(raw_v), max(raw_v), where=fault_mask, color="#d62728", alpha=0.15, label="True Fault Interval")
    axes[0].set_ylabel("Voltage [V]")
    axes[0].set_title("Naive Adaptation: Fault Absorbed into Baseline (SIMULATION-BASED EXPERIMENT)", fontsize=10, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="lower left")

    axes[1].plot(t_arr, raw_v, color="#cccccc", lw=1.0, label="Observed Voltage")
    axes[1].plot(t_arr, prot_base, color="#2ca02c", lw=2.0, label="Confidence-Gated & Frozen Baseline (Protected)")
    if np.any(fault_mask):
        axes[1].fill_between(t_arr, min(raw_v), max(raw_v), where=fault_mask, color="#2ca02c", alpha=0.15, label="True Fault Interval")
    axes[1].set_ylabel("Voltage [V]")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_title("Gated Adaptation: Baseline Frozen on Suspicion (SIMULATION-BASED EXPERIMENT)", fontsize=10, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="lower left")

    plt.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_ablation_study(
    ablation_df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = output_path or (FIGURES_DIR / "fig10_ablation_study.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    names = ablation_df["configuration"].to_numpy()
    f1_scores = ablation_df["f1"].to_numpy()
    bal_accs = ablation_df["balanced_accuracy"].to_numpy()

    x = np.arange(len(names))
    width = 0.35

    ax.bar(x - width / 2, f1_scores, width, label="F1-Score", color="#4e79a7", alpha=0.9)
    ax.bar(x + width / 2, bal_accs, width, label="Balanced Accuracy", color="#f28e2b", alpha=0.9)

    ax.set_title("Ablation Study: Contribution of Individual Calibration Safeguards (SIMULATION-BASED EXPERIMENT)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Score [0.0 - 1.0]")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="lower left")

    plt.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out
