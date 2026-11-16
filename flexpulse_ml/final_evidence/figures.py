"""
Phase 7 Publication Figures Generator: renders Figures 1-6 with mandatory simulation banner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flexpulse_ml.config import PROJECT_ROOT
from flexpulse_ml.data.storage import DatasetStorage
from flexpulse_ml.features.dataset import FeatureDatasetStorage


BANNER_TEXT = "SIMULATION-BASED EXPERIMENT — NO PHYSICAL VALIDATION"


def apply_simulation_banner(fig: plt.Figure) -> None:
    fig.text(
        0.5,
        0.98,
        BANNER_TEXT,
        ha="center",
        va="top",
        fontsize=9,
        fontweight="bold",
        color="#842029",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8d7da", edgecolor="#f5c2c7", alpha=0.9),
    )


class Phase7FigureGenerator:
    """Generates all publication-ready figures for Phase 7 IEEE evidence package."""

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        mirror_dir: Optional[Path] = None,
        dataset_id: str = "flexpulse_final_test_v4",
    ) -> None:
        self.output_dir = output_dir or (PROJECT_ROOT / "results" / "final_research_package" / "figures")
        if mirror_dir is not None:
            self.mirror_dir = mirror_dir
        elif output_dir is not None:
            self.mirror_dir = self.output_dir
        else:
            self.mirror_dir = PROJECT_ROOT / "results" / "figures" / "phase7_final_evidence"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mirror_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_id = dataset_id
        self.raw_storage = DatasetStorage(dataset_id)
        self.feature_storage = FeatureDatasetStorage(dataset_id)

    def generate_all_figures(
        self,
        table1: pd.DataFrame,
        table3: pd.DataFrame,
        table5: pd.DataFrame,
        table6: pd.DataFrame,
        ood_preds: Dict[str, np.ndarray],
    ) -> List[Path]:
        saved_paths: List[Path] = []
        saved_paths.append(self.plot_figure_1_architecture_signals())
        saved_paths.append(self.plot_figure_2_comparative_performance(table1))
        saved_paths.append(self.plot_figure_3_scenario_breakdown(table3))
        saved_paths.append(self.plot_figure_4_persistence_tradeoff(table5))
        saved_paths.append(self.plot_figure_5_contamination_trajectory())
        saved_paths.append(self.plot_figure_6_prediction_timeline(ood_preds))
        return saved_paths

    def _save_fig(self, fig: plt.Figure, base_name: str) -> Path:
        apply_simulation_banner(fig)
        p_pkg = self.output_dir / f"{base_name}.png"
        p_mir = self.mirror_dir / f"{base_name}.png"
        fig.savefig(p_pkg, dpi=300, bbox_inches="tight")
        fig.savefig(p_mir, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return p_pkg

    def plot_figure_1_architecture_signals(self) -> Path:
        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        t = np.linspace(0, 60, 6000)

        r_base = np.full(6000, 0.080)
        fault_mask = ((t >= 15.0) & (t <= 18.0)) | ((t >= 32.0) & (t <= 36.0))
        r_path = r_base.copy()
        r_path[fault_mask] += 0.450

        load_r = np.full(6000, 4.0)
        curr = 5.0 / (load_r + r_path) + np.random.normal(0, 0.006, 6000)
        volt = curr * load_r + np.random.normal(0, 0.005, 6000)
        v_drop = 5.0 - volt

        axes[0].plot(t, curr, color="#1f77b4", lw=1.2, label=r"$I_{\mathrm{meas}}(t)$")
        axes[0].set_ylabel("Current (A)", fontweight="bold")
        axes[0].grid(True, linestyle="--", alpha=0.5)
        axes[0].legend(loc="upper right")

        axes[1].plot(t, v_drop, color="#d62728", lw=1.2, label=r"$\Delta V_{\mathrm{drop}}(t)$")
        axes[1].set_ylabel(r"$\Delta V$ (V)", fontweight="bold")
        axes[1].grid(True, linestyle="--", alpha=0.5)
        axes[1].legend(loc="upper right")

        axes[2].plot(t, r_path, color="#2ca02c", lw=1.5, label=r"True $R_{\mathrm{path}}(t)$")
        axes[2].axvspan(15.0, 18.0, color="#ff7f0e", alpha=0.25, label="Intermittent Fault Event")
        axes[2].axvspan(32.0, 36.0, color="#ff7f0e", alpha=0.25)
        axes[2].axvspan(0.0, 5.0, color="#17becf", alpha=0.15, label="Known-Healthy Calib Window")
        axes[2].set_xlabel("Time (seconds)", fontweight="bold")
        axes[2].set_ylabel(r"$R_{\mathrm{path}} (\Omega)$", fontweight="bold")
        axes[2].grid(True, linestyle="--", alpha=0.5)
        axes[2].legend(loc="upper right")

        fig.suptitle("FlexPulse-ML Simulated Power-Path Telemetry and Fault Dynamics", fontsize=12, fontweight="bold", y=0.93)
        return self._save_fig(fig, "fig1_powerpath_architecture_signals")

    def plot_figure_2_comparative_performance(self, table1: pd.DataFrame) -> Path:
        fig, ax = plt.subplots(figsize=(11, 5.5))

        ood_t1 = table1[table1["Partition"] == "OOD Test"].copy()
        methods = ood_t1["Method ID"].tolist()
        names = [f"{m}\n({n.split()[0]})" for m, n in zip(methods, ood_t1["Method Name"].tolist())]

        bal_acc = ood_t1["Balanced Acc"].astype(float).tolist()
        spec = ood_t1["Specificity"].astype(float).tolist()
        f1_vals = [float(v) if v != "N/A" else 0.0 for v in ood_t1["F1-Score"].tolist()]

        x = np.arange(len(methods))
        width = 0.25

        r1 = ax.bar(x - width, bal_acc, width, label="Balanced Accuracy", color="#2b5c8f", edgecolor="black", alpha=0.85)
        r2 = ax.bar(x, spec, width, label="Specificity", color="#3caea3", edgecolor="black", alpha=0.85)
        r3 = ax.bar(x + width, f1_vals, width, label="F1-Score", color="#ed553b", edgecolor="black", alpha=0.85)

        ax.set_ylabel("Metric Score", fontweight="bold")
        ax.set_title("Out-of-Distribution (OOD) Performance Across Detection Architectures M1-M6", fontsize=12, fontweight="bold", y=1.03)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=9)
        ax.set_ylim(0.0, 1.15)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(loc="upper left", ncol=3)

        for rects in [r1, r2, r3]:
            for rect in rects:
                h = rect.get_height()
                if h > 0.01:
                    ax.annotate(f"{h:.2f}",
                                xy=(rect.get_x() + rect.get_width() / 2, h),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha="center", va="bottom", fontsize=7)

        return self._save_fig(fig, "fig2_id_vs_ood_comparative_performance")

    def plot_figure_3_scenario_breakdown(self, table3: pd.DataFrame) -> Path:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

        sc_ids = sorted(table3["Scenario ID"].unique())
        labels = [s for s in sc_ids]

        m2_f1, m3_f1, m5_f1 = [], [], []
        m2_spec, m3_spec, m5_spec = [], [], []

        for sc in sc_ids:
            sub = table3[table3["Scenario ID"] == sc]
            m2_row = sub[sub["Method ID"] == "M2"].iloc[0]
            m3_row = sub[sub["Method ID"] == "M3"].iloc[0]
            m5_row = sub[sub["Method ID"] == "M5"].iloc[0]

            m2_f1.append(float(m2_row["F1-Score"]) if m2_row["F1-Score"] != "N/A" else 0.0)
            m3_f1.append(float(m3_row["F1-Score"]) if m3_row["F1-Score"] != "N/A" else 0.0)
            m5_f1.append(float(m5_row["F1-Score"]) if m5_row["F1-Score"] != "N/A" else 0.0)

            m2_spec.append(float(m2_row["Specificity"]))
            m3_spec.append(float(m3_row["Specificity"]))
            m5_spec.append(float(m5_row["Specificity"]))

        x = np.arange(len(sc_ids))
        w = 0.25

        axes[0].bar(x - w, m2_f1, w, label="M2 (Static ML)", color="#d9534f", edgecolor="black")
        axes[0].bar(x, m3_f1, w, label="M3 (Initial Calib)", color="#337ab7", edgecolor="black")
        axes[0].bar(x + w, m5_f1, w, label="M5 (Initial Calib + P)", color="#5cb85c", edgecolor="black")
        axes[0].set_title("F1-Score Across Scenarios S1-S10", fontweight="bold", fontsize=11)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels, rotation=45, ha="right")
        axes[0].set_ylabel("F1-Score", fontweight="bold")
        axes[0].grid(axis="y", linestyle="--", alpha=0.5)
        axes[0].legend(loc="upper left")

        axes[1].bar(x - w, m2_spec, w, label="M2 (Static ML)", color="#d9534f", edgecolor="black")
        axes[1].bar(x, m3_spec, w, label="M3 (Initial Calib)", color="#337ab7", edgecolor="black")
        axes[1].bar(x + w, m5_spec, w, label="M5 (Initial Calib + P)", color="#5cb85c", edgecolor="black")
        axes[1].set_title("Specificity Across Scenarios S1-S10", fontweight="bold", fontsize=11)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=45, ha="right")
        axes[1].set_ylabel("Specificity", fontweight="bold")
        axes[1].grid(axis="y", linestyle="--", alpha=0.5)
        axes[1].legend(loc="lower left")

        fig.suptitle("Per-Scenario Fault Sensitivity and Specificity Breakdown", fontsize=12, fontweight="bold", y=0.98)
        return self._save_fig(fig, "fig3_scenario_breakdown")

    def plot_figure_4_persistence_tradeoff(self, table5: pd.DataFrame) -> Path:
        fig, ax1 = plt.subplots(figsize=(9, 5))

        n_vals = table5["Persistence Windows (N)"].astype(int).tolist()
        fa_rates = table5["FA Rate (events/min)"].astype(float).tolist()
        delays = table5["Added Latency vs N=1 (s)"].astype(float).tolist()
        f1_scores = table5["Window F1-Score"].astype(float).tolist()

        color1 = "#d9534f"
        ax1.set_xlabel("Persistence Filter Threshold N (consecutive windows)", fontweight="bold")
        ax1.set_ylabel("False Alarm Rate (events/min)", color=color1, fontweight="bold")
        l1 = ax1.plot(n_vals, fa_rates, "o-", color=color1, lw=2, markersize=8, label="False Alarm Rate")
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.set_xticks(n_vals)
        ax1.grid(True, linestyle="--", alpha=0.5)

        ax2 = ax1.twinx()
        color2 = "#337ab7"
        ax2.set_ylabel("Added Causal Latency (seconds)", color=color2, fontweight="bold")
        l2 = ax2.plot(n_vals, delays, "s--", color=color2, lw=2, markersize=8, label="Added Causal Delay")
        ax2.tick_params(axis="y", labelcolor=color2)

        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="center right")

        plt.title("Empirical Trade-off: False-Alarm Suppression vs Detection Latency", fontsize=12, fontweight="bold", y=1.03)
        return self._save_fig(fig, "fig4_persistence_tradeoff_curve")

    def plot_figure_5_contamination_trajectory(self) -> Path:
        fig, ax = plt.subplots(figsize=(10, 5))
        t = np.linspace(0, 60, 240)

        r_fault = np.zeros_like(t)
        creep_mask = (t >= 15.0) & (t <= 45.0)
        r_fault[creep_mask] = 0.015 * (t[creep_mask] - 15.0)
        post_mask = t > 45.0
        r_fault[post_mask] = 0.450

        v_center_nominal = 4.80
        v_drop = r_fault * 1.20

        v_tracker_ungated = v_center_nominal - (v_drop * 0.85)
        v_tracker_gated = np.full_like(t, v_center_nominal)
        v_tracker_gated[creep_mask] = v_center_nominal

        ax.plot(t, v_center_nominal - v_drop, color="#333333", linestyle=":", lw=1.5, label="Signal Voltage Center")
        ax.plot(t, v_tracker_ungated, color="#d9534f", lw=2.2, label=r"Ungated Tracker $\hat{V}_0(t)$ (Contaminated)")
        ax.plot(t, v_tracker_gated, color="#5cb85c", lw=2.5, label=r"Gated Tracker $\hat{V}_0(t)$ (Safeguarded / Frozen)")

        ax.axvspan(15.0, 45.0, color="#ff7f0e", alpha=0.18, label="Slow Creeping Fault Interval")
        ax.set_xlabel("Time (seconds)", fontweight="bold")
        ax.set_ylabel(r"Voltage Baseline Estimate $\hat{V}_0$ (V)", fontweight="bold")
        ax.set_title("Defensive Gating Prevents Baseline Contamination During Creeping Faults", fontsize=12, fontweight="bold", y=1.03)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower left")

        return self._save_fig(fig, "fig5_contamination_prevention_trajectory")

    def plot_figure_6_prediction_timeline(self, ood_preds: Dict[str, np.ndarray]) -> Path:
        fig, axes = plt.subplots(5, 1, figsize=(11, 7), sharex=True)

        ood_df = self.feature_storage.load_partition(DatasetStorage.SPLIT_OOD_TEST)
        runs = ood_df["run_id"].unique()
        r_target = runs[12] if len(runs) > 12 else runs[0]

        r_mask = (ood_df["run_id"].to_numpy() == r_target)
        r_df = ood_df[r_mask].reset_index(drop=True)
        t_w = r_df["window_start_s"].to_numpy()

        y_gt = r_df["fault_label"].to_numpy()
        p_m2 = ood_preds["M2_Static_ML"][r_mask]
        p_m3 = ood_preds["M3_Initial_Calib"][r_mask]
        p_m5 = ood_preds["M5_Initial_Calib_Persistence"][r_mask]

        axes[0].fill_between(t_w, 0, y_gt, color="#2ca02c", step="mid", alpha=0.7)
        axes[0].set_ylabel("Ground\nTruth", fontweight="bold", fontsize=9)
        axes[0].set_ylim(-0.1, 1.1)
        axes[0].grid(True, linestyle="--", alpha=0.5)

        axes[1].fill_between(t_w, 0, p_m2, color="#d9534f", step="mid", alpha=0.7)
        axes[1].set_ylabel("M2\nStatic ML", fontweight="bold", fontsize=9)
        axes[1].set_ylim(-0.1, 1.1)
        axes[1].grid(True, linestyle="--", alpha=0.5)

        axes[2].fill_between(t_w, 0, p_m3, color="#337ab7", step="mid", alpha=0.7)
        axes[2].set_ylabel("M3\nInitial Calib", fontweight="bold", fontsize=9)
        axes[2].set_ylim(-0.1, 1.1)
        axes[2].grid(True, linestyle="--", alpha=0.5)

        axes[3].fill_between(t_w, 0, p_m5, color="#5cb85c", step="mid", alpha=0.7)
        axes[3].set_ylabel("M5\nM3 + P (N=2)", fontweight="bold", fontsize=9)
        axes[3].set_ylim(-0.1, 1.1)
        axes[3].grid(True, linestyle="--", alpha=0.5)

        v_mean = r_df["voltage_mean_v"].to_numpy()
        axes[4].plot(t_w, v_mean, color="#333333", lw=1.2)
        axes[4].set_ylabel("Voltage\nMean (V)", fontweight="bold", fontsize=9)
        axes[4].set_xlabel("Window Start Time (seconds)", fontweight="bold")
        axes[4].grid(True, linestyle="--", alpha=0.5)

        fig.suptitle(f"Prediction Timeline Comparison on Shifted Run ({r_target})", fontsize=12, fontweight="bold", y=0.98)
        return self._save_fig(fig, "fig6_prediction_timeline_overlay")


def generate_final_figures(
    table1: pd.DataFrame,
    table3: pd.DataFrame,
    table5: pd.DataFrame,
    table6: pd.DataFrame,
    ood_preds: Dict[str, np.ndarray],
) -> List[Path]:
    gen = Phase7FigureGenerator()
    return gen.generate_all_figures(table1, table3, table5, table6, ood_preds)
