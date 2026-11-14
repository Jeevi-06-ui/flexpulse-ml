"""
Visual validation and plotting utility for simulated power-path scenarios.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flexpulse_ml.config import PROJECT_ROOT
from flexpulse_ml.simulation.electrical_model import SimulationResult
from flexpulse_ml.simulation.fault_injector import PowerPathSimulator

FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


def plot_simulation_scenario(
    result: SimulationResult,
    output_path: Optional[Path] = None,
    show_plot: bool = False,
) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = output_path or (FIGURES_DIR / f"simulated_{result.metadata.scenario.lower()}.png")

    obs = result.observations
    gt = result.ground_truth
    t = obs.timestamp_s

    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)

    fig.suptitle(
        f"FlexPulse-ML :: Scenario: {result.metadata.scenario} (SIMULATED DATA ONLY)",
        fontsize=13,
        fontweight="bold",
    )

    axes[0].plot(t, obs.voltage_measured_v, color="#1f77b4", label="Measured Voltage (V)")
    axes[0].plot(t, obs.source_voltage_v, color="#7f7f7f", linestyle="--", alpha=0.6, label="Source Voltage (V)")
    axes[0].set_ylabel("Voltage [V]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(t, obs.current_measured_a, color="#ff7f0e", label="Measured Current (A)")
    axes[1].set_ylabel("Current [A]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    axes[2].plot(t, gt.true_path_resistance_ohm, color="#2ca02c", label="True Path Resistance [Ω]")
    axes[2].set_ylabel("Resistance [Ω]")
    axes[2].set_xlabel("Time [s]")
    axes[2].grid(True, alpha=0.3)

    fault_indices = np.where(gt.fault_label == 1)[0]
    if len(fault_indices) > 0:
        fault_periods = np.split(fault_indices, np.where(np.diff(fault_indices) > 1)[0] + 1)
        for idx, period in enumerate(fault_periods):
            t_start = t[period[0]]
            t_end = t[period[-1]]
            label = "Ground-Truth Fault Interval" if idx == 0 else None
            for ax in axes:
                ax.axvspan(t_start, t_end, color="red", alpha=0.15, label=label)

    axes[2].legend(loc="upper right")

    fig.text(
        0.5,
        0.01,
        "SYNTHETIC DEMO SIMULATION: Resistive Equivalent Circuit Model (No Physical Hardware)",
        ha="center",
        fontsize=9,
        color="darkred",
        weight="bold",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(out, dpi=180)
    if not show_plot:
        plt.close(fig)

    return out


def generate_all_scenario_figures(output_dir: Optional[Path] = None) -> list[Path]:
    target_dir = output_dir or FIGURES_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    simulator = PowerPathSimulator()
    saved_paths = []

    for scenario_name in PowerPathSimulator.ALL_SCENARIOS:
        res = simulator.run_simulation(scenario=scenario_name)
        target_file = target_dir / f"simulated_{scenario_name.lower()}.png"
        saved = plot_simulation_scenario(res, output_path=target_file)
        saved_paths.append(saved)

    return saved_paths
