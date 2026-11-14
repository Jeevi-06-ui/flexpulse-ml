"""
Simulation package for FlexPulse-ML.
"""

from flexpulse_ml.simulation.electrical_model import (
    SimulationGroundTruth,
    SimulationMetadata,
    SimulationObservations,
    SimulationResult,
    compute_circuit_signals,
    validate_physical_parameters,
)
from flexpulse_ml.simulation.fault_injector import (
    PowerPathSimulator,
    SimulationParams,
)
from flexpulse_ml.simulation.plotting import (
    generate_all_scenario_figures,
    plot_simulation_scenario,
)

__all__ = [
    "PowerPathSimulator",
    "SimulationGroundTruth",
    "SimulationMetadata",
    "SimulationObservations",
    "SimulationParams",
    "SimulationResult",
    "compute_circuit_signals",
    "generate_all_scenario_figures",
    "plot_simulation_scenario",
    "validate_physical_parameters",
]
