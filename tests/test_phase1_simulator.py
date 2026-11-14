"""
Automated unit and integration tests for Phase 1: Electrical Power-Path Simulation Engine.
"""

import numpy as np
import pytest

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


def test_mathematical_equation_correctness():
    v_source = np.array([5.0, 5.0])
    r_path = np.array([0.100, 0.500])
    r_load = np.array([4.900, 4.500])
    v_offset = np.zeros(2)
    i_offset = np.zeros(2)
    v_noise = np.zeros(2)
    i_noise = np.zeros(2)

    v_load, i_true, v_meas, i_meas = compute_circuit_signals(
        source_voltage_v=v_source,
        path_resistance_ohm=r_path,
        load_resistance_ohm=r_load,
        voltage_offset_v=v_offset,
        current_offset_a=i_offset,
        voltage_noise_v=v_noise,
        current_noise_a=i_noise,
    )

    expected_i_0 = 5.0 / (4.900 + 0.100)
    expected_v_load_0 = expected_i_0 * 4.900

    expected_i_1 = 5.0 / (4.500 + 0.500)
    expected_v_load_1 = expected_i_1 * 4.500

    assert np.isclose(i_true[0], expected_i_0)
    assert np.isclose(v_load[0], expected_v_load_0)
    assert np.isclose(i_true[1], expected_i_1)
    assert np.isclose(v_load[1], expected_v_load_1)
    assert np.isclose(v_meas[0], v_load[0])
    assert np.isclose(i_meas[0], i_true[0])


def test_voltage_current_linear_relationship():
    v_source = np.full(100, 5.0)
    r_path = np.linspace(0.05, 0.50, 100)
    r_load = np.full(100, 3.0)
    v_offset = np.zeros(100)
    i_offset = np.zeros(100)
    v_noise = np.zeros(100)
    i_noise = np.zeros(100)

    v_load, i_true, _, _ = compute_circuit_signals(
        v_source, r_path, r_load, v_offset, i_offset, v_noise, i_noise
    )

    ohm_v = i_true * r_load
    assert np.allclose(v_load, ohm_v, atol=1e-12)


def test_healthy_scenario_labels_all_zero():
    sim = PowerPathSimulator()
    res = sim.run_simulation(scenario=PowerPathSimulator.SCENARIO_HEALTHY)

    assert np.all(res.ground_truth.fault_label == 0)
    assert np.all(res.ground_truth.fault_type == "HEALTHY")
    assert np.all(res.ground_truth.event_id == 0)


def test_increased_resistance_labels_and_timing():
    sim = PowerPathSimulator()
    start_s = 10.0
    duration_s = 15.0
    res = sim.run_simulation(
        scenario=PowerPathSimulator.SCENARIO_INCREASED_RESISTANCE,
        fault_start_time_s=start_s,
        fault_duration_s=duration_s,
        fault_resistance_ohms=0.300,
    )

    t = res.observations.timestamp_s
    labels = res.ground_truth.fault_label
    event_mask = (t >= start_s) & (t < start_s + duration_s)

    assert np.all(labels[event_mask] == 1)
    assert np.all(labels[~event_mask] == 0)
    assert np.all(res.ground_truth.fault_type[event_mask] == PowerPathSimulator.SCENARIO_INCREASED_RESISTANCE)


def test_intermittent_contact_labels():
    sim = PowerPathSimulator()
    starts = [10.0, 20.0, 30.0]
    duration_s = 0.5
    res = sim.run_simulation(
        scenario=PowerPathSimulator.SCENARIO_INTERMITTENT_CONTACT,
        event_count=3,
        event_duration_s=duration_s,
        event_start_times_s=starts,
        fault_resistance_ohms=100.0,
    )

    t = res.observations.timestamp_s
    labels = res.ground_truth.fault_label

    for ev_id, st in enumerate(starts, start=1):
        mask = (t >= st) & (t < st + duration_s)
        assert np.all(labels[mask] == 1)
        assert np.all(res.ground_truth.event_id[mask] == ev_id)
        assert np.all(res.ground_truth.fault_type[mask] == PowerPathSimulator.SCENARIO_INTERMITTENT_CONTACT)


def test_gradual_resistance_drift_labels():
    sim = PowerPathSimulator()
    drift_start = 8.0
    drift_end = 25.0
    res = sim.run_simulation(
        scenario=PowerPathSimulator.SCENARIO_GRADUAL_RESISTANCE_DRIFT,
        drift_start_s=drift_start,
        drift_end_s=drift_end,
        drift_rate_ohms_per_sec=0.01,
    )

    t = res.observations.timestamp_s
    labels = res.ground_truth.fault_label
    pre_drift = t < drift_start
    drift_and_post = t >= drift_start

    assert np.all(labels[pre_drift] == 0)
    assert np.all(labels[drift_and_post] == 1)


def test_measurement_offset_drift_remains_healthy_cable_label():
    sim = PowerPathSimulator()
    res = sim.run_simulation(
        scenario=PowerPathSimulator.SCENARIO_MEASUREMENT_OFFSET_DRIFT,
        voltage_offset_drift_v_per_sec=0.005,
    )

    assert np.all(res.ground_truth.fault_label == 0)
    assert np.all(res.ground_truth.fault_type == "HEALTHY")


def test_operating_condition_shift_remains_healthy_cable_label():
    sim = PowerPathSimulator()
    res = sim.run_simulation(
        scenario=PowerPathSimulator.SCENARIO_OPERATING_CONDITION_SHIFT,
        shift_time_s=15.0,
        shifted_load_ohms=1.5,
        noise_multiplier=3.0,
    )

    assert np.all(res.ground_truth.fault_label == 0)
    assert np.all(res.ground_truth.fault_type == "HEALTHY")


def test_timestamp_spacing_uniformity():
    p = SimulationParams(sampling_rate_hz=200.0, duration_seconds=10.0)
    sim = PowerPathSimulator(default_params=p)
    res = sim.run_simulation(scenario=PowerPathSimulator.SCENARIO_HEALTHY, params=p)

    t = res.observations.timestamp_s
    diffs = np.diff(t)
    expected_dt = 1.0 / 200.0

    assert np.allclose(diffs, expected_dt, atol=1e-9)


def test_identical_outputs_for_identical_seeds():
    sim = PowerPathSimulator()
    res1 = sim.run_simulation(scenario=PowerPathSimulator.SCENARIO_HEALTHY, seed=999)
    res2 = sim.run_simulation(scenario=PowerPathSimulator.SCENARIO_HEALTHY, seed=999)

    assert np.allclose(res1.observations.voltage_measured_v, res2.observations.voltage_measured_v)
    assert np.allclose(res1.observations.current_measured_a, res2.observations.current_measured_a)
    assert np.allclose(res1.ground_truth.true_path_resistance_ohm, res2.ground_truth.true_path_resistance_ohm)


def test_different_noise_for_different_seeds():
    sim = PowerPathSimulator()
    res1 = sim.run_simulation(scenario=PowerPathSimulator.SCENARIO_HEALTHY, seed=101)
    res2 = sim.run_simulation(scenario=PowerPathSimulator.SCENARIO_HEALTHY, seed=202)

    assert not np.allclose(res1.observations.voltage_measured_v, res2.observations.voltage_measured_v)


def test_invalid_parameters_rejection():
    with pytest.raises(ValueError):
        SimulationParams(sampling_rate_hz=-10.0).validate()

    with pytest.raises(ValueError):
        SimulationParams(duration_seconds=0.0).validate()

    with pytest.raises(ValueError):
        SimulationParams(source_voltage_nominal_v=-5.0).validate()

    with pytest.raises(ValueError):
        SimulationParams(load_resistance_nominal_ohms=0.0).validate()

    with pytest.raises(ValueError):
        SimulationParams(baseline_path_resistance_nominal_ohms=-0.01).validate()


def test_no_division_by_zero_or_nans_under_valid_inputs():
    sim = PowerPathSimulator()
    for scen in PowerPathSimulator.ALL_SCENARIOS:
        res = sim.run_simulation(scenario=scen)
        obs = res.observations
        gt = res.ground_truth

        assert not np.any(np.isnan(obs.voltage_measured_v))
        assert not np.any(np.isinf(obs.voltage_measured_v))
        assert not np.any(np.isnan(obs.current_measured_a))
        assert not np.any(np.isinf(obs.current_measured_a))
        assert not np.any(np.isnan(gt.true_path_resistance_ohm))
        assert not np.any(np.isinf(gt.true_path_resistance_ohm))


def test_separation_of_observations_and_ground_truth():
    sim = PowerPathSimulator()
    res = sim.run_simulation(scenario=PowerPathSimulator.SCENARIO_INCREASED_RESISTANCE)

    obs_df = res.observations.to_dataframe()
    gt_df = res.ground_truth.to_dataframe()

    assert "true_path_resistance_ohm" not in obs_df.columns
    assert "fault_label" not in obs_df.columns
    assert "fault_type" not in obs_df.columns
    assert "event_id" not in obs_df.columns

    assert "fault_label" in gt_df.columns
    assert "true_path_resistance_ohm" in gt_df.columns
    assert res.metadata.synthetic_data is True
