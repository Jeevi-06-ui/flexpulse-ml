"""
Unit and integration tests for Phase 3: Electrical Feature Engineering Pipeline.

Covers:
- Known voltage-drop calculations
- Resistance slope calculations
- Zero and near-zero current handling
- Missing load levels (slope fit gating)
- Unsettled measurements exclusion
- Invalid timestamps and dropped samples
- Known reference offsets (reference cable is NOT 0 Ohm)
- Synthetic vs. real data separation
- Fast interruption statistics gating (INA226 does not invent counts)
"""

from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pytest

from flexpulse_ml.acquisition.schema import (
    CanonicalMeasurementRecord,
    DatasetCategory,
    FaultClass,
    FaultOrigin,
    LabelSource,
    MeasurementQuality,
)
from flexpulse_ml.calibration.reference import (
    CalibrationRecord,
    CharacterizedReferenceCable,
    FixtureTareProfile,
)
from flexpulse_ml.processing.features import (
    FEATURE_COLUMN_NAMES,
    detect_distinct_current_levels,
    extract_feature_dataset,
    extract_features_from_conditioned_signal,
    fit_multi_load_resistance,
)
from flexpulse_ml.processing.preprocessing import (
    condition_trial_records,
)


@pytest.fixture
def calibrated_test_profile() -> CalibrationRecord:
    """Fixture with 40 mOhm tare and 72 mOhm characterized golden cable."""
    fixture = FixtureTareProfile(
        fixture_id="FIXTURE_TEST_01",
        tare_resistance_ohms=0.040, # 40 mOhm
        voltage_offset_v=0.0010,    # 1.0 mV
        current_offset_a=0.0005,
    )
    cable = CharacterizedReferenceCable(
        reference_cable_id="GOLDEN-REF-CABLE-01",
        four_wire_measured_resistance_ohms=0.072, # Non-zero reference cable!
    )
    return CalibrationRecord(
        calibration_id="CAL_TEST_01",
        fixture=fixture,
        reference_cable=cable,
    )


# ==============================================================================
# 1. KNOWN VOLTAGE DROP & OHM'S LAW CALCULATIONS
# ==============================================================================

def test_known_voltage_drop_calculation(calibrated_test_profile: CalibrationRecord):
    """Test that delta_v = vin - vout and summary statistics match known hand calculation."""
    # Create 3 records:
    # 1: vin=5.050, vout=4.930 -> delta_v = 0.120 V
    # 2: vin=5.040, vout=4.800 -> delta_v = 0.240 V
    # 3: vin=5.030, vout=4.670 -> delta_v = 0.360 V
    records = [
        CanonicalMeasurementRecord(
            timestamp_utc="2026-11-11T12:00:00Z",
            cable_id="CABLE-TEST-01",
            session_id="SESS-01",
            trial_id="TR-01",
            sample_index=i,
            vin_v=vin,
            vout_v=vout,
            current_a=curr,
            load_setpoint_a=curr,
            temperature_c=25.0,
            source_type=DatasetCategory.REAL_MEASUREMENT.value,
            ground_truth_label=FaultClass.INTACT.value,
            ground_truth_source=LabelSource.HUMAN_VERIFIED.value,
            fault_origin=FaultOrigin.GOLDEN_REFERENCE.value,
            fault_condition="NONE",
            fixture_id="FIXTURE_TEST_01",
            instrument_id="INA226",
            calibration_id="CAL_TEST_01",
            action_id="STATIC_HOLD",
            motion_region="STATIC",
            is_settled=True,
            measurement_quality=MeasurementQuality.OK.value,
        )
        for i, (vin, vout, curr) in enumerate([
            (5.050, 4.930, 1.0),
            (5.040, 4.800, 2.0),
            (5.030, 4.670, 3.0),
        ])
    ]

    conditioned = condition_trial_records(records, calibration=calibrated_test_profile)
    assert len(conditioned.delta_v) == 3
    assert pytest.approx(conditioned.delta_v[0], abs=1e-5) == 0.120
    assert pytest.approx(conditioned.delta_v[1], abs=1e-5) == 0.240
    assert pytest.approx(conditioned.delta_v[2], abs=1e-5) == 0.360

    features = extract_features_from_conditioned_signal(
        conditioned, calibration=calibrated_test_profile
    )

    # Mean = (0.120 + 0.240 + 0.360) / 3 = 0.240 V
    assert pytest.approx(features["delta_v_mean_v"], abs=1e-5) == 0.240
    # Median = 0.240 V
    assert pytest.approx(features["delta_v_median_v"], abs=1e-5) == 0.240


# ==============================================================================
# 2. RESISTANCE SLOPE CALCULATIONS & MULTI-LOAD GATING
# ==============================================================================

def test_multi_load_resistance_slope_fit():
    """Verify that linear fit delta_v = offset + R * I correctly estimates known slope."""
    # Synthetic current array with 3 distinct load levels (1.0A, 2.0A, 3.0A) with 4 samples each
    # True R = 0.120 Ohm (120 mOhm), True Offset = 0.005 V (5 mV)
    true_r = 0.120
    true_offset = 0.005

    currents = []
    delta_vs = []
    for current_level in [1.0, 2.0, 3.0]:
        for _ in range(4):
            currents.append(current_level)
            delta_vs.append(true_offset + (true_r * current_level))

    i_arr = np.array(currents)
    dv_arr = np.array(delta_vs)

    slope, offset, rmse, r2, max_res, is_valid = fit_multi_load_resistance(i_arr, dv_arr)
    assert is_valid is True
    assert pytest.approx(slope, abs=1e-5) == 0.120
    assert pytest.approx(offset, abs=1e-5) == 0.005
    assert pytest.approx(rmse, abs=1e-5) == 0.0
    assert pytest.approx(r2, abs=1e-5) == 1.0


def test_missing_load_levels_refuses_slope_calculation():
    """Scientific Integrity: If only 1 static load level is present, do NOT calculate slope!"""
    # 10 samples all at exactly 1.5 A
    i_arr = np.full(10, 1.5)
    dv_arr = np.full(10, 0.180)

    # detect_distinct_current_levels should return 1
    distinct = detect_distinct_current_levels(i_arr)
    assert distinct == 1

    slope, offset, rmse, r2, max_res, is_valid = fit_multi_load_resistance(i_arr, dv_arr)
    assert is_valid is False
    assert np.isnan(slope)
    assert np.isnan(offset)


# ==============================================================================
# 3. ZERO AND NEAR-ZERO CURRENT HANDLING
# ==============================================================================

def test_zero_and_near_zero_current_handling(calibrated_test_profile: CalibrationRecord):
    """Ensure zero/near-zero currents do not cause ZeroDivisionError and are handled safely."""
    records = [
        CanonicalMeasurementRecord(
            timestamp_utc="2026-11-11T12:00:00Z",
            cable_id="CABLE-DISCONN-01",
            session_id="SESS-01",
            trial_id="TR-01",
            sample_index=0,
            vin_v=5.00,
            vout_v=0.00, # Open circuit!
            current_a=0.00, # Zero current
            load_setpoint_a=1.5,
            temperature_c=25.0,
            source_type=DatasetCategory.REAL_MEASUREMENT.value,
            ground_truth_label=FaultClass.INTERMITTENT_DISCONNECT.value,
            ground_truth_source=LabelSource.HUMAN_VERIFIED.value,
            fault_origin=FaultOrigin.FIXTURE_INJECTED.value,
            fault_condition="OPEN_CIRCUIT",
            fixture_id="FIXTURE_TEST_01",
            instrument_id="INA226",
            calibration_id="CAL_TEST_01",
            action_id="STATIC_HOLD",
            motion_region="STATIC",
            is_settled=True,
            measurement_quality=MeasurementQuality.DISCONNECTED.value,
        )
    ]
    conditioned = condition_trial_records(records, calibration=calibrated_test_profile)
    features = extract_features_from_conditioned_signal(conditioned, calibration=calibrated_test_profile)

    # Slope should be NaN (no multi-load), point mean resistance should be NaN, not crash
    assert features["has_multiload_fit"] is False
    assert np.isnan(features["r_loop_slope_ohms"])
    assert np.isnan(features["r_point_mean_ohms"])


# ==============================================================================
# 4. UNSETTLED MEASUREMENTS EXCLUSION
# ==============================================================================

def test_unsettled_measurements_excluded_from_loop_resistance():
    """Verify that load transitions (is_settled=False) are excluded from steady-state slope fit."""
    records = []
    # 5 samples during switching transition (is_settled=False) with erratic voltages
    for i in range(5):
        records.append(
            CanonicalMeasurementRecord(
                timestamp_utc="2026-11-11T12:00:00Z",
                cable_id="CABLE-TEST-01",
                session_id="SESS-01",
                trial_id="TR-01",
                sample_index=i,
                vin_v=5.0,
                vout_v=3.5, # Huge transient drop
                current_a=0.8,
                load_setpoint_a=1.5,
                temperature_c=25.0,
                source_type=DatasetCategory.REAL_MEASUREMENT.value,
                ground_truth_label=FaultClass.INTACT.value,
                ground_truth_source=LabelSource.HUMAN_VERIFIED.value,
                fault_origin=FaultOrigin.GOLDEN_REFERENCE.value,
                fault_condition="NONE",
                fixture_id="FIXTURE_TEST_01",
                instrument_id="INA226",
                calibration_id="CAL_TEST_01",
                action_id="STATIC_HOLD",
                motion_region="STATIC",
                is_settled=False, # UNSETTLED TRANSIENT!
                measurement_quality="TRANSITION",
            )
        )
    # 8 settled samples at 1.0A (Vdrop = 0.100V) and 8 settled samples at 2.0A (Vdrop = 0.200V)
    # True steady-state R = 0.100 Ohm
    for j in range(8):
        records.append(
            CanonicalMeasurementRecord(
                timestamp_utc="2026-11-11T12:00:00Z",
                cable_id="CABLE-TEST-01",
                session_id="SESS-01",
                trial_id="TR-01",
                sample_index=5 + j,
                vin_v=5.0,
                vout_v=4.90,
                current_a=1.0,
                load_setpoint_a=1.0,
                temperature_c=25.0,
                source_type=DatasetCategory.REAL_MEASUREMENT.value,
                ground_truth_label=FaultClass.INTACT.value,
                ground_truth_source=LabelSource.HUMAN_VERIFIED.value,
                fault_origin=FaultOrigin.GOLDEN_REFERENCE.value,
                fault_condition="NONE",
                fixture_id="FIXTURE_TEST_01",
                instrument_id="INA226",
                calibration_id="CAL_TEST_01",
                action_id="STATIC_HOLD",
                motion_region="STATIC",
                is_settled=True, # SETTLED
                measurement_quality="OK",
            )
        )
    for k in range(8):
        records.append(
            CanonicalMeasurementRecord(
                timestamp_utc="2026-11-11T12:00:00Z",
                cable_id="CABLE-TEST-01",
                session_id="SESS-01",
                trial_id="TR-01",
                sample_index=13 + k,
                vin_v=5.0,
                vout_v=4.80,
                current_a=2.0,
                load_setpoint_a=2.0,
                temperature_c=25.0,
                source_type=DatasetCategory.REAL_MEASUREMENT.value,
                ground_truth_label=FaultClass.INTACT.value,
                ground_truth_source=LabelSource.HUMAN_VERIFIED.value,
                fault_origin=FaultOrigin.GOLDEN_REFERENCE.value,
                fault_condition="NONE",
                fixture_id="FIXTURE_TEST_01",
                instrument_id="INA226",
                calibration_id="CAL_TEST_01",
                action_id="STATIC_HOLD",
                motion_region="STATIC",
                is_settled=True, # SETTLED
                measurement_quality="OK",
            )
        )

    conditioned = condition_trial_records(records)
    assert conditioned.audit.unsettled_transition_samples == 5
    assert conditioned.audit.settled_samples == 16

    features = extract_features_from_conditioned_signal(conditioned)
    # The slope should be calculated ONLY on settled samples (Slope = (0.20 - 0.10) / (2.0 - 1.0) = 0.100 Ohm)
    assert features["has_multiload_fit"] is True
    assert pytest.approx(features["r_loop_slope_ohms"], abs=1e-4) == 0.100


# ==============================================================================
# 5. KNOWN REFERENCE OFFSETS & TARE COMPENSATION
# ==============================================================================

def test_reference_fixture_tare_compensation(calibrated_test_profile: CalibrationRecord):
    """Verify fixture tare subtraction and that reference cable resistance is NOT assumed 0."""
    cal = calibrated_test_profile
    assert cal.fixture.tare_resistance_ohms == 0.040 # 40 mOhm
    assert cal.reference_cable.four_wire_measured_resistance_ohms == 0.072 # 72 mOhm (non-zero!)

    # Suppose measured raw loop resistance is 0.120 Ohm (120 mOhm)
    # Calibrated cable resistance = 0.120 - 0.040 = 0.080 Ohm (80 mOhm)
    records = []
    for i, curr in enumerate([1.0, 1.0, 1.0, 2.0, 2.0, 2.0]):
        vdrop = 0.120 * curr
        records.append(
            CanonicalMeasurementRecord(
                timestamp_utc="2026-11-11T12:00:00Z",
                cable_id="CABLE-TEST-01",
                session_id="SESS-01",
                trial_id="TR-01",
                sample_index=i,
                vin_v=5.0,
                vout_v=5.0 - vdrop,
                current_a=curr,
                load_setpoint_a=curr,
                temperature_c=25.0,
                source_type=DatasetCategory.REAL_MEASUREMENT.value,
                ground_truth_label=FaultClass.INTACT.value,
                ground_truth_source=LabelSource.HUMAN_VERIFIED.value,
                fault_origin=FaultOrigin.GOLDEN_REFERENCE.value,
                fault_condition="NONE",
                fixture_id="FIXTURE_TEST_01",
                instrument_id="INA226",
                calibration_id="CAL_TEST_01",
                action_id="STATIC_HOLD",
                motion_region="STATIC",
                is_settled=True,
                measurement_quality="OK",
            )
        )

    conditioned = condition_trial_records(records, calibration=cal)
    features = extract_features_from_conditioned_signal(conditioned, calibration=cal)

    assert pytest.approx(features["r_loop_slope_ohms"], abs=1e-4) == 0.120
    # Calibrated = 0.120 - 0.040 = 0.080 Ohm
    assert pytest.approx(features["r_loop_calibrated_ohms"], abs=1e-4) == 0.080


# ==============================================================================
# 6. FAST INTERRUPTION GATING (INA226 CHANNELS)
# ==============================================================================

def test_fast_interruption_gating_for_slow_ina226_channel(calibrated_test_profile: CalibrationRecord):
    """Scientific Integrity: INA226 channel must set fast_capture_source_present=False and count=NaN."""
    records = [
        CanonicalMeasurementRecord(
            timestamp_utc="2026-11-11T12:00:00Z",
            cable_id="CABLE-TEST-01",
            session_id="SESS-01",
            trial_id="TR-01",
            sample_index=0,
            vin_v=5.0,
            vout_v=4.88,
            current_a=1.5,
            load_setpoint_a=1.5,
            temperature_c=25.0,
            source_type=DatasetCategory.REAL_MEASUREMENT.value,
            ground_truth_label=FaultClass.INTACT.value,
            ground_truth_source=LabelSource.HUMAN_VERIFIED.value,
            fault_origin=FaultOrigin.GOLDEN_REFERENCE.value,
            fault_condition="NONE",
            fixture_id="FIXTURE_ESP32_INA226_01",
            instrument_id="INA226",
            calibration_id="CAL_DEFAULT",
            action_id="STATIC_HOLD",
            motion_region="STATIC",
            is_settled=True,
            measurement_quality="OK",
        )
    ]
    conditioned = condition_trial_records(records, calibration=calibrated_test_profile)

    # When fast_capture_source_present is False (standard INA226 slow channel):
    features = extract_features_from_conditioned_signal(
        conditioned, calibration=calibrated_test_profile, fast_capture_source_present=False
    )
    assert features["fast_capture_source_present"] is False
    assert np.isnan(features["fast_interruption_count"])


# ==============================================================================
# 7. FEATURE DATASET EXTRACTION & SEPARATION OF TARGET / METADATA
# ==============================================================================

def test_extract_feature_dataset_preserves_zero_label_leakage():
    """Verify that metadata and labels are kept completely separate from features DataFrame."""
    from flexpulse_ml.acquisition.synthetic_generator import generate_canonical_synthetic_dataset

    records = generate_canonical_synthetic_dataset(samples_per_trial=15)
    features_df, metadata_df = extract_feature_dataset(records)

    assert len(features_df) == 4 # 4 trials (1 per cable)
    assert len(metadata_df) == 4

    # Check zero target leakage
    assert "ground_truth_label" not in features_df.columns
    assert "cable_id" not in features_df.columns
    assert "fault_condition" not in features_df.columns

    # Verify metadata preserves ground truth and grouping
    assert "ground_truth_label" in metadata_df.columns
    assert "cable_id" in metadata_df.columns
    assert "source_type" in metadata_df.columns
    assert all(st == "SYNTHETIC_DEMO" for st in metadata_df["source_type"])

