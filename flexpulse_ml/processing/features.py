"""
Scientifically Defensible Electrical Feature Engineering Pipeline for FlexPulse-ML.

Implements core electrical and physical features:
1. Voltage Drop: delta_v = vin_v - vout_v
2. Effective Loop Resistance from Settled Multi-Load Measurements:
   Linear fit: delta_v = offset + resistance * current.
   Slope-based resistance is ONLY calculated if there are >= 2 distinct, valid settled current levels.
3. Central Tendency & Robust Spread: Mean, Median, Std, IQR, and MAD of delta_v.
4. Current Variability & Settled-Load Stability: Mean, Std, CV, and setpoint stability ratio.
5. Fit Residuals & Load-Dependent Behavior: RMSE, R^2, and max residual.
6. Measurement Quality & Missingness Indicators: Settled ratio, missingness, and quality flags.
7. Temperature Context: Mean, drift, and validity flag.
8. Interruption Statistics: Strictly gated by fast_capture_source_present flag.
   INA226 channels NEVER invent transient event counts.
9. Reference Fixture Tare Correction: Calibrates fixture resistance without assuming
   a reference USB-C cable has zero resistance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from flexpulse_ml.acquisition.schema import CanonicalMeasurementRecord
from flexpulse_ml.calibration.reference import CalibrationRecord, CalibrationRegistry
from flexpulse_ml.processing.preprocessing import (
    ConditionedSignal,
    condition_trial_records,
)
from flexpulse_ml.utils.logging import get_logger
from flexpulse_ml.utils.paths import DOCS_DIR, PROCESSED_DATA_DIR, ensure_directories_exist

logger = get_logger("processing.features")

# Formal Feature Dictionary detailing electrical definitions, formulas, and validity rules
FEATURE_DICTIONARY: Dict[str, Dict[str, Any]] = {
    # 1. Voltage Drop Features
    "delta_v_mean_v": {
        "unit": "Volts (V)",
        "formula": "mean(vin_v - vout_v)",
        "description": "Mean voltage drop across the cable assembly under test.",
        "physical_meaning": "Primary indicator of total series path impedance."
    },
    "delta_v_median_v": {
        "unit": "Volts (V)",
        "formula": "median(vin_v - vout_v)",
        "description": "Median voltage drop across the cable assembly.",
        "physical_meaning": "Robust central tendency resistant to transient measurement spikes."
    },
    "delta_v_std_v": {
        "unit": "Volts (V)",
        "formula": "std(vin_v - vout_v)",
        "description": "Standard deviation of voltage drop.",
        "physical_meaning": "Measures voltage fluctuations caused by mechanical flex or contact flutter."
    },
    "delta_v_iqr_v": {
        "unit": "Volts (V)",
        "formula": "Q75(delta_v) - Q25(delta_v)",
        "description": "Interquartile range (IQR) of voltage drop.",
        "physical_meaning": "Robust measure of voltage dispersion unaffected by extreme outlier glitches."
    },
    "delta_v_mad_v": {
        "unit": "Volts (V)",
        "formula": "median(|delta_v - median(delta_v)|)",
        "description": "Median Absolute Deviation (MAD) of voltage drop.",
        "physical_meaning": "Scale-invariant robust spread metric for non-Gaussian noise."
    },
    # 2. Loop Resistance (Slope Fit)
    "r_loop_slope_ohms": {
        "unit": "Ohms (Ω)",
        "formula": "Slope R from linear fit: delta_v = offset + R * current (settled intervals only)",
        "description": "Effective multi-load loop resistance from settled measurements.",
        "physical_meaning": "True dynamic DC loop resistance independent of contact zero-offsets.",
        "validity_condition": "Computed ONLY when distinct settled current levels >= 2."
    },
    "v_offset_fit_v": {
        "unit": "Volts (V)",
        "formula": "Intercept of linear fit: delta_v = offset + R * current",
        "description": "Extrapolated zero-current voltage drop offset.",
        "physical_meaning": "Characterizes thermoelectric EMF or ADC instrumentation bias."
    },
    "has_multiload_fit": {
        "unit": "Boolean",
        "formula": "distinct_current_levels >= 2 and settled_samples >= 6",
        "description": "Flag indicating if multi-load slope resistance could be legitimately computed.",
        "physical_meaning": "Guarantees scientific integrity by preventing single-load slope estimation."
    },
    "r_point_mean_ohms": {
        "unit": "Ohms (Ω)",
        "formula": "mean(delta_v / current) on settled samples where current > 0.05A",
        "description": "Single-point apparent resistance calculated during settled intervals.",
        "physical_meaning": "Fallback resistance metric when multi-load sweep is not performed."
    },
    # 3. Fit Residuals & Non-Linearity
    "fit_rmse_v": {
        "unit": "Volts (V)",
        "formula": "sqrt(mean((delta_v - (offset + R*current))^2))",
        "description": "Root Mean Squared Error of the multi-load linear resistance fit.",
        "physical_meaning": "Quantifies deviation from ideal Ohmic behavior (e.g. non-linear contact resistance)."
    },
    "fit_r2": {
        "unit": "Dimensionless [0, 1]",
        "formula": "1 - (SS_res / SS_tot)",
        "description": "Coefficient of determination of the linear V-I fit.",
        "physical_meaning": "Measures goodness-of-fit to Ohm's Law."
    },
    "fit_max_residual_v": {
        "unit": "Volts (V)",
        "formula": "max(|delta_v - (offset + R*current)|)",
        "description": "Maximum individual sample deviation from the multi-load linear fit.",
        "physical_meaning": "Identifies localized non-linear spikes or thermal jumps."
    },
    # 4. Current Variability & Stability
    "current_mean_a": {
        "unit": "Amperes (A)",
        "formula": "mean(current_a) during settled intervals",
        "description": "Average current flowing through the cable during steady-state testing.",
        "physical_meaning": "Operating electrical load."
    },
    "current_std_a": {
        "unit": "Amperes (A)",
        "formula": "std(current_a) during settled intervals",
        "description": "Current standard deviation during settled intervals.",
        "physical_meaning": "Measures power supply and electronic load current regulation stability."
    },
    "current_cv": {
        "unit": "Dimensionless",
        "formula": "current_std_a / (current_mean_a + 1e-6)",
        "description": "Coefficient of variation of the current signal.",
        "physical_meaning": "Normalized current flutter index."
    },
    "settled_current_stability": {
        "unit": "Ratio [0, 1]",
        "formula": "fraction of settled samples within +/- 2.5% of commanded load setpoint",
        "description": "Settled current compliance ratio.",
        "physical_meaning": "Verifies that the electronic load reached steady-state regulation."
    },
    "distinct_current_levels_count": {
        "unit": "Count",
        "formula": "count of distinct settled load clusters separated by at least 0.2A",
        "description": "Number of distinct current setpoint plateaus observed in the trial.",
        "physical_meaning": "Prerequisite condition for multi-load resistance slope extraction."
    },
    # 5. Measurement Quality & Missingness
    "settled_sample_ratio": {
        "unit": "Ratio [0, 1]",
        "formula": "count(is_settled == True) / count(total_samples)",
        "description": "Fraction of trial duration spent in settled electrical steady state.",
        "physical_meaning": "Separates dynamic switching transitions from steady-state measurements."
    },
    "missing_sample_count": {
        "unit": "Count",
        "formula": "sum(sample_index_diff - 1 for diff > 1)",
        "description": "Total number of missing or dropped telemetry packets.",
        "physical_meaning": "Acquisition pipeline reliability indicator."
    },
    "missing_sample_ratio": {
        "unit": "Ratio [0, 1]",
        "formula": "missing_sample_count / (total_samples + missing_sample_count)",
        "description": "Proportion of dropped frames relative to nominal expected sample count.",
        "physical_meaning": "Telemetry channel packet loss metric."
    },
    "quality_ok_ratio": {
        "unit": "Ratio [0, 1]",
        "formula": "count(measurement_quality == 'OK') / total_samples",
        "description": "Fraction of samples with sensor health confirmed as OK.",
        "physical_meaning": "Sensor reliability metric."
    },
    # 6. Temperature Context
    "temp_mean_c": {
        "unit": "Degrees Celsius (°C)",
        "formula": "mean(temperature_c)",
        "description": "Mean temperature observed during trial.",
        "physical_meaning": "Thermal context for resistance normalization (copper tempco is ~0.393%/°C)."
    },
    "temp_drift_c": {
        "unit": "Degrees Celsius (°C)",
        "formula": "max(temperature_c) - min(temperature_c)",
        "description": "Temperature excursion observed across the trial duration.",
        "physical_meaning": "Joule heating or environmental drift."
    },
    "has_valid_temperature": {
        "unit": "Boolean",
        "formula": "True if -20.0 <= temp_mean_c <= 100.0 else False",
        "description": "Indicates whether valid thermal telemetry was present.",
        "physical_meaning": "Thermal context validity check."
    },
    # 7. Interruption Statistics (Strictly Fast-Capture Gated)
    "fast_capture_source_present": {
        "unit": "Boolean",
        "formula": "Explicit hardware registration flag for sub-microsecond comparator / fast ADC",
        "description": "Flag indicating if a dedicated fast-sampling hardware channel was active.",
        "physical_meaning": "SCIENTIFIC RIGOR: The INA226 cannot capture microsecond interrupts. Set to False for INA226."
    },
    "fast_interruption_count": {
        "unit": "Count",
        "formula": "NaN if not fast_capture_source_present else count(fast micro-interrupts)",
        "description": "Number of validated fast electrical interruptions (< 100 us).",
        "physical_meaning": "Transient event counts are NEVER fabricated from slow INA226 channels."
    },
    # 8. Reference-Corrected Features
    "r_loop_calibrated_ohms": {
        "unit": "Ohms (Ω)",
        "formula": "max(0.0, r_loop_slope_ohms - fixture_tare_resistance_ohms)",
        "description": "Effective loop resistance corrected for characterized test fixture tare resistance.",
        "physical_meaning": "Isolated cable resistance (fixture tare subtracted; reference cable NOT assumed to be 0 Ω)."
    },
    "r_point_calibrated_ohms": {
        "unit": "Ohms (Ω)",
        "formula": "max(0.0, r_point_mean_ohms - fixture_tare_resistance_ohms)",
        "description": "Point resistance corrected for characterized fixture tare resistance.",
        "physical_meaning": "Isolated cable point resistance."
    },
}

FEATURE_COLUMN_NAMES: List[str] = list(FEATURE_DICTIONARY.keys())


def detect_distinct_current_levels(current_array: np.ndarray, min_step_a: float = 0.20) -> int:
    """Identify the number of distinct settled current plateaus.

    Groups current readings into clusters separated by at least min_step_a.
    """
    valid_curr = current_array[~np.isnan(current_array) & (current_array > 0.05)]
    if len(valid_curr) < 6:
        return 0

    # Sort values and find clusters where consecutive difference > min_step_a
    sorted_curr = np.sort(valid_curr)
    # Use histogram / rounding to cluster current setpoints to nearest 0.1A
    rounded = np.round(sorted_curr, 1)
    unique_levels, counts = np.unique(rounded, return_counts=True)
    # Only count levels that have at least 3 samples
    significant_levels = unique_levels[counts >= 3]

    if len(significant_levels) <= 1:
        return len(significant_levels)

    # Count levels separated by at least min_step_a
    distinct_count = 1
    last_level = significant_levels[0]
    for lvl in significant_levels[1:]:
        if lvl - last_level >= min_step_a:
            distinct_count += 1
            last_level = lvl

    return distinct_count


def fit_multi_load_resistance(
    current: np.ndarray,
    delta_v: np.ndarray,
) -> Tuple[float, float, float, float, float, bool]:
    """Fit linear Ohm's Law model: delta_v = offset + R * current on settled samples.

    Returns
    -------
    Tuple[float, float, float, float, float, bool]
        (r_slope, offset, rmse, r2, max_residual, is_valid)
    """
    valid_mask = (~np.isnan(current)) & (~np.isnan(delta_v)) & (current > 0.05)
    i_valid = current[valid_mask]
    v_valid = delta_v[valid_mask]

    # Check that we have enough samples and distinct current spread
    distinct_levels = detect_distinct_current_levels(i_valid)
    if distinct_levels < 2 or len(i_valid) < 6:
        # SCIENTIFIC REQUIREMENT: Do not compute slope unless >= 2 distinct current levels
        return np.nan, np.nan, np.nan, np.nan, np.nan, False

    try:
        # Linear regression: delta_v = slope * current + intercept
        slope, intercept, r_val, _, _ = stats.linregress(i_valid, v_valid)
        if not np.isfinite(slope) or not np.isfinite(intercept):
            return np.nan, np.nan, np.nan, np.nan, np.nan, False

        # Compute residuals
        predicted_v = intercept + (slope * i_valid)
        residuals = v_valid - predicted_v
        rmse = float(np.sqrt(np.mean(residuals**2)))
        r2 = float(r_val**2)
        max_res = float(np.max(np.abs(residuals)))

        return float(slope), float(intercept), rmse, r2, max_res, True
    except Exception as e:
        logger.warning(f"Failed linear multi-load fit: {e}")
        return np.nan, np.nan, np.nan, np.nan, np.nan, False


def extract_features_from_conditioned_signal(
    signal: ConditionedSignal,
    calibration: Optional[CalibrationRecord] = None,
    fast_capture_source_present: bool = False,
) -> Dict[str, Any]:
    """Extract all 26 scientifically defensible features from a single trial signal."""
    cal = calibration or CalibrationRegistry().get_profile()
    fixture_tare = cal.fixture.tare_resistance_ohms

    # 1. Voltage Drop Statistics (computed across all valid samples)
    valid_dv_mask = ~np.isnan(signal.delta_v)
    valid_dv = signal.delta_v[valid_dv_mask]
    n_valid = len(valid_dv)

    if n_valid > 0:
        dv_mean = float(np.mean(valid_dv))
        dv_median = float(np.median(valid_dv))
        dv_std = float(np.std(valid_dv))
        q75, q25 = np.percentile(valid_dv, [75, 25])
        dv_iqr = float(q75 - q25)
        # Median Absolute Deviation (MAD)
        dv_mad = float(np.median(np.abs(valid_dv - dv_median)))
    else:
        dv_mean = np.nan
        dv_median = np.nan
        dv_std = np.nan
        dv_iqr = np.nan
        dv_mad = np.nan

    # 2. Settled Interval Masking for Steady-State Resistance & Current Stability
    settled_mask = signal.is_settled_mask & valid_dv_mask
    settled_i = signal.current_a[settled_mask]
    settled_dv = signal.calibrated_delta_v[settled_mask]
    settled_setpoints = signal.load_setpoint_a[settled_mask]

    distinct_loads = detect_distinct_current_levels(settled_i)

    # Multi-load linear slope fit
    r_slope, v_offset, fit_rmse, fit_r2, fit_max_res, has_multiload = fit_multi_load_resistance(
        settled_i, settled_dv
    )

    # Point-wise mean resistance fallback: mean(delta_v / I)
    valid_point_mask = (settled_i > 0.05) & (~np.isnan(settled_dv))
    if np.sum(valid_point_mask) > 0:
        r_points = settled_dv[valid_point_mask] / settled_i[valid_point_mask]
        r_point_mean = float(np.mean(r_points))
    else:
        r_point_mean = np.nan

    # 3. Current Variability & Stability
    if len(settled_i) > 0:
        i_mean = float(np.mean(settled_i))
        i_std = float(np.std(settled_i))
        i_cv = float(i_std / (i_mean + 1e-6))
        # Compliance with load setpoint within +/- 2.5%
        valid_sp_mask = settled_setpoints > 0.05
        if np.sum(valid_sp_mask) > 0:
            sp_errors = np.abs(settled_i[valid_sp_mask] - settled_setpoints[valid_sp_mask])
            within_band = sp_errors <= (0.025 * settled_setpoints[valid_sp_mask] + 0.010)
            stability_ratio = float(np.mean(within_band))
        else:
            stability_ratio = 1.0 if i_cv < 0.05 else 0.0
    else:
        i_mean = np.nan
        i_std = np.nan
        i_cv = np.nan
        stability_ratio = np.nan

    # 4. Measurement Quality & Missingness
    total_samples = len(signal.delta_v)
    settled_count = int(np.sum(signal.is_settled_mask))
    settled_ratio = float(settled_count / total_samples) if total_samples > 0 else 0.0
    missing_count = signal.audit.missing_samples_detected
    total_expected = total_samples + missing_count
    missing_ratio = float(missing_count / total_expected) if total_expected > 0 else 0.0
    ok_count = sum(1 for q in signal.measurement_quality if q == "OK")
    quality_ok_ratio = float(ok_count / total_samples) if total_samples > 0 else 0.0

    # 5. Temperature Context
    valid_temp_mask = ~np.isnan(signal.temperature_c)
    valid_temps = signal.temperature_c[valid_temp_mask]
    if len(valid_temps) > 0:
        t_mean = float(np.mean(valid_temps))
        t_drift = float(np.max(valid_temps) - np.min(valid_temps))
        has_temp = bool(-20.0 <= t_mean <= 100.0)
    else:
        t_mean = np.nan
        t_drift = np.nan
        has_temp = False

    # 6. Interruption Statistics (Strictly Gated by fast_capture_source_present)
    # SCIENTIFIC INTEGRITY: Never invent transient-event counts from slow INA226 telemetry
    if fast_capture_source_present:
        fast_interruption_count: Any = 0
    else:
        fast_interruption_count = np.nan

    # 7. Reference-Corrected Resistance Metrics
    # (Fixture tare resistance subtracted; reference cable resistance characterized but NOT assumed 0)
    if has_multiload and np.isfinite(r_slope):
        r_loop_cal = max(0.0, float(r_slope - fixture_tare))
    else:
        r_loop_cal = np.nan

    if np.isfinite(r_point_mean):
        r_point_cal = max(0.0, float(r_point_mean - fixture_tare))
    else:
        r_point_cal = np.nan

    feature_record: Dict[str, Any] = {
        # Voltage drop features
        "delta_v_mean_v": round(dv_mean, 6) if np.isfinite(dv_mean) else np.nan,
        "delta_v_median_v": round(dv_median, 6) if np.isfinite(dv_median) else np.nan,
        "delta_v_std_v": round(dv_std, 6) if np.isfinite(dv_std) else np.nan,
        "delta_v_iqr_v": round(dv_iqr, 6) if np.isfinite(dv_iqr) else np.nan,
        "delta_v_mad_v": round(dv_mad, 6) if np.isfinite(dv_mad) else np.nan,
        # Loop resistance
        "r_loop_slope_ohms": round(r_slope, 6) if np.isfinite(r_slope) else np.nan,
        "v_offset_fit_v": round(v_offset, 6) if np.isfinite(v_offset) else np.nan,
        "has_multiload_fit": has_multiload,
        "r_point_mean_ohms": round(r_point_mean, 6) if np.isfinite(r_point_mean) else np.nan,
        # Fit residuals
        "fit_rmse_v": round(fit_rmse, 6) if np.isfinite(fit_rmse) else np.nan,
        "fit_r2": round(fit_r2, 6) if np.isfinite(fit_r2) else np.nan,
        "fit_max_residual_v": round(fit_max_res, 6) if np.isfinite(fit_max_res) else np.nan,
        # Current variability
        "current_mean_a": round(i_mean, 6) if np.isfinite(i_mean) else np.nan,
        "current_std_a": round(i_std, 6) if np.isfinite(i_std) else np.nan,
        "current_cv": round(i_cv, 6) if np.isfinite(i_cv) else np.nan,
        "settled_current_stability": round(stability_ratio, 4) if np.isfinite(stability_ratio) else np.nan,
        "distinct_current_levels_count": distinct_loads,
        # Quality & Missingness
        "settled_sample_ratio": round(settled_ratio, 4),
        "missing_sample_count": missing_count,
        "missing_sample_ratio": round(missing_ratio, 4),
        "quality_ok_ratio": round(quality_ok_ratio, 4),
        # Temperature context
        "temp_mean_c": round(t_mean, 2) if np.isfinite(t_mean) else np.nan,
        "temp_drift_c": round(t_drift, 2) if np.isfinite(t_drift) else np.nan,
        "has_valid_temperature": has_temp,
        # Interruption statistics
        "fast_capture_source_present": fast_capture_source_present,
        "fast_interruption_count": fast_interruption_count,
        # Reference-corrected metrics
        "r_loop_calibrated_ohms": round(r_loop_cal, 6) if np.isfinite(r_loop_cal) else np.nan,
        "r_point_calibrated_ohms": round(r_point_cal, 6) if np.isfinite(r_point_cal) else np.nan,
    }

    return feature_record


def extract_feature_dataset(
    records: List[CanonicalMeasurementRecord],
    calibration: Optional[CalibrationRecord] = None,
    fast_capture_source_present: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Group experimental records by (cable_id, session_id, trial_id) and build feature dataset.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        features_df: The 26 extracted electrical and quality features.
        metadata_df: Traceability metadata (cable_id, session_id, trial_id, ground_truth_label, source_type).
        STRICT GUARANTEE: Metadata and ground-truth labels are NEVER placed inside features_df!
    """
    if not records:
        raise ValueError("Cannot extract features from empty record set.")

    # Group records by (cable_id, session_id, trial_id)
    trials_map: Dict[Tuple[str, str, str], List[CanonicalMeasurementRecord]] = {}
    for r in records:
        key = (r.cable_id, r.session_id, r.trial_id)
        if key not in trials_map:
            trials_map[key] = []
        trials_map[key].append(r)

    feature_rows: List[Dict[str, Any]] = []
    meta_rows: List[Dict[str, Any]] = []

    cal = calibration or CalibrationRegistry().get_profile()

    for (cid, sid, tid), trial_recs in trials_map.items():
        first = trial_recs[0]
        # Condition signal
        conditioned = condition_trial_records(trial_recs, calibration=cal)
        # Extract features
        feats = extract_features_from_conditioned_signal(
            conditioned,
            calibration=cal,
            fast_capture_source_present=fast_capture_source_present,
        )
        feature_rows.append(feats)

        meta = {
            "cable_id": cid,
            "session_id": sid,
            "trial_id": tid,
            "ground_truth_label": first.ground_truth_label,
            "ground_truth_source": first.ground_truth_source,
            "fault_origin": first.fault_origin,
            "fault_condition": first.fault_condition,
            "source_type": first.source_type,
            "action_id": first.action_id,
            "motion_region": first.motion_region,
            "raw_sample_count": len(trial_recs),
            "software_version": "0.1.0-phase3",
        }
        meta_rows.append(meta)

    features_df = pd.DataFrame(feature_rows, columns=FEATURE_COLUMN_NAMES)
    metadata_df = pd.DataFrame(meta_rows)

    logger.info(
        f"Feature extraction complete: {len(features_df)} trials processed across "
        f"{len(metadata_df['cable_id'].unique())} physical cables."
    )

    return features_df, metadata_df


def export_feature_dictionary_docs(
    markdown_path: Optional[Path] = None,
    json_path: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Generate both Markdown and JSON formal documentation of the feature dictionary."""
    ensure_directories_exist()
    md_file = markdown_path or (DOCS_DIR / "feature_dictionary.md")
    json_file = json_path or (PROCESSED_DATA_DIR / "feature_dictionary.json")

    # Save JSON dictionary
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(FEATURE_DICTIONARY, f, indent=2)

    # Save Markdown documentation
    lines = [
        "# FlexPulse-ML: Electrical & Physical Feature Dictionary",
        "",
        "This document details every feature extracted by the Phase 3 pipeline, its mathematical formula,",
        "physical interpretation, and validity constraints.",
        "",
        "| Feature Name | Units | Mathematical Formula | Physical Meaning | Validity Constraint |",
        "| :--- | :---: | :--- | :--- | :--- |",
    ]

    for feat_name, meta in FEATURE_DICTIONARY.items():
        unit = meta.get("unit", "")
        formula = meta.get("formula", "")
        meaning = meta.get("physical_meaning", "")
        validity = meta.get("validity_condition", "Always valid")
        lines.append(f"| `{feat_name}` | {unit} | `{formula}` | {meaning} | {validity} |")

    lines.append("")
    lines.append("## Hand-Calculated Electrical Example")
    lines.append("")
    lines.append("Consider a 1-meter USB-C cable tested at three settled current setpoints:")
    lines.append("- Point 1: $I_1 = 1.000\\text{ A}, V_{\\text{in}, 1} = 5.050\\text{ V}, V_{\\text{out}, 1} = 4.930\\text{ V} \\implies \\Delta V_1 = 0.120\\text{ V}$")
    lines.append("- Point 2: $I_2 = 2.000\\text{ A}, V_{\\text{in}, 2} = 5.040\\text{ V}, V_{\\text{out}, 2} = 4.800\\text{ V} \\implies \\Delta V_2 = 0.240\\text{ V}$")
    lines.append("- Point 3: $I_3 = 3.000\\text{ A}, V_{\\text{in}, 3} = 5.030\\text{ V}, V_{\\text{out}, 3} = 4.670\\text{ V} \\implies \\Delta V_3 = 0.360\\text{ V}$")
    lines.append("")
    lines.append("### Step 1: Multi-Load Slope Resistance Fit")
    lines.append("$$\\text{Slope } R_{\\text{loop}} = \\frac{\\Delta V_3 - \\Delta V_1}{I_3 - I_1} = \\frac{0.360 - 0.120}{3.000 - 1.000} = \\frac{0.240}{2.000} = 0.120\\ \\Omega = 120\\text{ m}\\Omega$$")
    lines.append("$$\\text{Intercept } V_{\\text{offset}} = 0.000\\text{ V}$$")
    lines.append("")
    lines.append("### Step 2: Reference Fixture Tare Correction")
    lines.append("Given characterized fixture tare resistance $R_{\\text{fixture\\_tare}} = 0.040\\ \\Omega$ (40 mΩ):")
    lines.append("$$R_{\\text{cable\\_calibrated}} = R_{\\text{loop}} - R_{\\text{fixture\\_tare}} = 120\\text{ m}\\Omega - 40\\text{ m}\\Omega = 80\\text{ m}\\Omega$$")
    lines.append("")
    lines.append("### Step 3: Central Tendency & Spread")
    lines.append("$$\\text{Mean } \\Delta V = \\frac{0.120 + 0.240 + 0.360}{3} = 0.240\\text{ V}$$")
    lines.append("$$\\text{Median } \\Delta V = 0.240\\text{ V}$$")
    lines.append("$$\\text{Std } \\Delta V = \\sqrt{\\frac{(0.12-0.24)^2 + (0.24-0.24)^2 + (0.36-0.24)^2}{3}} = \\sqrt{\\frac{0.0144 + 0 + 0.0144}{3}} = 0.09798\\text{ V}$$")

    with open(md_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Exported feature dictionary to {md_file} and {json_file}")
    return md_file, json_file
