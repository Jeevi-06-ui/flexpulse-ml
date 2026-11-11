# FlexPulse-ML: Electrical & Physical Feature Dictionary

This document details every feature extracted by the Phase 3 pipeline, its mathematical formula,
physical interpretation, and validity constraints.

| Feature Name | Units | Mathematical Formula | Physical Meaning | Validity Constraint |
| :--- | :---: | :--- | :--- | :--- |
| `delta_v_mean_v` | Volts (V) | `mean(vin_v - vout_v)` | Primary indicator of total series path impedance. | Always valid |
| `delta_v_median_v` | Volts (V) | `median(vin_v - vout_v)` | Robust central tendency resistant to transient measurement spikes. | Always valid |
| `delta_v_std_v` | Volts (V) | `std(vin_v - vout_v)` | Measures voltage fluctuations caused by mechanical flex or contact flutter. | Always valid |
| `delta_v_iqr_v` | Volts (V) | `Q75(delta_v) - Q25(delta_v)` | Robust measure of voltage dispersion unaffected by extreme outlier glitches. | Always valid |
| `delta_v_mad_v` | Volts (V) | `median(|delta_v - median(delta_v)|)` | Scale-invariant robust spread metric for non-Gaussian noise. | Always valid |
| `r_loop_slope_ohms` | Ohms (Ω) | `Slope R from linear fit: delta_v = offset + R * current (settled intervals only)` | True dynamic DC loop resistance independent of contact zero-offsets. | Computed ONLY when distinct settled current levels >= 2. |
| `v_offset_fit_v` | Volts (V) | `Intercept of linear fit: delta_v = offset + R * current` | Characterizes thermoelectric EMF or ADC instrumentation bias. | Always valid |
| `has_multiload_fit` | Boolean | `distinct_current_levels >= 2 and settled_samples >= 6` | Guarantees scientific integrity by preventing single-load slope estimation. | Always valid |
| `r_point_mean_ohms` | Ohms (Ω) | `mean(delta_v / current) on settled samples where current > 0.05A` | Fallback resistance metric when multi-load sweep is not performed. | Always valid |
| `fit_rmse_v` | Volts (V) | `sqrt(mean((delta_v - (offset + R*current))^2))` | Quantifies deviation from ideal Ohmic behavior (e.g. non-linear contact resistance). | Always valid |
| `fit_r2` | Dimensionless [0, 1] | `1 - (SS_res / SS_tot)` | Measures goodness-of-fit to Ohm's Law. | Always valid |
| `fit_max_residual_v` | Volts (V) | `max(|delta_v - (offset + R*current)|)` | Identifies localized non-linear spikes or thermal jumps. | Always valid |
| `current_mean_a` | Amperes (A) | `mean(current_a) during settled intervals` | Operating electrical load. | Always valid |
| `current_std_a` | Amperes (A) | `std(current_a) during settled intervals` | Measures power supply and electronic load current regulation stability. | Always valid |
| `current_cv` | Dimensionless | `current_std_a / (current_mean_a + 1e-6)` | Normalized current flutter index. | Always valid |
| `settled_current_stability` | Ratio [0, 1] | `fraction of settled samples within +/- 2.5% of commanded load setpoint` | Verifies that the electronic load reached steady-state regulation. | Always valid |
| `distinct_current_levels_count` | Count | `count of distinct settled load clusters separated by at least 0.2A` | Prerequisite condition for multi-load resistance slope extraction. | Always valid |
| `settled_sample_ratio` | Ratio [0, 1] | `count(is_settled == True) / count(total_samples)` | Separates dynamic switching transitions from steady-state measurements. | Always valid |
| `missing_sample_count` | Count | `sum(sample_index_diff - 1 for diff > 1)` | Acquisition pipeline reliability indicator. | Always valid |
| `missing_sample_ratio` | Ratio [0, 1] | `missing_sample_count / (total_samples + missing_sample_count)` | Telemetry channel packet loss metric. | Always valid |
| `quality_ok_ratio` | Ratio [0, 1] | `count(measurement_quality == 'OK') / total_samples` | Sensor reliability metric. | Always valid |
| `temp_mean_c` | Degrees Celsius (°C) | `mean(temperature_c)` | Thermal context for resistance normalization (copper tempco is ~0.393%/°C). | Always valid |
| `temp_drift_c` | Degrees Celsius (°C) | `max(temperature_c) - min(temperature_c)` | Joule heating or environmental drift. | Always valid |
| `has_valid_temperature` | Boolean | `True if -20.0 <= temp_mean_c <= 100.0 else False` | Thermal context validity check. | Always valid |
| `fast_capture_source_present` | Boolean | `Explicit hardware registration flag for sub-microsecond comparator / fast ADC` | SCIENTIFIC RIGOR: The INA226 cannot capture microsecond interrupts. Set to False for INA226. | Always valid |
| `fast_interruption_count` | Count | `NaN if not fast_capture_source_present else count(fast micro-interrupts)` | Transient event counts are NEVER fabricated from slow INA226 channels. | Always valid |
| `r_loop_calibrated_ohms` | Ohms (Ω) | `max(0.0, r_loop_slope_ohms - fixture_tare_resistance_ohms)` | Isolated cable resistance (fixture tare subtracted; reference cable NOT assumed to be 0 Ω). | Always valid |
| `r_point_calibrated_ohms` | Ohms (Ω) | `max(0.0, r_point_mean_ohms - fixture_tare_resistance_ohms)` | Isolated cable point resistance. | Always valid |

## Hand-Calculated Electrical Example

Consider a 1-meter USB-C cable tested at three settled current setpoints:
- Point 1: $I_1 = 1.000\text{ A}, V_{\text{in}, 1} = 5.050\text{ V}, V_{\text{out}, 1} = 4.930\text{ V} \implies \Delta V_1 = 0.120\text{ V}$
- Point 2: $I_2 = 2.000\text{ A}, V_{\text{in}, 2} = 5.040\text{ V}, V_{\text{out}, 2} = 4.800\text{ V} \implies \Delta V_2 = 0.240\text{ V}$
- Point 3: $I_3 = 3.000\text{ A}, V_{\text{in}, 3} = 5.030\text{ V}, V_{\text{out}, 3} = 4.670\text{ V} \implies \Delta V_3 = 0.360\text{ V}$

### Step 1: Multi-Load Slope Resistance Fit
$$\text{Slope } R_{\text{loop}} = \frac{\Delta V_3 - \Delta V_1}{I_3 - I_1} = \frac{0.360 - 0.120}{3.000 - 1.000} = \frac{0.240}{2.000} = 0.120\ \Omega = 120\text{ m}\Omega$$
$$\text{Intercept } V_{\text{offset}} = 0.000\text{ V}$$

### Step 2: Reference Fixture Tare Correction
Given characterized fixture tare resistance $R_{\text{fixture\_tare}} = 0.040\ \Omega$ (40 mΩ):
$$R_{\text{cable\_calibrated}} = R_{\text{loop}} - R_{\text{fixture\_tare}} = 120\text{ m}\Omega - 40\text{ m}\Omega = 80\text{ m}\Omega$$

### Step 3: Central Tendency & Spread
$$\text{Mean } \Delta V = \frac{0.120 + 0.240 + 0.360}{3} = 0.240\text{ V}$$
$$\text{Median } \Delta V = 0.240\text{ V}$$
$$\text{Std } \Delta V = \sqrt{\frac{(0.12-0.24)^2 + (0.24-0.24)^2 + (0.36-0.24)^2}{3}} = \sqrt{\frac{0.0144 + 0 + 0.0144}{3}} = 0.09798\text{ V}$$