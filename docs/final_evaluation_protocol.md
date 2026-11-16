# FlexPulse-ML: Frozen Final Evaluation Protocol

**Document ID:** `DOC-PROTOCOL-FROZEN-2026-V1`  
**Protocol Status:** **LOCKED AND FROZEN** (Pre-Test Specification)  
**Creation Timestamp:** 2026-11-16T04:50:00Z  
**Project Title:** *FlexPulse-ML: A Simulation-Based Self-Calibrating Machine Learning Framework for Intermittent Fault Detection in USB-C-Inspired Power Paths*  
**Academic Context:** Presidency University, Bengaluru  

---

## 1. Objective and Methodological Pre-Registration

This document establishes the frozen experimental protocol for the final independent evaluation of FlexPulse-ML. In accordance with rigorous scientific methodology:
1. All hyperparameters, threshold values, model weights, feature transformations, and evaluation rules are pre-specified and permanently frozen using training and validation development data only.
2. The final evaluation dataset (`flexpulse_final_test_v4`) will be generated under independent random seeds and evaluated exactly once using the frozen pipelines.
3. No hyperparameter or algorithmic tuning will be performed against final test data. All results (favorable or unfavorable) will be reported in full.

---

## 2. Frozen Model Versions and Pipeline Artifacts

The final comparative benchmark evaluates 6 standardized detection architectures:

| Method Identifier | Full Method Name | Serialized Model File | Preprocessing Representation |
| :--- | :--- | :--- | :--- |
| **M1 (A0)** | Fixed Threshold Baseline | `models/phase5/a0_fixed_threshold.joblib` | Raw unnormalized features ($\mu \pm 2.5\sigma$ envelope) |
| **M2 (A1)** | Static Machine Learning | `models/phase5/a1_static_random_forest.joblib` | Raw unnormalized features (100 trees, depth 8) |
| **M3 (A2)** | Initial Reference Calibration + ML | `models/phase5/a2_a3_normalized_random_forest.joblib` | Zero-centered features relative to initial 5.0 s baseline |
| **M4 (A3)** | Confidence-Gated Adaptive ML | `models/phase5/a2_a3_normalized_random_forest.joblib` | Online dynamic baseline centering with 6 safety gates |
| **M5 (A2+P)** | Initial Calibration + Persistence | `models/phase5/a2_a3_normalized_random_forest.joblib` | Initial baseline centering + causal 2-window persistence |
| **M6 (A3+P)** | Adaptive Calibration + Persistence | `models/phase5/a2_a3_normalized_random_forest.joblib` | Adaptive baseline centering + causal 2-window persistence |

---

## 3. Mathematical Simulation and Physical Parameter Ranges

### Power-Path Circuit Formulation
$$R_{\text{path}}(t) = R_0 + R_{\text{contact}}(t) + R_{\text{fault}}(t)$$
$$I(t) = \frac{V_s(t)}{R_{\text{load}}(t) + R_{\text{path}}(t)}, \quad V_{\text{load}}(t) = I(t) \cdot R_{\text{load}}(t)$$
$$V_{\text{meas}}(t) = V_{\text{load}}(t) + b_V(t) + \eta_V(t), \quad I_{\text{meas}}(t) = I(t) + b_I(t) + \eta_I(t)$$

### Physical Parameter Specification for Final Test Scenarios

| Scenario ID | Scenario Name | Base $R_0 \, (\Omega)$ | Nominal Load $R_L \, (\Omega)$ | Noise $\sigma_V, \sigma_I$ | Fault Type & Characteristics |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **S1** | `HEALTHY_NOMINAL` | 0.080 | 4.00 (1.25 A) | 0.005 V, 0.006 A | Healthy reference run (0 fault events) |
| **S2** | `INCREASED_PATH_RESISTANCE` | 0.220 | 4.00 (1.20 A) | 0.006 V, 0.008 A | Intermittent contact jumps ($\Delta R = 0.400\,\Omega$) |
| **S3** | `INTERMITTENT_CONTACT_FAULTS` | 0.080 | 4.00 (1.25 A) | 0.005 V, 0.006 A | Sharp micro-interruptions ($\Delta R = 0.500\,\Omega$, $0.25-2.0\,\text{s}$) |
| **S4** | `GRADUAL_RESISTANCE_DRIFT` | 0.080 | 4.00 (1.25 A) | 0.006 V, 0.008 A | Linear resistance creep ($+0.003\,\Omega/\text{s}$ after 5.0 s) |
| **S5** | `MEASUREMENT_OFFSET_DRIFT` | 0.080 | 4.00 (1.25 A) | 0.006 V, 0.008 A | Linear voltage sensor bias drift ($+0.002\,\text{V/s}$ after 5.0 s) |
| **S6** | `MEASUREMENT_NOISE_SHIFT` | 0.080 | 4.00 (1.25 A) | 0.016 V, 0.024 A | 3x elevated noise floor + intermittent contact faults |
| **S7** | `LOAD_OPERATING_SHIFT` | 0.080 | 1.90 (2.55 A) | 0.007 V, 0.010 A | Continuous heavy load + intermittent contact faults |
| **S8** | `ABRUPT_LOAD_CHANGE` | 0.080 | 4.00 $\to$ 2.00 | 0.006 V, 0.008 A | Sudden healthy load jump ($1.25\,\text{A} \to 2.45\,\text{A}$) at $t = 25.0\,\text{s}$ |
| **S9** | `SLOW_DEVELOPING_FAULT` | 0.080 | 4.00 (1.25 A) | 0.005 V, 0.006 A | Slow fault creep ($0 \to 0.450\,\Omega$ from $t=15\,\text{s}$ to $40\,\text{s}$) |
| **S10** | `COMBINED_OPERATING_SHIFTS` | 0.120 | 3.50 $\to$ 2.20 | 0.014 V, 0.020 A | Offset drift + load transition + elevated noise + faults |

---

## 4. Feature Extraction and Normalization Rules

* **Windowing:** Sampling rate $f_s = 100\,\text{Hz}$, duration $T_w = 0.50\,\text{s}$ (50 samples), stride $T_s = 0.25\,\text{s}$ (25 samples, 50% overlap).
* **26 Observable Features:** Extracted strictly from observable signals $V_{\text{meas}}$ and $I_{\text{meas}}$.
* **Level Subtraction Rule:**
  * Voltage level features (`voltage_mean_v`, `voltage_median_v`, `voltage_min_v`, `voltage_max_v`, `voltage_p10_v`, `voltage_p90_v`) undergo baseline centering: $\tilde{V} = V - \hat{V}_0$.
  * Current level features (`current_mean_a`, `current_median_a`, `current_min_a`, `current_max_a`, `current_p10_a`, `current_p90_a`) undergo baseline centering: $\tilde{I} = I - \hat{I}_0$.
  * Spread features (`std`, `range`, `mad`), rate features (`slope`), and difference features (`abs_mean`, `std_diff`, `abs_max`) remain unaltered.

---

## 5. Frozen Calibration Parameters and Multi-Stage Safety Gates

* **Initial Calibration Window:** $t \in [0.0, 5.0]\,\text{s}$ (minimum 8 valid windows).
* **Adaptation Update Factor:** $\alpha = 0.04$
* **Confidence Gate Threshold:** $\tau_{\text{conf}} = 0.85$ (calibrated from development data)
* **Current Stability Limit:** $\sigma_{I,\max} = 0.040\,\text{A}$
* **Outlier Residual Multiplier:** $2.5 \cdot \text{MAD}_V$
* **Maximum Step Limit:** $|\Delta \hat{V}_0| \le 0.03\,\text{V}$
* **Anomaly Cooldown:** $k = 4$ windows
* **Recovery Rule:** $N_{\text{recovery}} = 5$ consecutive valid healthy windows to exit `FROZEN` state.

---

## 6. Causal Detection and Temporal Persistence Specification

* **Causal Decision Timestamp:** A window spanning $[w_{\text{start}}, w_{\text{end}}]$ produces inference strictly at $t_{\text{decision}} = w_{\text{end}}$.
* **Causal Delay Definition:**
  $$\text{delay}_{\text{causal}} = \max(0.0, w_{\text{end}} - t_{\text{event\_start}})$$
* **Temporal Persistence Filter ($N=2$):** An alarm is triggered if and only if two consecutive sliding windows are predicted positive:
  $$\hat{y}_{\text{persist}}^{(k)} = \hat{y}^{(k)} \land \hat{y}^{(k-1)}$$
* **Event Matching Definition:** A ground-truth event $[t_{\text{start}}, t_{\text{end}}]$ is considered detected if a predicted interval overlaps with $[t_{\text{start}}, t_{\text{end}}]$ or has $|w_{\text{start}} - t_{\text{start}}| \le 0.50\,\text{s}$. Unmatched predicted intervals are counted as false alarm events. Gap merging threshold between consecutive predicted windows is $0.50\,\text{s}$.

---

## 7. Performance Metrics and Undefined Metric Policy

### Window-Level Metrics
* **Accuracy:** $(TP + TN) / (TP + TN + FP + FN)$
* **Balanced Accuracy:** $(\text{Recall} + \text{Specificity}) / 2$
* **Precision:** $TP / (TP + FP)$
* **Recall (Sensitivity):** $TP / (TP + FN)$
* **Specificity:** $TN / (TN + FP)$
* **False Positive Rate:** $FP / (FP + TN) = 1 - \text{Specificity}$
* **False Negative Rate:** $FN / (FN + TP) = 1 - \text{Recall}$
* **F1-Score:** $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$

### Healthy-Only Metric Policy (Part 4 Resolution)
In runs or scenarios where ground truth contains zero fault samples ($TP + FN = 0$):
* Recall, Precision, and F1-score are **mathematically undefined** ($0/0$).
* These metrics are reported strictly as **`N/A`** with an explanatory note.
* Specificity, False Positive Rate, false positive window counts, and false alarm events are reported as the primary valid performance indicators.

### Event-Level Metrics
* **Event Recall:** $\text{Detected Observable Events} / \text{Total Observable Events}$
* **False Alarms per Simulated Minute:** $\text{False Alarm Events} / (\text{Total Healthy Duration in Minutes})$
* **Median Causal Latency:** $\text{median}(\{w_{\text{end}} - t_{\text{start}}\})$

---

## 8. Statistical Comparison Methodology

* **Resampling Unit:** Complete independent simulation runs (never overlapping windows).
* **Iterations:** $B = 200$ bootstrap resamples.
* **Confidence Level:** 95% two-sided empirical percentile interval $[2.5\%, 97.5\%]$.
* **Paired Comparisons:**
  1. $\Delta \text{M2 vs M1}$ (Static ML vs Fixed Threshold)
  2. $\Delta \text{M3 vs M2}$ (Initial Calibration vs Static ML)
  3. $\Delta \text{M4 vs M3}$ (Adaptive Calibration vs Initial Calibration)
  4. $\Delta \text{M5 vs M3}$ (Persistence vs Raw Initial Calibration)
  5. $\Delta \text{M6 vs M4}$ (Persistence vs Raw Adaptive Calibration)
