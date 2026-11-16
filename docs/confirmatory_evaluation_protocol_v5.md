# FlexPulse-ML: Confirmatory Evaluation Protocol (v5)

**Document ID:** `DOC-PROTOCOL-CONFIRMATORY-2026-V5`  
**Protocol Status:** **LOCKED AND PRE-REGISTERED** (Pre-Test Specification)  
**Creation Date:** 16 November 2026  
**Project Title:** *FlexPulse-ML: A Simulation-Based Self-Calibrating Machine Learning Framework for Intermittent Fault Detection in USB-C-Inspired Power Paths*  
**Academic Context:** Presidency University, Bengaluru  
**Target Evaluation Dataset:** `flexpulse_final_test_v5` (Seed `20261202`)  

---

## 1. Executive Summary and Scope of Confirmatory Study

This protocol pre-registers the single, narrowly scoped confirmatory evaluation of the FlexPulse-ML framework on an independent evaluation dataset, designated `flexpulse_final_test_v5`. 

### Motivation
During the Phase 7 scientific audit and numerical verification, an implementation discrepancy was discovered in the synthetic data generator (`generator_v4.py`) for Scenario S4 (`GRADUAL_RESISTANCE_DRIFT`). While Scenario S4 was declared in the pre-registered protocol as a progressive power-path fault (linear resistance degradation), the generator modified `base_r` without activating ground truth fault indicators (`fault_labels = 1`) or registering event metadata (`events_meta`). Consequently, in `flexpulse_final_test_v4`, all simulated gradual degradation events during $t > 5.0\,\text{s}$ were evaluated against healthy ground-truth labels ($y=0$), artificially categorizing valid detections as false positives.

### Methodological Safeguards
To maintain strict scientific integrity and avoid p-hacking or retroactive tampering:
1. **No Retrospective Relabeling:** The historical dataset `flexpulse_final_test_v4` remains permanently frozen and locked.
2. **Zero Model Retraining:** All trained model artifacts (`models/phase5/`), decision thresholds, hyperparameters, feature extraction rules, and persistence filters remain strictly frozen.
3. **Independent Prospective Dataset:** An independent test dataset (`flexpulse_final_test_v5`) is generated using a distinct random seed (`20261202`) and the corrected S4 implementation.
4. **Transparent Comparative Reporting:** Results will be reported by contrasting historical v4 findings against prospective v5 confirmatory findings.

---

## 2. Physical and Mathematical Definition of Scenario S4

### Physical Grounding
In DC electrical power delivery paths (such as USB-C VBUS/GND contacts), connector wear, fretting corrosion, and surface oxidation cause contact resistance to drift upward over time. This represents an insidious fault condition that causes excessive thermal dissipation ($I^2 R$) and voltage drop under load.

### Mathematical Formulation
The instantaneous circuit power-path resistance is modeled as:
$$R_{\text{path}}(t) = R_{\text{base}} + R_{\text{contact}}(t) + R_{\text{fault}}(t)$$

In Scenario S4 (`GRADUAL_RESISTANCE_DRIFT`):
1. **Initial Calibration Window ($0.0 \le t \le 5.0\,\text{s}$):**
   $$R_{\text{fault}}(t) = 0, \quad y(t) = 0, \quad \text{Event ID} = \text{"NONE"}$$
   The circuit operates under healthy nominal conditions, providing a clean reference baseline for self-calibration.

2. **Gradual Degradation Period ($5.0 < t \le 60.0\,\text{s}$):**
   $$R_{\text{fault}}(t) = k_{\text{drift}} \cdot (t - 5.0)$$
   $$y(t) = 1, \quad \text{Event ID} = \text{"EV_GD_01"}, \quad \text{Fault Type} = \text{"GRADUAL_RESISTANCE_DRIFT"}$$
   where $k_{\text{drift}} \sim \mathcal{U}(0.002, 0.004)\,\Omega/\text{s}$.
   At $t = 60.0\,\text{s}$, the cumulative fault resistance added reaches $0.110\,\Omega$ to $0.220\,\Omega$.

3. **Event Metadata Record:**
   - `event_id`: `"EV_GD_01"`
   - `fault_type`: `"GRADUAL_RESISTANCE_DRIFT"`
   - `start_time_s`: $5.0\,\text{s}$
   - `end_time_s`: $60.0\,\text{s}$
   - `duration_s`: $55.0\,\text{s}$
   - `resistance_increase_ohm`: $k_{\text{drift}} \times 55.0$
   - `sampled_event`: `True`

All other scenarios (S1–S3, S5–S10) have been re-verified and remain unchanged.

---

## 3. Dataset Generation Specifications (`flexpulse_final_test_v5`)

- **Root Random Seed:** `20261202`
- **Total Simulation Runs:** 82 runs
  - **Train:** 6 nominal runs across base scenarios (reproducibility anchor)
  - **Val:** 6 nominal runs across base scenarios (reproducibility anchor)
  - **ID Test:** 20 runs (2 runs $\times$ 10 scenarios under nominal operating profile)
  - **OOD Test:** 50 runs (5 runs $\times$ 10 scenarios under shifted operating profile: elevated noise $\sigma_V=0.015\,\text{V}, \sigma_I=0.022\,\text{A}$, base resistance $R_{\text{base}}=0.140\,\Omega$, sensor biases)
- **Temporal Resolution:** $f_s = 100\,\text{Hz}$ ($0.01\,\text{s}$ timestep), total duration $T = 60.0\,\text{s}$ ($6,000$ points/run).
- **Feature Windowing:** Duration $T_w = 0.50\,\text{s}$ (50 points), stride $T_s = 0.25\,\text{s}$ (25 points), yielding 239 windows per 60 s run.
- **Storage Locations:**
  - Raw synthetic signals: `data/synthetic/flexpulse_final_test_v5/`
  - Extracted 26-features: `data/processed/features/flexpulse_final_test_v5/`

---

## 4. Evaluated Methods and Frozen Pipelines

| Identifier | Method Name | Architecture & Preprocessing | Frozen Artifact |
| :--- | :--- | :--- | :--- |
| **M1** | Fixed Threshold Envelope | Raw features, envelope $\mu \pm 2.5\sigma$ | `models/phase5/a0_fixed_threshold.joblib` |
| **M2** | Static Machine Learning | Raw unnormalized features, Random Forest (100 trees, depth 8) | `models/phase5/a1_static_random_forest.joblib` |
| **M3** | Initial Reference Calibration | Baseline-centered features ($\tilde{V} = V - \hat{V}_0, \tilde{I} = I - \hat{I}_0$) | `models/phase5/a2_a3_normalized_random_forest.joblib` |
| **M4** | Confidence-Gated Adaptive ML | Online calibrated features with 6 safety gating constraints | `models/phase5/a2_a3_normalized_random_forest.joblib` |
| **M5** | Initial Calibration + Persistence | Initial baseline centering + causal 2-window persistence filter | `models/phase5/a2_a3_normalized_random_forest.joblib` |
| **M6** | Adaptive Calibration + Persistence| Adaptive baseline centering + causal 2-window persistence filter | `models/phase5/a2_a3_normalized_random_forest.joblib` |

---

## 5. Statistical and Evaluation Criteria

1. **Window-Level Metrics:** Accuracy, Balanced Accuracy, Precision, Recall, Specificity, $\text{F}_1$-Score, Matthews Correlation Coefficient (MCC).
2. **Event-Level Metrics:** Ground truth event recall ($N_{\text{detected}} / N_{\text{total}}$), false alarm events, false alarms per minute ($N_{\text{FA}} / T_{\text{healthy}}$).
3. **Temporal Dynamics:** Causal detection latency relative to fault onset ($\tau_{\text{detect}} - \tau_{\text{onset}}$).
4. **Bootstrap Hypothesis Testing:** Paired run-level bootstrap ($B=1,000$ iterations, seed `20261202`) to compute 95% confidence intervals and two-sided empirical p-values for:
   - M3 vs M2 (Initial Calibration vs Static ML)
   - M4 vs M3 (Adaptive Update vs Initial Reference Calibration)
   - M5 vs M3 (Effect of 2-window Persistence on Initial Calibration)
   - M5 vs M2 (Initial Calibration + Persistence vs Static ML)

---

## 6. Pre-Test Hypotheses

1. **Hypothesis 1 (False Positive Reduction):** In v4, M3 and M5 exhibited $>1,000$ false positive windows on S4 due to the unlabelled degradation. In v5, because S4 windows after $5.0\,\text{s}$ are true faults, these detections will register as True Positives, markedly reducing false positive counts and elevating OOD specificity and $\text{F}_1$-score.
2. **Hypothesis 2 (Preservation of Equivalence M4 vs M3):** Under S4 degradation, the confidence gates will correctly prevent contaminated updates from updating the baseline. Consequently, M4 will maintain functional equivalence with M3 ($\Delta \text{F}_1 = 0.0000$).
3. **Hypothesis 3 (Persistence Superiority M5):** Persistence filtering ($N=2$) will suppress transient noise-induced false alarms in OOD regimes while preserving high event recall on gradual and intermittent faults.

---

## 7. Protocol Lock Record

- **Lock Status:** FROZEN AND LOCKED PRIOR TO SYNTHESIS
- **Lock Timestamp:** 2026-11-16T12:59:30+05:30
- **Pre-Registration Enforcement:** Frozen prior to dataset synthesis and model evaluation.


