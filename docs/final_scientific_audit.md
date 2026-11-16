# FlexPulse-ML: Final Scientific and Methodological Audit Report

**Document ID:** `DOC-AUDIT-FINAL-2026-V2`  
**Target Git Commit:** `dbd8587`  
**Audit Scope:** Complete Verification of Phases 0 through 7  
**Project Title:** *FlexPulse-ML: A Simulation-Based Self-Calibrating Machine Learning Framework for Intermittent Fault Detection in USB-C-Inspired Power Paths*  
**Academic Context:** Department of Computer Science & Engineering, Presidency University, Bengaluru  
**Notice:** SIMULATION-BASED EXPERIMENT — NO PHYSICAL VALIDATION CLAIMED

---

## 1. Executive Summary and Audit Scope

This document provides a comprehensive scientific, algorithmic, and software audit of the FlexPulse-ML research codebase. As the project transitions toward preparation of the IEEE research manuscript, all simulation mechanisms, feature derivations, model pipelines, calibration algorithms, and statistical conclusions are evaluated against strict metrological and machine learning standards.

### Primary Audit Verdicts
1. **Simulation Fidelity (Verified):** The electrical equivalent circuit model correctly implements Ohm's and Kirchhoff's laws for a single-rail resistive power path under load. All telemetry is generated via mathematical simulation and explicitly tagged with `synthetic_data: true`. No physical hardware, ESP32, USB-C Power Delivery (PD) negotiation, or laboratory bench instruments are involved.
2. **Train/Test Leakage Prevention (Verified):** Partitioning is strictly enforced at the complete simulation run level (`splits.json`). Feature windowing and normalization never cross run boundaries. Ground-truth state variables ($R_{\text{path}}$, $R_{\text{load}}$, true fault labels, scenario IDs) are strictly quarantined from observable feature matrices.
3. **Model Preprocessing Compliance (Verified):** The feature representation used for model evaluation strictly matches training representation. Static ML (M2/A1) was trained on raw unnormalized features. Initial Calibrated ML (M3/A2) and Adaptive ML (M4/A3) were trained on zero-centered features normalized by run-specific initial reference baselines.
4. **The "A2 vs. A3 Equivalence" Resolution (Audited & Clarified):**
   - Across development sets, Phase 5/6 controlled regimes, and the Phase 7 independent test set (`flexpulse_final_test_v4`), **M4/A3 produces identical classification predictions to M3/A2 ($\Delta \text{F1} = 0.0000$, 95% CI $[0.0000, 0.0000]$, empirical $p = 1.0000$)**.
   - **Scientific Finding:** Initial reference calibration (M3/A2) performs 100% of the effective work in overcoming DC operational shifts. Incremental online adaptive updates (M4/A3) provide zero measurable classification or detection benefit in the tested simulation domain.
5. **Safeguard Utility (Verified Defensive Role):** While adaptive updates do not improve classification, multi-stage confidence ($\tau_{\text{conf}} = 0.85$) and stability ($\sigma_I \le 0.040\,\text{A}$) gates prevent baseline poisoning during slow creeping faults (Scenario S9), reducing contamination from $2.12\%$ down to $0.00\%$. The safeguards are strictly defensive.
6. **False-Alarm Mitigation (Verified):** In the Phase 7 final test dataset, a causal two-window persistence filter ($N=2$, M5) reduces false alarm events by $40.88\%$ ($543 \to 321$ events, $12.25 \to 7.24\,\text{FA/min}$) and increases F1 from $0.5104$ to $0.5873$ (observed $\Delta \text{F1} = +0.0769$, bootstrap mean $\Delta \text{F1} = +0.0776$ [$B=1,000$] / $+0.0765$ [$B=200$], empirical $p < 0.01$). It preserves all 53 observable fault events detected by M3 (Event Recall = $53/58 = 91.38\%$), at an added median causal latency of $0.2228\,\text{s}$.
7. **Sensitivity vs. F1 Trade-Off:** On `flexpulse_final_test_v4`, initial reference calibration (M3) restores Event Recall to $91.38\%$ ($53/58$ events, vs. $74.14\%$ for static ML M2) and Balanced Accuracy to $0.8458$. However, because M3 flags subtle voltage deviations, false alarms also increase (FA Rate = $12.25\,\text{FA/min}$, Specificity = $0.7575$), resulting in an OOD F1 of $0.5104$ compared to $0.5424$ for M2. The paired bootstrap difference is $\Delta \text{F1} = -0.0360$ (95% CI $[-0.1379, 0.0597]$, $p = 0.4680$ with $B=1,000$; $-0.0391$, $p = 0.3700$ with $B=200$). Initial calibration recovers missed fault events, but does not statistically increase window F1 over static ML unless combined with persistence filtering.

---

## 2. Eleven-Point Architectural and Pipeline Verification

### 1. Electrical Power-Path Simulation
- **Model Equation:**
  $$R_{\text{path}}(t) = R_0 + R_{\text{contact}}(t) + R_{\text{fault}}(t)$$
  $$I(t) = \frac{V_s(t)}{R_{\text{load}}(t) + R_{\text{path}}(t)}, \quad V_{\text{load}}(t) = I(t) \cdot R_{\text{load}}(t)$$
  $$V_{\text{meas}}(t) = V_{\text{load}}(t) + b_V(t) + \eta_V(t), \quad I_{\text{meas}}(t) = I(t) + b_I(t) + \eta_I(t)$$
- **Verification:** Validated in `flexpulse_ml/simulation/electrical_model.py`. The vectorized function `compute_circuit_signals()` enforces positive source voltage, non-negative resistances, and caps open-circuit conditions to $100\,\text{k}\Omega$ to prevent division by zero.

### 2. Fault Generation and Ground-Truth Labels
- **Mechanism:** Contact degradation and intermittent micro-interruptions are injected via `PowerPathSimulator` (`fault_injector.py`) and `Phase7DatasetGenerator` (`generator_v4.py`).
- **Binary Labeling Rule:** $y(t) = 1$ if and only if a physical fault resistance is active ($R_{\text{fault}}(t) > 0$). Quiescent sensor offset drifts and healthy load transitions maintain $y(t) = 0$.
- **Quarantine:** Ground-truth files (`ground_truth.csv`, `events.json`) are stored separately from observable telemetry (`observations.csv`).

### 3. Dataset Partitioning
- **Hierarchy:** Split strictly by complete runs (`train/`, `validation/`, `id_test/`, `ood_test/`).
- **Independence:** Verified that no run seed or run ID appears in more than one partition. Zero window leakage exists.

### 4. Feature Extraction
- **Window Duration:** $T_w = 0.50\,\text{s}$ (50 samples at $100\,\text{Hz}$).
- **Stride:** $T_s = 0.25\,\text{s}$ (25 samples, 50% overlap).
- **Schema:** 26 observable features (13 voltage, 13 current).
- **Dimensional Consistency:** Verified that only the 6 level-valued voltage features and 6 level-valued current features undergo baseline centering. Spreads, slopes, and difference metrics remain unaltered.

### 5. Train/Test Leakage Prevention
- Scalers and models are fitted strictly on development data.
- Evaluation on test partitions (`id_test`, `ood_test`) uses frozen pipelines.

### 6. ML Model Training
- Evaluated classifiers:
  - M1: Fixed Threshold Baseline (`FixedThresholdBaseline`): $\mu \pm 2.5\sigma$ envelope.
  - M2: Static Random Forest (100 bagged trees, max depth 8, balanced class weighting on raw features).
  - M3: Initial Reference Calibration + Random Forest (trained on zero-centered features).
  - M4: Confidence-Gated Adaptive Random Forest (online dynamic centering with 6 safety gates).
  - M5: Initial Reference Calibration + Causal Temporal Persistence Filter ($N=2$).
  - M6: Adaptive Calibration + Causal Temporal Persistence Filter ($N=2$).

### 7. Initial Reference Normalization
- **Interval:** $t \in [0.0, 5.0]\,\text{s}$ across all runs.
- **Estimation:** Robust medians and median absolute deviations:
  $$\hat{V}_0 = \text{median}(\{V_{\text{mean}}^{(k)}\}), \quad \hat{I}_0 = \text{median}(\{I_{\text{mean}}^{(k)}\})$$
  $$\text{MAD}_V = 1.4826 \cdot \text{median}(\|V_{\text{mean}}^{(k)} - \hat{V}_0\|)$$
- **Assumption:** The physical power path is known to be in a healthy connection state during this initial 5-second interval.

### 8. Adaptive Baseline Calibration
- **Update Equation:**
  $$\hat{V}_0^{(k)} = (1 - \alpha)\hat{V}_0^{(k-1)} + \alpha V_{\text{mean}}^{(k)}, \quad \alpha = 0.04$$
- **Multi-Stage Gating (6 Gates):**
  1. Initialization Gate ($t > 5.0\,\text{s}$)
  2. Confidence Gate ($P(\text{healthy}) \ge \tau_{\text{conf}}$ and $\hat{y} = 0$, with $\tau_{\text{conf}} = 0.85$)
  3. Current Stability Gate ($\sigma_I \le \sigma_{\max} = 0.040\,\text{A}$)
  4. Anomaly Cooldown Gate ($k=4$ windows)
  5. Outlier Residual Gate ($\|V_{\text{mean}} - \hat{V}_0\| \le 2.5 \cdot \text{MAD}_V$)
  6. Maximum Step Limiter ($|\Delta \hat{V}_0| \le 0.03\,\text{V}$)
- **State Machine:** States `INITIALIZING`, `NORMAL`, `FROZEN`. Transition to `FROZEN` occurs upon consecutive anomalies; recovery requires $N_{\text{recovery}} = 5$ consecutive valid healthy windows.

### 9. Event Detection and Causal Latency
- **Decision Timestamp:** Fixed to window end time $w_{\text{end}}$. Decisions cannot occur at $w_{\text{start}}$ because window features require all 50 samples to be recorded.
- **Causal Delay Definition:**
  $$\text{delay}_{\text{causal}} = \max(0.0, w_{\text{end}} - t_{\text{event\_start}})$$
- **Event Matching:** A ground-truth event is matched if a predicted positive interval overlaps with it, or occurs within $\pm 0.5\,\text{s}$. Unmatched predicted intervals are flagged as false alarm events.

### 10. Temporal Persistence Filtering
- **Filter Rule ($N=2$):** Positive alarm triggers only when the current window and immediately preceding window are both positive ($\hat{y}_k = 1 \land \hat{y}_{k-1} = 1$).
- **Latency Overhead:** Adds $0.2228\,\text{s}$ of median causal delay.

### 11. Statistical Evaluation
- **Bootstrap Resampling:** Evaluated on complete simulation runs (resampling runs with replacement, $B = 200$), preserving internal time-series correlation.
- **Resolution Limit:** With $B=200$, the minimum non-zero empirical resolution is $1/B = 0.005$. Empirical p-values are reported with appropriate bounds ($p < 0.01$).

---

## 3. Recomputation Verification of Historical Benchmark Results

Historical metrics from Phases 4, 5, and 6 were independently verified against saved records:

| Experiment / Metric | Previously Reported | Recomputed From Saved Records | Status |
| :--- | :---: | :---: | :---: |
| **Phase 4 RF In-Distribution (ID) F1** | 0.9967 | 0.9967 | **Exact Match** |
| **Phase 4 RF ID Balanced Accuracy** | 0.9967 | 0.9967 | **Exact Match** |
| **Phase 4 RF Out-of-Distribution (OOD) F1** | 0.0518 | 0.0518 | **Exact Match** |
| **Phase 4 RF Causal Median Delay** | 0.2400 s | 0.2400 s | **Exact Match** |
| **Phase 5 A0 Fixed Threshold OOD F1** | 0.3341 | 0.3341 | **Exact Match** |
| **Phase 5 A1 Static ML OOD F1** | 0.0419 | 0.0419 | **Exact Match** |
| **Phase 5 A2 Initial Calibration OOD F1** | 0.9022 | 0.9022 | **Exact Match** |
| **Phase 5 A3 Adaptive Calibration OOD F1** | 0.9022 | 0.9022 | **Exact Match** |
| **Phase 6 Persistence N=2 OOD F1** | 0.9382 | 0.9382 | **Exact Match** |
| **Phase 6 Persistence N=2 False Alarms** | 78 events | 78 events | **Exact Match** |
| **Phase 6 A3 vs A2 Bootstrap Delta F1** | 0.0000 | 0.0000 | **Exact Match** |

*Important Methodological Clarification:* Each phase evaluated specific, distinct synthetic datasets. Phase 4 evaluated `flexpulse_research_v1` (150 runs). Phase 5 evaluated `flexpulse_calibration_v2`. Phase 6 evaluated controlled regimes E1–E10. Phase 7 evaluated `flexpulse_final_test_v4` (82 runs, seed `20261201`). Metrics from different datasets must never be combined into a single claim of performance decline or improvement.

---

## 4. Phase 7 Audit Discovery: Scenario S4 Labeling Semantic Divergence

During this final audit of `flexpulse_final_test_v4` (`flexpulse_ml/data/generator_v4.py`), a critical semantic labeling discrepancy was identified in Scenario S4 (Gradual Resistance Drift):

```python
# Code snippet from flexpulse_ml/data/generator_v4.py
elif scenario == self.SCENARIO_S4_GRADUAL_RESISTANCE_DRIFT:
    post_mask = t > 5.0
    drift_rate = float(rng.uniform(0.002, 0.004))
    base_r[post_mask] += drift_rate * (t[post_mask] - 5.0)
```

### Audit Findings:
1. **The Semantic Divergence:** Physical resistance drift was added directly to `base_r` rather than through the fault injection helper `add_event()`. Consequently, `fault_r` remained zero, `fault_labels` remained all zeros ($0$), and no entries were appended to `events_meta`.
2. **Impact on Metrics:** In `table3_scenario_breakdown.csv`, Scenario S4 has `GT Pos Windows = 0`. As detectors (M3, M4, M5, M6) detect the rising loop resistance, every positive window prediction is scored against zero ground-truth positives, generating 1,105 false positive windows for M3 (Specificity = 0.0753) and 1,083 for M5 (Specificity = 0.0937).
3. **Audit Verdict & Pre-Registration Compliance:** In accordance with strict pre-registration rules, final test data labels must not be retroactively modified after running evaluations. This generator divergence is formally documented as an empirical limitation and anomaly.

---

## 5. Master Correction Ledger

This ledger details all discrepancies identified between preliminary documentation and saved empirical CSV tables, and confirms their resolution:

| Item # | Target Section / Topic | Original Inaccurate Statement | Corrected Accurate Statement | Exact Supporting Artifact | Underlying Data Changed? |
| :---: | :--- | :--- | :--- | :--- | :---: |
| **1** | M3 vs M2 OOD F1 Comparison | Initial calibration significantly boosts OOD F1 over static ML | M2 achieves OOD F1=0.5424; M3 achieves OOD F1=0.5104 ($\Delta \text{F1} = -0.0360$, 95% CI $[-0.1379, 0.0597]$, $p = 0.4680$ with $B=1,000$; $-0.0391$, $p = 0.3700$ with $B=200$). M3 improves event recall ($91.4\%$ vs $74.1\%$) but does not improve window F1. | `table1_overall_performance.csv`, `table2_bootstrap_significance.csv` | **No** (Corrects text to match table) |
| **2** | Cross-Dataset Comparison | Static RF F1 drops from 0.9967 (Phase 4) to 0.5424 (Phase 7 OOD) | Within `flexpulse_final_test_v4`, Static RF achieves ID F1=0.5193 and OOD F1=0.5424. Static ML exhibits inadequate fault sensitivity, missing 25.86% of events. | `table1_overall_performance.csv` | **No** (Eliminates cross-dataset mixing) |
| **3** | Persistence False-Alarm Reduction | Persistence N=2 cuts false alarm events by 47.0% | Persistence N=2 cuts false alarm events by **40.88%** ($543 \to 321$ events, $12.25 \to 7.24\,\text{FA/min}$). | `table5_persistence_ablation.csv` | **No** (Corrects historical 47% to exact 40.88%) |
| **4** | Event Recall Accuracy | Persistence achieves 100% event detection | Persistence preserves all 53 detected events, maintaining an Event Recall of **53/58 = 91.38%**. | `table4_event_level_metrics.csv`, `table5_persistence_ablation.csv` | **No** (Removes inaccurate 100% detection claim) |
| **5** | Median Added Causal Latency | Added causal delay is +0.25 s | Median causal latency increases from 0.1093 s ($N=1$) to 0.3321 s ($N=2$), an added delay of exactly **+0.2228 s**. | `table5_persistence_ablation.csv` | **No** (Corrects round 0.25 s to exact 0.2228 s) |
| **6** | Creeping Fault Contamination | Ungated contamination is 100% vs 0% gated | Ungated tracking contamination during creeping faults is **2.12%** (5.12 mV shift), whereas fully gated tracking reduces it to **0.00%** (5.06 mV shift). | `table6_gating_ablation.csv` | **No** (Corrects 100% claim to exact 2.12%) |
| **7** | Scenario S8 Specificity | S8 reference calibration achieves 1.0000 specificity | Under healthy load steps (S8), M3 achieves Specificity = **0.9054** (113 FP windows), improving to **0.9548** (54 FP windows) with M5. | `table3_scenario_breakdown.csv` | **No** (Corrects 1.0000 claim to exact 0.9054) |
| **8** | Persistence Optimality Analysis | N=2 is mathematically proven universal Pareto optimum; N>=3 drops recall | Both $N=2$ and $N=3$ retain 91.38% event recall (53/58 events). Event recall drops at $N=4$ (89.66%) and $N=5$ (86.21%). $N=2$ was chosen as a sub-stride latency compromise (+0.2228 s vs +0.4728 s for $N=3$). | `table5_persistence_ablation.csv` | **No** (Reconciles persistence analysis with table) |
| **9** | Scenario S4 Labeling Semantics | Gradual drift reported with 0 positives without explanation | Documented as a generator semantic divergence: physical drift was added to `base_r` without activating `fault_labels = 1`. | `table3_scenario_breakdown.csv`, Audit Sec 4 | **No** (Documented as formal limitation) |
| **10** | Bootstrap P-Value Precision | Reported as p < 0.0001 without empirical resolution bound | Empirical resolution is $1/B$ ($0.001$ for $B=1,000$; $0.005$ for $B=200$). Zero crossings are reported as **p < 0.01**. | `table2_bootstrap_significance.csv` | **No** (Adjusts p-value precision reporting) |
| **11** | Pre-Registration Protocol Claim | Stored SHA-256 hash is proof of independent preregistration | Clarified as an internal pre-test configuration lock and tamper-evident SHA-256 hash. | `protocol/protocol_hash_sha256.txt` | **No** (Corrects scientific governance claim) |
| **12** | Domain Extrapolation | Extrapolated findings to physical USB-C cables and production firmware | Scope strictly restricted to a software-based lumped-element simulation of a resistive power path. | `limitations/scientific_limitations.md` | **No** (Removes physical hardware claims) |

---

## 6. Unsupported Claims vs. Supported Scientific Conclusions

### Unsupported Claims (Strictly Prohibited in IEEE Manuscript)
- **NO Hardware Claims:** Never claim "validated on physical USB-C cables", "tested on hardware test fixtures", or "proven compliant with USB-IF Power Delivery specifications".
- **NO Generalization Overreach:** Never claim that "adaptive baseline calibration outperforms initial calibration". The evidence demonstrates that online adaptation produced zero gain over initial reference calibration.
- **NO False F1 Improvement Claims:** Never claim that initial calibration significantly improves window F1 over static ML on OOD data ($\Delta \text{F1} = -0.0391$, $p = 0.3700$). Initial calibration improves event-level sensitivity and balanced accuracy, but increases window false alarms.
- **NO Universal Optimality Claims:** Do not claim $N=2$ persistence is a "mathematically proven universal optimum." It is an engineering latency trade-off.

### Supported Conclusions (Defensible with Numerical Evidence)
- **Static ML Misses Fault Events:** Static Random Forest misses 25.86% of observable fault events (15/58 missed) under OOD operating shifts.
- **Initial Calibration Recovers Event Sensitivity:** Initial reference calibration recovers Event Recall from 74.14% to 91.38% (+10 detected events) and Balanced Accuracy to 0.8458.
- **Adaptive Baseline Updates Provide No Additional Benefit:** Online confidence-gated adaptive updates provide zero incremental F1 benefit over initial reference calibration ($\Delta \text{F1} = 0.0000$, 95% CI $[0.0000, 0.0000]$, $p = 1.0000$).
- **Multi-Stage Safeguards Prevent Contamination:** In the presence of slow creeping faults, unprotected adaptive updating causes baseline contamination ($2.12\%$ rate), whereas confidence and stability gates reduce inappropriate updates to $0.00\%$.
- **Temporal Persistence Mitigates False Alarms:** A two-window causal persistence filter cuts false alarm events by $40.88\%$ ($543 \to 321$ events) and raises F1 from $0.5104$ to $0.5873$ ($\Delta \text{F1} = +0.0765$, $p < 0.01$) while maintaining 91.38% event recall (53/58 events), with a $0.2228\,\text{s}$ added latency cost.
