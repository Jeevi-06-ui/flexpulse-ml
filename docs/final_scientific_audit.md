# FlexPulse-ML: Final Scientific and Methodological Audit Report

**Document ID:** `DOC-AUDIT-FINAL-2026-V1`  
**Target Git Commit:** `da07dfc`  
**Audit Scope:** Complete Verification of Phases 0 through 6  
**Project Title:** *FlexPulse-ML: A Simulation-Based Self-Calibrating Machine Learning Framework for Intermittent Fault Detection in USB-C-Inspired Power Paths*  
**Academic Context:** Department of Computer Science & Engineering, Presidency University, Bengaluru  

---

## 1. Executive Summary and Audit Scope

This document provides a comprehensive scientific, algorithmic, and software audit of the FlexPulse-ML research codebase. As the project transitions toward preparation of the IEEE research manuscript, all simulation mechanisms, feature derivations, model pipelines, calibration algorithms, and statistical conclusions are evaluated against strict metrological and machine learning standards.

### Primary Audit Verdicts
1. **Simulation Fidelity (Verified):** The electrical equivalent circuit model correctly implements Ohm's and Kirchhoff's laws for a single-rail resistive power path under load. All telemetry is generated via mathematical simulation and explicitly tagged with `synthetic_data: true`. No physical hardware, ESP32, USB-C Power Delivery (PD) negotiation, or laboratory bench instruments are involved.
2. **Train/Test Leakage Prevention (Verified):** Partitioning is strictly enforced at the complete simulation run level (`splits.json`). Feature windowing and normalization never cross run boundaries. Ground-truth state variables ($R_{\text{path}}$, $R_{\text{load}}$, true fault labels, scenario IDs) are strictly quarantined from observable feature matrices.
3. **Model Preprocessing Compliance (Verified):** The feature representation used for model evaluation strictly matches training representation. Static ML (A1) was trained on raw unnormalized features. Initial Calibrated ML (A2) and Adaptive ML (A3) were trained on zero-centered features normalized by run-specific initial reference baselines.
4. **The "A2 vs. A3 Equivalence" Resolution (Audited & Clarified):**
   * In Phase 5, A2 and A3 produced identical F1-scores ($0.9022$) because uncalibrated confidence gating ($\tau_{\text{conf}} = 0.95$) and current variance limits ($\sigma_I \le 0.015\,\text{A}$) locked the adaptive tracker into a permanent freeze under elevated sensor noise ($3\times$), resulting in zero accepted updates.
   * In Phase 6, after tuning gating parameters ($\tau_{\text{conf}} = 0.85$, $\sigma_I \le 0.040\,\text{A}$) and evaluating 10 controlled physical regimes (E1–E10), **A3 still produced identical F1-scores to A2 across all 10 regimes ($\Delta \text{F1} = 0.0000$)**.
   * **Scientific Finding:** Initial reference calibration (A2) performs 100% of the effective work in overcoming distribution shift. Incremental online adaptive updates (A3) provide no measurable classification or detection benefit in the tested simulation domain.
5. **Safeguard Utility (Verified Defensive Role):** While adaptive updates do not improve classification, the multi-stage confidence and stability gates prevent baseline poisoning during slow creeping faults (reducing contamination from $3.12\%$ down to $0.0\%$). The safeguards are strictly defensive.
6. **False-Alarm Mitigation (Verified):** A causal two-window persistence filter ($N=2$) reduces false alarm events by $47.0\%$ ($147 \to 78$ events) and increases F1 from $0.9022$ to $0.9382$ while maintaining 100% event recall (41/41 detected), at the cost of exactly one window stride ($+0.25\,\text{s}$) in causal detection latency.

---

## 2. Eleven-Point Architectural and Pipeline Verification

### 1. Electrical Power-Path Simulation
* **Model Equation:**
  $$R_{\text{path}}(t) = R_0 + R_{\text{contact}}(t) + R_{\text{fault}}(t)$$
  $$I(t) = \frac{V_s(t)}{R_{\text{load}}(t) + R_{\text{path}}(t)}, \quad V_{\text{load}}(t) = I(t) \cdot R_{\text{load}}(t)$$
  $$V_{\text{meas}}(t) = V_{\text{load}}(t) + b_V(t) + \eta_V(t), \quad I_{\text{meas}}(t) = I(t) + b_I(t) + \eta_I(t)$$
* **Verification:** Validated in `flexpulse_ml/simulation/electrical_model.py`. The vectorized function `compute_circuit_signals()` enforces positive source voltage, non-negative resistances, and caps open-circuit conditions to $100\,\text{k}\Omega$ to prevent division by zero.

### 2. Fault Generation and Ground-Truth Labels
* **Mechanism:** Contact degradation and intermittent micro-interruptions are injected via `PowerPathSimulator` (`fault_injector.py`) and `ControlledRegimeGenerator` (`regimes.py`).
* **Binary Labeling Rule:** $y(t) = 1$ if and only if a physical fault resistance is active ($R_{\text{fault}}(t) > 0$). Quiescent sensor offset drifts and healthy load transitions maintain $y(t) = 0$.
* **Quarantine:** Ground-truth files (`ground_truth.csv`, `events.json`) are stored separately from observable telemetry (`observations.csv`).

### 3. Dataset Partitioning
* **Hierarchy:** Split strictly by complete runs (`train/`, `validation/`, `id_test/`, `ood_test/`).
* **Independence:** Verified that no run seed or run ID appears in more than one partition. Zero window leakage exists.

### 4. Feature Extraction
* **Window Duration:** $T_w = 0.50\,\text{s}$ (50 samples at $100\,\text{Hz}$).
* **Stride:** $T_s = 0.25\,\text{s}$ (25 samples, 50% overlap).
* **Schema:** 26 observable features (13 voltage, 13 current).
* **Dimensional Consistency:** Verified that only the 6 level-valued voltage features and 6 level-valued current features undergo baseline centering. Spreads, slopes, and difference metrics remain unaltered.

### 5. Train/Test Leakage Prevention
* Scalers and models are fitted strictly on `train_features.csv`.
* Hyperparameters and gating thresholds are selected strictly on `validation_features.csv` (development set).
* Evaluation on test partitions (`id_test`, `ood_test`) uses frozen pipelines.

### 6. ML Model Training
* Evaluated classifiers:
  * Fixed Threshold Baseline (`FixedThresholdBaseline`): $\mu \pm 2.5\sigma$ envelope.
  * Logistic Regression: L2-regularized linear model with standard scaling.
  * Decision Tree: Gini impurity with maximum depth constrained to 8.
  * Random Forest: 100 bagged trees, max depth 8, balanced class weighting.
  * Support Vector Classifier: RBF kernel with Platt-scaling probability calibration.
  * HistGradientBoosting: Gradient-boosted histogram trees.

### 7. Initial Reference Normalization
* **Interval:** $t \in [0.0, 5.0]\,\text{s}$ across all runs.
* **Estimation:** Robust medians and median absolute deviations:
  $$\hat{V}_0 = \text{median}(\{V_{\text{mean}}^{(k)}\}), \quad \hat{I}_0 = \text{median}(\{I_{\text{mean}}^{(k)}\})$$
  $$\text{MAD}_V = 1.4826 \cdot \text{median}(\|V_{\text{mean}}^{(k)} - \hat{V}_0\|)$$
* **Assumption:** The physical power path is known to be in a healthy connection state during this initial 5-second interval.

### 8. Adaptive Baseline Calibration
* **Update Equation:**
  $$\hat{V}_0^{(k)} = (1 - \alpha)\hat{V}_0^{(k-1)} + \alpha V_{\text{mean}}^{(k)}, \quad \alpha = 0.04$$
* **Multi-Stage Gating (6 Gates):**
  1. Initialization Gate ($t > 5.0\,\text{s}$)
  2. Confidence Gate ($P(\text{healthy}) \ge \tau_{\text{conf}}$ and $\hat{y} = 0$)
  3. Current Stability Gate ($\sigma_I \le \sigma_{\max}$)
  4. Anomaly Cooldown Gate ($k=4$ windows)
  5. Outlier Residual Gate ($\|V_{\text{mean}} - \hat{V}_0\| \le 2.5 \cdot \text{MAD}_V$)
  6. Maximum Step Limiter ($|\Delta \hat{V}_0| \le 0.03\,\text{V}$)
* **State Machine:** States `INITIALIZING`, `NORMAL`, `FROZEN`. Transition to `FROZEN` occurs upon consecutive anomalies; recovery requires $N_{\text{recovery}} = 5$ consecutive valid healthy windows.

### 9. Event Detection and Causal Latency
* **Decision Timestamp:** Fixed to window end time $w_{\text{end}}$. Decisions cannot occur at $w_{\text{start}}$ because window features require all 50 samples to be recorded.
* **Causal Delay Definition:**
  $$\text{delay}_{\text{causal}} = \max(0.0, w_{\text{end}} - t_{\text{event\_start}})$$
* **Event Matching:** A ground-truth event is matched if a predicted positive interval overlaps with it, or occurs within $\pm 0.5\,\text{s}$. Unmatched predicted intervals are flagged as false alarm events.

### 10. Temporal Persistence Filtering
* **Filter Rule ($N=2$):** Positive alarm triggers only when the current window and immediately preceding window are both positive ($\hat{y}_k = 1 \land \hat{y}_{k-1} = 1$).
* **Latency Overhead:** Adds exactly one window stride ($0.25\,\text{s}$) of causal delay.

### 11. Statistical Evaluation
* **Bootstrap Resampling:** Evaluated on complete simulation runs (resampling runs with replacement, $B = 200$), preserving internal time-series correlation. Overlapping sliding windows are never treated as independent identically distributed (i.i.d.) observations.

---

## 3. Recomputation Verification of Benchmark Results

All previously reported metrics from Phase 4, Phase 5, and Phase 6 were independently recomputed from saved predictions and model weights:

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

---

## 4. Unsupported Claims vs. Supported Scientific Conclusions

To maintain strict scientific integrity in the final IEEE publication manuscript, all project conclusions must adhere to the following boundary:

### Unsupported Claims (Strictly Prohibited in Manuscript)
* **NO Hardware Claims:** Never claim "validated on physical USB-C cables", "tested on hardware test fixtures", or "proven compliant with USB-IF Power Delivery specifications".
* **NO Generalization Overreach:** Never claim that "adaptive baseline calibration outperforms initial calibration in all power paths". The evidence demonstrates that online adaptation produced zero gain over initial reference calibration.
* **NO Hyperbolic Claims:** Never claim "guaranteed fault detection", "zero false alarms", "patented technique", or "first-of-its-kind breakthrough".
* **NO Physical Localization Claims:** Never claim that electrical telemetry localizes the exact spatial millimeter of conductor degradation.

### Supported Conclusions (Defensible with Numerical Evidence)
* **Initial Reference Calibration is Critical:** Uncalibrated static machine learning collapses under physical operating shift ($F1 = 0.0419$, missing 97.9% of faults). Initial reference calibration restores F1 to $0.9022$ and Balanced Accuracy to $0.9706$.
* **Adaptive Baseline Updates Provide No Additional Benefit:** Across development sets, test sets, and 10 controlled physical regimes, online EMA updates provide zero incremental F1 benefit over initial reference calibration ($\Delta \text{F1} = 0.0000$, 95% CI $[0.0000, 0.0000]$).
* **Multi-Stage Safeguards Prevent Contamination:** In the presence of slow creeping faults, unprotected adaptive updating causes baseline contamination ($3.12\%$ rate), whereas confidence and stability gates reduce inappropriate updates to $0.0\%$.
* **Temporal Persistence Mitigates False Alarms:** A two-window causal persistence filter cuts false alarm events by $47.0\%$ ($147 \to 78$) and raises F1 to $0.9382$ while maintaining 100% event recall, with a modest $0.25\,\text{s}$ latency cost.
