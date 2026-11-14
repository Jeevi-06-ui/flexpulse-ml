# Phase 4 Scientific Robustness and Methodology Audit Report

**Project Title:** FlexPulse-ML: A Simulation-Based Self-Calibrating Machine Learning Framework for Intermittent Fault Detection in USB-C-Inspired Power Paths  
**Audit Date:** 2026-11-14  
**Audit Target:** Phase 4 Machine Learning Training and Benchmarking Deliverables  
**Historical Commit:** `50f5c80`  
**Dataset Examined:** `flexpulse_research_v1`  

---

## 1. Executive Summary of Audit Findings

A rigorous audit of Phase 4 deliverables was conducted prior to implementing the Phase 5 adaptive self-calibration architecture. The primary findings are:
1. **Metric Reproducibility Verified:** All window classification metrics (Accuracy, Balanced Accuracy, Precision, Recall, Specificity, F1-Score, Confusion Matrices, ROC-AUC) saved in `results/tables/phase4_benchmark/` recompute to the exact fourth decimal place from frozen model pipelines (`models/phase4/*.joblib`) and feature matrices.
2. **Event Detection Delay Non-Causality Identified and Corrected:** Phase 4 event delay calculations matched predicted intervals using window start timestamps ($w_{\text{start}}$), yielding non-causal zero-second median detection delays. In a real-time system, a 0.5-second observation window cannot produce an inference until the window end ($w_{\text{end}}$) when all 50 samples have been received. Correcting this latency definition yields a true causal median detection delay of **0.2400 s** and a causal mean delay of **0.3233 s** for Random Forest.
3. **Deterministic Fault Timing Disclosed:** While sensor noise and drift trajectories were randomized with pseudo-random seeds, fault event onset and duration schedules were held constant across runs, explaining the deterministic 20.99% positive window fraction.
4. **Observable Input Policy Conformance:** Confirmed that all 26 feature inputs access strictly `voltage_measured_v` and `current_measured_a`. No hidden state or ground-truth leakage occurred.
5. **Clarification of Out-of-Distribution Shift Modeling:** Confirmed that the mathematical electrical simulation does not contain a thermal physics differential equation. Mentions of "temperature shift" were conceptual analogies for ambient drift rather than modeled physical states. True modeled shifts consisted of higher initial path resistance ($0.180\,\Omega$), heavier load current ($2.6\,\text{A}$), elevated noise floor ($3\times$), and faster sensor offset drift.
6. **Integrity Preservation:** Historical Phase 4 benchmark files were left untouched. Corrected versioned benchmark tables are preserved under `results/tables/phase4_corrected/`.

---

## 2. Recomputation Verification of Saved Benchmark Metrics

Using the frozen artifacts in `models/phase4/` and `data/processed/features/flexpulse_research_v1/id_test_features.csv`, metrics were independently recomputed:

| Model | Historical Stored F1 | Recomputed F1 | Historical Balanced Acc | Recomputed Balanced Acc | Recomputation Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Fixed Threshold** | 0.8349 | 0.8349 | 0.8628 | 0.8628 | Exact Match |
| **Logistic Regression** | 0.9944 | 0.9944 | 0.9945 | 0.9945 | Exact Match |
| **Decision Tree** | 0.9961 | 0.9961 | 0.9969 | 0.9969 | Exact Match |
| **Random Forest** | 0.9967 | 0.9967 | 0.9967 | 0.9967 | Exact Match |
| **SVM (RBF Kernel)** | 0.9933 | 0.9933 | 0.9934 | 0.9934 | Exact Match |
| **Gradient Boosting** | 0.9961 | 0.9961 | 0.9965 | 0.9965 | Exact Match |

---

## 3. Audit of Fault Timing Schedules

### Injected Event Envelopes
In `flexpulse_research_v1`, the generator configured default timing envelopes:
- `INCREASED_RESISTANCE`: Fixed onset at $t = 10.0\,\text{s}$, duration $20.0\,\text{s}$ (ends $t = 30.0\,\text{s}$).
- `GRADUAL_RESISTANCE_DRIFT`: Fixed onset at $t = 7.0\,\text{s}$, duration $48.0\,\text{s}$ (ends $t = 55.0\,\text{s}$).
- `INTERMITTENT_CONTACT`: 4 discrete 0.15s contact interruptions starting at $t = 7.0\,\text{s}$ spaced across the run.
- `HEALTHY`, `MEASUREMENT_OFFSET_DRIFT`, `OPERATING_CONDITION_SHIFT`: Physical path remains healthy (0 fault events).

### Stochastic vs. Deterministic Parameter Breakdown:
- **Deterministic:** Event start times, event durations, and scenario allocation per block.
- **Randomized:** Unique pseudo-random seeds per run governing Gaussian sensor noise realizations ($\sigma_V, \sigma_I$), sensor bias random walk trajectories, and initial load timing jitter.
- **Audit Requirement for Phase 5:** In Phase 5 protocols, event onsets and durations must be randomized across runs to prevent models from learning temporal placement artifacts.

---

## 4. Audit of Event Detection Timestamps and Causal Latency

### Mechanism of the Latency Distortion:
In Phase 4 `evaluate_event_level_detections`:
```python
delay = max(0.0, p_st - inj["st"])
```
Here `p_st` was recorded as the start time of the first window triggering a positive prediction. For example, if a fault begins at $t = 10.0\,\text{s}$, the window spanning $[9.75\,\text{s}, 10.25\,\text{s}]$ contains 0.25s of fault data. When the classifier flags this window, setting decision time to $w_{\text{start}} = 9.75\,\text{s}$ yielded:
$$\text{delay} = \max(0.0, 9.75 - 10.0) = 0.0000\,\text{s}$$

### Physical and Mathematical Correction:
A sliding window spanning $[t_{\text{start}}, t_{\text{end}}]$ with duration $T_w = 0.5\,\text{s}$ requires all 50 samples to be observed before feature extraction and inference can execute. Therefore, the earliest physically realizable decision timestamp is $t_{\text{decision}} = t_{\text{end}}$.

The true causal detection delay is defined as:
$$\text{delay}_{\text{causal}} = \max(0.0, t_{\text{decision}} - t_{\text{event\_start}}) = \max(0.0, w_{\text{end}} - t_{\text{event\_start}})$$

### Corrected Detection Delay Comparison:
| Model | Historical Uncorrected Mean Delay | Corrected Causal Mean Delay | Historical Median Delay | Corrected Causal Median Delay | Latency Offset ($\Delta$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Fixed Threshold** | 3.3333 s | **3.6150 s** | 0.0000 s | **0.2400 s** | +0.2817 s |
| **Logistic Regression** | 0.0972 s | **0.3789 s** | 0.0000 s | **0.2400 s** | +0.2817 s |
| **Decision Tree** | 0.0278 s | **0.3094 s** | 0.0000 s | **0.2400 s** | +0.2817 s |
| **Random Forest** | 0.0417 s | **0.3233 s** | 0.0000 s | **0.2400 s** | +0.2817 s |
| **SVM (RBF Kernel)** | 0.1250 s | **0.4067 s** | 0.0000 s | **0.2400 s** | +0.2817 s |
| **Gradient Boosting** | 0.0417 s | **0.3233 s** | 0.0000 s | **0.2400 s** | +0.2817 s |

*Note: All corrected tables have been saved under `results/tables/phase4_corrected/` without altering historical benchmark files.*

---

## 5. Audit of Observable Telemetry vs. Modeled State

A line-by-line inspection of `flexpulse_ml/features/extractor.py` and `flexpulse_ml/ml/train.py` confirmed:
- Inputs to feature extraction consist strictly of `voltage_measured_v` and `current_measured_a`.
- Target labels (`fault_label`) are isolated during inference.
- Hidden simulator states ($R_{\text{path}}(t)$, $R_{\text{load}}(t)$, $b_V(t)$, $b_I(t)$, `scenario`, `event_id`) are never passed to the ML pipeline.

---

## 6. Audit of Out-of-Distribution (OOD) Shift Mechanisms

The actual shifts implemented in `generator.py` for OOD runs were:
1. Nominal source voltage: $5.2\,\text{V}$ (vs $5.0\,\text{V}$).
2. Baseline path resistance: $0.180\,\Omega$ (vs $0.080\,\Omega$, a $125\%$ increase).
3. Load resistance: $1.80\,\Omega$ (vs $3.333\,\Omega$, increasing nominal load current from $\approx 1.5\,\text{A}$ to $\approx 2.6\,\text{A}$).
4. Sensor noise standard deviation: $\sigma_V = 0.015\,\text{V}$ ($3\times$), $\sigma_I = 0.025\,\text{A}$ ($3.1\times$).
5. Sensor offset drift rate: $0.0004\,\text{V/s}$ ($4\times$) and $0.0002\,\text{A/s}$ ($4\times$).

**Clarification on Temperature:** The mathematical equivalent circuit in `electrical_model.py` contains no thermal state equation or temperature feature. Descriptions implying temperature modeling were conceptual analogies. The underlying physical simulation is an electrical equivalent circuit subject to parameter drift and observation noise.

---

## 7. Root Causes of Static Model OOD Collapse

The observed collapse of static classifiers under OOD shift ($F1 \le 0.052$) stems from two distinct mathematical failure modes:
1. **Linear and Threshold False Alarm Explosion ($FP > 3,700$):** In nominal training, healthy voltage drop was $\Delta V \approx 1.5\,\text{A} \times 0.080\,\Omega = 0.12\,\text{V}$. In OOD runs, healthy voltage drop was $\Delta V \approx 2.6\,\text{A} \times 0.180\,\Omega = 0.468\,\text{V}$. Because static threshold and linear models classify $\Delta V > 0.20\,\text{V}$ as a fault, every healthy shifted window is misclassified as faulty.
2. **Tree Ensemble and Kernel Silent Failure ($FN = 1,505$):** Decision trees partition feature space into orthogonal hyper-rectangles. In the OOD regime, the simultaneous combination of higher current, higher voltage drop, and larger noise fell into regions of the feature space unpopulated during nominal training. The leaf nodes covering these unobserved regions defaulted to majority class 0, producing zero fault detections.

This divergence confirms that static ML boundaries cannot generalize across physical parameter shifts without an adaptive baseline normalization mechanism.
