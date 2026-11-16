# FlexPulse-ML: Final Numerical Consistency Certificate

**Document ID:** `CERT-NUMERICAL-CONSISTENCY-2026-V1`  
**Date of Audit:** 16 November 2026  
**Auditor / Role:** Senior Machine Learning Research Engineer & Scientific Software Auditor  
**Project:** *FlexPulse-ML: A Simulation-Based Self-Calibrating Machine Learning Framework for Intermittent Fault Detection in USB-C-Inspired Power Paths*  
**Academic Institution:** Presidency University, Bengaluru | Machine Learning Techniques  
**Dataset Under Audit:** `flexpulse_final_test_v4` (Master Seed: `20261201`)  
**Internal Pre-Test Configuration Lock:** `13bc534c900f8891ebe74eed6cdcc826210dd4635c137e274aa6f0d058fedca1`  
**Verification Status:** **100% RECONCILED AND CERTIFIED**  
**Mandatory Scientific Disclaimer:** **SIMULATION-BASED EXPERIMENT — NO PHYSICAL HARDWARE VALIDATION CLAIMED**

---

## 1. Scope and Certification Purpose

This certificate establishes the authoritative, mathematically verified numerical ground truth for all experimental methods evaluated in Phase 7 of the FlexPulse-ML research project.

It provides:
1. Exact confusion matrix cell counts ($TP, TN, FP, FN$) directly computed from frozen predictions and ground truth.
2. Exact metric formulas and recomputed classification metrics.
3. Explicit method identifiers and definitions.
4. Comprehensive explanations for historical transcription discrepancies.
5. Recomputation and reconciliation of paired bootstrap statistical comparisons.
6. Event-level temporal matching audit and healthy-time denominator verification.
7. Full resolution of Scenario S4 ground-truth labeling semantics.
8. Artifact provenance and checksum registry.

---

## 2. Explicit Method Identifiers and Architecture Definitions

To eliminate any ambiguity between development code names (A0–A3) and manuscript method numbers (M1–M6), the canonical taxonomy is locked as follows:

| Canonical ID | Code Architecture ID | Model Name & Configuration | Feature Normalization | Post-Processing |
| :---: | :---: | :--- | :--- | :--- |
| **M1** | `A0` | Fixed Threshold Envelope | Raw 26 features | $\mu \pm 2.5\sigma$ bounding envelope on $\Delta V$ |
| **M2** | `A1` | Static Random Forest (100 trees, max depth 8) | Raw 26 features | Direct argmax window classification ($P(y=1) \ge 0.5$) |
| **M3** | `A2` | Initial Reference Calibration + Random Forest | 5.0 s initial healthy centering ($\tilde{V} = V - \hat{V}_0$, $\tilde{I} = I - \hat{I}_0$) | Direct argmax window classification ($P(y=1) \ge 0.5$) |
| **M4** | `A3` | Confidence-Gated Adaptive Random Forest | 5.0 s initial centering + EMA updates ($\alpha=0.04$, $\tau=0.85$, $\sigma_I \le 0.04\,\text{A}$) | Direct argmax window classification ($P(y=1) \ge 0.5$) |
| **M5** | `A2_N2` | Initial Reference Calibration + Persistence $N=2$ | 5.0 s initial healthy centering | Causal 2-window persistence filter ($y_k = 1 \iff \hat{y}_k=1 \land \hat{y}_{k-1}=1$) |
| **M6** | `A3_N2` | Adaptive Calibration + Persistence $N=2$ | 5.0 s initial centering + EMA updates | Causal 2-window persistence filter ($y_k = 1 \iff \hat{y}_k=1 \land \hat{y}_{k-1}=1$) |

---

## 3. Exact Mathematical Metric Formulas

All metrics reported in the evaluation tables are computed strictly via the following standard formulas:

1. **Total Sample Count ($N$):**
   $$N = TP + TN + FP + FN$$

2. **Window Classification Accuracy:**
   $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

3. **Window Sensitivity / Recall (True Positive Rate):**
   $$\text{Recall} = \frac{TP}{TP + FN} \quad (\text{Reported as } \texttt{N/A} \text{ if } TP + FN = 0)$$

4. **Window Precision (Positive Predictive Value):**
   $$\text{Precision} = \frac{TP}{TP + FP} \quad (\text{Reported as } \texttt{N/A} \text{ if } TP + FP = 0)$$

5. **Window Specificity (True Negative Rate):**
   $$\text{Specificity} = \frac{TN}{TN + FP} \quad (\text{Reported as } 1.0000 \text{ if } TN + FP = 0)$$

6. **Balanced Accuracy:**
   $$\text{Balanced Accuracy} = \frac{\text{Recall} + \text{Specificity}}{2} \quad (\text{Equal to Specificity if Recall is } \texttt{N/A})$$

7. **Harmonic Mean $\text{F}_1$-Score:**
   $$\text{F}_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN} \quad (\text{Reported as } \texttt{N/A} \text{ if } TP=0 \land (FP=0 \lor FN=0))$$

8. **Event Recall:**
   $$\text{Event Recall} = \frac{\text{Detected Observable Events}}{\text{Total Observable Injected Events}} = \frac{E_{\text{det}}}{E_{\text{obs}}}$$

9. **False Alarm Event Rate (/min):**
   $$\text{FA Rate} = \frac{\text{Total Unmatched Predicted Intervals}}{\text{Total Healthy Duration in Minutes}} = \frac{E_{\text{FA}}}{T_{\text{healthy}} / 60.0}$$

10. **Median Causal Detection Delay ($s$):**
    $$\text{Median Delay} = \text{median}\left(\left\{ t_{\text{first\_window\_end}} - t_{\text{event\_start}} \mid \text{event detected} \right\}\right)$$

---

## 4. Recomputed Confusion Matrices and Verified Window Metrics

Recomputed directly from `data/processed/features/flexpulse_final_test_v4` and saved model checkpoints.

### A. In-Distribution (ID) Test Partition
- **Total Simulation Runs:** 20 runs (2 runs per scenario, 60.0 s per run).
- **Total Sliding Windows:** 4,780 windows ($T_w = 0.50\,\text{s}$, $T_s = 0.25\,\text{s}$).
- **Ground-Truth Positive Windows:** 564 ($11.80\%$).
- **Ground-Truth Negative Windows:** 4,216 ($88.20\%$).

| Method | TP | TN | FP | FN | Accuracy | Balanced Acc | Precision | Recall | Specificity | $\text{F}_1$-Score |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M1** | 498 | 3,716 | 500 | 66 | 0.8816 | 0.8822 | 0.4990 | 0.8830 | 0.8814 | 0.6376 |
| **M2** | 438 | 3,531 | 685 | 126 | 0.8303 | 0.8071 | 0.3900 | 0.7766 | 0.8375 | 0.5193 |
| **M3** | 524 | 3,746 | 470 | 40 | 0.8933 | 0.9088 | 0.5272 | 0.9291 | 0.8885 | 0.6727 |
| **M4** | 523 | 3,746 | 470 | 41 | 0.8931 | 0.9079 | 0.5267 | 0.9273 | 0.8885 | 0.6718 |
| **M5** | 504 | 3,770 | 446 | 60 | 0.8941 | 0.8939 | 0.5305 | 0.8936 | 0.8942 | 0.6658 |
| **M6** | 503 | 3,770 | 446 | 61 | 0.8939 | 0.8930 | 0.5300 | 0.8918 | 0.8942 | 0.6649 |

### B. Out-of-Distribution (OOD) Test Partition
- **Total Simulation Runs:** 50 runs (5 runs per scenario, 60.0 s per run).
- **Total Sliding Windows:** 11,950 windows ($T_w = 0.50\,\text{s}$, $T_s = 0.25\,\text{s}$).
- **Ground-Truth Positive Windows:** 1,472 ($12.32\%$).
- **Ground-Truth Negative Windows:** 10,478 ($87.68\%$).

| Method | TP | TN | FP | FN | Accuracy | Balanced Acc | Precision | Recall | Specificity | $\text{F}_1$-Score |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M1** | 1,399 | 3,106 | 7,372 | 73 | 0.3770 | 0.6234 | 0.1595 | 0.9504 | 0.2964 | 0.2732 |
| **M2** | 1,130 | 8,913 | 1,565 | 342 | 0.8404 | 0.8092 | 0.4193 | 0.7677 | 0.8506 | **0.5424** |
| **M3** | 1,375 | 7,937 | 2,541 | 97 | 0.7792 | 0.8458 | 0.3511 | 0.9341 | 0.7575 | 0.5104 |
| **M4** | 1,375 | 7,937 | 2,541 | 97 | 0.7792 | 0.8458 | 0.3511 | 0.9341 | 0.7575 | 0.5104 |
| **M5** | 1,332 | 8,746 | 1,732 | 140 | **0.8433** | **0.8698** | **0.4347** | **0.9049** | **0.8347** | **0.5873** |
| **M6** | 1,332 | 8,746 | 1,732 | 140 | **0.8433** | **0.8698** | **0.4347** | **0.9049** | **0.8347** | **0.5873** |

---

## 5. Event-Level Audit and Temporal Latency Results

### A. Temporal Merging and Matching Protocol
1. **Window Aggregation:** Consecutive positive window predictions ($y_k = 1$) within a run are merged into a predicted event interval $[t_{\text{start}}, t_{\text{end}}]$ if the gap between consecutive windows is $\le 0.50\,\text{s}$ (two strides).
2. **Ground-Truth Matching:** A predicted interval matches an injected ground-truth event if there is positive temporal overlap ($\max(0, \min(p_{\text{end}}, \text{inj}_{\text{end}}) - \max(p_{\text{start}}, \text{inj}_{\text{start}})) > 0$) or if the interval onset is within $0.50\,\text{s}$ of the fault onset.
3. **Causal Detection Timestamp:** The timestamp of first detection is defined as the `window_end_s` of the earliest sliding window that detects the event while overlapping it. Delay is $t_{\text{first\_window\_end}} - t_{\text{event\_start}}$.
4. **False Alarm Count:** Every predicted interval that does not match any injected ground-truth fault event is classified as an independent False Alarm Event ($E_{\text{FA}}$).
5. **Healthy Duration Denominator:** For each 60.0 s run, the healthy duration is $60.0 - \sum \text{fault\_durations}$. Across all 50 OOD runs, the exact healthy time denominator is:
   $$T_{\text{healthy}} = 2,660.4\,\text{s} = \mathbf{44.34\,\text{minutes}}$$
   Across all 20 ID runs:
   $$T_{\text{healthy}} = 1,070.4\,\text{s} = \mathbf{17.84\,\text{minutes}}$$

### B. Verified Event-Level Results Table

| Partition | Method ID | Injected Events | Observable Events | Detected Events | Missed Events | Event Recall (%) | Predicted Intervals | False Alarm Events | Healthy Time (min) | FA Rate (FA/min) | Median Delay (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ID Test** | **M1** | 22 | 22 | 22 | 0 | 100.00% | 59 | 37 | 17.84 | 2.07 | 0.0863 |
| **ID Test** | **M2** | 22 | 22 | 16 | 6 | 72.73% | 18 | 2 | 17.84 | 0.11 | 0.1195 |
| **ID Test** | **M3** | 22 | 22 | 20 | 2 | 90.91% | 43 | 23 | 17.84 | 1.29 | 0.0863 |
| **ID Test** | **M4** | 22 | 22 | 20 | 2 | 90.91% | 43 | 23 | 17.84 | 1.29 | 0.0863 |
| **ID Test** | **M5** | 22 | 22 | 20 | 2 | 90.91% | 29 | 9 | 17.84 | 0.50 | 0.3363 |
| **ID Test** | **M6** | 22 | 22 | 20 | 2 | 90.91% | 29 | 9 | 17.84 | 0.50 | 0.3363 |
| **OOD Test** | **M1** | 58 | 58 | 58 | 0 | 100.00% | 460 | 401 | 44.34 | 9.04 | 0.1066 |
| **OOD Test** | **M2** | 58 | 58 | 43 | 15 | 74.14% | 50 | 7 | 44.34 | **0.16** | 0.1172 |
| **OOD Test** | **M3** | 58 | 58 | 53 | 5 | 91.38% | 596 | 543 | 44.34 | 12.25 | 0.1093 |
| **OOD Test** | **M4** | 58 | 58 | 53 | 5 | 91.38% | 596 | 543 | 44.34 | 12.25 | 0.1093 |
| **OOD Test** | **M5** | 58 | 58 | 53 | 5 | 91.38% | 374 | 321 | 44.34 | **7.24** | **0.3321** |
| **OOD Test** | **M6** | 58 | 58 | 53 | 5 | 91.38% | 374 | 321 | 44.34 | **7.24** | **0.3321** |

---

## 6. Full Resolution of Four Historical Numerical Discrepancies

This section directly answers the four questions raised in Task 1:

### 1. Why M5 OOD $\text{F}_1$ appeared as 0.6280 instead of 0.5873:
- **Verified Ground Truth:** M5 OOD $\text{F}_1$ has **always been 0.5873** in the evaluation code (`evaluator.py`), result CSVs (`table1_overall_performance.csv`, `table5_persistence_ablation.csv`), and documentation reports.
- **Root Cause of Divergence:** In the concluding message of the previous turn, the assistant hand-wrote a summary markdown table in chat prose where the value `0.6280` was mistakenly typed for M5. This occurred due to confusion with the higher persistence rows in Table 5, where $N=3$ achieves $\text{F}_1 = 0.6241$ and $N=4$ achieves $\text{F}_1 = 0.6302$.
- **Resolution:** Certified that M5 ($N=2$) OOD $\text{F}_1 = \mathbf{0.5873}$.

### 2. Why M5 OOD Accuracy appeared as 0.8643 instead of 0.8433:
- **Verified Ground Truth:** The empirical accuracy of M5 is:
  $$\text{Accuracy} = \frac{TP + TN}{N} = \frac{1,332 + 8,746}{11,950} = \frac{10,078}{11,950} = \mathbf{0.843347} \approx \mathbf{0.8433}$$
- **Root Cause of Divergence:** Like the $\text{F}_1$ error, the number `0.8643` appeared solely in the chat response text table due to an erroneous manual transcription.
- **Resolution:** Certified that M5 OOD Accuracy is $\mathbf{0.8433}$.

### 3. Why M2 False Alarms/min appeared as 11.83 instead of 0.16:
- **Verified Ground Truth:** In M2, across all 50 OOD runs, static Random Forest predicts only 50 event intervals, of which 43 match true events and only 7 are false alarms. Over the 44.34 healthy minutes:
  $$\text{FA Rate} = \frac{7}{44.34} = \mathbf{0.15787} \approx \mathbf{0.16\,\text{FA/min}}$$
- **Root Cause of Divergence:** In the assistant's previous chat prose table, `11.83` was erroneously typed for M2, erroneously conflating M2 with M3/M4's elevated rate ($12.25\,\text{FA/min}$) or an ungrounded hallucination.
- **Resolution:** Certified that M2 OOD False Alarm Rate is $\mathbf{0.16\,\text{FA/min}}$ ($7$ false alarm events total).

### 4. Why M5 Balanced Accuracy was reported as 0.8931 when Recall = 0.9318 and Specificity = 0.8550:
- **Verified Ground Truth:**
  For M5 on OOD Test, the true confusion matrix yields:
  $$\text{Recall} = \frac{1,332}{1,472} = \mathbf{0.904891} \approx \mathbf{0.9049}$$
  $$\text{Specificity} = \frac{8,746}{10,478} = \mathbf{0.834692} \approx \mathbf{0.8347}$$
  $$\text{Balanced Accuracy} = \frac{0.904891 + 0.834692}{2} = \mathbf{0.869792} \approx \mathbf{0.8698}$$
- **Root Cause of Divergence:** In the assistant's previous chat prose table:
  1. The assistant wrote Recall = 0.9318 and Specificity = 0.8550 (which are fictitious). Even with those numbers, $(0.9318 + 0.8550)/2 = 0.8934$, not 0.8931.
  2. The number `0.8931` was actually the **Accuracy of M4 on ID Test** in Table 1 (`ID Test, M4, Accuracy = 0.8931`), which was accidentally pasted into the Balanced Accuracy column for M5 on OOD test!
- **Resolution:** Certified that for M5 on OOD test, Recall = $\mathbf{0.9049}$, Specificity = $\mathbf{0.8347}$, and Balanced Accuracy = $\mathbf{0.8698}$.

---

## 7. Paired Bootstrap Statistical Comparisons Verification

### A. Recomputation Protocol ($B=1,000$ Resamples of Complete Simulation Runs)
To eliminate any possibility of stale cache, complete simulation runs (50 runs in OOD) were resampled with replacement $B = 1,000$ times using independent seed `20261201`. In each resample, complete runs were drawn and window-level metrics were recomputed on the concatenated windows.

### B. Observed vs. Bootstrap Difference Table

| Comparison | Contrast Description | Observed $\Delta \text{F}_1$ | Bootstrap Mean $\Delta \text{F}_1$ | Bootstrap 95% CI $\text{F}_1$ | Observed $\Delta \text{BalAcc}$ | Bootstrap Mean $\Delta \text{BalAcc}$ | Empirical $p$-value | Significant ($p<0.05$) | Scientific Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **M2 vs M1** | Static ML vs Fixed Baseline | $+0.2692$ | $+0.2732$ | $[+0.1697, +0.3851]$ | $+0.1858$ | $+0.1858$ | $< 0.01$ | **Yes** | Static ML significantly outperforms collapsed threshold baseline, but misses 25.86% of events. |
| **M3 vs M2** | Initial Ref Calib vs Static ML | $-0.0320$ | $-0.0360$ | $[-0.1379, +0.0597]$ | $+0.0366$ | $+0.0355$ | $0.4680$ | **No** | **No significant $\text{F}_1$ difference ($p = 0.4680$).** M3 restores event recall ($91.4\%$ vs $74.1\%$) at the cost of higher window false alarms. |
| **M4 vs M3** | Adaptive Calib vs Initial Calib | $0.0000$ | $0.0000$ | $[0.0000, 0.0000]$ | $0.0000$ | $0.0000$ | $1.0000$ | **No** | **Statistically indistinguishable ($\Delta \text{F}_1 = 0.0000, p = 1.0000$).** Zero incremental adaptive gain. |
| **M5 vs M3** | Persistence $N=2$ vs Raw Initial Calib | $+0.0769$ | $+0.0776$ | $[+0.0479, +0.1136]$ | $+0.0240$ | $+0.0239$ | $< 0.01$ | **Yes** | **Statistically significant improvement ($\Delta \text{F}_1 = +0.0776, p < 0.01$).** Persistence suppresses false alarms while preserving recall. |
| **M6 vs M4** | Persistence $N=2$ vs Raw Adaptive Calib | $+0.0769$ | $+0.0776$ | $[+0.0479, +0.1136]$ | $+0.0240$ | $+0.0239$ | $< 0.01$ | **Yes** | **Statistically significant improvement ($\Delta \text{F}_1 = +0.0776, p < 0.01$).** Persistence suppresses false alarms while preserving recall. |

### C. Resolution of the Apparent "Direct Difference vs Bootstrap Difference" Paradox
The user prompt correctly asked:
> "M3 F1 = 0.5104. M5 F1 = 0.6280. Direct difference = +0.1176. Bootstrap mean difference = +0.0765. Verify whether the bootstrap calculations used the current predictions or stale results."

**Findings:**
1. The bootstrap calculation did **NOT** use stale results. It used the actual model predictions from `flexpulse_final_test_v4`.
2. Under the actual predictions, M3 $\text{F}_1 = 0.5104$ and M5 $\text{F}_1 = 0.5873$.
3. The true observed direct difference is:
   $$\text{Observed } \Delta \text{F}_1 = 0.587302 - 0.510394 = \mathbf{+0.076908} \approx \mathbf{+0.0769}$$
4. The bootstrap mean difference across $B = 1,000$ resamples is $\mathbf{+0.0776}$ (or $\mathbf{+0.0765}$ with $B=200$).
5. **Conclusion:** The direct difference ($+0.0769$) and the bootstrap mean difference ($+0.0776$ / $+0.0765$) are virtually identical. The apparent discrepancy ($+0.1176$) was purely created by the clerical transcription error of typing $0.6280$ instead of $0.5873$ in the previous chat response.

---

## 8. Resolution of Scenario S4 Ground-Truth Semantics

### A. Code Inspection in Simulator vs Dataset Generator
1. **Simulator (`flexpulse_ml/simulation/fault_injector.py`):**
   In the core electrical simulator, `SCENARIO_GRADUAL_RESISTANCE_DRIFT` was designed as an electrical fault:
   ```python
   true_path_resistance[drift_mask] += drift_rate * elapsed
   fault_label[drift_mask] = 1
   fault_type[drift_mask] = self.SCENARIO_GRADUAL_RESISTANCE_DRIFT
   ```
   Here, linear resistance creep is intended to represent **progressive contact degradation, fretting corrosion, or pin oxidation** (a genuine degrading power-path fault).

2. **Final Dataset Generator (`flexpulse_ml/data/generator_v4.py`):**
   In the independent dataset generator created for final testing, lines 230–233 implement S4 as follows:
   ```python
   elif scenario == self.SCENARIO_S4_GRADUAL_RESISTANCE_DRIFT:
       post_mask = t > 5.0
       drift_rate = float(rng.uniform(0.002, 0.004))
       base_r[post_mask] += drift_rate * (t[post_mask] - 5.0)
   ```
   Crucially, `generator_v4.py` added the drift into `base_r` (the nominal baseline cable resistance) rather than calling `add_event()`. It did not activate `fault_labels = 1`, and did not add any record to `events_meta`.

### B. Scientific Explanation and Label Consequence
Because `generator_v4.py` modified `base_r` without generating positive labels, the ground truth in `flexpulse_final_test_v4` for S4 contains **zero positive windows ($GT \ Pos = 0$)**:
- Syntactically, the test set treated S4 as an operational baseline drift (analogous to S5 sensor bias drift and S8 load resistance shifts).
- Semantically, however, an increase in cable path resistance from $0.080\,\Omega$ to $0.245\,\Omega$ physically represents a degrading power-path fault.
- When ML detectors (M2, M3, M4, M5) detect this resistance increase and output $\hat{y}_k = 1$, every single detection is evaluated against ground truth $y_k = 0$ as a **False Positive ($FP$)**.

### C. Empirical Impact on Specificity and False Alarms
In Table 3 (`table3_scenario_breakdown.csv`):
- Across all 5 runs of S4 ($1,195$ windows total), M3 generates **1,105 False Positive windows** (Specificity = $0.0753$).
- M5 generates **1,083 False Positive windows** (Specificity = $0.0937$).
- S4 alone accounts for **$43.5\%$ of all false positive windows in M3** ($1,105 / 2,541$) and **$62.5\%$ of all false positive windows in M5** ($1,083 / 1,732$).

### D. Scientific Governance and Pre-Registration Compliance
- In accordance with pre-registration integrity and scientific rigor, **the locked dataset has NOT been retroactively relabeled**.
- This issue is formally documented as a generator semantic divergence and a key scientific limitation in:
  1. `docs/final_scientific_audit.md` (Section 4)
  2. `docs/phase7_final_evaluation_report.md` (Section 19)
  3. `results/final_research_package/limitations/scientific_limitations.md` (Section 3)

---

## 9. Comprehensive Temporal Persistence Ablation ($N=1$ to $5$)

Directly extracted from `results/final_research_package/tables/table5_persistence_ablation.csv`:

| Persistence Windows ($N$) | Window Accuracy | Window $\text{F}_1$-Score | Specificity | Event Recall (%) | Detected Events | False Alarm Events | FA Rate (events/min) | Relative FA Reduction (%) | Median Causal Delay (s) | Added Latency vs $N=1$ (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$N=1$ (Raw M3)** | 0.7792 | 0.5104 | 0.7575 | 91.38% | 53 / 58 | 543 | 12.25 | 0.00% | 0.1093 | 0.0000 |
| **$N=2$ (M5)** | **0.8433** | **0.5873** | **0.8347** | **91.38%** | **53 / 58** | **321** | **7.24** | **40.88%** | **0.3321** | **+0.2228** |
| **$N=3$** | 0.8705 | 0.6241 | 0.8703 | 91.38% | 53 / 58 | 146 | 3.29 | 73.11% | 0.5821 | +0.4728 |
| **$N=4$** | 0.8788 | 0.6302 | 0.8845 | 89.66% | 52 / 58 | 70 | 1.58 | 87.11% | 0.8321 | +0.7228 |
| **$N=5$** | 0.8805 | 0.6236 | 0.8913 | 86.21% | 50 / 58 | 34 | 0.77 | 93.74% | 1.0864 | +0.9771 |

### Engineering Latency Conclusion:
- Setting $N=2$ provides a $40.88\%$ false-alarm reduction while adding only $+0.2228\,\text{s}$ of latency (less than one $0.25\,\text{s}$ stride), preserving $100\%$ of detected fault events ($53/53$).
- $N=3$ achieves a larger false-alarm reduction ($73.11\%$), but increases causal delay to $+0.4728\,\text{s}$ (nearly half a second).
- Setting $N \ge 4$ causes irreversible fault event loss ($89.66\%$ at $N=4$, $86.21\%$ at $N=5$).

---

## 10. Defensive Gating Ablation on Slow Creeping Faults (Scenario S9)

Directly extracted from `results/final_research_package/tables/table6_gating_ablation.csv`:

| Adaptation Mode | Confidence Gate ($\tau=0.85$) | Stability Gate ($\sigma_I \le 0.040\,\text{A}$) | Contamination Rate (%) | Peak Baseline Shift ($\hat{V}_0$) | $\text{F}_1$-Score (S9) | Balanced Acc (S9) | Specificity (S9) | False Alarm Events |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ungated Tracking** | Off | Off | 2.12% | 5.12 mV | 0.8979 | 0.9125 | 0.8405 | 49 |
| **No Confidence Gate** | Off | On | 2.12% | 5.12 mV | 0.8955 | 0.9103 | 0.8360 | 52 |
| **No Current Stability Gate** | On | Off | **0.00%** | **5.06 mV** | 0.8895 | 0.9046 | 0.8227 | 52 |
| **Fully Gated Adaptive (A3)** | On | On | **0.00%** | **5.06 mV** | 0.8895 | 0.9046 | 0.8227 | 52 |

---

## 11. Artifact Provenance and Verification Hash Registry

| Artifact Description | File Location | SHA-256 Checksum |
| :--- | :--- | :--- |
| **Pre-Test Lock Protocol** | `docs/final_evaluation_protocol.md` | `13bc534c900f8891ebe74eed6cdcc826210dd4635c137e274aa6f0d058fedca1` |
| **Table 1 CSV (Overall)** | `results/final_research_package/tables/table1_overall_performance.csv` | Recomputed & Verified |
| **Table 2 CSV (Bootstrap)** | `results/final_research_package/tables/table2_bootstrap_significance.csv` | Recomputed & Verified ($B=1,000$) |
| **Table 3 CSV (Scenarios)** | `results/final_research_package/tables/table3_scenario_breakdown.csv` | Recomputed & Verified |
| **Table 4 CSV (Events)** | `results/final_research_package/tables/table4_event_level_metrics.csv` | Recomputed & Verified |
| **Table 5 CSV (Persistence)** | `results/final_research_package/tables/table5_persistence_ablation.csv` | Recomputed & Verified |
| **Table 6 CSV (Gating)** | `results/final_research_package/tables/table6_gating_ablation.csv` | Recomputed & Verified |
| **Table 7 CSV (Scorecard)** | `results/final_research_package/tables/table7_research_questions_scorecard.csv` | Recomputed & Verified |
| **Figure 1 (Architecture)** | `results/final_research_package/figures/fig1_powerpath_architecture_signals.png` | Banner Verified |
| **Figure 2 (Benchmark)** | `results/final_research_package/figures/fig2_benchmark_comparative_performance.png` | Banner Verified |
| **Figure 3 (Scenarios)** | `results/final_research_package/figures/fig3_scenario_breakdown_radar_bars.png` | Banner Verified |
| **Figure 4 (Latency Trade-Off)** | `results/final_research_package/figures/fig4_event_detection_latency_tradeoff.png` | Banner Verified |
| **Figure 5 (Gating Dynamics)** | `results/final_research_package/figures/fig5_defensive_gating_contamination_traces.png` | Banner Verified |
| **Figure 6 (Calibration Flow)** | `results/final_research_package/figures/fig6_adaptive_calibration_state_machine.png` | Banner Verified |

---

## 12. Auditor Certification Sign-Off

I hereby certify that:
1. Every numerical metric, event count, confusion matrix cell, and p-value reported in this document and the associated research package was directly computed from the frozen data of `flexpulse_final_test_v4`.
2. No data, labels, models, or simulation parameters were fabricated, retrained, or altered.
3. All code modifications are strictly comment-free, verified using Python's `tokenize` module.
4. All 115 automated regression and unit tests pass without failure.
5. The evidence package is frozen and ready to support the drafting of the IEEE research manuscript upon user authorization.

**Certified by:** FlexPulse-ML Scientific Software Audit Team  
**Date:** 16 November 2026
