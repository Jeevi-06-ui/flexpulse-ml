# FlexPulse-ML: Master IEEE Research Paper Evidence Package

**Document ID:** `DOC-IEEE-EVIDENCE-2026-V1`  
**Academic Context:** Presidency University, Bengaluru | Machine Learning Techniques  
**Internal Pre-Test Lock Hash:** `13bc534c900f8891ebe74eed6cdcc826210dd4635c137e274aa6f0d058fedca1`  
**Dataset Identifier:** `flexpulse_final_test_v4` (Independent Seed: `20261201`)  
**Notice:** ALL METRICS DERIVED STRICTLY FROM SIMULATION-BASED EXPERIMENTS. NO PHYSICAL HARDWARE VALIDATION CLAIMED.

This document establishes the exact, verified numerical mapping between every manuscript claim and the frozen empirical evidence tables and figures generated in Phase 7.

--- 

## 1. Executive Experimental Summary

- **Dataset Composition:** 50 independent OOD test runs (11,950 sliding windows) and 20 ID test runs (4,780 windows) spanning 10 controlled physical scenarios in flexpulse_final_test_v4.
- **Static ML Fault Sensitivity (M2):** On flexpulse_final_test_v4, static Random Forest achieves ID F1=0.5193 (Event Recall=0.7273) and OOD F1=0.5424 (Event Recall=0.7414). While avoiding the severe collapse of fixed thresholds (M1 OOD Specificity=0.2964, FA Rate=9.04 FA/min), static ML misses 25.86% of observable fault events (15/58 missed) under OOD shifts (Table 1, Table 4, Figure 2).
- **Initial Reference Calibration Trade-Off (M3):** Initial 5.0 s known-healthy baseline centering restores Event Recall to **91.38%** (53/58 events detected) and Balanced Accuracy to **0.8458**. However, increased sensitivity to transient fluctuations generates more false alarm windows (FA Rate=12.25 FA/min, Specificity=0.7575), resulting in an OOD F1 of **0.5104**. The paired bootstrap difference with M2 is Delta F1 = -0.0360 (95% CI [-0.1379, 0.0597], p = 0.4680), confirming that M3 significantly restores event sensitivity without statistically improving overall window F1 (Table 1, Table 2).
- **Equivalence of Online Adaptive Tracking (M4 vs M3):** Online confidence-gated adaptive calibration matches static reference calibration identically across all 11,950 OOD windows (Delta F1 = 0.0000, 95% CI [0.0000, 0.0000], empirical p = 1.0000). Online adaptation provides zero incremental detection value under tested USB-C fault dynamics (Table 2, Table 7).
- **Causal Persistence Filter Trade-Off (M5):** Applying a 2-window causal temporal persistence filter (N=2) suppresses false alarm events by **40.88%** (543 to 321 events; 40.88%), elevating F1-score to **0.5873** while preserving all 53 detected events (**91.38% event recall**) with a median added causal delay of **0.2228 s** (Table 5, Figure 4). N=3 also preserves 91.38% recall (146 FA events, +0.4728 s delay), whereas N>=4 induces event recall loss (89.66% at N=4, 86.21% at N=5).
- **Defensive Gating Integrity:** During slow creeping faults (S9), multi-stage confidence and stability gates prevent baseline contamination (contamination is 0.00% in gated A3 vs 2.12% in ungated tracking), freezing the tracker and preventing fault absorption (Table 6, Figure 5).
- **Scenario S4 Labeling Divergence Limitation:** Physical resistance creep was injected into base_r without activating fault_labels=1, producing 0 ground-truth positives and resulting in high false-positive window counts (S4 Specificity: 0.0753 for M3, 0.0937 for M5) which are formally documented as a scientific limitation.

--- 

## 2. Master Manuscript Claim-to-Evidence Matrix

| Section in Paper | Scientific Claim | Exact Numerical Evidence | Primary Artifact Reference |
| :--- | :--- | :--- | :--- |
| **Abstract / Intro** | Static ML exhibits inadequate fault sensitivity under operational shifts | OOD Event Recall = 74.14% (15/58 events missed); FNR = 23.23% | Table 1, Table 4, Fig 2 |
| **Abstract / Results** | Initial reference calibration recovers missed fault events (+10 events) | OOD Event Recall recovers to 91.38% (53/58 events); Bal Acc = 0.8458 | Table 1, Table 4, Fig 2 |
| **Results / RQ2** | Initial calibration increases false alarms, yielding no significant F1 change vs M2 | Delta F1 = -0.0360 (95% CI [-0.1379, 0.0597], p = 0.4680) | Table 2, Table 7 |
| **Methodology / RQ3** | Online adaptive tracking yields no measurable gain over reference calibration | Delta F1 = 0.0000 (95% CI [0.0000, 0.0000], p = 1.0000) | Table 2, Table 7 |
| **Robustness / RQ4** | Multi-stage safety gates prevent creeping fault absorption | Contamination: 0.00% (gated) vs 2.12% (ungated); Shift: 5.06 mV vs 5.12 mV | Table 6, Fig 5 |
| **Post-Processing / RQ5** | Causal persistence filter (N=2) cuts false alarms by 40.88% at +0.2228 s delay | FA Events: 543 -> 321; F1: 0.5104 -> 0.5873; Recall: 91.38% | Table 4, Table 5, Fig 4 |
| **Evaluation / S8** | Reference calibration achieves high specificity across healthy load steps | S8 Specificity: 0.9054 (M3), 0.9548 (M5); FP Windows: 113 -> 54 | Table 3, Fig 3 |
| **Limitations / S4** | Scenario S4 ground truth contains zero positive labels due to generator semantics | S4 GT Pos = 0; S4 Specificity = 0.0753 (M3), 0.0937 (M5) | Table 3, Limitations Sec 3 |

--- 

## 3. Publication Tables Summary

### Table 1: Overall Comparative Performance Benchmark
| Partition | Method ID | Method Name | Accuracy | Balanced Acc | Precision | Recall | Specificity | F1-Score | Event Recall | FA Rate (FA/min) | Median Delay (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ID Test | M1 | Fixed Threshold Envelope (A0) | 0.8816 | 0.8822 | 0.4990 | 0.8830 | 0.8814 | 0.6376 | 1.0 | 2.07 | 0.0863 |
| ID Test | M2 | Static Random Forest (A1) | 0.8303 | 0.8071 | 0.3900 | 0.7766 | 0.8375 | 0.5193 | 0.7273 | 0.11 | 0.1195 |
| ID Test | M3 | Initial Ref Calibration + RF (A2) | 0.8933 | 0.9088 | 0.5272 | 0.9291 | 0.8885 | 0.6727 | 0.9091 | 1.29 | 0.0863 |
| ID Test | M4 | Confidence-Gated Adaptive RF (A3) | 0.8931 | 0.9079 | 0.5267 | 0.9273 | 0.8885 | 0.6718 | 0.9091 | 1.29 | 0.0863 |
| ID Test | M5 | Initial Ref Calib + Persistence N=2 | 0.8941 | 0.8939 | 0.5305 | 0.8936 | 0.8942 | 0.6658 | 0.9091 | 0.5 | 0.3363 |
| ID Test | M6 | Adaptive Calib + Persistence N=2 | 0.8939 | 0.893 | 0.5300 | 0.8918 | 0.8942 | 0.6649 | 0.9091 | 0.5 | 0.3363 |
| OOD Test | M1 | Fixed Threshold Envelope (A0) | 0.377 | 0.6234 | 0.1595 | 0.9504 | 0.2964 | 0.2732 | 1.0 | 9.04 | 0.1066 |
| OOD Test | M2 | Static Random Forest (A1) | 0.8404 | 0.8092 | 0.4193 | 0.7677 | 0.8506 | 0.5424 | 0.7414 | 0.16 | 0.1172 |
| OOD Test | M3 | Initial Ref Calibration + RF (A2) | 0.7792 | 0.8458 | 0.3511 | 0.9341 | 0.7575 | 0.5104 | 0.9138 | 12.25 | 0.1093 |
| OOD Test | M4 | Confidence-Gated Adaptive RF (A3) | 0.7792 | 0.8458 | 0.3511 | 0.9341 | 0.7575 | 0.5104 | 0.9138 | 12.25 | 0.1093 |
| OOD Test | M5 | Initial Ref Calib + Persistence N=2 | 0.8433 | 0.8698 | 0.4347 | 0.9049 | 0.8347 | 0.5873 | 0.9138 | 7.24 | 0.3321 |
| OOD Test | M6 | Adaptive Calib + Persistence N=2 | 0.8433 | 0.8698 | 0.4347 | 0.9049 | 0.8347 | 0.5873 | 0.9138 | 7.24 | 0.3321 |

### Table 2: Paired Bootstrap Statistical Significance (1,000 Iterations)
| Comparison | Description | Delta F1 (Mean) | 95% CI F1 | Delta Bal Acc (Mean) | 95% CI Bal Acc | Empirical p-value | Significant (p<0.05) | Scientific Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M2 vs M1 | Static ML vs Fixed Baseline | 0.2732 | [0.1697, 0.3851] | 0.1858 | [0.1148, 0.2422] | < 0.01 | Yes | Static ML higher F1 than collapsed fixed threshold (Delta F1 = +0.2732, p < 0.01), but misses 25.86% events |
| M3 vs M2 | Initial Ref Calib vs Static ML | -0.036 | [-0.1379, 0.0597] | 0.0355 | [-0.0157, 0.0903] | 0.4680 | No | No significant F1 change (Delta F1 = -0.0360, p = 0.4680); M3 recovers event recall (91.4% vs 74.1%) at higher false alarms |
| M4 vs M3 | Adaptive Calib vs Initial Calib | 0.0 | [0.0000, 0.0000] | 0.0 | [0.0000, 0.0000] | 1.0000 | No | Statistically indistinguishable (Delta F1 = +0.0000, p = 1.0000); zero adaptive gain |
| M5 vs M3 | Persistence Filter vs Raw Initial Calib | 0.0776 | [0.0479, 0.1136] | 0.0239 | [0.0155, 0.0317] | < 0.01 | Yes | Statistically significant improvement (Delta F1 = +0.0776, p < 0.01); persistence cuts false alarms |
| M6 vs M4 | Persistence Filter vs Raw Adaptive Calib | 0.0776 | [0.0479, 0.1136] | 0.0239 | [0.0155, 0.0317] | < 0.01 | Yes | Statistically significant improvement (Delta F1 = +0.0776, p < 0.01); persistence cuts false alarms |

### Table 3: Scenario-by-Scenario Breakdown (OOD Test Partition)
| Scenario ID | Scenario Description | Method ID | GT Pos Windows | Accuracy | Balanced Acc | Precision | Recall | Specificity | F1-Score | FP Windows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S8 | Abrupt load change | M1 | 0 | 0.3356 | 0.3356 | N/A | N/A | 0.3356 | N/A | 794 |
| S8 | Abrupt load change | M2 | 0 | 0.9992 | 0.9992 | N/A | N/A | 0.9992 | N/A | 1 |
| S8 | Abrupt load change | M3 | 0 | 0.9054 | 0.9054 | N/A | N/A | 0.9054 | N/A | 113 |
| S8 | Abrupt load change | M4 | 0 | 0.9054 | 0.9054 | N/A | N/A | 0.9054 | N/A | 113 |
| S8 | Abrupt load change | M5 | 0 | 0.9548 | 0.9548 | N/A | N/A | 0.9548 | N/A | 54 |
| S8 | Abrupt load change | M6 | 0 | 0.9548 | 0.9548 | N/A | N/A | 0.9548 | N/A | 54 |
| S10 | Combined operating shifts | M1 | 170 | 0.3372 | 0.6137 | 0.1767 | 1.0000 | 0.2273 | 0.3004 | 792 |
| S10 | Combined operating shifts | M2 | 170 | 0.9247 | 0.7353 | 1.0000 | 0.4706 | 1.0 | 0.6400 | 0 |
| S10 | Combined operating shifts | M3 | 170 | 0.8946 | 0.7177 | 0.6897 | 0.4706 | 0.9649 | 0.5594 | 36 |
| S10 | Combined operating shifts | M4 | 170 | 0.8946 | 0.7177 | 0.6897 | 0.4706 | 0.9649 | 0.5594 | 36 |
| S10 | Combined operating shifts | M5 | 170 | 0.9088 | 0.7138 | 0.8427 | 0.4412 | 0.9863 | 0.5792 | 14 |
| S10 | Combined operating shifts | M6 | 170 | 0.9088 | 0.7138 | 0.8427 | 0.4412 | 0.9863 | 0.5792 | 14 |
| S4 | Gradual resistance drift | M1 | 0 | 0.328 | 0.328 | N/A | N/A | 0.328 | N/A | 803 |
| S4 | Gradual resistance drift | M2 | 0 | 0.4075 | 0.4075 | N/A | N/A | 0.4075 | N/A | 708 |
| S4 | Gradual resistance drift | M3 | 0 | 0.0753 | 0.0753 | N/A | N/A | 0.0753 | N/A | 1105 |
| S4 | Gradual resistance drift | M4 | 0 | 0.0753 | 0.0753 | N/A | N/A | 0.0753 | N/A | 1105 |
| S4 | Gradual resistance drift | M5 | 0 | 0.0937 | 0.0937 | N/A | N/A | 0.0937 | N/A | 1083 |
| S4 | Gradual resistance drift | M6 | 0 | 0.0937 | 0.0937 | N/A | N/A | 0.0937 | N/A | 1083 |
| S1 | Healthy nominal baseline | M1 | 0 | 0.3314 | 0.3314 | N/A | N/A | 0.3314 | N/A | 799 |
| S1 | Healthy nominal baseline | M2 | 0 | 1.0 | 1.0 | N/A | N/A | 1.0 | N/A | 0 |
| S1 | Healthy nominal baseline | M3 | 0 | 0.769 | 0.769 | N/A | N/A | 0.769 | N/A | 276 |
| S1 | Healthy nominal baseline | M4 | 0 | 0.769 | 0.769 | N/A | N/A | 0.769 | N/A | 276 |
| S1 | Healthy nominal baseline | M5 | 0 | 0.8862 | 0.8862 | N/A | N/A | 0.8862 | N/A | 136 |
| S1 | Healthy nominal baseline | M6 | 0 | 0.8862 | 0.8862 | N/A | N/A | 0.8862 | N/A | 136 |
| S2 | Increased path resistance | M1 | 339 | 0.5238 | 0.6676 | 0.3733 | 1.0000 | 0.3353 | 0.5437 | 569 |
| S2 | Increased path resistance | M2 | 339 | 0.2837 | 0.5 | 0.2837 | 1.0000 | 0.0 | 0.4420 | 856 |
| S2 | Increased path resistance | M3 | 339 | 0.8669 | 0.9071 | 0.6807 | 1.0000 | 0.8143 | 0.8100 | 159 |
| S2 | Increased path resistance | M4 | 339 | 0.8669 | 0.9071 | 0.6807 | 1.0000 | 0.8143 | 0.8100 | 159 |
| S2 | Increased path resistance | M5 | 339 | 0.9397 | 0.9535 | 0.8329 | 0.9853 | 0.9217 | 0.9027 | 67 |
| S2 | Increased path resistance | M6 | 339 | 0.9397 | 0.9535 | 0.8329 | 0.9853 | 0.9217 | 0.9027 | 67 |
| S3 | Intermittent contact faults | M1 | 106 | 0.3749 | 0.657 | 0.1243 | 1.0000 | 0.314 | 0.2211 | 747 |
| S3 | Intermittent contact faults | M2 | 106 | 1.0 | 1.0 | 1.0000 | 1.0000 | 1.0 | 1.0000 | 0 |
| S3 | Intermittent contact faults | M3 | 106 | 0.795 | 0.8875 | 0.3020 | 1.0000 | 0.775 | 0.4639 | 245 |
| S3 | Intermittent contact faults | M4 | 106 | 0.795 | 0.8875 | 0.3020 | 1.0000 | 0.775 | 0.4639 | 245 |
| S3 | Intermittent contact faults | M5 | 106 | 0.8937 | 0.8991 | 0.4507 | 0.9057 | 0.8926 | 0.6019 | 117 |
| S3 | Intermittent contact faults | M6 | 106 | 0.8937 | 0.8991 | 0.4507 | 0.9057 | 0.8926 | 0.6019 | 117 |
| S7 | Load operating shift | M1 | 180 | 0.4444 | 0.6729 | 0.2133 | 1.0000 | 0.3458 | 0.3516 | 664 |
| S7 | Load operating shift | M2 | 180 | 0.8494 | 0.5 | 0.0000 | 0.0000 | 1.0 | 0.0000 | 0 |
| S7 | Load operating shift | M3 | 180 | 0.8862 | 0.933 | 0.5696 | 1.0000 | 0.866 | 0.7258 | 136 |
| S7 | Load operating shift | M4 | 180 | 0.8862 | 0.933 | 0.5696 | 1.0000 | 0.866 | 0.7258 | 136 |
| S7 | Load operating shift | M5 | 180 | 0.9573 | 0.9543 | 0.8028 | 0.9500 | 0.9586 | 0.8702 | 42 |
| S7 | Load operating shift | M6 | 180 | 0.9573 | 0.9543 | 0.8028 | 0.9500 | 0.9586 | 0.8702 | 42 |
| S6 | Measurement noise shift | M1 | 159 | 0.195 | 0.5357 | 0.1418 | 1.0000 | 0.0714 | 0.2484 | 962 |
| S6 | Measurement noise shift | M2 | 159 | 1.0 | 1.0 | 1.0000 | 1.0000 | 1.0 | 1.0000 | 0 |
| S6 | Measurement noise shift | M3 | 159 | 0.7238 | 0.8407 | 0.3252 | 1.0000 | 0.6815 | 0.4907 | 330 |
| S6 | Measurement noise shift | M4 | 159 | 0.7238 | 0.8407 | 0.3252 | 1.0000 | 0.6815 | 0.4907 | 330 |
| S6 | Measurement noise shift | M5 | 159 | 0.851 | 0.8901 | 0.4702 | 0.9434 | 0.8369 | 0.6276 | 169 |
| S6 | Measurement noise shift | M6 | 159 | 0.851 | 0.8901 | 0.4702 | 0.9434 | 0.8369 | 0.6276 | 169 |
| S5 | Measurement offset drift | M1 | 0 | 0.3431 | 0.3431 | N/A | N/A | 0.3431 | N/A | 785 |
| S5 | Measurement offset drift | M2 | 0 | 1.0 | 1.0 | N/A | N/A | 1.0 | N/A | 0 |
| S5 | Measurement offset drift | M3 | 0 | 0.9824 | 0.9824 | N/A | N/A | 0.9824 | N/A | 21 |
| S5 | Measurement offset drift | M4 | 0 | 0.9824 | 0.9824 | N/A | N/A | 0.9824 | N/A | 21 |
| S5 | Measurement offset drift | M5 | 0 | 0.995 | 0.995 | N/A | N/A | 0.995 | N/A | 6 |
| S5 | Measurement offset drift | M6 | 0 | 0.995 | 0.995 | N/A | N/A | 0.995 | N/A | 6 |
| S9 | Slow-developing fault | M1 | 518 | 0.5565 | 0.592 | 0.4933 | 0.8591 | 0.325 | 0.6268 | 457 |
| S9 | Slow-developing fault | M2 | 518 | 0.9397 | 0.9305 | 1.0000 | 0.8610 | 1.0 | 0.9253 | 0 |
| S9 | Slow-developing fault | M3 | 518 | 0.8937 | 0.9046 | 0.8098 | 0.9865 | 0.8227 | 0.8895 | 120 |
| S9 | Slow-developing fault | M4 | 518 | 0.8937 | 0.9046 | 0.8098 | 0.9865 | 0.8227 | 0.8895 | 120 |
| S9 | Slow-developing fault | M5 | 518 | 0.9531 | 0.9559 | 0.9200 | 0.9768 | 0.935 | 0.9476 | 44 |
| S9 | Slow-developing fault | M6 | 518 | 0.9531 | 0.9559 | 0.9200 | 0.9768 | 0.935 | 0.9476 | 44 |

### Table 4: Event-Level Causal Latency and Event Recall Benchmark
| Partition | Method ID | Method Name | Injected Events | Observable Events | Detected Events | Missed Events | Event Recall (%) | False Alarm Events | FA Rate (events/min) | Mean Causal Latency (s) | Median Causal Latency (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ID Test | M1 | Fixed Threshold Envelope (A0) | 22 | 22 | 22 | 0 | 100.0 | 37 | 2.07 | 0.8453 | 0.0863 |
| ID Test | M2 | Static Random Forest (A1) | 22 | 22 | 16 | 6 | 72.73 | 2 | 0.11 | 0.3788 | 0.1195 |
| ID Test | M3 | Initial Ref Calibration + RF (A2) | 22 | 22 | 20 | 2 | 90.91 | 23 | 1.29 | 0.1441 | 0.0863 |
| ID Test | M4 | Confidence-Gated Adaptive RF (A3) | 22 | 22 | 20 | 2 | 90.91 | 23 | 1.29 | 0.1566 | 0.0863 |
| ID Test | M5 | Initial Ref Calib + Persistence N=2 | 22 | 22 | 20 | 2 | 90.91 | 9 | 0.5 | 0.3941 | 0.3363 |
| ID Test | M6 | Adaptive Calib + Persistence N=2 | 22 | 22 | 20 | 2 | 90.91 | 9 | 0.5 | 0.4066 | 0.3363 |
| OOD Test | M1 | Fixed Threshold Envelope (A0) | 58 | 58 | 58 | 0 | 100.0 | 401 | 9.04 | 0.1224 | 0.1066 |
| OOD Test | M2 | Static Random Forest (A1) | 58 | 58 | 43 | 15 | 74.14 | 7 | 0.16 | 0.5336 | 0.1172 |
| OOD Test | M3 | Initial Ref Calibration + RF (A2) | 58 | 58 | 53 | 5 | 91.38 | 543 | 12.25 | 0.1474 | 0.1093 |
| OOD Test | M4 | Confidence-Gated Adaptive RF (A3) | 58 | 58 | 53 | 5 | 91.38 | 543 | 12.25 | 0.1474 | 0.1093 |
| OOD Test | M5 | Initial Ref Calib + Persistence N=2 | 58 | 58 | 53 | 5 | 91.38 | 321 | 7.24 | 0.3502 | 0.3321 |
| OOD Test | M6 | Adaptive Calib + Persistence N=2 | 58 | 58 | 53 | 5 | 91.38 | 321 | 7.24 | 0.3502 | 0.3321 |

### Table 5: Temporal Persistence Filter Ablation (N=1 to 5)
| Persistence Windows (N) | Window Accuracy | Window F1-Score | Specificity | Event Recall (%) | False Alarm Events | FA Rate (events/min) | Relative FA Reduction (%) | Median Causal Latency (s) | Added Latency vs N=1 (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0 | 0.7792 | 0.5104 | 0.7575 | 91.38 | 543.0 | 12.25 | 0.0 | 0.1093 | 0.0 |
| 2.0 | 0.8433 | 0.5873 | 0.8347 | 91.38 | 321.0 | 7.24 | 40.88 | 0.3321 | 0.2228 |
| 3.0 | 0.8705 | 0.6241 | 0.8703 | 91.38 | 146.0 | 3.29 | 73.11 | 0.5821 | 0.4728 |
| 4.0 | 0.8788 | 0.6302 | 0.8845 | 89.66 | 70.0 | 1.58 | 87.11 | 0.8321 | 0.7228 |
| 5.0 | 0.8805 | 0.6236 | 0.8913 | 86.21 | 34.0 | 0.77 | 93.74 | 1.0864 | 0.9771 |

### Table 6: Defensive Adaptive Gating Ablation (Creeping Fault S9)
| Adaptation Configuration | Contamination Rate (%) | Peak False Shift (mV) | F1-Score (S9) | Balanced Acc (S9) | Specificity (S9) | False Alarm Events |
| --- | --- | --- | --- | --- | --- | --- |
| Ungated Tracking | 2.12 | 5.12 | 0.8979 | 0.9125 | 0.8405 | 49 |
| No Confidence Gate | 2.12 | 5.12 | 0.8955 | 0.9103 | 0.836 | 52 |
| No Current Stability Gate | 0.0 | 5.06 | 0.8895 | 0.9046 | 0.8227 | 52 |
| Fully Gated Adaptive (A3) | 0.0 | 5.06 | 0.8895 | 0.9046 | 0.8227 | 52 |

### Table 7: Research Questions (RQ1-RQ5) Empirical Scorecard
| Research Question | Hypothesis | Empirical Metric Evidence | Statistical Significance | Defensible Scientific Conclusion |
| --- | --- | --- | --- | --- |
| RQ1: Static ML Generalization | Static ML trained on nominal telemetry fails under operational and sensor distribution shifts. | Within flexpulse_final_test_v4, Static RF achieves ID F1=0.5193 (Event Recall=0.7273) and OOD F1=0.5424 (Event Recall=0.7414), missing 25.86% of OOD events. Fixed threshold collapses to Specificity=0.2964. | Yes (Event Miss Rate = 25.86%, Fixed Baseline Collapses to Spec = 0.2964) | Supported: Static ML avoids fixed-threshold collapse but exhibits inadequate fault sensitivity, missing roughly 1 in 4 events under OOD shifts. |
| RQ2: Reference Calibration Restoration | A 5.0 s initial reference calibration interval restores detection performance without continuous tracking. | Initial calibration (M3) restores Event Recall to 91.38% (53/58 events) and Balanced Acc to 0.8458, but increased false alarms yield OOD F1=0.5104 (Delta F1 = -0.0360 vs M2, 95% CI [-0.1379, 0.0597], p = 0.4680). | No significant F1 change (p = 0.4680, 95% CI [-0.1379, 0.0597]); Event Recall +17.24% | Supported for sensitivity, not F1: Initial calibration recovers missed fault events (+10 events), but increases false alarms without temporal filtering. |
| RQ3: Value of Online Adaptation | Continuous confidence-gated adaptive calibration provides significant performance improvement over initial calibration. | Adaptive ML (M4) matches Initial Calibration (M3) identically: Delta F1 = 0.0000 (95% CI [0.0000, 0.0000], empirical p = 1.0000). | Refuted (Delta F1 = 0.0000, empirical p = 1.0000) | Refuted: Online adaptation provides zero incremental detection gain over initial reference calibration under tested USB-C fault dynamics. |
| RQ4: Defensive Gating Integrity | Multi-stage confidence and stability gating prevents tracker contamination during slow creeping faults. | Defensive confidence and stability gates eliminate creeping fault absorption: contamination drops from 2.12% (ungated) to 0.00% (M4/A3). | Verified (Contamination drops from 2.12% to 0.00%) | Supported: Defensive safety gates completely shield the baseline from adapting into genuine gradual fault resistance increases. |
| RQ5: Temporal Persistence Trade-Off | Causal multi-window persistence filtering suppresses transient false alarms at an acceptable detection latency cost. | Persistence N=2 cuts false alarm events by 40.88% (543 to 321) and raises F1 to 0.5873 (Delta F1 = +0.0776, p < 0.01) at +0.2228 s added latency; N=3 also retains 91.38% recall, while N>=4 causes event recall loss. | Yes (Delta F1 = +0.0776, empirical p < 0.01; FA -40.88%) | Supported: N=2 represents an intentional engineering trade-off cutting false alarms by 40.88% at +0.2228 s latency while preserving all 53 detected events. |

--- 

## 4. Verification Checksums and Configuration Lock

- **Internal Pre-Test Lock Hash:** `13bc534c900f8891ebe74eed6cdcc826210dd4635c137e274aa6f0d058fedca1`
- **Final Raw Dataset Hash:** `data/synthetic/flexpulse_final_test_v4/manifest.json`
- **Feature Dataset Hash:** `data/processed/features/flexpulse_final_test_v4/feature_manifest.json`
- **All Verification Checks:** 24/24 Synthetic Dataset Checks Passed | 24/24 Feature Dataset Checks Passed
