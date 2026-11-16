| Partition | Method ID | Method Name | Accuracy | Balanced Acc | Precision | Recall | Specificity | F1-Score | Event Recall | FA Rate (FA/min) | Median Delay (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ID Test | M1 | Fixed Threshold Envelope (A0) | 0.8395 | 0.773 | 0.6123 | 0.6578 | 0.8883 | 0.6342 | 1.0 | 1.88 | 0.1196 |
| ID Test | M2 | Static Random Forest (A1) | 0.8858 | 0.8545 | 0.7016 | 0.8002 | 0.9087 | 0.7477 | 0.76 | 0.0 | 0.1196 |
| ID Test | M3 | Initial Ref Calibration + RF (A2) | 0.978 | 0.9687 | 0.9441 | 0.9525 | 0.9849 | 0.9483 | 0.92 | 1.69 | 0.1196 |
| ID Test | M4 | Confidence-Gated Adaptive RF (A3) | 0.9778 | 0.9682 | 0.9441 | 0.9515 | 0.9849 | 0.9478 | 0.92 | 1.69 | 0.1196 |
| ID Test | M5 | Initial Ref Calib + Persistence N=2 | 0.9805 | 0.9623 | 0.9761 | 0.9308 | 0.9939 | 0.9529 | 0.92 | 0.56 | 0.3696 |
| ID Test | M6 | Adaptive Calib + Persistence N=2 | 0.9803 | 0.9618 | 0.9761 | 0.9298 | 0.9939 | 0.9524 | 0.92 | 0.56 | 0.3696 |
| OOD Test | M1 | Fixed Threshold Envelope (A0) | 0.4165 | 0.5725 | 0.2488 | 0.8465 | 0.2984 | 0.3846 | 1.0 | 9.0 | 0.1239 |
| OOD Test | M2 | Static Random Forest (A1) | 0.8664 | 0.8115 | 0.6810 | 0.7148 | 0.9081 | 0.6975 | 0.7656 | 0.05 | 0.1322 |
| OOD Test | M3 | Initial Ref Calibration + RF (A2) | 0.8722 | 0.9028 | 0.6350 | 0.9565 | 0.8491 | 0.7633 | 0.9219 | 12.95 | 0.1322 |
| OOD Test | M4 | Confidence-Gated Adaptive RF (A3) | 0.8722 | 0.9028 | 0.6350 | 0.9565 | 0.8491 | 0.7633 | 0.9219 | 12.95 | 0.1322 |
| OOD Test | M5 | Initial Ref Calib + Persistence N=2 | 0.9321 | 0.9335 | 0.7885 | 0.9359 | 0.9311 | 0.8559 | 0.9219 | 7.92 | 0.3407 |
| OOD Test | M6 | Adaptive Calib + Persistence N=2 | 0.9321 | 0.9335 | 0.7885 | 0.9359 | 0.9311 | 0.8559 | 0.9219 | 7.92 | 0.3407 |
