| Comparison | Description | Delta F1 (Mean) | 95% CI F1 | Delta Bal Acc (Mean) | 95% CI Bal Acc | Empirical p-value | Significant (p<0.05) | Scientific Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M2 vs M1 | Static ML vs Fixed Baseline | 0.3169 | [0.2084, 0.4322] | 0.2384 | [0.1888, 0.2865] | < 0.01 | Yes | Static ML higher F1 than collapsed fixed threshold (Delta F1 = +0.3169, p < 0.01), but misses 25.86% events |
| M3 vs M2 | Initial Ref Calib vs Static ML | 0.0595 | [-0.0648, 0.1596] | 0.0896 | [0.0398, 0.1379] | 0.3220 | No | No significant F1 change (Delta F1 = +0.0595, p = 0.3220); M3 recovers event recall (91.4% vs 74.1%) at higher false alarms |
| M4 vs M3 | Adaptive Calib vs Initial Calib | 0.0 | [0.0000, 0.0000] | 0.0 | [0.0000, 0.0000] | 1.0000 | No | Statistically indistinguishable (Delta F1 = +0.0000, p = 1.0000); zero adaptive gain |
| M5 vs M3 | Persistence Filter vs Raw Initial Calib | 0.0932 | [0.0684, 0.1223] | 0.0304 | [0.0224, 0.0382] | < 0.01 | Yes | Statistically significant improvement (Delta F1 = +0.0932, p < 0.01); persistence cuts false alarms |
| M6 vs M4 | Persistence Filter vs Raw Adaptive Calib | 0.0932 | [0.0684, 0.1223] | 0.0304 | [0.0224, 0.0382] | < 0.01 | Yes | Statistically significant improvement (Delta F1 = +0.0932, p < 0.01); persistence cuts false alarms |
