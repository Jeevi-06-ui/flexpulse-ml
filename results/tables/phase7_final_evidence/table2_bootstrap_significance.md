| Comparison | Description | Delta F1 (Mean) | 95% CI F1 | Delta Bal Acc (Mean) | 95% CI Bal Acc | Empirical p-value | Significant (p<0.05) | Scientific Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M2 vs M1 | Static ML vs Fixed Baseline | 0.2723 | [0.1853, 0.3712] | 0.1864 | [0.1189, 0.2402] | < 0.01 | Yes | Static ML higher F1 than collapsed fixed threshold, but misses 25.86% events |
| M3 vs M2 | Initial Ref Calib vs Static ML | -0.0391 | [-0.1347, 0.0509] | 0.0332 | [-0.0182, 0.0852] | 0.3700 | No | No significant F1 change (p=0.3700); M3 recovers event recall (91.4% vs 74.1%) at higher false alarms |
| M4 vs M3 | Adaptive Calib vs Initial Calib | 0.0 | [0.0000, 0.0000] | 0.0 | [0.0000, 0.0000] | 1.0000 | No | Statistically indistinguishable (Delta F1 = 0.0000, p = 1.0000); zero adaptive gain |
| M5 vs M3 | Persistence Filter vs Raw Initial Calib | 0.0765 | [0.0485, 0.1054] | 0.0238 | [0.0157, 0.0310] | < 0.01 | Yes | Statistically significant improvement (Delta F1 = +0.0765, p < 0.01); persistence cuts false alarms |
| M6 vs M4 | Persistence Filter vs Raw Adaptive Calib | 0.0765 | [0.0485, 0.1054] | 0.0238 | [0.0157, 0.0310] | < 0.01 | Yes | Statistically significant improvement (Delta F1 = +0.0765, p < 0.01); persistence cuts false alarms |
