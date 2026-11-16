| Comparison | Description | Delta F1 (Mean) | 95% CI F1 | Delta Bal Acc (Mean) | 95% CI Bal Acc | Empirical p-value | Significant (p<0.05) | Scientific Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M2 vs M1 | Static ML vs Fixed Baseline | 0.2732 | [0.1697, 0.3851] | 0.1858 | [0.1148, 0.2422] | < 0.01 | Yes | Static ML higher F1 than collapsed fixed threshold (Delta F1 = +0.2732, p < 0.01), but misses 25.86% events |
| M3 vs M2 | Initial Ref Calib vs Static ML | -0.036 | [-0.1379, 0.0597] | 0.0355 | [-0.0157, 0.0903] | 0.4680 | No | No significant F1 change (Delta F1 = -0.0360, p = 0.4680); M3 recovers event recall (91.4% vs 74.1%) at higher false alarms |
| M4 vs M3 | Adaptive Calib vs Initial Calib | 0.0 | [0.0000, 0.0000] | 0.0 | [0.0000, 0.0000] | 1.0000 | No | Statistically indistinguishable (Delta F1 = +0.0000, p = 1.0000); zero adaptive gain |
| M5 vs M3 | Persistence Filter vs Raw Initial Calib | 0.0776 | [0.0479, 0.1136] | 0.0239 | [0.0155, 0.0317] | < 0.01 | Yes | Statistically significant improvement (Delta F1 = +0.0776, p < 0.01); persistence cuts false alarms |
| M6 vs M4 | Persistence Filter vs Raw Adaptive Calib | 0.0776 | [0.0479, 0.1136] | 0.0239 | [0.0155, 0.0317] | < 0.01 | Yes | Statistically significant improvement (Delta F1 = +0.0776, p < 0.01); persistence cuts false alarms |
