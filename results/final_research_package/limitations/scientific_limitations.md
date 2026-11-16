# FlexPulse-ML: Scientific and Experimental Limitations

**Document Classification:** Scientific Scope, Empirical Limitations, and Validity Boundaries
**Notice:** SIMULATION-BASED EXPERIMENT — NO PHYSICAL HARDWARE VALIDATION

## 1. Pure Simulation Scope and Hardware Boundaries
All empirical data and metrics presented in this research package originate from a software-based lumped-element mathematical simulator of a simplified resistive electrical power path. No measurements were collected from physical USB-C cables, physical receptacles, test fixtures, electronic DC loads, or physical laboratory oscilloscopes. The results must not be extrapolated to physical cable reliability, manufacturing quality, or operational safety.

## 2. Power Delivery Protocol and High-Frequency Decoupling
FlexPulse-ML models DC and low-frequency electrical telemetry (V_meas, I_meas) at 100 Hz. It does not simulate USB Power Delivery (USB-PD 3.x) BMC communication on the CC line, programmable power supply (PPS) negotiation, e-marker identity validation, or high-speed transient contact chatter on sub-millisecond timescales (<10 ms).

## 3. Scenario S4 Ground-Truth Labeling Semantic Divergence
During dataset generation for flexpulse_final_test_v4, Scenario S4 (Gradual Resistance Drift) was configured with physical base resistance drift (post_mask base_r += drift_rate * (t - 5.0)). However, unlike Scenario S9, the generator did not activate fault_labels = 1 or append an entry to events_meta. Consequently, ground truth in S4 contains zero positive fault windows (GT Pos Windows = 0). When detectors identify the rising path resistance, every predicted positive window is scored as a false positive against ground truth, causing reported S4 Specificity to drop (0.0753 for M3, 0.0937 for M5). In accordance with strict scientific pre-registration integrity, test data labels were not retroactively modified. This discrepancy is formally documented here as a generator labeling divergence and an empirical limitation.

## 4. Initial Calibration Sensitivity vs Precision Trade-Off
Initial reference calibration (M3) boosts Event Recall from 74.14% (M2) to 91.38% (53/58 events detected). However, by removing DC level offsets, M3 increases sensitivity to transient fluctuations, generating more window-level false positives (OOD Specificity: 0.7575 for M3 vs 0.8506 for M2). Consequently, raw window F1 drops slightly from 0.5424 (M2) to 0.5104 (M3), a difference that is not statistically significant (Delta F1 = -0.0391, 95% CI [-0.1347, 0.0509], p = 0.3700). Window F1 improves only when coupled with causal persistence filtering (M5 F1 = 0.5873).

## 5. Statistical Bootstrap Resolution Limits
Bootstrap paired comparisons are evaluated using B = 1,000 resamples at the complete-run level. With 1,000 resamples, the minimum empirical non-zero frequency resolution is 1/1,000 = 0.001. Where zero resample differences cross the null boundary, empirical p-values are reported with their appropriate bound (p < 0.01) rather than unsupported infinite precision (e.g. p < 0.0001).

## 6. Pre-Registration Protocol Status
The protocol document (docs/final_evaluation_protocol.md) was cryptographically locked using SHA-256 prior to executing final model evaluations. This hash serves as an internal tamper-evident configuration audit trail, not an independently hosted third-party pre-registration.
