# FlexPulse-ML: Scientific and Experimental Limitations

**Document Classification:** Scientific Scope & Physical Boundary Analysis
**Notice:** SIMULATION-BASED EXPERIMENT — NO PHYSICAL HARDWARE VALIDATION

## 1. Simulation Nature and Hardware Disclaimer
All empirical results presented in this research package originate from a software-based mathematical simulator of a resistive power path. The experimental data were not acquired from physical USB-C cables, connectors, electronic loads, or physical oscilloscope instrumentation.

## 2. Power Delivery (USB-PD) Protocol Scope
FlexPulse-ML models the resistive power path and low-frequency electrical telemetry ($V, I$). It does not model BMC physical layer signaling, USB Power Delivery 3.x packet arbitration, programmable power supply (PPS) feedback negotiation, or cable e-marker ID validation.

## 3. Sampling Frequency and Transient Faults
Features are extracted from sliding windows with $f_s = 100\,\text{Hz}$ ($T_w = 0.5\,\text{s}$, $T_s = 0.25\,\text{s}$). Sub-millisecond micro-interruptions or contact chatter occurring on timescales faster than $10\,\text{ms}$ are unresolvable by the telemetry window without dedicated high-speed transient capture circuits.

## 4. Value of Online Adaptation Negative Finding
The experimental results demonstrate that confidence-gated adaptive calibration (M4) provides zero statistically significant F1-score improvement over static reference calibration (M3) ($\Delta \text{F1} = 0.0000$). In USB-C-inspired power paths, initial baseline centering accounts for 100% of the operational robustness. Online adaptation introduces computational complexity without classification benefit, representing a rigorous scientific negative finding that prevents overclaiming in the final manuscript.
