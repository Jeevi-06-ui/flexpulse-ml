# Pre-Submission Review Checklist for Undergraduate Research Paper

**Project Title:** *FlexPulse-ML: A Simulation-Based Evaluation of Reference-Calibrated Machine Learning for Intermittent Fault Detection in USB-C-Inspired Power Paths*  
**Course / Subject:** Machine Learning Techniques  
**Institution:** Presidency University, Bengaluru | Department of Computer Science and Engineering  
**Target Deadline:** 1 December 2026  
**Compiled Manuscript:** `research_paper/flexpulse_ieee_paper.pdf` (10 pages, IEEE conference format)  
**LaTeX Source:** `research_paper/flexpulse_ieee_paper.tex`  

---

## 1. Author and Institutional Placeholders to Update

Before submitting this manuscript to your faculty guide or course instructor, open `research_paper/flexpulse_ieee_paper.tex` and replace the following bracketed placeholders with your actual details:

- [x] **Author Details Updated:** Set to Jeevith P. (`jeevithp18@gmail.com`). Update with institutional student ID email if required by university.
- [ ] **Supervisor Details:** Confirm faculty guide role and insert supervisor name & email upon departmental assignment.
- [ ] **Line 29:** Replace `[Faculty Guide / Supervisor Name]` with your faculty mentor's full academic name and title (e.g., *Dr. S. Kumar, Associate Professor*).
- [ ] **Line 34:** Replace `[guide.email@presidencyuniversity.in]` with your faculty mentor's email address.
- [ ] **Section \textit{Acknowledgment}:** Add any specific course, lab, or department acknowledgment requested by your faculty guide.
- [ ] **Re-compile PDF:** Run `/opt/homebrew/bin/tectonic flexpulse_ieee_paper.tex` inside `research_paper/` to generate the final personalized PDF.

---

## 2. Key Academic Findings to Discuss with Your Faculty Guide

When presenting this research paper to your faculty supervisor, emphasize the following major technical points:

1. **The Realistic Problem Formulation:**
   - In DC power paths (like USB-C VBUS/GND), true internal path resistance cannot be measured directly from load-side voltage and current without disconnecting the circuit or knowing source and load parameters. Apparent resistance ($V_{\text{meas}}/I_{\text{meas}}$) merely calculates load resistance ($R_{\text{load}}$).
2. **The Failure Modes of Conventional Machine Learning (RQ1):**
   - Static Random Forest (M2) achieves high accuracy on nominal data but collapses under operational shifts:
     - On **Scenario S7 (Heavy Load Shift)**: Recall drops to **0.0000** (misses 100% of faults).
     - On **Scenario S2 (Elevated Conductor Resistance)**: Specificity drops to **0.0000** (triggers 860 false positive windows).
     - Across all OOD runs: Misses 15 of 64 fault events (**Event Recall = 76.56%**).
3. **The Value of Initial Reference Calibration (RQ2):**
   - A short, known-healthy 5.0-second baseline normalization interval ($\tilde{V} = V - \hat{V}_0$) removes static operating offsets, restoring fault sensitivity and recovering event recall from 76.56% to **92.19%** (59 of 64 events detected, $p < 0.01$).
4. **The Critical Negative Result on Online Adaptation (RQ3):**
   - Continuous confidence-gated adaptive calibration (M4) showed **zero incremental gain** over initial calibration (M3) ($\Delta\text{F1} = 0.0000$, $p = 1.0000$).
   - *Why?* To prevent the tracker from adapting into slow-developing faults (creep), defensive safety gates must be strict. When gates are strict, they correctly freeze updates during abnormal behavior, leaving the baseline identical to the initial calibration. When gates are loose, the tracker contaminates (2.11% contamination on S9).
   - This negative result demonstrates scientific maturity: do not hide it, celebrate it as evidence of algorithmic rigor.
5. **The Temporal Persistence Solution (RQ5):**
   - Initial calibration restores sensitivity but increases noise-induced false alarms (12.95 FA/min).
   - Adding a causal two-window persistence filter ($N=2$, M5) suppresses false-alarm events by **38.83%** (515 to 315) and elevates OOD F1 to **0.8559**, while preserving 100% of detected events (59/64) at an added median latency of $+0.2085\,\text{s}$ (total median detection delay $0.3407\,\text{s}$).
   - While longer filters ($N=3, 4$) yield higher window-level F1 (up to 0.9163), they introduce larger delays ($+0.46\,\text{s}$ and $+0.71\,\text{s}$) and drop event recall ($N \ge 4$ misses events). Thus $N=2$ is an effective lower-latency compromise rather than a universally optimal setting.

---

## 3. Mathematical Equations You Must Understand and Be Ready to Defend

Your viva or project review committee will likely ask you to explain these core equations:

1. **Circuit Model:**
   $$R_{\text{path}}(t) = R_0 + R_{\text{contact}}(t) + R_{\text{fault}}(t)$$
   $$I(t) = \frac{V_s(t)}{R_{\text{path}}(t) + R_{\text{load}}(t)}, \quad V_{\text{load}}(t) = I(t) \cdot R_{\text{load}}(t)$$
2. **Non-Identifiability Principle:**
   $$\frac{V_{\text{meas}}(t)}{I_{\text{meas}}(t)} \approx R_{\text{load}}(t)$$
   Explain that a terminal voltage drop can be caused by either rising connector resistance ($R_{\text{fault}}$) OR rising load current ($I_{\text{load}}$). Static classifiers confuse the two.
3. **Reference Baseline Estimation:**
   $$\hat{V}_0 = \text{median}(\{V_{\text{mean}, k}\}_{k=1}^{19}), \quad \hat{I}_0 = \text{median}(\{I_{\text{mean}, k}\}_{k=1}^{19})$$
   Calculated over the initial $5.0\,\text{s}$ (19 sliding windows of $0.50\,\text{s}$ duration with $0.25\,\text{s}$ stride).
4. **Exponential Moving Average (EMA) Adaptive Tracking:**
   $$\hat{V}_k = (1 - \alpha)\hat{V}_{k-1} + \alpha V_{\text{mean}, k}, \quad \alpha = 0.04$$
5. **Causal Persistence Filter:**
   $$\hat{y}_{\text{filt}, k} = \prod_{j=0}^{N-1} \hat{y}_{\text{raw}, k-j}$$
   Requires positive classification across $N=2$ consecutive sliding windows before triggering an alert.

---

## 4. Software and Test Suite Verification

- [x] **Automated Test Suite:** 122/122 unit and regression tests pass (`pytest tests/ -v`).
- [x] **Zero Python Code Comments:** Strictly 0 `#` comments across all Python source files in the entire project repository.
- [x] **Dataset Reproducibility:** Confirmatory evaluation dataset `flexpulse_final_test_v5` generated with locked root seed `20261202`.
- [x] **Bootstrap Statistical Significance:** $B=1,000$ paired run-level resamples computed and verified.
- [x] **No Hardware Fabrication:** Manuscript strictly disclaims hardware validation; mathematical equivalent-circuit simulation only.

---

## 5. University Submission Protocol

1. **Review with Guide:** Schedule an appointment with your faculty guide to review `flexpulse_ieee_paper.pdf`.
2. **Review Claim Evidence Audit:** Keep `research_paper/claim_evidence_audit.md` on hand during the viva to point directly to data files if any number is questioned.
3. **Internal Review Only:** As per project instructions, do NOT submit to external IEEE conferences, journals, or online preprint servers without formal departmental approval.
