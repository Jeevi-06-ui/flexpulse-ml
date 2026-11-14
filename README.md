# FlexPulse-ML

**FlexPulse-ML: A Simulation-Based Self-Calibrating Machine Learning Framework for Intermittent Fault Detection in USB-C-Inspired Power Paths**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Type: Simulation Only](https://img.shields.io/badge/Methodology-Mathematical%20Simulation-success)](#scientific-integrity)

---

## 1. Scientific Integrity and Simulation Notice

* **Software-Only Research:** This is a software-only project. It does not require ESP32, Arduino, Raspberry Pi, test fixtures, physical cables, sensors, or oscilloscopes.
* **No Physical Control:** The system does not access or control physical USB-C ports or power delivery hardware.
* **Pure Mathematical Modeling:** All telemetry is generated using a resistive equivalent circuit. Every dataset is labeled `SYNTHETIC` or `SIMULATED`.
* **Zero Fabrication Policy:** Model metrics, detection delays, and comparison tables are derived strictly from actual executed experiments.

---

## 2. Research Objective and Mathematical Model

The objective is to investigate whether a **confidence-gated adaptive baseline calibration** improves intermittent fault detection when simulated operating conditions (load impedance, baseline drift, sensor noise) change.

### Equivalent Circuit Model
$$R_{\text{path}}(t) = R_0 + R_{\text{contact}}(t) + R_{\text{fault}}(t)$$
$$I(t) = \frac{V_s(t)}{R_{\text{load}}(t) + R_{\text{path}}(t)}$$
$$V_{\text{load}}(t) = I(t) \cdot R_{\text{load}}(t)$$

### Simulated Observations
$$V_{\text{measured}}(t) = V_{\text{load}}(t) + b_V(t) + \text{noise}_V(t)$$
$$I_{\text{measured}}(t) = I(t) + b_I(t) + \text{noise}_I(t)$$

---

## 3. Directory Structure

```
flexpulse_ml/
├── __init__.py
├── app.py             # Streamlit research dashboard
├── config.py          # YAML configuration loader and validator
├── logging.py         # Structured logging
├── seed.py            # Reproducible random-seed manager
└── cli.py             # Command-line interface

configs/
└── default_config.yaml # Master experiment configuration

data/
├── raw/               # Simulated run time-series
├── processed/         # Extracted feature matrices
└── metadata/          # Manifests, event metadata, audit logs

models/                # Serialized trained model pipelines
results/
├── figures/           # Generated evaluation plots
├── tables/            # LaTeX and Markdown metrics tables
└── experiments/       # Machine-readable experiment outputs

tests/                 # Automated Pytest suite
```

---

## 4. Setup and Quickstart (macOS / Apple Silicon)

### Environment Setup
```bash
# Clone the repository
cd "ML Research paper"

# Create and activate Python 3.12 virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies and local package in editable mode
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Running Tests
```bash
pytest -v
```

### Starting the Streamlit Dashboard
```bash
streamlit run flexpulse_ml/app.py
```

---

## 5. Development Phases

* **Phase 0 (Completed):** Project foundation, configuration management, seed manager, directory topology, Streamlit navigation placeholders.
* **Phase 1:** Electrical power-path mathematical simulator.
* **Phase 2:** Synthetic dataset generator with run-level isolation.
* **Phase 3:** Feature engineering pipeline.
* **Phase 4:** Machine learning training and prediction pipeline.
* **Phase 5:** Confidence-gated adaptive baseline self-calibration.
* **Phase 6:** Experiment execution and evaluation engine.
* **Phase 7:** Interactive research dashboard implementation.
* **Phase 8:** Verification, audit, and research report export.
