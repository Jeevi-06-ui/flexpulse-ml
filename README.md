# FlexPulse-ML: An Active Self-Calibrating Machine Learning System for Early Detection and Localization of Intermittent USB-C Cable Faults

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Academic Institution](https://img.shields.io/badge/Institution-Presidency%20University%2C%20Bengaluru-orange)](https://presidencyuniversity.in/)
[![Scientific Integrity](https://img.shields.io/badge/Integrity-Zero%20Data%20Fabrication-success)](#scientific-integrity-framework)

**FlexPulse-ML** is a scientific machine learning research project developed at **Presidency University, Bengaluru** under the *Machine Learning Techniques* curriculum. The project targets reproducible, physics-guided detection of intermittent contact degradation in USB Type-C power lines, leading to an **IEEE research manuscript submission by 1 December 2026**.

---

## 1. Research Motivation & Physics Context

USB-C cables are subjected to repetitive mechanical strain (bending, torsion, plug cycling). Rather than failing abruptly, cables experience **intermittent micro-contact degradations**:
* Copper strand micro-fractures exhibiting angle-dependent resistive spikes.
* Contact oxidation and plating wear altering interfacial resistance ($R_{\text{contact}}$).
* Thermal drift and contact bounce under dynamic current loads.

Standard black-box ML pipelines suffer from severe data leakage when samples from the same physical cable are split randomly into train and test sets. **FlexPulse-ML** introduces:
1. **Leave-One-Cable-Out (LOCO) Cross-Validation**: Models are evaluated exclusively on physical cables excluded entirely from training.
2. **Reference-Based Calibration**: Fixture tare resistance and baseline offsets are calibrated to isolate genuine cable properties.
3. **Calibrated Probability & Abstention**: The model outputs an explicit *"Abstain / Inconclusive"* verdict when classification entropy exceeds an uncertainty threshold ($\tau$), preventing false alarms in hardware testing.
4. **Active Test Selection**: An information-theoretic diagnostic policy that guides the operator to optimal follow-up electrical tests.

---

## 2. Scientific Integrity Framework

To uphold the highest ethical and academic standards:
* **Zero Fabrication Policy**: Model accuracies, F1-scores, and physical measurements are never invented.
* **Strict Synthetic Isolation**: Software verification uses mock data explicitly marked `DatasetCategory.SYNTHETIC_DEMO`. Synthetic data is programmatically forbidden from real experimental benchmark runs.
* **Real Hardware Status**: When physical test fixtures are offline or data is not yet gathered, real-data experiments are reported honestly as `NOT RUN (Awaiting physical hardware test fixture)`.
* **Zero Label Leakage**: Injected fault commands, fixture IDs, and ground truth are excluded from feature matrices.

---

## 3. Directory Layout

```
flexpulse_ml/
├── acquisition/         # Data schemas, validation rules, CSV loaders, hardware abstraction
├── processing/          # Signal cleaning, jitter analysis, physics-informed feature extraction
├── ml/                  # DT, RF, SVM, GBDT, Logistic Regression & LOCO cross-validation
├── calibration/         # Reference fixture zeroing, Platt/Isotonic probability calibration
├── active_testing/      # Uncertainty-aware diagnostic test selection policy
├── reporting/           # IEEE LaTeX tables, publication figures, reproducibility logs
└── utils/              # Project paths, structured logging, audit trails

data/
├── raw/                 # Immutable experimental captures (real and lab captures)
├── processed/           # Feature matrices and preprocessed datasets
├── demo/                # Strictly labeled SYNTHETIC_DEMO datasets for offline testing
└── metadata/            # Audit logs (quarantined invalid records)

models/                  # Serialized model pipelines (.joblib) with provenance hashes
results/
├── figures/             # Camera-ready publication plots (PDF, PNG)
├── tables/              # IEEE manuscript LaTeX tables
└── experiments/         # Parameter configs and execution logs

configs/                 # Reproducible JSON/YAML experiment configurations
tests/                   # Pytest automated unit and integration tests
docs/                    # Academic research documentation and math formulations
scripts/                 # Automation and test runners
```

---

## 4. Quickstart Guide (macOS / Apple Silicon)

### Prerequisites
* macOS (optimized for Apple Silicon `arm64`)
* Python 3.11 or later
* Git

### Setup Virtual Environment
```bash
# Clone the repository
git clone <repository_url> flexpulse-ml
cd flexpulse-ml

# Create and activate Python 3.12 virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies and local flexpulse CLI in editable mode
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### CLI Commands
```bash
# 1. Verify your local environment and dependencies
flexpulse env-check

# 2. Check the project research progression and hardware readiness
flexpulse status

# 3. Generate a sample synthetic demo dataset for offline testing
flexpulse generate-demo --output data/demo/synthetic_sample.csv --samples 150

# 4. Audit and validate a dataset against physical and schema constraints
flexpulse validate-data data/demo/synthetic_sample.csv

# 5. Run automated pytest test suite
flexpulse test
```

---

## 5. Development Phases

* **Phase 1 (Completed)**: Project foundation, typed schema, validator, synthetic generator, and CLI.
* **Phase 2 (Next)**: Signal preprocessing, INA226 jitter analysis, and physics-guided electrical feature engineering.
* **Phase 3**: Classical ML benchmark suite (DT, RF, SVM, GBDT, LR) with Leave-One-Cable-Out CV.
* **Phase 4**: Reference fixture calibration, probability calibration, and uncertainty abstention.
* **Phase 5**: Uncertainty-aware active test selection policy.
* **Phase 6**: Streamlit research dashboard.
* **Phase 7**: IEEE manuscript export engine and end-to-end verification.

---

## 6. Academic Citation & License

Developed at **Presidency University, Bengaluru**. Released under the MIT License for open scientific reproduction.
