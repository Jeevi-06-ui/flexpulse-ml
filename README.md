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
├── cli.py             # Command-line interface
├── simulation/        # Phase 1: Electrical model and scenario fault injector
│   ├── electrical_model.py
│   ├── fault_injector.py
│   └── plotting.py
└── data/              # Phase 2: Synthetic dataset generation, storage & validation
    ├── generator.py
    ├── manifest.py
    ├── storage.py
    └── validator.py

configs/
└── default_config.yaml # Master experiment configuration

data/
├── synthetic/         # Phase 2: Generated multi-run datasets
│   └── <dataset_id>/
│       ├── manifest.json
│       ├── splits.json
│       ├── train/runs/<run_id>/
│       ├── validation/runs/<run_id>/
│       ├── id_test/runs/<run_id>/
│       └── ood_test/runs/<run_id>/
├── raw/               # Telemetry captures
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

## 4. Synthetic Dataset Generation and Validation (Phase 2)

### Dataset Directory Architecture
Datasets are partitioned by complete simulation runs to prevent cross-window data leakage:

```
data/synthetic/<dataset_id>/
├── manifest.json                       # Global summary, class balance, SHA-256 checksums
├── splits.json                         # Complete disjoint mapping of run IDs to splits
├── train/runs/<run_id>/                # Training partition (70% nominal runs)
│   ├── observations.csv                # Strictly observable signals only
│   ├── conditions.csv                  # Operating setpoints (source voltage, load)
│   ├── ground_truth.csv                # Quarantined fault labels, true path resistance
│   ├── events.json                     # Ground-truth event intervals & sampling status
│   └── metadata.json                   # Provenance flags, seed, scenario parameters
├── validation/runs/<run_id>/           # Validation partition (15% nominal runs)
├── id_test/runs/<run_id>/              # In-distribution test partition (15% nominal runs)
└── ood_test/runs/<run_id>/             # Out-of-distribution shifted test partition
```

### Strict Observable vs. Hidden Data Segregation
* **`observations.csv` (Public to ML):** strictly contains `timestamp_s`, `voltage_measured_v`, `current_measured_a`, `source_voltage_v`.
* **`ground_truth.csv` (Quarantined):** contains `timestamp_s`, `true_path_resistance_ohm`, `fault_label` $\in \{0, 1\}$, `fault_type`, `event_id`.
* **Sensor Drift & Shift Integrity:** For `MEASUREMENT_OFFSET_DRIFT` and `OPERATING_CONDITION_SHIFT`, `fault_label` is strictly 0 because the physical path remains healthy.

### Run-Level Partitioning
1. **Train Split (70%):** Stratified nominal scenario runs for model optimization.
2. **Validation Split (15%):** Stratified nominal scenario runs for hyperparameter tuning.
3. **In-Distribution Test (15%):** Unseen nominal runs with standard parameter distributions.
4. **Out-of-Distribution Test (OOD):** Dedicated runs with shifted baseline resistance ($0.180\,\Omega$), higher current load ($1.80\,\Omega$), and $3\times$ higher noise floor ($0.015\,\text{V}$, $0.025\,\text{A}$).

### Python API Usage
```python
from flexpulse_ml.data import generate_synthetic_dataset, validate_synthetic_dataset
from flexpulse_ml.data.storage import SYNTHETIC_DATA_ROOT

# Generate a balanced dataset with nominal and OOD runs
manifest = generate_synthetic_dataset(
    dataset_id="flexpulse_demo_dataset",
    runs_per_scenario=2,
    ood_runs_per_scenario=1,
    root_seed=42,
    mode="SCENARIO_BALANCED",
)

# Run the 20-check scientific validation engine
report = validate_synthetic_dataset(SYNTHETIC_DATA_ROOT / "flexpulse_demo_dataset")
report.print_summary()
```

### Scientific Validation Engine
`DatasetValidator` executes 20 automated checks on every generated dataset:
1. `manifest_file_exists`: Verifies existence of root manifest.json.
2. `splits_file_exists`: Verifies existence of root splits.json.
3. `all_splits_present`: Confirms train, validation, id_test, and ood_test definitions.
4. `unique_run_ids_splits`: Guarantees run IDs are unique across all partitions.
5. `no_run_overlap_between_splits`: Enforces zero data leakage across splits.
6. `unique_run_seeds`: Asserts every simulation run was generated from a unique seed.
7. `file_structure_integrity`: Confirms all 5 required files exist in every run directory.
8. `observations_schema`: Validates observable column names and types.
9. `conditions_schema`: Validates condition setpoint column names and types.
10. `ground_truth_schema`: Validates ground-truth column names and types.
11. `metadata_schema`: Validates required provenance and simulation metadata fields.
12. `no_ground_truth_leakage`: Confirms observations.csv contains ZERO ground-truth columns.
13. `no_nan_or_inf`: Enforces clean signals with zero NaN or infinite values.
14. `binary_labels_valid`: Guarantees fault labels are strictly binary $\{0, 1\}$.
15. `sensor_drift_label_integrity`: Confirms sensor drift and operating shifts have label 0.
16. `timestamp_monotonic_and_aligned`: Verifies strictly increasing and synchronized timestamps.
17. `event_sampling_consistency`: Asserts ground-truth labels align with event time windows.
18. `synthetic_provenance_flag`: Ensures every run is explicitly marked `synthetic_data=True`.
19. `checksum_integrity`: Validates SHA-256 cryptographic hashes for every file.
20. `manifest_counts_consistency`: Reconciles sample, fault, and run counts against manifest.

---

## 5. Setup and Quickstart (macOS / Apple Silicon)

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

## 6. Development Phases

* **Phase 0 (Completed):** Project foundation, YAML config, structured logging, seed manager, Streamlit dashboard shell.
* **Phase 1 (Completed):** Electrical power-path mathematical simulator (`electrical_model.py`), scenario fault injection engine (`fault_injector.py`), and visualization (`plotting.py`).
* **Phase 2 (Completed):** Synthetic dataset generator, run-level partitioning (Train 70%, Val 15%, ID Test 15%, OOD Test), strict ground-truth quarantine, 20-check validation engine, and Streamlit Data Explorer controls.
* **Phase 3:** Feature engineering pipeline.
* **Phase 4:** Machine learning training and prediction pipeline.
* **Phase 5:** Confidence-gated adaptive baseline self-calibration.
* **Phase 6:** Experiment execution and evaluation engine.
* **Phase 7:** Interactive research dashboard implementation.
* **Phase 8:** Verification, audit, and research report export.
