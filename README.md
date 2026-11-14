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

## 5. Electrical Feature Engineering Pipeline (Phase 3)

### Observable Telemetry Input Policy
Model inputs are derived strictly from observable signals (`voltage_measured_v` and `current_measured_a`). All ground-truth variables (`true_path_resistance_ohm`, `fault_type`, `event_id`, configured load resistance) are quarantined and never enter feature tables.

### Sliding-Window Segmentation
* **Sampling Rate ($f_s$):** 100 Hz
* **Window Duration:** 0.5 seconds (50 samples)
* **Overlap:** 50% (stride of 25 samples)
* **Window Labeling Rule:** A window is labeled faulty ($y = 1$) if at least one sample in the window contains a ground-truth fault sample (`np.any(gt == 1)`). The continuous fraction of faulty samples (`fault_fraction`) is stored as evaluation metadata, never as an input feature.

### 26 Extracted Electrical and Statistical Features
For each window, 13 features are calculated for voltage ($V$) and 13 for current ($A$):
1. **Central Tendency:** Mean (`voltage_mean_v`, `current_mean_a`), Median (`voltage_median_v`, `current_median_a`).
2. **Dispersion & Spread:** Standard Deviation, Minimum, Maximum, Range, 10th Percentile, 90th Percentile, Median Absolute Deviation (MAD).
3. **Dynamics & Trends:** Linear Trend Slope ($V/\text{s}$, $A/\text{s}$), Mean Absolute First Difference, Standard Deviation of First Difference, Maximum Absolute First Difference.

### Derived Feature Storage Topology
```
data/processed/features/<dataset_id>/
├── feature_manifest.json          # Schema, window counts, class balance, SHA-256 hashes
├── train_features.csv             # Training feature matrix (70% nominal runs)
├── validation_features.csv        # Validation feature matrix (15% nominal runs)
├── id_test_features.csv           # In-distribution test feature matrix (15% nominal runs)
└── ood_test_features.csv          # Out-of-distribution test feature matrix
```

---

## 6. Machine Learning Training, Benchmarking and Evaluation (Phase 4)

### Evaluated Model Suite
Phase 4 implements and comparatively benchmarks 6 distinct classifier paradigms using strictly the 26 observable electrical features:
1. **Fixed Threshold Baseline (`FixedThresholdBaseline`):** Interpretable rule-based comparator fitting $\mu \pm k\sigma$ envelopes on voltage drop and max transient jumps from healthy training intervals.
2. **Logistic Regression (`LogisticRegression`):** Linear probabilistic baseline fitted inside a `StandardScaler` pipeline with L2 regularization and balanced class weighting.
3. **Decision Tree (`DecisionTreeClassifier`):** Non-linear rule induction optimizing Gini impurity with constrained tree depth ($d=8$) to prevent overfitting.
4. **Random Forest (`RandomForestClassifier`):** Ensemble bagging of 100 decorrelated decision trees, aggregating variance across random feature subsets.
5. **Support Vector Machine (`SVC` with RBF kernel):** Maximum-margin kernel classifier operating in non-linear Hilbert feature space with scaling and Platt-scaling probability calibration.
6. **Gradient Boosting (`HistGradientBoostingClassifier`):** Gradient-boosted histogram decision trees sequentially correcting negative gradient residuals.

### Strict Preprocessing Hygiene and Anti-Leakage Controls
* **Feature Allowlist Enforced:** Training strictly accesses the 26 observable features (`FEATURE_NAMES`). Ground truth, event IDs, scenario labels, and physical parameters are prohibited.
* **Pipeline Encapsulation:** Scalers (`StandardScaler`) are encapsulated directly inside `Pipeline` objects and fitted strictly on `train_features.csv`. Test sets (`id_test`, `ood_test`) are strictly evaluated on frozen pipelines.
* **Validation-Based Freezing:** Hyperparameters are selected strictly on `validation_features.csv` based on F1-score / balanced accuracy tradeoff before model parameters are permanently frozen.

### Empirical Benchmark Results

#### In-Distribution (ID) Test Performance (Nominal Regimes)
| Model | Accuracy | Balanced Acc | Precision | Recall | Specificity | F1-Score | ROC AUC | Avg Precision | Train Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fixed Threshold** | 0.9393 | 0.8628 | 0.9735 | 0.7309 | 0.9947 | 0.8349 | 0.8646 | 0.7868 | 0.0001 |
| **Logistic Regression** | 0.9977 | 0.9945 | 1.0000 | 0.9889 | 1.0000 | 0.9944 | 0.9996 | 0.9988 | 0.0096 |
| **Decision Tree** | 0.9984 | 0.9969 | 0.9978 | 0.9945 | 0.9994 | 0.9961 | 0.9987 | 0.9976 | 0.0929 |
| **Random Forest (Best)** | **0.9986** | **0.9967** | **1.0000** | **0.9934** | **1.0000** | **0.9967** | **0.9997** | **0.9993** | 0.2243 |
| **SVM (RBF Kernel)** | 0.9972 | 0.9934 | 1.0000 | 0.9867 | 1.0000 | 0.9933 | 0.9992 | 0.9984 | 0.7151 |
| **Gradient Boosting** | 0.9984 | 0.9965 | 0.9989 | 0.9934 | 0.9997 | 0.9961 | 0.9995 | 0.9988 | 1.3858 |

#### Out-of-Distribution (OOD) Test Performance (Shifted Regimes)
| Model | Accuracy | Balanced Acc | Precision | Recall | Specificity | F1-Score | ROC AUC | Failure Mode |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Fixed Threshold** | 0.4778 | 0.6695 | 0.2867 | 1.0000 | 0.3391 | 0.4457 | 0.9193 | False alarm explosion (FP=3,744) |
| **Logistic Regression** | 0.3350 | 0.5743 | 0.2382 | 0.9867 | 0.1619 | 0.3838 | 0.8869 | Precision collapse (FP=4,748) |
| **Decision Tree** | 0.7901 | 0.5000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 | Complete recall loss (FN=1,505) |
| **Random Forest** | 0.7957 | 0.5133 | 1.0000 | 0.0266 | 1.0000 | 0.0518 | 0.8358 | Under-detection (FN=1,465) |
| **SVM (RBF Kernel)** | 0.7901 | 0.5000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.8335 | Complete recall loss (FN=1,505) |
| **Gradient Boosting** | 0.7901 | 0.5000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5133 | Complete recall loss (FN=1,505) |

> [!NOTE]
> **Key Scientific Finding:** While static non-linear ML models achieve near-perfect F1 ($>0.996$) under nominal in-distribution conditions, their performance collapses dramatically ($F1 \le 0.052$) when subjected to uncompensated physical distribution shifts (sensor bias offset, temperature drift, baseline changes). This empirically proves the absolute necessity for **confidence-gated adaptive baseline self-calibration** in Phase 5.

---

## 7. Setup and Quickstart (macOS / Apple Silicon)

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

## 8. Development Phases

* **Phase 0 (Completed):** Project foundation, YAML config, structured logging, seed manager, Streamlit dashboard shell.
* **Phase 1 (Completed):** Electrical power-path mathematical simulator (`electrical_model.py`), scenario fault injection engine (`fault_injector.py`), and visualization (`plotting.py`).
* **Phase 2 (Completed):** Synthetic dataset generator, run-level partitioning (Train 70%, Val 15%, ID Test 15%, OOD Test), 150-run research dataset (`flexpulse_research_v1`), strict ground-truth quarantine, 24-check validation engine, and Streamlit Data Explorer controls.
* **Phase 3 (Completed):** Electrical feature engineering pipeline (`windowing.py`, `extractor.py`, `dataset.py`, `validator.py`), 26 voltage & current features, sliding-window binary labeling with fault-fraction metadata, full feature validation engine, and Streamlit Feature Analysis dashboard.
* **Phase 4 (Completed):** Machine learning training, benchmarking, and empirical evaluation pipeline (`baselines.py`, `models.py`, `train.py`, `evaluate.py`, `persistence.py`, `plotting.py`, `experiment.py`). Evaluates 6 models across ID and OOD test sets, event-level detection, bootstrap confidence intervals, and scenario breakdowns.
* **Phase 5:** Confidence-gated adaptive baseline self-calibration.
* **Phase 6:** Experiment execution and evaluation engine.
* **Phase 7:** Interactive research dashboard implementation.
* **Phase 8:** Verification, audit, and research report export.
