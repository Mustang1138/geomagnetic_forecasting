# Project Progress Notes

## Week 1 (19–25 January 2026)

### 🎯 Objectives

- Establish development environment and tooling
- Define project scope and architecture
- Identify appropriate geomagnetic data sources
- Implement initial data acquisition pipeline

---

### ✅ Completed

#### Environment & Tooling

- ✓ Installed PyCharm (via JetBrains Toolbox)
- ✓ Created project with Python 3.13 virtual environment
- ✓ Initialised Git repository with structured `.gitignore`
- ✓ Installed core dependencies:
    - NumPy, Pandas, SciPy
    - scikit-learn
    - PyTorch (CPU-only)
    - Matplotlib, Seaborn
    - Jupyter

#### Architecture & Configuration

- ✓ Designed modular project directory hierarchy
- ✓ Implemented centralised configuration via `config.yaml`
- ✓ Wrote and refined comprehensive `README.md`
- ✓ Developed shared helper utilities (`utils.py`) including:
    - Logging setup
    - Configuration loading
    - Directory management

#### Data Acquisition & Parsing

- ✓ Implemented `data_loader.py` to manage data acquisition workflow
- ✓ Integrated historical **OMNI2** hourly solar wind and geomagnetic data (2010–2026)
- ✓ Implemented fixed-width file parser for OMNI2 annual datasets
- ✓ Integrated NOAA **DSCOVR** real-time magnetic field and plasma JSON endpoints
- ✓ Implemented robust missing-value handling using documented OMNI2 fill values

#### Data Validation

- ✓ Implemented schema validation for OMNI2 datasets
- ✓ Added physical range checks for key parameters (Bz GSM, speed, density, Dst)
- ✓ Implemented temporal continuity checks for hourly cadence
- ✓ Performed exploratory statistical validation to confirm physical plausibility

#### Documentation

- ✓ Documented data sources and assumptions
- ✓ Established formal progress tracking for dissertation audit trail

---

### 🎓 Reflections

Early investment in data validation and project structure significantly reduced downstream complexity. OMNI2 fixed-width
parsing required strict adherence to documentation to avoid silent errors. Configuration-driven pipelines improved
reproducibility and experimental flexibility.

---

### ⏱️ Time Tracking

| Activity                          | Hours    |
|-----------------------------------|----------|
| Environment setup & tooling       | 2.0      |
| Project structure & configuration | 1.5      |
| Data source research              | 2.0      |
| Data acquisition & parsing        | 4.0      |
| Utility code implementation       | 1.5      |
| Validation & debugging            | 2.5      |
| Documentation                     | 2.0      |
| **Week 1 Total**                  | **15.5** |

---

### 📝 Technical Notes

#### Data Sources

- **OMNI2 (NASA SPDF):**
    - Hourly, near-Earth solar wind and geomagnetic parameters
    - Used as primary historical dataset
    - Time span: 2010–2026

- **NOAA DSCOVR:**
    - Near-real-time solar wind magnetic field and plasma measurements
    - JSON API
    - Used for methodological comparison (not real-time forecasting)

#### Technology Decisions

- **Python 3.13:** Modern language features and performance
- **PyTorch (CPU-only):** Sufficient for dataset scale, avoids GPU dependency
- **Configuration-driven design:** Supports reproducibility and experimentation
- **Jupyter Notebooks:** Used for EDA and visual validation

#### Risk Assessment

- **Low:** Environment and tooling (completed)
- **Low:** Data availability (historical OMNI2 is stable)
- **Medium:** Model performance uncertainty (inherent to research question)
- **Low:** Project timeline (buffer available)

---

## Week 2 (26 January – 1 February 2026)

### 🎯 Objectives

- Finalise data preprocessing pipeline
- Validate OMNI2 dataset integrity
- Implement baseline ML models
- Introduce automated testing infrastructure

---

### ✅ Completed

#### Data Preprocessing

- ✓ Implemented unified preprocessing pipeline (`preprocess.py`)
- ✓ Enforced chronological train/validation/test split (no leakage)
- ✓ Implemented forward/backward filling for short missing intervals
- ✓ Applied domain-informed physical limits via `config.yaml`
- ✓ Standardised features using training data only
- ✓ Generated both tabular (baseline) and sequence (LSTM) datasets
- ✓ Persisted fitted scalers for inverse transformation during evaluation

#### Data Validation

- ✓ Implemented schema validation for OMNI2 data
- ✓ Added missing data analysis and reporting
- ✓ Implemented temporal continuity checks
- ✓ Added physical outlier detection
- ✓ Introduced post-preprocessing validation safeguards

#### Baseline Models

- ✓ Implemented Linear Regression baseline
- ✓ Implemented Random Forest Regressor baseline with fixed hyperparameters
- ✓ Ensured models consume frozen preprocessing outputs only
- ✓ Persisted trained models and predictions for all splits
- ✓ Centralised baseline hyperparameters in `config.yaml`

#### Testing Infrastructure

- ✓ Introduced `pytest` for automated testing
- ✓ Added unit tests for preprocessing pipeline
- ✓ Added unit tests for OMNI2 validation logic
- ✓ Added end-to-end tests for baseline model training
- ✓ Configured project-wide test behaviour via `pytest.ini`
- ✓ Validated sequence shapes and minimum sample requirements

---

### 🎓 Reflections

Unified preprocessing is critical for fair model comparison. Freezing preprocessing outputs improved experimental
validity and eliminated hidden coupling between data handling and model training. Automated tests significantly reduced
regression risk during rapid iteration. Clear separation of data, models, and evaluation simplifies analysis and
improves reproducibility.

---

### ⏱️ Time Tracking

| Activity                           | Hours    |
|------------------------------------|----------|
| Preprocessing & validation         | 4.0      |
| Baseline model implementation      | 3.0      |
| Testing infrastructure & debugging | 2.5      |
| Documentation updates              | 1.0      |
| **Week 2 Total**                   | **10.5** |

---

## Week 3 (2–8 February 2026)

### 🎯 Objectives

- Extend historical dataset coverage
- Finalise derived feature set
- Implement and lock baseline evaluation pipeline
- Improve test coverage and pipeline robustness
- Establish a clean “baseline checkpoint” for future models

---

### ✅ Completed

#### Dataset Extension

- ✓ Extended historical OMNI2 dataset coverage back to **2000**
- ✓ Updated data ingestion and preprocessing logic to support longer time span
- ✓ Revalidated temporal continuity and missing data characteristics
- ✓ Confirmed preprocessing scalability over 25+ years of hourly data

#### Feature Engineering

- ✓ Implemented derived feature computation (`derived_features.py`)
- ✓ Added physically motivated transformations and aggregations
- ✓ Ensured derived features are computed **after** temporal split
- ✓ Added unit tests for derived feature correctness and stability

#### Validation & Safeguards

- ✓ Extended validation utilities to cover derived features
- ✓ Added consistency checks for preprocessing outputs
- ✓ Strengthened failure modes for invalid or empty datasets
- ✓ Ensured all validation logic is test-covered

#### Baseline Evaluation

- ✓ Implemented standalone evaluation pipeline (`evaluate.py`)
- ✓ Computed standard regression metrics:
    - RMSE
    - MAE
    - R²
- ✓ Generated diagnostic plots:
    - Predicted vs true SSI time series
    - Predicted vs true scatter plots
- ✓ Saved consolidated metrics to `metrics_baselines.csv`
- ✓ Ensured evaluation is strictly read-only with respect to data

#### Testing & Stability

- ✓ Added evaluation-specific tests using synthetic predictions
- ✓ Ensured compatibility with sklearn API changes
- ✓ Fixed edge cases where no prediction files are present
- ✓ Confirmed all tests pass cleanly (`pytest`)

Current test status:
> ✅ All tests passing

---

### 🔒 Locked Components

The following components are now considered **frozen reference implementations**:

- Preprocessing pipeline
- Derived feature computation
- Baseline model definitions
- Baseline evaluation metrics and plots
- Preprocessing pipeline (inputs, splits, scaling)

Any future model must:

- Consume identical preprocessing outputs
- Be evaluated against the same locked baselines
- Report results relative to these reference metrics

---

### 🎓 Reflections

Locking preprocessing and baselines before introducing temporal models significantly strengthens experimental validity.
Extending the dataset back to 2000 improved statistical robustness and ensured that baseline performance is not an
artefact of a limited solar cycle. Treating evaluation as a pure consumer of model predictions reduced coupling and
simplified testing. At this point, improvements in performance can be attributed confidently to model design rather than
pipeline changes.

---

### ⏱️ Time Tracking

| Activity                         | Hours    |
|----------------------------------|----------|
| Dataset extension & revalidation | 2.5      |
| Derived feature implementation   | 2.0      |
| Evaluation & plotting pipeline   | 2.5      |
| Testing & debugging              | 2.0      |
| Documentation & cleanup          | 1.0      |
| **Week 3 Total**                 | **10.0** |

---

### 📋 Next Steps

1. Design temporal windowing strategy for sequence models
2. Implement LSTM / GRU data loaders
3. Define validation-driven early stopping
4. Train first sequence baseline
5. Compare temporal models against frozen baselines
6. Analyse error behaviour during geomagnetic storm events

---

### 🚧 Blockers / Issues

None currently identified.

---

*Last Updated: 06-02-2026*  
*Next Review: 10-02-2026*