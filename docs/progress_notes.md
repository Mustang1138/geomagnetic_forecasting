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

### 📋 Next Steps

1. Implement evaluation metrics (RMSE, MAE, R²)
2. Generate baseline performance plots
3. Analyse baseline model performance
4. Design LSTM architecture
5. Implement temporal model training loop
6. Compare temporal vs non-temporal approaches

---

### ❓ Questions for Supervisor (Meeting: 28-01-2026)

1. **Data Handling**
    - Preferred strategy for handling missing data in geomagnetic time series?
    - Acceptability of linear interpolation versus masking missing intervals?

2. **Train/Test Strategy**
    - Recommended temporal split strategy for time-series evaluation?
    - Suitable validation window size for this dataset?

3. **Preprocessing**
    - Any domain-specific feature transformations recommended?
    - Preferred normalisation approach for solar wind parameters?

4. **Evaluation**
    - Minimum baseline performance expectations?
    - Value of comparison with published geomagnetic forecasting benchmarks?

---

### 🚧 Blockers / Issues

None currently identified.

---

*Last Updated: 26-01-2026*  
*Next Review: 28-01-2026*