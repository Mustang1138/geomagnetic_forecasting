# Project Progress Notes

## Week 1 (19–25 January 2026)

### 🎯 Objectives

* Set up development environment
* Create project structure
* Research suitable geomagnetic and solar wind data sources
* Implement and validate initial data acquisition pipeline

---

### ✅ Completed

#### Environment Setup

* ✓ Installed PyCharm (via JetBrains Toolbox)
* ✓ Created project with Python 3.13 virtual environment
* ✓ Initialised Git repository with appropriate `.gitignore`
* ✓ Installed core dependencies:

    * NumPy, Pandas, SciPy
    * scikit-learn
    * PyTorch (CPU-only)
    * Matplotlib, Seaborn
    * Jupyter

#### Project Structure & Configuration

* ✓ Designed and created modular project directory hierarchy
* ✓ Implemented configuration management via `config.yaml`
* ✓ Wrote and refined comprehensive `README.md`
* ✓ Implemented shared helper utilities (`utils.py`) including:

    * Logging setup
    * Configuration loading
    * Directory management

#### Data Acquisition & Parsing

* ✓ Implemented `data_loader.py` to manage data acquisition workflow
* ✓ Integrated historical **OMNI2** hourly solar wind and geomagnetic data (2010–2026)
* ✓ Implemented fixed-width file parser for OMNI2 annual datasets
* ✓ Integrated NOAA **DSCOVR** real-time magnetic field and plasma JSON endpoints
* ✓ Implemented robust missing-value handling using documented OMNI2 fill values

#### Data Validation

* ✓ Implemented schema validation for OMNI2 datasets
* ✓ Added physical range checks for key parameters (Bz GSM, speed, density, Dst)
* ✓ Implemented temporal continuity checks for hourly cadence
* ✓ Performed exploratory statistical validation to confirm physical plausibility

#### Documentation

* ✓ Documented data sources and assumptions
* ✓ Began formal progress tracking for dissertation audit trail

---

### 🔄 In Progress

* Refinement of OMNI2 parsing and validation logic
* Initial exploratory data analysis (EDA)
* Preparation for preprocessing and feature engineering stage

---

### 📋 Next Steps

#### Week 2 Planning (26 January – 1 February 2026)

1. Finalise preprocessing pipeline (`preprocess.py`)
2. Handle missing data (interpolation vs masking strategy)
3. Perform exploratory data analysis in Jupyter notebooks
4. Define baseline forecasting targets (e.g. Dst prediction horizon)
5. Implement baseline regression models

---

### 🚧 Blockers / Issues Encountered

* Initial OMNI2 parsing issues due to incorrect column specifications (resolved)
* Handling of documented OMNI2 fill values required careful validation (resolved)

---

### ❓ Questions for Supervisor (Meeting: 28-01-2026)

1. **Data Handling**

    * Preferred strategy for handling missing data in geomagnetic time series?
    * Acceptability of linear interpolation versus masking missing intervals?

2. **Train/Test Strategy**

    * Recommended temporal split strategy for time-series evaluation?
    * Suitable validation window size for this dataset?

3. **Preprocessing**

    * Any domain-specific feature transformations recommended?
    * Preferred normalisation approach for solar wind parameters?

4. **Evaluation**

    * Minimum baseline performance expectations?
    * Value of comparison with published geomagnetic forecasting benchmarks?

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

* **OMNI2 (NASA SPDF):**

    * Hourly, near-Earth solar wind and geomagnetic parameters
    * Used as primary historical dataset
    * Time span: 2010–2026

* **NOAA DSCOVR:**

    * Near-real-time solar wind magnetic field and plasma measurements
    * JSON API
    * Used for methodological comparison (not real-time forecasting)

#### Technology Decisions

* **Python 3.13:** Modern language features and performance
* **PyTorch (CPU-only):** Sufficient for dataset scale, avoids GPU dependency
* **Configuration-driven design:** Supports reproducibility and experimentation
* **Jupyter Notebooks:** Used for EDA and visual validation

#### Risk Assessment

* **Low:** Environment and tooling (completed)
* **Low:** Data availability (historical OMNI2 is stable)
* **Medium:** Model performance uncertainty (inherent to research question)
* **Low:** Project timeline (buffer available)

---

### 🎓 Learning & Insights

* Early focus on data validation prevents downstream modelling issues
* OMNI2 fixed-width parsing requires strict adherence to documentation
* Configuration-driven pipelines improve reproducibility
* Clear separation of data acquisition, validation, and modelling is beneficial

---

## Week 2 (26 January – 1 February 2026)

### 🎯 Objectives

* Implement preprocessing and feature engineering pipeline
* Conduct exploratory data analysis
* Define forecasting targets and baselines

#### Data Preprocessing

* ✓ Implemented unified preprocessing pipeline (`preprocess.py`)
* ✓ Enforced chronological train/validation/test splits (no leakage)
* ✓ Implemented forward/backward filling for missing values
* ✓ Applied physically motivated outlier filtering via `config.yaml`
* ✓ Standardised features and target using training data only
* ✓ Generated both:
    * Tabular datasets for baseline models
    * Fixed-length sequences for LSTM models
* ✓ Persisted fitted scalers for inverse transformation during evaluation

#### Baseline Models

* ✓ Implemented Linear Regression baseline
* ✓ Implemented Random Forest Regressor baseline
* ✓ Trained models using frozen, preprocessed datasets
* ✓ Saved trained models and predictions for train/validation/test splits
* ✓ Centralised baseline hyperparameters in `config.yaml`

#### Testing & Validation

* ✓ Added unit tests for preprocessing pipeline
* ✓ Added unit tests for data validation utilities
* ✓ Validated sequence shapes and minimum sample requirements
* ✓ Ensured preprocessing outputs meet ML training constraints

---

### 📋 Next Steps

1. Implement evaluation and metric calculation pipeline
2. Compute RMSE, MAE, and R² for baseline models
3. Generate comparative plots for baseline performance
4. Implement LSTM temporal model
5. Compare baseline vs temporal model results

---

*Last Updated: 26-01-2026*
*Next Review: 28-01-2026*