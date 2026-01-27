# Geomagnetic Forecasting – Machine Learning Evaluation Project

**An offline, research-focused investigation evaluating whether temporal machine learning models provide measurable
performance improvements over classical baselines for geomagnetic activity forecasting.**

---

## 📋 Project Information

- **Student:** Mark Lewis (25214071)
- **Supervisor:** Prof. Ella Pereira
- **Module:** CIS3425 – Research and Development Project
- **Programme:** B.Sc. (Hons) Software Engineering
- **Academic Year:** 2025/2026

---

## 🎯 Project Aim

The primary aim of this project is to **critically evaluate whether temporal machine learning models (specifically
LSTMs) outperform established non-temporal baseline models** when forecasting geomagnetic activity.

The project focuses on *methodological fairness, reproducibility, and scientific validity* rather than operational
deployment.

Specifically, the project:

- Develops a fully reproducible offline forecasting pipeline
- Establishes conservative classical baselines for comparison
- Applies a unified preprocessing strategy to eliminate data-handling bias
- Evaluates models using standard regression metrics on historical space-weather data

---

## 📊 Project Scope

This project is intentionally scoped as an **offline, proof-of-concept research study**.

### Included

- ✅ Historical data analysis (2010–2026)
- ✅ Rigorous data ingestion, validation, and preprocessing
- ✅ Feature engineering for both tabular and temporal models
- ✅ Controlled comparison of baseline and temporal models
- ✅ Quantitative performance evaluation

### Excluded

- ❌ Real-time forecasting or alerting
- ❌ Production deployment
- ❌ Operational space-weather services
- ❌ GPU-dependent or distributed systems

---

## 🗂️ Project Structure

```
geomagnetic_forecasting/
│
├── data/
│   ├── raw/                    # Raw downloaded datasets (OMNI2, DSCOVR)
│   └── processed/              # Frozen, preprocessed datasets
│
├── src/
│   ├── __init__.py             # Package marker for src module
│   ├── data_loader.py          # Data acquisition and consolidation
│   ├── parsers.py              # OMNI2 and DSCOVR parsing utilities
│   ├── validators.py           # Schema, continuity, and physical validation
│   ├── preprocess.py           # Cleaning and feature engineering
│   ├── baseline_models.py      # Linear regression and ensemble baselines
│   ├── temporal_model.py       # LSTM-based temporal model
│   ├── train.py                # Model training pipeline
│   ├── evaluate.py             # Metric calculation and comparison
│   └── utils.py                # Logging and helper utilities
│
├── results/
│   ├── plots/                  # Evaluation visualisations
│   └── metrics/                # Model performance metrics (CSV)
│
├── tests/                      # Unit tests for preprocessing and validation
│   ├── test_preprocess.py
│   ├── test_validators.py
│   └── test_baseline_models.py
│
├── notebooks/                  # Exploratory analysis notebooks
├── docs/                       # Project documentation and progress logs
│   └── progress_notes.md
│
├── config.yaml                 # Centralised configuration
├── requirements.txt            # Python dependencies
└── README.md                   # Project overview
```

---

## 📡 Data Sources

| Dataset    | Parameters                                    | Provider            |
|------------|-----------------------------------------------|---------------------|
| **OMNI2**  | IMF Bz (GSM), solar wind speed & density, Dst | NASA SPDF / OMNIWeb |
| **DSCOVR** | Near-real-time magnetic field & plasma        | NOAA SWPC           |

### Notes on Data Handling

- OMNI2 hourly data are used as the primary historical dataset for modelling
- Fill values are replaced with NaN according to official documentation
- Physical plausibility checks are applied during validation
- All modelling is performed using historical OMNI2 data to ensure complete temporal coverage and reproducibility

---

## 🧹 Preprocessing & Experimental Control

A **single unified preprocessing pipeline** is used for *all* models.

This design is a deliberate methodological choice to ensure that:

- No model benefits from privileged data handling
- Data leakage is strictly prevented
- Observed performance differences arise from model architecture, not preprocessing

Key preprocessing steps include:

- Chronological validation and sorting
- Physically motivated range filtering using domain-informed bounds
- Forward/backward filling of short missing intervals
- Strict chronological train/validation/test splitting
- Feature standardisation using training data only

Preprocessed outputs are **frozen to disk** and reused unchanged by all models.

The same cleaned and scaled datasets are consumed by:

- Classical baseline models (tabular format)
- Temporal models (sequence format for LSTM)

---

## 🤖 Models

### Baseline Models

Baseline models establish conservative reference performance using classical, non-temporal machine learning methods.

- **Linear Regression**
  - Interpretable linear benchmark
- **Random Forest Regressor**
  - Non-linear ensemble with fixed hyperparameters

Baseline models:

- Consume frozen preprocessing outputs
- Perform no additional cleaning or scaling
- Are intentionally not exhaustively tuned

### Temporal Model

- **LSTM (Long Short-Term Memory)**
- Captures temporal dependencies in solar wind–geomagnetic interactions
- Trained on fixed-length sequences derived from the same preprocessed data

---

## 📈 Evaluation

Models are evaluated using standard regression metrics:

- **RMSE** – Root Mean Square Error
- **MAE** – Mean Absolute Error
- **R²** – Coefficient of Determination

Results are analysed quantitatively and visually to assess both average performance and temporal behaviour. Qualitative
assessment is supported via time-series visualisations of predictions versus observations.

---

## 🧪 Testing & Validation

### Data Validation

Automated validation checks include:

- Schema verification
- Missing data analysis
- Hourly time-continuity checks
- Physical plausibility bounds for solar wind and geomagnetic parameters

These checks ensure scientific consistency prior to modelling.

### Unit Testing

Automated tests validate:

- Preprocessing correctness and sequence construction
- Schema and physical validation logic
- Baseline model training and artefact persistence
- Minimum dataset size and shape constraints

Tests are implemented using `pytest` and operate entirely on synthetic data to ensure isolation and reproducibility.

Run all tests with:

```bash
python -m pytest
```

---

## 🛠️ Installation

### Requirements

- Python **3.13**
- pip

### Setup

```bash
# Clone repository
git clone https://github.com/Mustang1138/geomagnetic_forecasting.git
cd geomagnetic_forecasting

# Create and activate virtual environment
python3.13 -m venv venv
source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Install PyTorch (CPU-only)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## 🚀 Usage

Example end-to-end workflow:

```bash
# 1. Download and parse raw datasets
python -m src.data_loader

# 2. Preprocess and engineer features
python -m src.preprocess

# 3. Train baseline and temporal models
python -m src.train

# 4. Evaluate and compare performance
python -m src.evaluate
```

---

## 📝 Progress Tracking

Development progress, design decisions, and weekly milestones are documented in:

```
docs/progress_notes.md
```

---

## 📄 Licence

MIT Licence — Academic research project for Edge Hill University.

---

## 🙏 Acknowledgements

- Prof. Ella Pereira (Project Supervisor)
- NASA Space Physics Data Facility (SPDF)
- NOAA Space Weather Prediction Center

---

*Last updated: January 2026*