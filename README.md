# Geomagnetic Forecasting – Machine Learning Evaluation Project

**An offline, research-focused proof-of-concept evaluating whether temporal machine learning models outperform classical
baselines for geomagnetic activity forecasting.**

---

## 📋 Project Information

- **Student:** Mark Lewis (25214071)
- **Supervisor:** Prof. Ella Pereira
- **Module:** CIS3425 - Research and Development Project
- **Programme:** B.Sc. (Hons) Software Engineering
- **Academic Year:** 2025/2026

---

## 🎯 Project Aim

The primary aim of this project is to evaluate whether machine learning–based temporal models can improve predictive
performance for geomagnetic activity forecasting when compared with established baseline methods.

Specifically, the project:

- Develops a reproducible offline forecasting pipeline
- Compares baseline statistical and ensemble models against temporal deep learning approaches
- Evaluates predictive performance using standard regression metrics on historical space‑weather data

---

## 📊 Project Scope

This project is intentionally scoped as an **offline, proof‑of‑concept research study.**

### Included

- ✅ Historical data analysis (2010-2026)
- ✅ Data ingestion, validation, and preprocessing
- ✅ Feature engineering for time‑series modelling
- ✅ Baseline vs temporal model comparison
- ✅ Quantitative performance evaluation

### Excluded

- ❌ Real‑time or operational forecasting
- ❌ Production‑ready deployment
- ❌ Live API services or dashboards
- ❌ Space‑weather alerting systems

---

## 🗂️ Project Structure
```
geomagnetic_forecasting/
│
├── data/
│   ├── raw/                # Raw downloaded datasets (OMNI2, DSCOVR)
│   └── processed/          # Cleaned and feature-engineered datasets
│
├── src/
│   ├── __init__.py         # Package marker for src module
│   ├── data_loader.py      # Data acquisition and consolidation
│   ├── parsers.py          # OMNI2 and DSCOVR parsing utilities
│   ├── validators.py       # Schema, continuity, and physical validation
│   ├── preprocess.py       # Cleaning and feature engineering
│   ├── baseline_models.py  # Linear regression and ensemble baselines
│   ├── temporal_model.py   # LSTM-based temporal model
│   ├── train.py            # Model training pipeline
│   ├── evaluate.py         # Metric calculation and comparison
│   └── utils.py            # Logging and helper utilities
│
├── results/
│   ├── plots/              # Evaluation visualisations
│   └── metrics/            # Model performance metrics (CSV)
│
├── tests/                  # Unit tests for preprocessing and validation
│   ├── test_preprocess.py
│   └── test_validators.py
│
├── notebooks/              # Exploratory analysis notebooks
├── docs/                   # Project documentation and progress logs
│
├── config.yaml             # Centralised configuration
├── requirements.txt        # Python dependencies
└── README.md               # Project overview
```

---

## 📡 Data Sources

| Dataset    | Parameters                                    | Provider            |
|------------|-----------------------------------------------|---------------------|
| **OMNI2**  | IMF Bz (GSM), solar wind speed & density, Dst | NASA SPDF / OMNIWeb |
| **DSCOVR** | Near‑real‑time magnetic field & plasma        | NOAA SWPC           |

### Notes on Data Handling

* OMNI2 hourly data are used as the primary historical dataset
* Fill values are replaced with NaN according to official documentation
* Physical plausibility checks are applied during validation
* All analysis is performed offline

---

## 🤖 Models

### Baseline Models

Baseline models are intentionally not exhaustively tuned and are used to establish lower-bound reference performance.

Used as benchmarking references:

* **Linear Regression** – Simple linear predictor
* **Random Forest Regressor** – Non‑linear ensemble baseline

### Temporal Model

* **LSTM (Long Short‑Term Memory)** – Deep learning model designed to capture temporal dependencies in geomagnetic time
  series

---

## 📈 Evaluation Metrics

Models are evaluated using standard regression metrics:

* **RMSE** – Root Mean Square Error
* **MAE** – Mean Absolute Error
* **R²** – Coefficient of Determination

Qualitative assessment is supported via time‑series visualisations of predictions versus observations.

---

## 🛠️ Installation

### Requirements

* Python **3.13**
* pip

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

Example end‑to‑end workflow:

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

## 🧪 Data Validation

Automated validation checks include:

* Schema verification
* Missing data analysis
* Hourly time‑continuity checks
* Physical plausibility bounds for solar wind and geomagnetic parameters

These checks ensure scientific consistency prior to modelling.

---

## 🧪 Testing

Automated tests are implemented to ensure correctness and reproducibility:

* Preprocessing pipeline output shapes and sequence construction
* Validation logic for schema, continuity, and physical constraints
* Minimum dataset size checks for model training

Tests are written using `pytest` and are designed to catch silent
data leakage, shape mismatches, and invalid preprocessing outputs
before model training.

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

* Prof. Ella Pereira (Project Supervisor)
* NASA Space Physics Data Facility (SPDF)
* NOAA Space Weather Prediction Center

---

*Last updated: January 2026*
