# Geomagnetic Forecasting - ML Evaluation Project

**An offline proof-of-concept investigating whether temporal ML models outperform baseline methods for geomagnetic 
activity prediction.**

---

## 📋 Project Information

- **Student:** Mark Lewis (25214071)
- **Supervisor:** Prof. Ella Pereira
- **Module:** CIS3425 - Research and Development Project
- **Programme:** B.Sc. (Hons) Software Engineering
- **Academic Year:** 2025/2026

---

## 🎯 Project Aim

To investigate whether machine learning-based temporal modelling approaches can provide better predictive performance 
for geomagnetic forecasting compared to established baseline methods, and to develop and evaluate a suitable forecasting
model using historical space-weather data.

---

## 📊 Project Scope

This is an **offline, proof-of-concept** evaluation project:
- ✅ Historical data analysis (2010-2025)
- ✅ Model comparison (baseline vs temporal)
- ✅ Performance evaluation
- ❌ No real-time deployment
- ❌ Not production-ready system
- ❌ No live API integration

---

## 🗂️ Project Structure
```
geomagnetic_forecasting/
│
├── data/
│   ├── raw/                # Raw data from APIs
│   └── processed/          # Cleaned, preprocessed datasets
│
├── src/
│   ├── data_loader.py      # Data acquisition from APIs
│   ├── preprocess.py       # Data cleaning & feature engineering
│   ├── baseline_models.py  # Linear Regression, Random Forest
│   ├── temporal_model.py   # LSTM implementation
│   ├── train.py            # Model training pipeline
│   ├── evaluate.py         # Performance metrics & comparison
│   └── utils.py            # Helper functions
│
├── results/
│   ├── plots/              # Visualisations
│   └── metrics/            # Performance metrics (CSV)
│
├── notebooks/              # Jupyter notebooks for exploration
├── docs/                   # Progress notes & documentation
│
├── config.yaml             # Configuration parameters
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 📡 Data Sources

| Source | Data Type | Provider |
|--------|-----------|----------|
| Solar Wind Parameters | Velocity, Density, Temperature | NOAA DSCOVR |
| Dst Index | Geomagnetic disturbance | Kyoto University WDC |
| Kp Index | Planetary geomagnetic activity | GFZ Potsdam |

All data sources are publicly available and documented.

---

## 🤖 Models

### Baseline Models (Benchmarks)
- **Linear Regression**         - Simple linear relationship
- **Random Forest Regressor**   - Non-linear ensemble method

### Temporal Model
- **LSTM** (Long Short-Term Memory) - Captures time-series dependencies

---

## 📈 Evaluation Metrics
- **RMSE** (Root Mean Square Error) - Overall prediction accuracy
- **MAE** (Mean Absolute Error) - Average prediction error
- **R²** (Coefficient of Determination) - Model fit quality

Visual comparisons via time-series plots.

---

## 🛠️ Installation
### Requirements
- Python 3.13
- pip (package manager)

### Setup
```bash
# Clone repository
git clone https://github.com/Mustang1138/geomagnetic_forecasting.git
cd geomagnetic_forecasting

# Create virtual environment
python3.13 -m venv venv
source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Install PyTorch (CPU-only)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## 🚀 Usage
*To be completed as development progresses*
```bash
# Example workflow (future):
# 1. Acquire data
python src/data_loader.py

# 2. Preprocess
python src/preprocess.py

# 3. Train models
python src/train.py

# 4. Evaluate
python src/evaluate.py
```

---

## 📝 Progress Tracking
See `docs/progress_notes.md` for detailed weekly updates and development log.

---

## 📄 License
MIT License - Academic project for Edge Hill University

---

## 🙏 Acknowledgments
- Prof. Ella Pereira (Project Supervisor)
- NOAA Space Weather Prediction Center
- Kyoto University World Data Center
- GFZ Potsdam University

---
*Last Updated: January 2026*