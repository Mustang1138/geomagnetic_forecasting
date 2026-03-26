# 🌌 Geomagnetic Forecasting – Machine Learning Evaluation & Visualisation Platform

**A controlled research study evaluating whether temporal deep learning models (LSTM, GRU) provide measurable
performance improvements over classical baselines for geomagnetic activity forecasting - extended with an interactive 3D
auroral visualisation web app.**

---

## 📋 Project Information

- **Student:** Mark Lewis (25214071)
- **Supervisor:** Prof. Ella Pereira
- **Module:** CIS3425 – Research and Development Project
- **Programme:** B.Sc. (Hons) Software Engineering
- **Academic Year:** 2025/2026

---

## 🎯 Research Question

> Do temporal sequence models (LSTM, GRU) meaningfully improve geomagnetic forecasting performance compared to classical
> baselines under controlled preprocessing conditions?

The project prioritises **methodological fairness**, **reproducibility**, and **scientific validity**, while adding a
rich visual analytics layer to explore model predictions and auroral phenomena.

---

## ✨ Key Features

### Machine Learning Pipeline

- Unified preprocessing applied to **all models** (eliminates data-handling bias)
- Classical baselines: Linear Regression, Random Forest, Persistence
- Temporal deep learning models: LSTM & GRU (PyTorch)
- Comprehensive evaluation with RMSE, MAE, R², residual analysis, feature importance and time-series plots
- Fully reproducible with frozen datasets

### Interactive Visualisation (Frontend)

- 3D Earth globe with dynamic **auroral oval** rendering
- Auroral latitude and intensity driven by model predictions
- Storm severity colour mapping (Green → Quiet … Purple → Extreme)
- Northern / Southern / Global views
- Country overlays with visibility weighting
- Interactive timeline with playback, scrubbing and storm markers
- Easy model switching (Baseline vs GRU vs LSTM)

---

## 🗂️ Project Structure

```bash
geomagnetic_forecasting/
├── src/                    # Python backend (ML + FastAPI)
│   ├── api/                # FastAPI routes and prediction serving
│   ├── data/               # Data loading & sequence creation
│   ├── preprocessing/      # Data cleaning and feature engineering
│   ├── features/
│   ├── models/             # Baselines + LSTM/GRU + training
│   ├── evaluation/
│   └── utils.py
├── frontend/               # React + Vite + TypeScript + Three.js
├── data/                   # raw/ and processed/
├── outputs/                # plots, metrics, model artifacts
├── results/
├── logs/
├── tests/
├── config.yaml
├── requirements.txt
└── README.md
```

---

## 📡 Data Sources

| Dataset    | Parameters                                  | Provider            |
|------------|---------------------------------------------|---------------------|
| **OMNI2**  | IMF Bz, Bt, solar wind speed & density, Dst | NASA SPDF / OMNIWeb |
| **DSCOVR** | Near-real-time magnetic field & plasma      | NOAA SWPC           |

---

## 🔄 Unified Preprocessing Pipeline

A single shared pipeline ensures fairness:

1. Schema & physical plausibility validation
2. Missing value handling
3. Strict chronological train/validation/test split
4. Feature scaling (fit on training data only)
5. Frozen preprocessed artifacts reused by every model

---

## 🤖 Models Implemented

### Baseline Models

- **Linear Regression** – Interpretable benchmark
- **Random Forest Regressor** – Non-linear with feature importance
- **Persistence** – Last-value predictor

### Temporal Models (PyTorch)

- **LSTM Regressor**
- **GRU Regressor**

All models use identical data splits and output predictions in physical units.

---

## 📈 Evaluation Strategy

- **Metrics:** RMSE, MAE, R²
- **Diagnostics:** Residual scatter & histograms, error distributions, model ranking chart
- **Visualisations:** Time-series comparisons and feature importance plots

---

## 🛠️ Installation & Running

### 1. Setup Environment

```bash
# From project root
python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 2. Download and Prepare Data

```bash
# Download raw data from NASA/NOAA sources
python -m src.data.data_loader

# Run full preprocessing pipeline (cleaning, feature engineering, scaling, splitting)
python -m src.preprocessing.prepare_data
```

### 3. Train Models

```bash
# Train classical baselines (Linear Regression + Random Forest)
python -m src.models.training.train_baselines

# Train LSTM model
python -m src.models.training.train_lstm

# Train GRU model
python -m src.models.training.train_gru
```

### 4. Evaluate All Models

```bash
python -m src.evaluation.evaluate
```

### 5. Start the Web Application

**Backend (FastAPI)**

```bash
uvicorn src.api.main:app --reload --port 8000
```

API available at: `http://127.0.0.1:8000`

**Frontend (React + Vite + Three.js)**

```bash
cd frontend
npm install
npm run dev
```

Frontend available at: `http://localhost:5173`

---

## 🧪 Testing

```bash
pytest
```

---

## 🔮 Future Extensions

- Transformer-based models
- Probabilistic forecasting with uncertainty
- Real-time data ingestion
- Improved auroral physics modelling

---

## 📝 Documentation

- Detailed progress notes: `docs/progress_notes.md`
- References: `REFERENCES.md`

---

## 📄 Licence

MIT Licence – Academic research project for Edge Hill University.

---

## 🙏 Acknowledgements

- Prof. Ella Pereira – Project Supervisor
- NASA Space Physics Data Facility (SPDF)
- NOAA Space Weather Prediction Center

---

*Last updated: March 2026*
