# Geomagnetic Forecasting - Machine Learning Evaluation & Visualisation Platform

A controlled research study evaluating whether temporal deep learning models (LSTM, GRU) provide measurable
performance improvements over classical baselines for geomagnetic activity forecasting, extended with an
interactive 3D auroral visualisation web application.

---

## Project Information

| Field            | Detail                                        |
|------------------|-----------------------------------------------|
| Student          | Mark Lewis (25214071)                         |
| Supervisor       | Prof. Ella Pereira                            |
| Module           | CIS3425 – Research and Development Project    |
| Programme        | B.Sc. (Hons) Software Engineering             |
| Academic Year    | 2025/2026                                     |

---

## Research Question

> Do temporal sequence models (LSTM, GRU) meaningfully improve geomagnetic forecasting performance compared to
> classical baselines under controlled preprocessing conditions?

The project prioritises **methodological fairness**, **reproducibility**, and **scientific validity**, ensuring
all models are trained and evaluated under identical conditions with no data leakage.

---

## Key Features

### Machine Learning Pipeline

- Unified preprocessing pipeline applied to **all models** - eliminates data-handling bias
- Classical baselines: Linear Regression, Random Forest, Persistence (last-value)
- Temporal deep learning: LSTM and GRU regressors (PyTorch), trained on 6-hourly sequences
- Physics-informed Storm Severity Index (SSI) target: composite [0, 1] score from Bz, Bt, solar wind speed,
  density, and Dst, computed at hourly resolution **before** resampling to preserve storm peaks
- Hyperparameter optimisation via Optuna (`tune.py`)
- Comprehensive evaluation: RMSE, MAE, R², residual analysis, feature importance, model ranking chart

### Interactive Web Application

- 3D Earth globe with dynamic auroral oval rendering driven by live model predictions
- Northern, Southern, and global views with geomagnetically-correct oval placement (IGRF-13)
- Storm severity colour scale: Green (Quiet) → Yellow → Orange → Red → Purple (Extreme)
- Country border overlay showing historical aurora visibility probability at the current SSI level
- **History tab**: scrubable playback through the test-set predictions (Dec 2022 – present)
- **Forecast tab**: live 7-day, 6-hourly forecast generated from real-time DSCOVR solar wind observations
- Per-model metrics panel with SSI value, auroral latitude estimate, and error vs observation

---

## Data Sources

| Dataset    | Parameters                                         | Provider         |
|------------|----------------------------------------------------|------------------|
| OMNI2      | Hourly IMF Bz, Bt, solar wind speed & density, Dst | NASA SPDF        |
| DSCOVR     | Near-real-time magnetic field and plasma            | NOAA SWPC        |

OMNI2 data covers the period 2000-01-01 to the end of the previous calendar month (resolved automatically at
runtime). DSCOVR data is fetched live and used solely for inference - it is excluded from training to avoid
distribution shift between the historical and operational data streams.

---

## Project Structure

```
geomagnetic_forecasting/
├── src/
│   ├── api/
│   │   ├── main.py                   # FastAPI application entry point
│   │   ├── data_loader.py            # Loads pre-trained artefacts for serving
│   │   ├── forecast.py               # Autoregressive 7-day DSCOVR forecast
│   │   └── routes/
│   │       ├── predictions.py        # GET /api/predictions
│   │       ├── snapshot.py           # GET /api/snapshot
│   │       ├── models.py             # GET /api/models
│   │       └── forecast_route.py     # GET /api/forecast
│   ├── data/
│   │   ├── data_loader.py            # Downloads OMNI2 annual files
│   │   ├── data_sources.py           # HTTP clients (OMNI2, DSCOVR)
│   │   ├── build_aurora_lookup.py    # Builds country visibility index
│   │   ├── sequence_datasets.py      # Sliding-window sequence builder
│   │   └── torch_datasets.py        # PyTorch Dataset wrappers
│   ├── preprocessing/
│   │   ├── prepare_data.py           # Pipeline entry point
│   │   ├── preprocess.py             # DataPreprocessor class
│   │   ├── parsers.py                # OMNI2 fixed-width file parser
│   │   └── realtime_pipeline.py      # DSCOVR → scaled seed window
│   ├── features/
│   │   └── derived_features.py       # SSI, auroral latitude, storm class
│   ├── models/
│   │   ├── baseline_models.py        # Linear Regression, Random Forest
│   │   ├── temporal_model.py         # LSTMRegressor, GRURegressor (PyTorch)
│   │   ├── persistence.py            # Last-value baseline
│   │   └── training/
│   │       ├── train_baselines.py
│   │       ├── train_lstm.py
│   │       ├── train_gru.py
│   │       ├── train_utils.py        # Trainer class and TrainingConfig
│   │       └── tune.py               # Optuna hyperparameter search
│   ├── evaluation/
│   │   ├── evaluate.py               # Evaluation orchestrator
│   │   ├── plots.py                  # Matplotlib plotting functions
│   │   └── validators.py             # Preprocessed data validation
│   └── utils.py                      # Shared utilities (logging, config, I/O)
│
├── frontend/
│   └── src/
│       ├── App.tsx                   # Root component; History / Forecast tabs
│       ├── utils.ts                  # Shared types and SSI colour/label helpers
│       ├── components/
│       │   ├── AuroralMap.tsx        # 3D globe with auroral oval (Three.js)
│       │   ├── CountryOverlay.tsx    # Country border visibility layer
│       │   ├── ForecastChart.tsx     # Canvas SSI forecast chart (all models)
│       │   ├── ForecastTab.tsx       # 7-day forecast tab
│       │   ├── TimelineChart.tsx     # Canvas history playback chart
│       │   ├── ModelSelector.tsx     # Model switcher with live metrics
│       │   ├── StatsPanel.tsx        # Per-timestep model comparison sidebar
│       │   └── Controls.tsx          # Play/pause/speed/view controls
│       ├── hooks/
│       │   ├── usePredictions.ts     # useModels, usePredictions, useSnapshot
│       │   ├── useForecast.ts        # useForecast (live DSCOVR forecast)
│       │   └── useVisibilityLookup.ts # Country aurora visibility data
│       ├── geometry/
│       │   └── aurora.ts             # 3D auroral oval geometry (ring generators)
│       └── canvas/
│           └── chartUtils.ts         # Shared canvas coordinate helpers
│
├── data/
│   ├── raw/                          # Downloaded OMNI2 .dat files
│   └── processed/                    # Scaled arrays, scalers, sequence files
├── outputs/
│   ├── baselines/                    # Baseline model artefacts and plots
│   ├── temporal/                     # LSTM/GRU model artefacts and plots
│   ├── metrics_all_models.csv
│   └── model_ranking_rmse.png
├── tests/
├── config.yaml                       # All hyperparameters, paths, and URLs
├── requirements.txt
└── README.md
```

---

## Models

| Model              | Type         | Library     | Input                             |
|--------------------|--------------|-------------|-----------------------------------|
| Linear Regression  | Baseline     | scikit-learn | Tabular features (1 row)         |
| Random Forest      | Baseline     | scikit-learn | Tabular features (1 row)         |
| Persistence        | Baseline     | -           | Last observed SSI value           |
| LSTM Regressor     | Temporal     | PyTorch     | Sequence `(batch, seq_len, 5)`    |
| GRU Regressor      | Temporal     | PyTorch     | Sequence `(batch, seq_len, 5)`    |

Feature columns (in order): `bt`, `bz_gsm`, `speed`, `density`, `dst`.  
Target: Storm Severity Index (SSI), scaled at training time and inverse-scaled for evaluation.

---

## API Endpoints

All routes are served under `/api`.

| Method | Path                         | Description                                         |
|--------|------------------------------|-----------------------------------------------------|
| GET    | `/api/predictions?model=KEY` | Full test-set time series for one model (`rf`, `lr`, `ls`, `gr`, `pe`) |
| GET    | `/api/snapshot?idx=N`        | Single-timestep snapshot of all model predictions   |
| GET    | `/api/models`                | Available models with evaluation metrics            |
| GET    | `/api/forecast`              | Live 7-day forecast from current DSCOVR conditions  |

In production, the compiled React SPA (`frontend/dist/`) is served directly by the FastAPI application.

---

## Installation and Setup

### 1. Python Environment

```bash
# From project root
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

### 2. Run the ML Pipeline

Run these steps in order.

```bash
# Download raw OMNI2 data from NASA SPDF
python -m src.data.data_fetcher

# Preprocess: clean, compute SSI, scale, split → data/processed/
python -m src.preprocessing.prepare_data

# Train models
python -m src.models.training.train_baselines
python -m src.models.training.train_lstm
python -m src.models.training.train_gru

# (Optional) Hyperparameter tuning - updates suggested values in config.yaml
python -m src.models.training.tune

# Evaluate all models → outputs/
python -m src.evaluation.evaluate

# Build the country aurora visibility index (one-time, run after evaluation)
python -m src.data.build_aurora_lookup
```

### 3. Start the Web Application

**Backend - FastAPI (development)**

```bash
uvicorn src.api.main:app --reload --port 8000
```

API available at `http://127.0.0.1:8000`.

**Frontend - React + Vite (development)**

```bash
cd frontend
npm install
npm run dev        # Dev server at http://localhost:5173
npm run build      # Production build → frontend/dist/ (served by FastAPI)
npm run lint       # ESLint
```

---

## Testing

```bash
# All tests
pytest

# Single test file
pytest tests/test_baseline_models.py

# Single test by name
pytest tests/test_baseline_models.py::test_function_name
```

---

## Key Design Constraints

**SSI computed before resampling.** The Storm Severity Index is computed at hourly resolution and *then*
6-hourly averages are taken. Computing SSI after averaging dilutes storm peaks and understates severity.

**Chronological splits only.** Time-series data is split train → validation → test in temporal order with no
shuffling. The split proportions are configured in `config.yaml` (`training.test_split`,
`training.validation_split`).

**Scalers fitted on training data only.** `scaler_X.pkl` and `scaler_y.pkl` are fit on the training split and
reused for validation, test, and live inference. Fitting on the full dataset would constitute data leakage.

**6-hourly training cadence matches inference.** OMNI2 training data is resampled to 6-hourly to match the
DSCOVR inference pipeline. Changing this setting (`resample_6h` in `config.yaml`) breaks the distribution.

---

## References

See [REFERENCES.md](REFERENCES.md) for the full bibliography.

---

## Acknowledgements

- Prof. Ella Pereira - Project Supervisor, Edge Hill University
- David Broome - Doctoral Researcher of Environment and Natural Resources at the University of Iceland
- NASA Space Physics Data Facility (SPDF) - OMNI2 dataset
- NOAA Space Weather Prediction Center - DSCOVR real-time data

---

*Last updated: April 2026*
