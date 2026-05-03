# Geomagnetic Forecasting - Machine Learning Evaluation & Visualisation Platform

A reproducible, leakage-controlled comparison study of machine-learning models for short-horizon geomagnetic
storm-severity prediction, extended with an interactive 3D auroral visualisation web application. Under
controlled conditions, temporal sequence architectures provide no measurable forecasting advantage: Random
Forest achieves R² = 0.97 and a skill score of 0.91, whilst LSTM and GRU perform within measurement error of
the trivial persistence baseline.

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

> Do temporal sequence architectures (LSTM, GRU) provide a measurable forecasting advantage over classical
> regression baselines (Random Forest, Linear Regression, Persistence) for 6-hourly geomagnetic storm-severity
> prediction under strictly controlled, leakage-free experimental conditions?

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
- Hyperparameter optimisation via Optuna (`tune.py`); tuned values committed to `config.yaml`
- Random Forest **90 % prediction intervals** from per-tree quantile aggregation (Meinshausen 2006), exposed
  via the API and rendered as a CI band in the forecast view
- Comprehensive evaluation: RMSE, MAE, R², skill score, residual analysis, feature importance, model ranking
- **Statistical significance testing**: pairwise Diebold–Mariano with Harvey small-sample correction
- **Stratified analysis**: per-SSI-class skill score against a class-restricted persistence baseline
- **Ablation study**: 4-feature (Dst-withheld) re-training to quantify Dst's marginal contribution
- Dissertation figures regenerated end-to-end with `python -m src.evaluation.figures`

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
│   │   ├── data_fetcher.py           # Downloads OMNI2 annual files
│   │   ├── data_sources.py           # HTTP clients (OMNI2, DSCOVR)
│   │   ├── build_aurora_lookup.py    # Builds country visibility index
│   │   ├── sequence_datasets.py      # Sliding-window sequence builder
│   │   └── torch_datasets.py         # PyTorch Dataset wrappers
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
│   │   ├── rf_quantile.py            # RF prediction intervals from estimators_
│   │   └── training/
│   │       ├── train_baselines.py
│   │       ├── train_lstm.py
│   │       ├── train_gru.py
│   │       ├── train_utils.py        # Trainer class and TrainingConfig
│   │       └── tune.py               # Optuna hyperparameter search
│   ├── evaluation/
│   │   ├── evaluate.py               # Evaluation orchestrator
│   │   ├── plots.py                  # Per-model plots and shared palette
│   │   ├── dm_test.py                # Pairwise Diebold–Mariano significance
│   │   ├── stratified_metrics.py     # Per-SSI-class skill scoring
│   │   ├── run_ablation.py           # 4-feature Dst-withheld experiment
│   │   ├── validators.py             # Preprocessed data validation
│   │   └── figures/                  # Dissertation figure generators
│   │       ├── __main__.py           # `python -m src.evaluation.figures`
│   │       ├── _common.py            # Paths, palette, prediction loaders
│   │       ├── main_results.py       # Headline test-set figures
│   │       ├── storm_window.py       # Storm-epoch detail figures
│   │       ├── diagnostics.py        # SSI ACF, Dst-vs-SSI, class distribution
│   │       ├── significance.py       # DM heatmap, stratified skill bars
│   │       └── loss_curves.py        # LSTM/GRU train/val loss curves
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
| GET    | `/api/forecast`              | Live 7-day forecast from current DSCOVR conditions; the `rf` entry includes 90 % prediction-interval bounds |

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

# (Optional) Run the Dst-withheld ablation, significance tests, and stratified metrics
python -m src.evaluation.run_ablation
python -m src.evaluation.dm_test
python -m src.evaluation.stratified_metrics

# Regenerate the dissertation figure set → outputs/
python -m src.evaluation.figures

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
```

---

## Testing

The repository ships with 122 tests covering preprocessing, derived features, every model, the evaluation
pipeline, the API routes, the autoregressive forecast engine, and the real-time DSCOVR pipeline.

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

## Acknowledgements

- Prof. Ella Pereira - Project Supervisor, Edge Hill University
- David Broome - Doctoral Researcher of Environment and Natural Resources at the University of Iceland
- NASA Space Physics Data Facility (SPDF) - OMNI2 dataset
- NOAA Space Weather Prediction Center - DSCOVR real-time data

---

*Last updated: May 2026*
