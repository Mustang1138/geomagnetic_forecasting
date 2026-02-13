# 🌌 Geomagnetic Forecasting – Machine Learning Evaluation Project

**A controlled research study evaluating whether temporal deep learning models provide measurable performance improvements over classical baselines for geomagnetic activity forecasting.**

---

## 📋 Project Information

- **Student:** Mark Lewis (25214071)  
- **Supervisor:** Prof. Ella Pereira  
- **Module:** CIS3425 – Research and Development Project  
- **Programme:** B.Sc. (Hons) Software Engineering  
- **Academic Year:** 2025/2026  

---

## 🎯 Research Question

> Do temporal sequence models (LSTM, GRU) meaningfully improve geomagnetic forecasting performance compared to classical baselines under controlled preprocessing conditions?

The project focuses on **methodological fairness, reproducibility, and scientific validity** rather than operational deployment.

Specifically, the project:

- Develops a fully reproducible offline forecasting pipeline
- Establishes conservative classical baselines for comparison
- Applies a unified preprocessing strategy to eliminate data-handling bias
- Evaluates models using standard regression metrics on historical space-weather data

---

## 📊 Project Scope

This project is intentionally scoped as an **offline, proof-of-concept research study**.

### ✅ Included

- Historical data analysis (2010–2026)
- Rigorous data ingestion, validation, and preprocessing
- Feature engineering for both tabular and temporal models
- Baseline regression models (Linear, Random Forest, Persistence)
- Temporal sequence models (LSTM & GRU)
- Controlled model comparison using consistent splits
- Quantitative performance evaluation and visualisation

### ❌ Excluded

- Real-time forecasting or alerting systems
- Production deployment
- Operational space-weather services
- GPU-dependent or distributed training systems

---

## 🗂️ Project Structure

```
geomagnetic_forecasting/
│
├── data/
│   ├── raw/                    # Raw downloaded datasets (OMNI2, DSCOVR)
│   └── processed/              # Frozen, preprocessed datasets & scalers
│
├── src/
│   ├── __init__.py             # Package marker for src module
│   │
│   ├── data/
│   │   ├── data_loader.py      # Data acquisition and consolidation
│   │   ├── data_sources.py     # Data source configurations
│   │   ├── sequence_datasets.py # Sequence window construction
│   │   └── torch_datasets.py   # PyTorch dataset adapters
│   │
│   ├── preprocessing/
│   │   ├── parsers.py          # OMNI2 and DSCOVR parsing utilities
│   │   ├── prepare_data.py     # Data preparation pipeline
│   │   └── preprocess.py       # Cleaning and feature engineering
│   │
│   ├── features/
│   │   └── derived_features.py # Feature engineering utilities
│   │
│   ├── models/
│   │   ├── baseline_models.py  # Linear and Random Forest models
│   │   ├── persistence.py      # Persistence baseline
│   │   ├── temporal_model.py   # LSTMRegressor, GRURegressor
│   │   └── training/           # Training procedures
│   │       ├── train_baselines.py
│   │       ├── train_lstm.py
│   │       ├── train_gru.py
│   │       └── train_utils.py
│   │
│   ├── evaluation/
│   │   ├── evaluate.py         # Metric calculation and comparison
│   │   └── validators.py       # Schema, continuity, and physical validation
│   │
│   ├── run_baselines.py        # Baseline execution script
│   └── utils.py                # Logging and helper utilities
│
├── results/
│   ├── baselines/
│   │   ├── models/             # Trained baseline model artifacts
│   │   ├── plots/              # Baseline visualisations
│   │   └── predictions/        # Baseline predictions (CSV)
│   └── metrics_baselines.csv   # Baseline performance metrics
│
├── outputs/
│   ├── plots/                  # Additional evaluation plots
│   └── metrics/                # Additional metrics
│
├── tests/                      # Unit tests for all components
│
├── docs/                       # Project documentation
│   └── progress_notes.md
│
├── logs/                       # Execution logs
├── config.yaml                 # Centralised configuration
├── requirements.txt            # Python dependencies
├── pytest.ini                  # Pytest configuration
├── LICENSE                     # MIT Licence
├── REFERENCES.md               # Project references
└── README.md                   # Project overview
```

**Design Principles:**

- Separation of concerns (data vs models vs training vs evaluation)
- Stateless model classes for testability
- Deterministic preprocessing with frozen artifacts
- Reproducible end-to-end execution
- Extensibility for future architectures (e.g., Transformers)

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

## 🔄 Unified Preprocessing Pipeline

A **single unified preprocessing pipeline** is applied to **all models** to eliminate bias.

This design is a deliberate methodological choice to ensure that:

- No model benefits from privileged data handling
- Data leakage is strictly prevented
- Observed performance differences arise from model architecture, not preprocessing

**Key Steps:**

1. Chronological validation and sorting
2. Schema and physical plausibility checks
3. Forward/backward filling of short missing intervals
4. **Explicit assertion: no null values reach model training**
5. Strict chronological train/validation/test split
6. Feature scaling fit on training data only
7. Frozen output persistence for reproducibility

Preprocessed outputs are **frozen to disk** and reused unchanged by all models.

The same cleaned and scaled datasets are consumed by:

- Classical baseline models (tabular format)
- Temporal models (sequence format for LSTM/GRU)

---

## 🤖 Models Implemented

### Baseline Models

Baseline models establish conservative reference performance using classical, non-temporal machine learning methods.

- **Linear Regression** – Interpretable linear benchmark
- **Random Forest Regressor** – Non-linear ensemble with fixed hyperparameters
- **Persistence Model** – Last value predictor

Baseline models:

- Consume frozen preprocessing outputs
- Perform no additional cleaning or scaling
- Are intentionally not exhaustively tuned

### Temporal Deep Learning Models

Implemented in PyTorch:

- **LSTM Regressor** (Long Short-Term Memory)
- **GRU Regressor** (Gated Recurrent Unit)

Both models:

- Capture temporal dependencies in solar wind–geomagnetic interactions
- Enforce explicit input shape validation
- Accept input of shape `(batch, seq_len, n_features)`
- Produce scalar sequence-to-one predictions
- Share consistent interfaces for fair comparison
- Are trained on fixed-length sequences derived from the same preprocessed data

All models use identical training/validation/test splits and evaluation metrics.

---

## 📈 Evaluation Strategy

Models are evaluated using standard regression metrics:

- **RMSE** – Root Mean Square Error
- **MAE** – Mean Absolute Error
- **R²** – Coefficient of Determination

**Evaluation Process:**

- Identical test sets across all models
- Inverse scaling of predictions for fair comparison
- Quantitative performance tables
- Visual comparison plots and time-series visualisations

Results are analysed quantitatively and visually to assess both average performance and temporal behaviour.

The objective is controlled architectural comparison, not hyperparameter maximisation.

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
- Baseline model training and artifact persistence
- LSTM and GRU input validation
- Torch dataset adapters
- Minimum dataset size and shape constraints

Tests are implemented using `pytest` and operate entirely on synthetic data to ensure isolation and reproducibility.

Run all tests with:

```bash
pytest
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

Run full pipeline (TBC)

```bash
python -m src.run_all
```

This executes:
- Data ingestion
- Preprocessing
- Baseline model training
- Temporal model training (LSTM & GRU)
- Evaluation and visualisation

*Note: A unified run script to execute the complete pipeline will be implemented upon project completion.*

---

## 🔮 Future Extensions

- Hyperparameter optimisation
- Transformer-based sequence models
- Probabilistic forecasting (prediction intervals)
- Interactive web-based visualisation
- Geographical auroral activity mapping
- Unified experiment execution script

---

## 📝 Progress Tracking

Development progress, design decisions, and weekly milestones are documented in:

```
docs/progress_notes.md
```

---

## 📄 Licence

MIT Licence – Academic research project for Edge Hill University.

---

## 🙏 Acknowledgements

- Prof. Ella Pereira – Project Supervisor
- NASA Space Physics Data Facility (SPDF)
- NOAA Space Weather Prediction Center

---

*Last updated: February 2026*