# Regime-Adaptive HRP

A regime-adaptive Hierarchical Risk Parity (HRP) portfolio optimization framework combining Hidden Markov Models (HMM) for regime detection with XGBoost machine learning for regime prediction.

## Overview

This project implements a two-stage HRP portfolio that:
1. Aggregates CRSP stocks into 12 value-weighted Fama-French industry ETFs
2. Applies HRP to these 12 industries (ensuring N=12 < T=60 for full-rank covariance)
3. Uses a 2-state HMM to identify Bull/Bear market regimes
4. Trains XGBoost in walk-forward fashion to predict next-month regimes
5. Adjusts portfolio leverage based on P(Bull) predictions

## Setup Instructions

### 1. Clone/Download the Repository

```bash
git clone <repository-url>
cd HRP
```

### 2. Install Required Libraries

```bash
pip install cupy-cuda12x scikit-learn scipy tqdm pandas numpy matplotlib xgboost hmmlearn ta joblib pyyaml pytest optuna
```

**Required Libraries:**
| Library | Purpose |
|---------|---------|
| `pandas`, `numpy` | Data manipulation |
| `scipy` | Hierarchical clustering, statistical tests |
| `scikit-learn` | Cross-validation, metrics |
| `xgboost` | Regime prediction model |
| `hmmlearn` | Hidden Markov Model for regime detection |
| `matplotlib` | Visualization |
| `cupy-cuda12x` | GPU acceleration (optional, falls back to CPU) |
| `optuna` | Hyperparameter tuning (optional) |
| `pyyaml` | Configuration file parsing |
| `joblib` | Model serialization |

### 3. Add Required Data Files

The large data files are not included due to size/licensing. Add them to:

- **CRSP data** → `DATA/CRSP/`:
  - `CRSP_selected_columns.csv` (stock-level returns, prices, volume)
  - `CRSP_value_weighted_returns.csv` (market benchmark)

- **Compustat data** → `DATA/Compustat/`:
  - `compustat_selected_columns.csv` (book equity for valuation spread)

FRED and Fama-French data are included in the repository.

## Project Structure

```
HRP/
├── Codes/                      # Jupyter notebooks
│   ├── CUDA-HRP-MULTIVAR-ML-V3.ipynb   # Main notebook (run this)
│   └── code archive/           # Previous versions
├── DATA/                       # Data folder
│   ├── CRSP/                   # CRSP market data (add files here)
│   ├── Compustat/              # Compustat fundamentals (add files here)
│   └── FRED/                   # FRED macroeconomic data (included)
├── Report/                     # LaTeX project report
│   ├── project_report.tex      # Main report
│   ├── references.bib          # Bibliography
│   └── figures/                # Report figures
├── Super_Agent_Output/         # Model outputs and visualizations
├── tests/                      # Unit tests
├── utils/                      # Python utility modules
└── config.yaml                 # Configuration file
```

## Utility Modules (`utils/`)

| Module | Description |
|--------|-------------|
| `hrp_data.py` | Data loading, filtering, universe construction, industry ETF computation |
| `hrp_functions.py` | HRP weight computation, covariance estimation, hierarchical clustering |
| `hrp_pipeline.py` | HRP orchestration, backtesting loop |
| `hrp_hmm.py` | HMM regime detection, forward-only filter (no look-ahead bias) |
| `hrp_features.py` | ML feature engineering (13 predictors), Z-scoring |
| `hrp_ml.py` | XGBoost training, walk-forward prediction, metrics |
| `hrp_strategy.py` | Portfolio overlay, two-stage transaction costs |
| `hrp_viz.py` | Visualization functions |
| `hrp_analytics.py` | Performance analytics, subsample analysis |
| `hrp_logger.py` | Logging utilities |
| `hrp_setup.py` | Environment setup, GPU detection |

## How to Run

### Execution Order

1. **Open the main notebook**: `Codes/CUDA-HRP-MULTIVAR-ML-V3.ipynb`

2. **Run cells sequentially** (the notebook is organized in phases):

   | Cell | Phase | Description |
   |------|-------|-------------|
   | 1-2 | Setup | Install packages, import modules, load config |
   | 3 | Data | Load CRSP, FRED, Compustat data |
   | 4 | HRP | Compute two-stage HRP weights (12 industries) |
   | 5 | Backtest | Performance metrics for static HRP |
   | 6-7 | Analysis | Weight distributions, dendrogram |
   | 8-9 | Phase 1-2 | HMM regime detection (Bull/Bear labels) |
   | 10-11 | Phase 3 | Walk-forward XGBoost prediction |
   | 12-13 | Phase 4 | Portfolio integration (GROSS returns) |
   | 14 | Phase 5 | Transaction costs (NET returns) |
   | 15-17 | Analysis | Subsample analysis, statistical tests, save results |

3. **Outputs** are saved to `Super_Agent_Output/`:
   - `all_hrp_weights.csv` - Stock-level HRP weights
   - `hrp_industry_weights.csv` - Industry-level HRP weights
   - `regime_strategy_results.csv` - Strategy returns (NET)
   - `wf_predictions.csv` - Walk-forward predictions
   - Various figures (`.png`)

### Configuration

Edit `config.yaml` to adjust parameters:

```yaml
data:
  window: 60              # Lookback window for covariance
  rebalance_freq: 1M      # Monthly rebalancing

hmm:
  min_train: 120          # Minimum HMM training months
  refit_freq: 12          # HMM refit frequency

ml:
  min_train_months: 120   # Minimum XGBoost training
  purge_gap: 12           # Purge gap for CV
  embargo_pct: 0.01       # Embargo (1% of test data)

strategy:
  tx_cost_bps: 10         # Transaction cost per trade
  financing_spread_bps: 50 # Borrowing spread when leveraged
```

## Running Tests

```bash
cd HRP
pytest tests/ -v
```

## Key Results

- **Sharpe Ratio**: 0.85 (P(Bull) Scaled) vs 0.56 (Static HRP)
- **Max Drawdown**: 25.2% vs 47.1%
- **Out-of-Sample Accuracy**: 76.5% (438 months, 1988-2024)

## Note

The large data files (CRSP and Compustat) are not included in this repository due to size limitations and licensing restrictions. Please contact the repository owner to obtain the necessary data files.
