# Regime-Adaptive HRP

A regime-adaptive Hierarchical Risk Parity (HRP) portfolio optimization framework combining Hidden Markov Models (HMM) for regime detection with machine learning-enhanced portfolio allocation.

## Setup Instructions

1. **Download the HRP folder** from this repository

2. **Add the required data files** (received by mail) to the following locations:

   - **CRSP data** → Place in `HRP/DATA/CRSP/`:
     - `CRSP_selected_columns.csv`
     - `CRSP_value_weighted_returns.csv`

   - **Compustat data** → Place in `HRP/DATA/Compustat/`:
     - `compustat_selected_columns.csv`

3. **Install dependencies** and run the notebooks in the `Codes/` folder

## Project Structure

```
HRP/
├── Codes/                  # Jupyter notebooks for analysis
├── DATA/                   # Data folder
│   ├── CRSP/              # CRSP market data (add files here)
│   ├── Compustat/         # Compustat fundamentals (add files here)
│   └── FRED/              # FRED macroeconomic data
├── Report/                 # LaTeX project report
├── Super_Agent_Output/     # Model outputs and visualizations
├── tests/                  # Unit tests
├── utils/                  # Python utility modules
└── config.yaml            # Configuration file
```

## Note

The large data files (CRSP and Compustat) are not included in this repository due to size limitations and licensing restrictions. Please contact the repository owner to obtain the necessary data files.
