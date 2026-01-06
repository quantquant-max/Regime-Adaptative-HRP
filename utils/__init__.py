"""
HRP Utils Package
=================
Hierarchical Risk Parity portfolio optimization utilities.

This package provides modules for:
- Data loading and preprocessing (hrp_data)
- HRP weight computation (hrp_functions)
- Pipeline orchestration (hrp_pipeline)
- Analytics and visualization (hrp_analytics, hrp_viz)
- Feature engineering (hrp_features)
- ML regime prediction (hrp_ml, hrp_hmm)
- Strategy computation (hrp_strategy)
- Covariance shrinkage (cov_shrinkage)
- Logging (hrp_logger)
"""

# Common imports used across modules
import numpy as np
import pandas as pd
import os
import sys

# Core modules
from .hrp_setup import setup_environment, load_config, get_file_paths
from .hrp_data import load_market_data, load_risk_free_rate, load_fred_data
from .hrp_functions import compute_hrp_weights, compute_hrp_weights_batch
from .hrp_pipeline import run_hrp_computation, run_backtest
from .hrp_analytics import (
    plot_universe_size,
    plot_portfolio_size,
    plot_hrp_dendrogram,
    plot_weight_distribution,
    plot_industry_exposure_over_time,
    load_permno_to_ff12_mapping,
    FF12_NAMES
)
from .hrp_logger import setup_logger

# ML modules (imported on demand due to heavier dependencies)
def get_ml_modules():
    """Lazy import of ML modules to avoid loading heavy dependencies unnecessarily."""
    from . import hrp_features, hrp_hmm, hrp_ml, hrp_strategy, hrp_viz
    return hrp_features, hrp_hmm, hrp_ml, hrp_strategy, hrp_viz

__version__ = "1.0.0"
__all__ = [
    # Setup
    'setup_environment', 'load_config', 'get_file_paths',
    # Data
    'load_market_data', 'load_risk_free_rate', 'load_fred_data',
    # HRP
    'compute_hrp_weights', 'compute_hrp_weights_batch',
    'run_hrp_computation', 'run_backtest',
    # Analytics
    'plot_universe_size', 'plot_portfolio_size', 'plot_hrp_dendrogram',
    'plot_weight_distribution', 'plot_industry_exposure_over_time',
    'load_permno_to_ff12_mapping', 'FF12_NAMES',
    # Logging
    'setup_logger',
    # ML (lazy)
    'get_ml_modules',
]
