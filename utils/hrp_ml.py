"""
HRP ML Module - Machine Learning for Regime Prediction
=======================================================
Provides functions for:
- Feature alignment and preparation
- Purged Time-Series Cross-Validation (MdLP methodology)
- Permutation importance feature selection with purged CV
- Optuna hyperparameter optimization with purged CV
- Walk-forward XGBoost prediction (GPU-accelerated with CPU fallback)
- Performance metrics with bootstrap confidence intervals

References:
- López de Prado, M. (2018) Advances in Financial Machine Learning, Ch. 7
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix)
from collections import defaultdict
from tqdm import tqdm
from hrp_logger import setup_logger

# Import centralized seed management
try:
    from hrp_setup import get_random_state, get_rng
except ImportError:
    # Fallback if hrp_setup not available
    def get_random_state():
        return 42
    def get_rng():
        return np.random.default_rng(42)

logger = setup_logger()

# Suppress Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# GPU Detection for XGBoost
# =============================================================================
XGB_GPU_AVAILABLE = False
XGB_DEVICE = 'cpu'

def _detect_xgb_gpu():
    """Detect if GPU is available for XGBoost."""
    global XGB_GPU_AVAILABLE, XGB_DEVICE
    import warnings
    import json
    
    try:
        # Suppress XGBoost warnings during detection
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            
            # Try to create a small test model with GPU
            test_params = {'device': 'cuda', 'tree_method': 'hist', 'verbosity': 0}
            test_data = xgb.DMatrix(np.random.randn(10, 2), label=np.array([0, 1]*5))
            
            # Train and check if it actually ran on GPU (not silently fallback to CPU)
            booster = xgb.train(test_params, test_data, num_boost_round=1, verbose_eval=False)
            
            # Parse config JSON to check actual device used
            config = json.loads(booster.save_config())
            actual_device = config.get('learner', {}).get('generic_param', {}).get('device', 'cpu')
            
            if actual_device == 'cuda':
                XGB_GPU_AVAILABLE = True
                XGB_DEVICE = 'cuda'
                logger.info("✓ XGBoost GPU (CUDA) available")
            else:
                # XGBoost silently fell back to CPU
                XGB_GPU_AVAILABLE = False
                XGB_DEVICE = 'cpu'
                logger.info("[!] XGBoost GPU not available (fallback to CPU detected)")
                
    except Exception as e:
        XGB_GPU_AVAILABLE = False
        XGB_DEVICE = 'cpu'
        logger.info(f"[!] XGBoost GPU not available ({type(e).__name__}), using CPU")

# Detect GPU on module load
_detect_xgb_gpu()

def get_xgb_device_params() -> dict:
    """Get XGBoost parameters for current device (GPU/CPU)."""
    if XGB_GPU_AVAILABLE:
        return {'device': 'cuda', 'tree_method': 'hist'}
    else:
        return {'device': 'cpu', 'tree_method': 'hist'}


# =============================================================================
# Feature Engineering
# =============================================================================

def align_features_with_regimes(X_features: pd.DataFrame, 
                                 df_regimes: pd.DataFrame,
                                 lag: int = 1) -> tuple[pd.DataFrame, pd.Series]:
    """
    Align features with regime labels and apply lag for prediction.
    
    Goal: Features at time t predict if t+1 is Bear (0) or Bull (1).
    
    Parameters
    ----------
    X_features : pd.DataFrame
        Feature matrix
    df_regimes : pd.DataFrame
        Regime labels from HMM (0=Bear, 1=Bull)
    lag : int
        Number of periods to lag (1 = predict next month's regime)
        
    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Aligned and lagged X and y (y: 0=Bear next month, 1=Bull next month)
    """
    # Find common dates
    common_dates = X_features.index.intersection(df_regimes.index)
    
    X_aligned = X_features.loc[common_dates].copy()
    y_aligned = df_regimes.loc[common_dates, 'regime'].copy()
    
    # Shift target forward (predict next month's regime)
    y_target = y_aligned.shift(-lag)
    
    # Drop rows without target
    valid_mask = ~y_target.isna()
    X_final = X_aligned[valid_mask].copy()
    y_final = y_target[valid_mask].astype(int).copy()
    
    logger.info(f"Aligned data: {len(X_final)} observations")
    logger.info(f"Target distribution: Bear={sum(y_final==0)}, Bull={sum(y_final==1)}")
    
    return X_final, y_final


# =============================================================================
# Cross-Validation for Feature Selection
# =============================================================================

def purged_time_series_cv_splits(n_samples: int, n_splits: int = 5, 
                                  min_train: int = 120,
                                  purge_gap: int = 3,
                                  embargo_pct: float = 0.01) -> list:
    """
    Generate Purged Time-Series CV splits to prevent data leakage.
    
    This avoids information leakage by:
    1. Purge gap: Removes `purge_gap` observations between train and test
    2. Embargo: Additional buffer after test set before it can be used in training
    
    Parameters
    ----------
    n_samples : int
        Total number of samples
    n_splits : int
        Number of CV folds
    min_train : int
        Minimum training set size
    purge_gap : int
        Number of periods to purge between train and test (default=3)
        Removes label overlap from lagged predictions
    embargo_pct : float
        Embargo period as fraction of test size (default=0.01)
        
    Returns
    -------
    list
        List of (train_idx, test_idx) tuples with purge gap applied
        
    References
    ----------
    López de Prado, M. (2018) Advances in Financial Machine Learning, Ch. 7
    """
    # Calculate test size accounting for purge gaps
    usable_samples = n_samples - min_train - (n_splits * purge_gap)
    test_size = max(1, usable_samples // n_splits)
    embargo = int(test_size * embargo_pct)
    
    splits = []
    for i in range(n_splits):
        # Training ends before purge gap
        train_end = min_train + i * (test_size + purge_gap)
        
        # Test starts after purge gap
        test_start = train_end + purge_gap
        test_end = min(test_start + test_size, n_samples - embargo)
        
        if test_end > test_start and train_end > 0:
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            splits.append((train_idx, test_idx))
    
    return splits


# =============================================================================
# Walk-Forward Prediction
# =============================================================================

def _tune_hyperparameters_for_training(X_train: np.ndarray, y_train: np.ndarray,
                                        purge_gap: int = 12, embargo_pct: float = 0.01,
                                        n_cv_splits: int = 3, n_trials: int = 20) -> tuple[dict, int]:
    """
    Tune XGBoost hyperparameters using only training data (no look-ahead bias).
    
    This is called inside walk-forward to avoid look-ahead bias in hyperparameter selection.
    Uses Purged CV on the training window only.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features (up to time t)
    y_train : np.ndarray
        Training labels
    purge_gap : int
        Purge gap for CV
    embargo_pct : float
        Embargo percentage
    n_cv_splits : int
        Number of CV splits for tuning
    n_trials : int
        Number of Optuna trials (reduced for efficiency)
        
    Returns
    -------
    tuple[dict, int]
        Best parameters and best n_rounds
    """
    n_samples = len(X_train)
    min_train_inner = max(24, n_samples // 3)
    
    splits = purged_time_series_cv_splits(n_samples, n_cv_splits, min_train_inner,
                                          purge_gap, embargo_pct)
    
    if len(splits) < 2:
        # Not enough data for CV, return conservative defaults
        default_params = {
            'max_depth': 3,
            'min_child_weight': 5,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'learning_rate': 0.05,
        }
        return default_params, 100
    
    def objective(trial):
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'verbosity': 0,
            'random_state': get_random_state(),
            'max_depth': trial.suggest_int('max_depth', 2, 5),
            'min_child_weight': trial.suggest_int('min_child_weight', 3, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 0.3),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 2.0),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.15, log=True),
            **get_xgb_device_params()  # Add GPU/CPU device params
        }
        
        n_rounds = trial.suggest_int('n_rounds', 50, 150)
        cv_scores = []
        
        for train_idx, val_idx in splits:
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            n_bull = (y_tr == 1).sum()
            n_bear = (y_tr == 0).sum()
            params['scale_pos_weight'] = n_bear / n_bull if n_bull > 0 else 1.0
            
            dtrain = xgb.DMatrix(X_tr, label=y_tr)
            dval = xgb.DMatrix(X_val, label=y_val)
            
            model = xgb.train(params, dtrain, num_boost_round=n_rounds,
                             evals=[(dval, 'val')], early_stopping_rounds=15,
                             verbose_eval=False)
            
            y_pred = (model.predict(dval) >= 0.5).astype(int)
            cv_scores.append(f1_score(y_val, y_pred, average='macro'))
        
        return np.mean(cv_scores) if cv_scores else 0.0
    
    # Run smaller Optuna search
    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=get_random_state()))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    best_params = study.best_params.copy()
    best_n_rounds = best_params.pop('n_rounds')
    
    return best_params, best_n_rounds


def _select_features_for_training(X_train: pd.DataFrame, y_train: np.ndarray,
                                   purge_gap: int = 12, embargo_pct: float = 0.01,
                                   importance_threshold: float = 0.0,
                                   n_cv_splits: int = 3) -> list:
    """
    Select features using permutation importance on training data only.
    
    This is called inside walk-forward to avoid look-ahead bias in feature selection.
    Uses Purged CV on the training window only.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features (up to time t)
    y_train : np.ndarray
        Training labels
    purge_gap : int
        Purge gap for CV
    embargo_pct : float
        Embargo percentage
    importance_threshold : float
        Threshold for feature selection
    n_cv_splits : int
        Number of CV splits for feature selection
        
    Returns
    -------
    list
        Selected feature names
    """
    xgb_params = {
        'objective': 'binary:logistic',
        'max_depth': 3,
        'learning_rate': 0.05,
        'colsample_bytree': 0.7,
        'subsample': 0.8,
        'min_child_weight': 5,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': get_random_state(),
        'eval_metric': 'logloss',
        'verbosity': 0,
        **get_xgb_device_params()  # Add GPU/CPU device params
    }
    
    n_samples = len(X_train)
    min_train_inner = max(24, n_samples // 3)  # Use 1/3 of data for inner training
    
    splits = purged_time_series_cv_splits(n_samples, n_cv_splits, min_train_inner,
                                          purge_gap, embargo_pct)
    
    if len(splits) == 0:
        # Not enough data for CV, return all features
        return list(X_train.columns)
    
    fold_importances = []
    X_arr = X_train.values
    feature_names = list(X_train.columns)
    
    for train_idx, test_idx in splits:
        X_tr = X_arr[train_idx]
        y_tr = y_train[train_idx]
        X_test = X_arr[test_idx]
        y_test = y_train[test_idx]
        
        # Handle class imbalance
        n_bull = (y_tr == 1).sum()
        n_bear = (y_tr == 0).sum()
        params = xgb_params.copy()
        params['scale_pos_weight'] = n_bear / n_bull if n_bull > 0 else 1.0
        
        # Train model
        dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_names)
        model = xgb.train(params, dtrain, num_boost_round=100, verbose_eval=False)
        
        # Baseline score
        dtest = xgb.DMatrix(X_test, feature_names=feature_names)
        y_pred_proba = model.predict(dtest)
        
        try:
            baseline_score = roc_auc_score(y_test, y_pred_proba)
        except ValueError:
            # Only one class in test set
            continue
        
        # Permutation importance
        feature_importance = {}
        for feat_idx, feat_name in enumerate(feature_names):
            X_test_permuted = X_test.copy()
            np.random.seed(get_random_state() + feat_idx)
            X_test_permuted[:, feat_idx] = np.random.permutation(X_test_permuted[:, feat_idx])
            
            dtest_perm = xgb.DMatrix(X_test_permuted, feature_names=feature_names)
            try:
                perm_score = roc_auc_score(y_test, model.predict(dtest_perm))
                feature_importance[feat_name] = baseline_score - perm_score
            except ValueError:
                feature_importance[feat_name] = 0.0
        
        fold_importances.append(feature_importance)
    
    if len(fold_importances) == 0:
        return feature_names
    
    # Aggregate importance
    importance_df = pd.DataFrame(fold_importances)
    mean_importance = importance_df.mean()
    
    # Select features above threshold
    selected = mean_importance[mean_importance > importance_threshold].index.tolist()
    
    # Always keep at least 3 features
    if len(selected) < 3:
        selected = mean_importance.nlargest(3).index.tolist()
    
    return selected


def walk_forward_predict(X_final: pd.DataFrame, y_final: pd.Series,
                         xgb_params: dict = None, n_boost_rounds: int = None,
                         min_train: int = 120, 
                         refit_freq: int = 12,
                         select_features: bool = True,
                         tune_hyperparams: bool = True,
                         purge_gap: int = 12,
                         embargo_pct: float = 0.01,
                         importance_threshold: float = 0.0,
                         optuna_trials_per_refit: int = 20) -> tuple[pd.DataFrame, list]:
    """
    Walk-forward XGBoost prediction for Bear/Bull regime prediction.
    
    Predicts if NEXT MONTH will be Bear (0) or Bull (1) regime.
    Both feature selection AND hyperparameter tuning are performed INSIDE 
    the walk-forward loop at each refit to avoid look-ahead bias.
    
    Parameters
    ----------
    X_final : pd.DataFrame
        Feature matrix (all candidate features)
    y_final : pd.Series
        Target labels (0=Bear next month, 1=Bull next month)
    xgb_params : dict, optional
        Initial XGBoost parameters (used if tune_hyperparams=False)
    n_boost_rounds : int, optional
        Number of boosting rounds (used if tune_hyperparams=False)
    min_train : int
        Minimum training months
    refit_freq : int
        Model refit frequency
    select_features : bool
        Whether to perform feature selection inside walk-forward
    tune_hyperparams : bool
        Whether to re-tune hyperparameters at each refit
    purge_gap : int
        Purge gap for CV (default=12)
    embargo_pct : float
        Embargo percentage for CV
    importance_threshold : float
        Threshold for feature selection
    optuna_trials_per_refit : int
        Number of Optuna trials per refit (default=20, reduced for efficiency)
        
    Returns
    -------
    tuple[pd.DataFrame, list]
        Predictions DataFrame and feature importances list
    """
    logger.info("="*70)
    logger.info("WALK-FORWARD BEAR/BULL PREDICTION")
    logger.info("  Goal: Predict if NEXT MONTH is Bear (0) or Bull (1)")
    logger.info(f"  XGBoost device: {XGB_DEVICE} {'(GPU)' if XGB_GPU_AVAILABLE else '(CPU)'}")
    logger.info("  ✓ Winsorization INSIDE walk-forward (training bounds only)")
    if select_features:
        logger.info("  ✓ Feature selection INSIDE walk-forward")
    if tune_hyperparams:
        logger.info(f"  ✓ Hyperparameter tuning INSIDE walk-forward ({optuna_trials_per_refit} trials/refit)")
    logger.info("="*70)
    
    wf_results = defaultdict(list)
    feature_importances_list = []
    feature_selection_history = []
    hyperparam_history = []  # Track hyperparams at each refit
    last_model = None
    last_fit_idx = -refit_freq
    dates = X_final.index.tolist()
    current_features = list(X_final.columns)
    current_winsor_bounds = {}  # Store winsorization bounds from training data
    
    # Default parameters if not tuning (includes GPU/CPU device params)
    current_params = xgb_params.copy() if xgb_params else {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'verbosity': 0,
        'random_state': get_random_state(),
        'max_depth': 3,
        'learning_rate': 0.05,
        'min_child_weight': 5,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        **get_xgb_device_params()  # Add GPU/CPU device params
    }
    current_n_rounds = n_boost_rounds if n_boost_rounds else 100
    
    for t in tqdm(range(min_train, len(X_final)), desc="Walk-Forward", leave=False):
        current_date = dates[t]
        
        # Refit periodically
        if t - last_fit_idx >= refit_freq:
            X_train_full = X_final.iloc[:t].copy()
            y_train = y_final.iloc[:t].values
            
            # Winsorization inside walk-forward - compute bounds from training only
            current_winsor_bounds = {}
            for col in X_train_full.columns:
                current_winsor_bounds[col] = {
                    'lower': X_train_full[col].quantile(0.01),
                    'upper': X_train_full[col].quantile(0.99)
                }
                X_train_full[col] = X_train_full[col].clip(
                    lower=current_winsor_bounds[col]['lower'],
                    upper=current_winsor_bounds[col]['upper']
                )
            
            # Feature selection inside walk-forward (no look-ahead bias)
            if select_features:
                current_features = _select_features_for_training(
                    X_train_full, y_train,
                    purge_gap=purge_gap, embargo_pct=embargo_pct,
                    importance_threshold=importance_threshold,
                    n_cv_splits=3
                )
                feature_selection_history.append({
                    'date': current_date, 
                    'n_features': len(current_features),
                    'features': current_features
                })
            
            # Hyperparameter tuning inside walk-forward (no look-ahead bias)
            if tune_hyperparams:
                X_train_selected = X_train_full[current_features].values
                tuned_params, tuned_n_rounds = _tune_hyperparameters_for_training(
                    X_train_selected, y_train,
                    purge_gap=purge_gap, embargo_pct=embargo_pct,
                    n_cv_splits=3, n_trials=optuna_trials_per_refit
                )
                current_params = {
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss',
                    'verbosity': 0,
                    'random_state': get_random_state(),
                    **tuned_params,
                    **get_xgb_device_params()  # Add GPU/CPU device params
                }
                current_n_rounds = tuned_n_rounds
                hyperparam_history.append({
                    'date': current_date,
                    'max_depth': tuned_params.get('max_depth'),
                    'learning_rate': tuned_params.get('learning_rate'),
                    'n_rounds': tuned_n_rounds
                })
            
            # Train on selected features (already winsorized)
            X_train = X_train_full[current_features].values
            
            n_bull = (y_train == 1).sum()
            n_bear = (y_train == 0).sum()
            
            params = current_params.copy()
            params['scale_pos_weight'] = n_bear / n_bull if n_bull > 0 else 1.0
            
            dtrain = xgb.DMatrix(X_train, label=y_train, 
                                feature_names=current_features)
            
            try:
                last_model = xgb.train(params, dtrain, num_boost_round=current_n_rounds,
                                      verbose_eval=False)
                last_fit_idx = t
                
                importance = last_model.get_score(importance_type='gain')
                feature_importances_list.append({'date': current_date, **importance})
            except Exception as e:
                logger.warning(f"Training failed at {current_date}: {e}")
                continue
        
        if last_model is None:
            continue
        
        # Predict using current selected features
        # Apply winsorization bounds from training data (no look-ahead)
        X_test_raw = X_final.iloc[t:t+1][current_features].copy()
        for col in current_features:
            if col in current_winsor_bounds:
                X_test_raw[col] = X_test_raw[col].clip(
                    lower=current_winsor_bounds[col]['lower'],
                    upper=current_winsor_bounds[col]['upper']
                )
        X_test = X_test_raw.values
        dtest = xgb.DMatrix(X_test, feature_names=current_features)
        
        prob_bull = last_model.predict(dtest)[0]
        
        wf_results['date'].append(current_date)
        wf_results['prob_bull'].append(prob_bull)
        wf_results['prob_bear'].append(1 - prob_bull)
        wf_results['predicted_regime'].append(1 if prob_bull >= 0.5 else 0)
        wf_results['actual_regime'].append(y_final.iloc[t])
    
    wf_df = pd.DataFrame(wf_results).set_index('date')
    
    # Log summaries
    if select_features and feature_selection_history:
        avg_n_features = np.mean([fs['n_features'] for fs in feature_selection_history])
        logger.info(f"Feature selection: avg {avg_n_features:.1f} features per refit")
    
    if tune_hyperparams and hyperparam_history:
        avg_depth = np.mean([h['max_depth'] for h in hyperparam_history])
        avg_lr = np.mean([h['learning_rate'] for h in hyperparam_history])
        avg_rounds = np.mean([h['n_rounds'] for h in hyperparam_history])
        logger.info(f"Hyperparams: avg max_depth={avg_depth:.1f}, lr={avg_lr:.3f}, n_rounds={avg_rounds:.0f}")
    
    logger.info(f"Walk-forward complete: {len(wf_df)} predictions")
    
    return wf_df, feature_importances_list


# =============================================================================
# Performance Metrics
# =============================================================================

def compute_prediction_metrics(wf_df: pd.DataFrame) -> dict:
    """
    Compute prediction accuracy metrics.
    
    Parameters
    ----------
    wf_df : pd.DataFrame
        Walk-forward prediction results
        
    Returns
    -------
    dict
        Dictionary of metrics
    """
    y_true = wf_df['actual_regime'].values
    y_pred = wf_df['predicted_regime'].values
    y_prob = wf_df['prob_bull'].values
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'precision_bear': precision_score(y_true, y_pred, pos_label=0),
        'precision_bull': precision_score(y_true, y_pred, pos_label=1),
        'recall_bear': recall_score(y_true, y_pred, pos_label=0),
        'recall_bull': recall_score(y_true, y_pred, pos_label=1),
        'f1_bear': f1_score(y_true, y_pred, pos_label=0),
        'f1_bull': f1_score(y_true, y_pred, pos_label=1),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    return metrics


def _block_bootstrap_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate block bootstrap indices of exactly length n.
    
    Uses circular block bootstrap to handle edge cases cleanly.
    
    Parameters
    ----------
    n : int
        Length of original sample
    block_size : int
        Size of each block
    rng : np.random.Generator
        Random number generator
        
    Returns
    -------
    np.ndarray
        Bootstrap indices of length n
    """
    n_blocks_needed = int(np.ceil(n / block_size))
    # Allow blocks to start anywhere (circular bootstrap handles wraparound)
    block_starts = rng.integers(0, n, size=n_blocks_needed)
    
    # Build indices with circular wraparound
    indices = []
    for start in block_starts:
        block_idx = np.arange(start, start + block_size) % n  # Circular wrap
        indices.append(block_idx)
    
    boot_idx = np.concatenate(indices)[:n]  # Truncate to exact length
    return boot_idx


def compute_metrics_with_ci(returns: pd.Series, n_bootstrap: int = 10000, 
                            ci: float = 0.95, random_state: int = 42,
                            block_size: int = 12, rf_rate: pd.Series = None) -> dict:
    """
    Compute performance metrics with block bootstrap confidence intervals.
    
    Uses circular block bootstrap to preserve autocorrelation structure
    in financial time series data.
    
    Parameters
    ----------
    returns : pd.Series
        Strategy returns
    n_bootstrap : int
        Number of bootstrap samples (default 10000 for robust CIs)
    ci : float
        Confidence level
    random_state : int
        Random seed
    block_size : int
        Block size for bootstrap (default 12 for monthly data)
    rf_rate : pd.Series, optional
        Monthly risk-free rate for proper excess return Sharpe calculation.
        If None, uses mean Rf = 0 (raw return Sharpe).
        
    Returns
    -------
    dict
        Metrics with confidence intervals
    """
    rng = np.random.default_rng(random_state)
    n = len(returns)
    
    # Align risk-free rate with returns
    if rf_rate is not None:
        rf_aligned = rf_rate.reindex(returns.index).fillna(rf_rate.mean())
        mean_rf_monthly = rf_aligned.mean()
        excess_returns = returns - rf_aligned
    else:
        mean_rf_monthly = 0.0
        excess_returns = returns
    
    # Point estimates
    cum_ret = (1 + returns).prod() - 1
    n_months = len(returns)
    ann_ret = (1 + cum_ret) ** (12 / n_months) - 1
    ann_vol = returns.std() * np.sqrt(12)
    
    # Sharpe: (mean excess return × 12) / (vol × sqrt(12))
    # Academic standard: use excess returns in numerator
    ann_rf = mean_rf_monthly * 12
    sharpe = (excess_returns.mean() * 12) / ann_vol if ann_vol > 0 else 0
    
    # Sortino: use downside deviation of excess returns
    downside_excess = excess_returns[excess_returns < 0]
    downside_vol = downside_excess.std() * np.sqrt(12) if len(downside_excess) > 0 else 0
    sortino = (excess_returns.mean() * 12) / downside_vol if downside_vol > 0 else 0
    
    cum_rets = (1 + returns).cumprod()
    rolling_max = cum_rets.expanding().max()
    drawdowns = cum_rets / rolling_max - 1
    max_dd = drawdowns.min()
    
    win_rate = (returns > 0).mean()
    
    # Block Bootstrap
    boot_metrics = {'sharpe': [], 'sortino': [], 'ann_ret': [], 'ann_vol': [], 'max_dd': []}
    returns_arr = returns.values
    excess_arr = excess_returns.values
    
    for _ in range(n_bootstrap):
        boot_idx = _block_bootstrap_indices(n, block_size, rng)
        boot_returns = returns_arr[boot_idx]
        boot_excess = excess_arr[boot_idx]
        
        b_cum_ret = (1 + boot_returns).prod() - 1
        b_ann_ret = (1 + b_cum_ret) ** (12 / len(boot_returns)) - 1
        b_ann_vol = boot_returns.std() * np.sqrt(12)
        
        # Bootstrap Sharpe with excess returns
        b_sharpe = (boot_excess.mean() * 12) / b_ann_vol if b_ann_vol > 0 else 0
        
        b_downside = boot_excess[boot_excess < 0]
        b_down_vol = b_downside.std() * np.sqrt(12) if len(b_downside) > 0 else b_ann_vol
        b_sortino = (boot_excess.mean() * 12) / b_down_vol if b_down_vol > 0 else 0
        
        b_cum_rets = (1 + boot_returns).cumprod()
        b_rolling_max = np.maximum.accumulate(b_cum_rets)
        b_max_dd = (b_cum_rets / b_rolling_max - 1).min()
        
        boot_metrics['sharpe'].append(b_sharpe)
        boot_metrics['sortino'].append(b_sortino)
        boot_metrics['ann_ret'].append(b_ann_ret)
        boot_metrics['ann_vol'].append(b_ann_vol)
        boot_metrics['max_dd'].append(b_max_dd)
    
    alpha = (1 - ci) / 2
    
    return {
        'Cum Return': cum_ret,
        'Ann Return': ann_ret,
        'Ann Return CI': (np.percentile(boot_metrics['ann_ret'], alpha*100), 
                          np.percentile(boot_metrics['ann_ret'], (1-alpha)*100)),
        'Ann Vol': ann_vol,
        'Ann Vol CI': (np.percentile(boot_metrics['ann_vol'], alpha*100), 
                       np.percentile(boot_metrics['ann_vol'], (1-alpha)*100)),
        'Sharpe': sharpe,
        'Sharpe CI': (np.percentile(boot_metrics['sharpe'], alpha*100), 
                      np.percentile(boot_metrics['sharpe'], (1-alpha)*100)),
        'Sortino': sortino,
        'Sortino CI': (np.percentile(boot_metrics['sortino'], alpha*100), 
                       np.percentile(boot_metrics['sortino'], (1-alpha)*100)),
        'Max DD': max_dd,
        'Max DD CI': (np.percentile(boot_metrics['max_dd'], alpha*100), 
                      np.percentile(boot_metrics['max_dd'], (1-alpha)*100)),
        'Win Rate': win_rate
    }


def bootstrap_sharpe_test(baseline_returns: np.ndarray, 
                          strategy_returns: np.ndarray,
                          n_bootstrap: int = 10000,
                          random_state: int = 42,
                          block_size: int = 12) -> tuple[float, tuple, str]:
    """
    Block bootstrap test for Sharpe ratio difference.
    
    Uses circular block bootstrap with paired resampling to preserve
    autocorrelation and correlation between strategies.
    
    Parameters
    ----------
    baseline_returns : np.ndarray
        Baseline strategy returns
    strategy_returns : np.ndarray
        Comparison strategy returns
    n_bootstrap : int
        Number of bootstrap samples (default 10000 for robust CIs)
    random_state : int
        Random seed
    block_size : int
        Block size for bootstrap (default 12 for monthly data)
        
    Returns
    -------
    tuple[float, tuple, str]
        Mean difference, confidence interval, significance string
    """
    rng = np.random.default_rng(random_state)
    n = len(baseline_returns)
    sharpe_diffs = []
    
    for _ in range(n_bootstrap):
        # Use same block bootstrap indices for both strategies (paired resampling)
        boot_idx = _block_bootstrap_indices(n, block_size, rng)
        base_boot = baseline_returns[boot_idx]
        strat_boot = strategy_returns[boot_idx]
        
        base_sharpe = (base_boot.mean() * 12) / (base_boot.std() * np.sqrt(12)) if base_boot.std() > 0 else 0
        strat_sharpe = (strat_boot.mean() * 12) / (strat_boot.std() * np.sqrt(12)) if strat_boot.std() > 0 else 0
        sharpe_diffs.append(strat_sharpe - base_sharpe)
    
    diff_mean = np.mean(sharpe_diffs)
    diff_ci = (np.percentile(sharpe_diffs, 2.5), np.percentile(sharpe_diffs, 97.5))
    
    if diff_ci[0] > 0:
        significance = "Significantly BETTER"
    elif diff_ci[1] < 0:
        significance = "Significantly WORSE"
    else:
        significance = "Not significant"
    
    return diff_mean, diff_ci, significance


def get_top_n_drawdowns(returns: pd.Series, n: int = 3) -> list:
    """
    Find the top N distinct drawdown periods.
    
    Parameters
    ----------
    returns : pd.Series
        Strategy returns
    n : int
        Number of top drawdowns to return
        
    Returns
    -------
    list
        List of dicts with keys: max_dd, start, trough, end
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1
    
    dd_periods = []
    in_drawdown = False
    period_start = None
    period_trough = None
    period_min = 0
    
    for date, dd in drawdown.items():
        if dd < 0 and not in_drawdown:
            in_drawdown = True
            period_start = date
            period_trough = date
            period_min = dd
        elif dd < 0 and in_drawdown:
            if dd < period_min:
                period_min = dd
                period_trough = date
        elif dd == 0 and in_drawdown:
            dd_periods.append({
                'max_dd': period_min,
                'start': period_start,
                'trough': period_trough,
                'end': date
            })
            in_drawdown = False
    
    if in_drawdown:
        dd_periods.append({
            'max_dd': period_min,
            'start': period_start,
            'trough': period_trough,
            'end': returns.index[-1]
        })
    
    dd_periods_sorted = sorted(dd_periods, key=lambda x: x['max_dd'])
    return dd_periods_sorted[:n]


def calc_metrics_extended(returns: pd.Series, rf_rate: pd.Series = None, 
                          n_drawdowns: int = 5) -> dict:
    """
    Calculate annualized performance metrics with top N drawdowns.
    
    Parameters
    ----------
    returns : pd.Series
        Strategy returns
    rf_rate : pd.Series, optional
        Risk-free rate for excess return Sharpe
    n_drawdowns : int
        Number of top drawdowns to include
        
    Returns
    -------
    dict
        Performance metrics including Return, Vol, Sharpe, and top N drawdowns
    """
    n_months = len(returns)
    
    if rf_rate is not None:
        rf_aligned = rf_rate.reindex(returns.index).fillna(rf_rate.mean())
        excess_returns = returns - rf_aligned
    else:
        excess_returns = returns
    
    cum_ret = (1 + returns).prod() - 1
    ann_ret = (1 + cum_ret) ** (12 / n_months) - 1
    ann_vol = returns.std() * np.sqrt(12)
    sharpe = (excess_returns.mean() * 12) / ann_vol if ann_vol > 0 else 0
    
    top_dds = get_top_n_drawdowns(returns, n=n_drawdowns)
    
    result = {
        'Ann Return': ann_ret, 
        'Ann Vol': ann_vol, 
        'Sharpe': sharpe,
    }
    
    for i, dd_info in enumerate(top_dds):
        result[f'DD{i+1}'] = dd_info['max_dd']
        result[f'DD{i+1}_period'] = f"{dd_info['start']:%Y-%m} to {dd_info['trough']:%Y-%m}"
    
    for i in range(len(top_dds), n_drawdowns):
        result[f'DD{i+1}'] = np.nan
        result[f'DD{i+1}_period'] = 'N/A'
    
    return result


def print_performance_summary(strategy_results: pd.DataFrame, 
                               strategies: dict,
                               market_returns: pd.Series,
                               rf_rate: pd.Series) -> dict:
    """
    Print detailed performance summary comparing strategies to market.
    
    Parameters
    ----------
    strategy_results : pd.DataFrame
        DataFrame with strategy return columns
    strategies : dict
        Dict mapping display names to column names
    market_returns : pd.Series
        Market benchmark returns
    rf_rate : pd.Series
        Risk-free rate series
        
    Returns
    -------
    dict
        Dictionary of metrics for each strategy
    """
    # Align market and rf with strategy results
    market_aligned = market_returns.reindex(strategy_results.index).dropna()
    common_idx = strategy_results.index.intersection(market_aligned.index)
    rf_aligned = rf_rate.reindex(strategy_results.index).fillna(rf_rate.mean())
    rf_for_market = rf_rate.reindex(common_idx).fillna(rf_rate.mean())
    
    # Calculate metrics for all strategies + market
    metrics_summary = {}
    for name, col in strategies.items():
        metrics_summary[name] = calc_metrics_extended(
            strategy_results[col], rf_rate=rf_aligned
        )
    metrics_summary['CRSP VW Index'] = calc_metrics_extended(
        market_aligned.loc[common_idx], rf_rate=rf_for_market
    )
    
    # Print summary table
    print("\n" + "="*110)
    print("PERFORMANCE SUMMARY (Annualized) - Excess Return Sharpe")
    print("="*110)
    print(f"Date Range: {strategy_results.index.min():%Y-%m} to {strategy_results.index.max():%Y-%m} ({len(strategy_results)} months)")
    print(f"Risk-Free Rate: Mean = {rf_aligned.mean()*12:.2%} annualized")
    print(f"\n{'Strategy':<18} {'Return':>10} {'Vol':>8} {'Sharpe':>8} {'DD1':>10} {'DD2':>10} {'DD3':>10} {'DD4':>10} {'DD5':>10}")
    print("-"*105)
    for name, m in metrics_summary.items():
        dd1 = f"{m['DD1']:.1%}" if not np.isnan(m['DD1']) else 'N/A'
        dd2 = f"{m['DD2']:.1%}" if not np.isnan(m['DD2']) else 'N/A'
        dd3 = f"{m['DD3']:.1%}" if not np.isnan(m['DD3']) else 'N/A'
        dd4 = f"{m['DD4']:.1%}" if not np.isnan(m['DD4']) else 'N/A'
        dd5 = f"{m['DD5']:.1%}" if not np.isnan(m['DD5']) else 'N/A'
        print(f"{name:<18} {m['Ann Return']:>9.1%} {m['Ann Vol']:>7.1%} {m['Sharpe']:>8.2f} {dd1:>10} {dd2:>10} {dd3:>10} {dd4:>10} {dd5:>10}")
    print("-"*105)
    
    # Print drawdown periods detail
    print("\n" + "="*110)
    print("TOP 5 DRAWDOWN PERIODS (by magnitude)")
    print("="*110)
    for name, m in metrics_summary.items():
        print(f"\n{name}:")
        for i in range(1, 6):
            dd_key = f'DD{i}'
            period_key = f'DD{i}_period'
            if dd_key in m and not np.isnan(m[dd_key]):
                print(f"  {i}. {m[dd_key]:>7.1%}  ({m[period_key]})")
            else:
                print(f"  {i}. N/A")
    
    return metrics_summary


def print_xgb_summary(metrics: dict, feature_importances: list, top_n: int = 5):
    """
    Print XGBoost prediction summary (confusion matrix + top features).
    
    Parameters
    ----------
    metrics : dict
        Dictionary from compute_prediction_metrics (includes confusion_matrix)
    feature_importances : list
        List of dicts with feature importance by date
    top_n : int
        Number of top features to display
    """
    cm = metrics['confusion_matrix']
    print(f"\nConfusion Matrix:")
    print(f"                    Predicted Bear  Predicted Bull")
    print(f"Actual Bear             {cm[0,0]:>8}          {cm[0,1]:>8}")
    print(f"Actual Bull             {cm[1,0]:>8}          {cm[1,1]:>8}")
    
    if feature_importances:
        fi_df = pd.DataFrame(feature_importances).set_index('date')
        avg_imp = fi_df.mean().sort_values(ascending=False)
        print(f"\nTop {top_n} Features (Avg Gain):")
        for feat, imp in avg_imp.head(top_n).items():
            print(f"  {feat}: {imp:.2f}")


def print_performance_with_ci(strategy_results: pd.DataFrame, 
                               strategies: dict, 
                               rf_rate: pd.Series) -> dict:
    """
    Print performance metrics with bootstrap confidence intervals.
    
    Parameters
    ----------
    strategy_results : pd.DataFrame
        DataFrame with strategy return columns
    strategies : dict
        Dict mapping display names to column names
    rf_rate : pd.Series
        Risk-free rate series (aligned to strategy_results)
        
    Returns
    -------
    dict
        Dictionary of metrics with CI for each strategy
    """
    rf_aligned = rf_rate.reindex(strategy_results.index).fillna(rf_rate.mean())
    
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON (95% CI) - Excess Return Sharpe")
    print("="*70)
    print(f"Risk-Free Rate: Mean = {rf_aligned.mean()*12:.2%} annualized")
    
    performance_ci = {}
    for name, col in strategies.items():
        perf_metrics = compute_metrics_with_ci(strategy_results[col], rf_rate=rf_aligned)
        perf_metrics['Strategy'] = name
        performance_ci[name] = perf_metrics
    
    print(f"\n{'Strategy':<22} {'Ann Return':>18} {'Sharpe':>18} {'Max DD':>18}")
    print("-"*80)
    for name, m in performance_ci.items():
        ann_ret = f"{m['Ann Return']:.1%} [{m['Ann Return CI'][0]:.1%}, {m['Ann Return CI'][1]:.1%}]"
        sharpe = f"{m['Sharpe']:.2f} [{m['Sharpe CI'][0]:.2f}, {m['Sharpe CI'][1]:.2f}]"
        max_dd = f"{m['Max DD']:.1%}"
        print(f"{name:<22} {ann_ret:>18} {sharpe:>18} {max_dd:>18}")
    
    return performance_ci