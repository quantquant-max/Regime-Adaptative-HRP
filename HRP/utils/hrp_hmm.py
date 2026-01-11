"""
HRP HMM Module - Hidden Markov Model for Bear/Bull Regime Detection
===================================================================
Goal: Identify if the market is in a Bear or Bull/Consolidation regime.

Regime Definitions:
- Bear (0): Negative expected returns, elevated volatility, correlation convergence
- Bull (1): Positive expected returns, normal volatility, diversification works

Provides functions for:
- Rolling downside deviation calculation
- HMM data preparation (log returns + downside deviation)
- Expanding-window HMM fitting with TRUE FILTERED probabilities (forward pass only)
- Regime labeling and transition matrix computation

IMPORTANT: Uses custom forward-only filter to avoid look-ahead bias.
           hmmlearn's predict_proba() uses forward-backward (smoothed posteriors),
           which leaks future information. We implement P(q_t | O_1:t) directly.

Note: The HMM labels are used as ground truth for XGBoost prediction.
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from tqdm import tqdm
from hrp_logger import setup_logger

# Import centralized seed management
try:
    from hrp_setup import get_random_state
except ImportError:
    def get_random_state():
        return 42

logger = setup_logger()

# =============================================================================
# GPU Detection for CuPy
# =============================================================================
GPU_AVAILABLE = False
cp = None

def _detect_gpu():
    """Detect if CuPy is available for GPU acceleration."""
    global GPU_AVAILABLE, cp
    try:
        import cupy as _cp
        _cp.array([1, 2, 3])  # Test GPU
        cp = _cp
        GPU_AVAILABLE = True
        logger.info("✓ hrp_hmm: CuPy GPU available")
    except Exception:
        GPU_AVAILABLE = False
        logger.info("✗ hrp_hmm: CuPy not available, using CPU")

_detect_gpu()


def _compute_log_likelihood(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """
    Compute log emission probabilities for each observation under each state.
    
    Parameters
    ----------
    model : GaussianHMM
        Fitted HMM model
    X : np.ndarray, shape (n_samples, n_features)
        Observation sequence
        
    Returns
    -------
    np.ndarray, shape (n_samples, n_components)
        Log P(O_t | q_t = i) for each t and state i
    """
    from scipy.stats import multivariate_normal
    
    n_samples = X.shape[0]
    n_components = model.n_components
    log_prob = np.zeros((n_samples, n_components))
    
    for i in range(n_components):
        mean = model.means_[i]
        
        # Handle different covariance types
        if model.covariance_type == 'full':
            cov = model.covars_[i]
        elif model.covariance_type == 'diag':
            cov = np.diag(model.covars_[i])
        elif model.covariance_type == 'spherical':
            cov = np.eye(X.shape[1]) * model.covars_[i]
        elif model.covariance_type == 'tied':
            cov = model.covars_
        else:
            raise ValueError(f"Unknown covariance type: {model.covariance_type}")
        
        try:
            rv = multivariate_normal(mean=mean, cov=cov, allow_singular=True)
            log_prob[:, i] = rv.logpdf(X)
        except Exception:
            # Fallback: use very small probability
            log_prob[:, i] = -1e10
    
    return log_prob


def _forward_filter(model: GaussianHMM, X: np.ndarray, use_gpu: bool = None) -> np.ndarray:
    """
    Compute FILTERED probabilities using forward algorithm only.
    
    This computes P(q_t = i | O_1, O_2, ..., O_t) - the probability of being
    in state i at time t given ONLY observations up to and including time t.
    
    NO LOOK-AHEAD BIAS: Does not use backward pass or future observations.
    
    Algorithm:
    1. α_t(i) = P(O_1:t, q_t = i)  [forward variable, unnormalized]
    2. P(q_t = i | O_1:t) = α_t(i) / Σ_j α_t(j)  [filtered probability]
    
    Parameters
    ----------
    model : GaussianHMM
        Fitted HMM model with startprob_, transmat_, means_, covars_
    X : np.ndarray, shape (n_samples, n_features)
        Observation sequence
    use_gpu : bool, optional
        If True, use GPU. If False, use CPU. If None, auto-detect.
        
    Returns
    -------
    np.ndarray, shape (n_samples, n_components)
        Filtered probabilities P(q_t | O_1:t) for each timestep
    """
    if use_gpu is None:
        use_gpu = GPU_AVAILABLE
    
    if use_gpu and GPU_AVAILABLE:
        return _forward_filter_gpu(model, X)
    else:
        return _forward_filter_cpu(model, X)


def _forward_filter_cpu(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """CPU implementation of forward filter."""
    n_samples = X.shape[0]
    n_components = model.n_components
    
    # Get log emission probabilities: log P(O_t | q_t = i)
    log_frameprob = _compute_log_likelihood(model, X)
    
    # Get log transition and start probabilities
    log_startprob = np.log(model.startprob_ + 1e-10)
    log_transmat = np.log(model.transmat_ + 1e-10)
    
    # Forward pass in log space for numerical stability
    # log_alpha[t, i] = log P(O_1:t, q_t = i)
    log_alpha = np.zeros((n_samples, n_components))
    
    # Initialize: log α_0(i) = log π_i + log P(O_0 | q_0 = i)
    log_alpha[0] = log_startprob + log_frameprob[0]
    
    # Forward recursion: α_t(j) = P(O_t | q_t=j) * Σ_i [α_{t-1}(i) * P(q_t=j | q_{t-1}=i)]
    for t in range(1, n_samples):
        for j in range(n_components):
            # log Σ_i [α_{t-1}(i) * A_{ij}] = logsumexp over i of [log_alpha[t-1, i] + log_transmat[i, j]]
            log_alpha[t, j] = logsumexp(log_alpha[t-1] + log_transmat[:, j]) + log_frameprob[t, j]
    
    # Normalize to get filtered probabilities: P(q_t | O_1:t) = α_t / Σ_j α_t(j)
    # In log space: log P(q_t=i | O_1:t) = log_alpha[t, i] - logsumexp(log_alpha[t, :])
    log_filtered = log_alpha - logsumexp(log_alpha, axis=1, keepdims=True)
    
    # Convert back to probability space
    filtered_probs = np.exp(log_filtered)
    
    # Ensure proper normalization (handle numerical issues)
    filtered_probs = filtered_probs / filtered_probs.sum(axis=1, keepdims=True)
    
    return filtered_probs


def _forward_filter_gpu(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """
    GPU-accelerated forward filter using CuPy.
    
    Vectorizes the inner loop over states using GPU matrix operations.
    Note: The outer loop over time is sequential (inherent HMM dependency).
    """
    n_samples = X.shape[0]
    n_components = model.n_components
    
    # Get log emission probabilities (computed on CPU via scipy)
    log_frameprob_np = _compute_log_likelihood(model, X)
    
    # Transfer to GPU
    log_frameprob = cp.asarray(log_frameprob_np)
    log_startprob = cp.log(cp.asarray(model.startprob_) + 1e-10)
    log_transmat = cp.log(cp.asarray(model.transmat_) + 1e-10)
    
    # Forward pass in log space
    log_alpha = cp.zeros((n_samples, n_components), dtype=cp.float64)
    
    # Initialize
    log_alpha[0] = log_startprob + log_frameprob[0]
    
    # Forward recursion - vectorized over states
    # α_t(j) = P(O_t | q_t=j) * Σ_i [α_{t-1}(i) * A_{ij}]
    # In log space: log_alpha[t, j] = log_frameprob[t, j] + logsumexp_i(log_alpha[t-1, i] + log_transmat[i, j])
    for t in range(1, n_samples):
        # Vectorized logsumexp: for each j, compute logsumexp over i
        # log_alpha[t-1, :, None] + log_transmat has shape (n_components, n_components)
        # where [i, j] = log_alpha[t-1, i] + log_transmat[i, j]
        log_transition = log_alpha[t-1, :, cp.newaxis] + log_transmat  # (n_components, n_components)
        
        # logsumexp over axis=0 (sum over i) -> shape (n_components,)
        max_log = cp.max(log_transition, axis=0)
        log_sum = max_log + cp.log(cp.sum(cp.exp(log_transition - max_log), axis=0))
        
        log_alpha[t] = log_sum + log_frameprob[t]
    
    # Normalize to get filtered probabilities
    max_alpha = cp.max(log_alpha, axis=1, keepdims=True)
    log_sum_alpha = max_alpha + cp.log(cp.sum(cp.exp(log_alpha - max_alpha), axis=1, keepdims=True))
    log_filtered = log_alpha - log_sum_alpha
    
    # Convert back to probability space
    filtered_probs = cp.exp(log_filtered)
    filtered_probs = filtered_probs / cp.sum(filtered_probs, axis=1, keepdims=True)
    
    # Transfer back to CPU
    return cp.asnumpy(filtered_probs)


def rolling_downside_deviation(returns: pd.Series, window: int = 12, 
                                threshold: float = 0, min_obs: int = 2) -> pd.Series:
    """
    Calculate rolling downside deviation.
    
    Only considers returns below the threshold (default=0).
    If fewer than min_obs negative returns in window, uses overall std as fallback.
    
    Parameters
    ----------
    returns : pd.Series
        Return series (log returns or simple returns)
    window : int
        Rolling window size in periods
    threshold : float
        Threshold below which returns are considered "downside"
    min_obs : int
        Minimum number of observations below threshold required
        
    Returns
    -------
    pd.Series
        Rolling downside deviation
    """
    def downside_std(x):
        below_threshold = x[x < threshold]
        if len(below_threshold) < min_obs:
            # Fallback: use full std * 0.5 as proxy (less volatile periods)
            return x.std() * 0.5 if len(x) >= min_obs else np.nan
        return below_threshold.std()
    
    return returns.rolling(window=window, min_periods=window).apply(downside_std, raw=False)


def load_umich_sentiment(fred_path: str) -> pd.Series:
    """
    Load Michigan Consumer Sentiment Index from tbmics.csv.
    
    The file format is: Month,YYYY,ICS_ALL (e.g., "February,1953,90.7")
    
    Parameters
    ----------
    fred_path : str
        Path to FRED data directory containing tbmics.csv
        
    Returns
    -------
    pd.Series
        Monthly sentiment index, datetime-indexed
    """
    import os
    import calendar
    
    filepath = os.path.join(fred_path, 'tbmics.csv')
    if not os.path.exists(filepath):
        logger.warning(f"UMICH sentiment file not found: {filepath}")
        return None
    
    df = pd.read_csv(filepath)
    
    # Parse month names to month numbers
    month_map = {name: num for num, name in enumerate(calendar.month_name) if num}
    df['month_num'] = df['Month'].map(month_map)
    
    # Create datetime index (end of month)
    df['date'] = pd.to_datetime(df['YYYY'].astype(str) + '-' + df['month_num'].astype(str) + '-01')
    df['date'] = df['date'] + pd.offsets.MonthEnd(0)  # Snap to month-end
    
    df = df.set_index('date').sort_index()
    sentiment = df['ICS_ALL'].astype(float)
    
    logger.info(f"Loaded UMICH sentiment: {len(sentiment)} observations ({sentiment.index.min().date()} to {sentiment.index.max().date()})")
    
    return sentiment


def compute_sentiment_features(sentiment: pd.Series) -> pd.DataFrame:
    """
    Compute sentiment features for HMM input.
    
    Features:
    - sent_change: Year-over-Year % change = (S_t / S_{t-12}) - 1
    - sent_gap: Deviation from 12-month MA = S_t - MA_12(S)
    
    Both features are LAGGED BY 1 MONTH to account for publication delay.
    
    Parameters
    ----------
    sentiment : pd.Series
        Raw sentiment index (monthly)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with sent_change and sent_gap columns
    """
    df = pd.DataFrame(index=sentiment.index)
    
    # YoY change: (S_t / S_{t-12}) - 1
    df['sent_change'] = sentiment.pct_change(12)
    
    # Gap from trend: S_t - MA_12(S)
    df['sent_gap'] = sentiment - sentiment.rolling(12).mean()
    
    # LAG BY 1 MONTH for publication delay (data at t published after t)
    df['sent_change'] = df['sent_change'].shift(1)
    df['sent_gap'] = df['sent_gap'].shift(1)
    
    logger.info(f"Computed sentiment features: sent_change, sent_gap (lagged 1M for publication)")
    
    return df


def _expanding_zscore(series: pd.Series, min_periods: int = 24) -> pd.Series:
    """
    Compute expanding-window Z-score to avoid look-ahead bias.
    
    z_t = (x_t - mean(x_1:t-1)) / std(x_1:t-1)
    
    Uses only PAST data to normalize each observation.
    
    Parameters
    ----------
    series : pd.Series
        Input series
    min_periods : int
        Minimum periods before computing z-score (default=24 months)
        
    Returns
    -------
    pd.Series
        Z-scored series with no look-ahead bias
    """
    # Expanding mean and std using only past data (shift by 1)
    expanding_mean = series.shift(1).expanding(min_periods=min_periods).mean()
    expanding_std = series.shift(1).expanding(min_periods=min_periods).std()
    
    # Z-score: (current value - past mean) / past std
    z_score = (series - expanding_mean) / (expanding_std + 1e-8)
    
    return z_score


def prepare_hmm_data(market_returns: pd.Series, 
                     downside_window: int = 12,
                     min_periods_zscore: int = 24,
                     sentiment: pd.Series = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare data for HMM regime detection.
    
    Uses EXPANDING Z-SCORE to avoid look-ahead bias (no future data leakage).
    
    HMM Input Features:
    1. log_return_z: Log returns (expanding z-scored)
    2. downside_dev_z: Downside deviation (expanding z-scored)
    3. sent_change_z: UMICH sentiment YoY change (expanding z-scored, lagged 1M)
    
    Parameters
    ----------
    market_returns : pd.Series
        Market returns series (e.g., CRSP VW returns)
    downside_window : int
        Window for rolling downside deviation calculation
    min_periods_zscore : int
        Minimum periods before computing z-score (default=24 months)
    sentiment : pd.Series, optional
        Michigan Consumer Sentiment Index (raw levels)
        If provided, adds sent_change_z to HMM features
        
    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        df_hmm_raw: Raw HMM input (log_return, downside_dev, sent_change)
        df_hmm: Standardized HMM input (all features z-scored) - NO LOOK-AHEAD
    """
    logger.info(f"Preparing HMM data from {len(market_returns)} observations")
    logger.info(f"Using EXPANDING Z-SCORE (no look-ahead bias)")
    
    # Calculate log returns
    log_returns = np.log1p(market_returns)
    
    # Calculate downside deviation
    downside_dev = rolling_downside_deviation(log_returns, window=downside_window)
    
    # Construct raw DataFrame
    df_hmm_raw = pd.DataFrame({
        'log_return': log_returns,
        'downside_dev': downside_dev
    })
    
    # Add sentiment features if available
    if sentiment is not None:
        logger.info("Adding UMICH Consumer Sentiment to HMM features")
        sent_features = compute_sentiment_features(sentiment)
        
        # Align sentiment to market returns index
        df_hmm_raw = df_hmm_raw.join(sent_features['sent_change'], how='left')
        
        # Forward-fill sentiment for any missing months (survey not always monthly in early years)
        df_hmm_raw['sent_change'] = df_hmm_raw['sent_change'].ffill()
        logger.info(f"Sentiment coverage: {df_hmm_raw['sent_change'].notna().sum()} months")
    
    # Drop rows with any NaN
    df_hmm_raw = df_hmm_raw.dropna()
    
    logger.info(f"HMM input data: {len(df_hmm_raw)} observations after dropping NaN")
    logger.info(f"Date range: {df_hmm_raw.index.min().date()} to {df_hmm_raw.index.max().date()}")
    
    # Standardize features using EXPANDING Z-SCORE (no look-ahead bias)
    # Each observation is standardized using only past data
    df_hmm = pd.DataFrame(index=df_hmm_raw.index)
    df_hmm['log_return_z'] = _expanding_zscore(df_hmm_raw['log_return'], min_periods=min_periods_zscore)
    df_hmm['downside_dev_z'] = _expanding_zscore(df_hmm_raw['downside_dev'], min_periods=min_periods_zscore)
    
    # Add sentiment z-score if available
    if sentiment is not None and 'sent_change' in df_hmm_raw.columns:
        df_hmm['sent_change_z'] = _expanding_zscore(df_hmm_raw['sent_change'], min_periods=min_periods_zscore)
        logger.info("Added sent_change_z to HMM input (3 features total)")
    
    # Drop NaN from burn-in period
    df_hmm = df_hmm.dropna()
    df_hmm_raw = df_hmm_raw.loc[df_hmm.index]
    
    logger.info(f"After expanding z-score burn-in: {len(df_hmm)} observations")
    logger.info(f"HMM features: {list(df_hmm.columns)}")
    
    return df_hmm_raw, df_hmm


def check_hmm_data_quality(df_hmm: pd.DataFrame, verbose: bool = True) -> dict:
    """
    Check for missing values in HMM input features.
    
    Parameters
    ----------
    df_hmm : pd.DataFrame
        Standardized HMM input features (log_return_z, downside_dev_z, sent_change_z)
    verbose : bool
        If True, print results to console
        
    Returns
    -------
    dict
        Dictionary with check results:
        - 'all_valid': bool - True if no missing values in any feature
        - 'feature_status': dict - {feature_name: {'valid': bool, 'n_missing': int, 'missing_dates': list}}
    """
    hmm_feature_cols = ['log_return_z', 'downside_dev_z', 'sent_change_z']
    
    results = {
        'all_valid': True,
        'feature_status': {}
    }
    
    if verbose:
        print("-"*70)
        print("DATA QUALITY CHECK: HMM Input Features")
        print("-"*70)
    
    for col in hmm_feature_cols:
        if col in df_hmm.columns:
            missing_mask = df_hmm[col].isna()
            n_missing = missing_mask.sum()
            
            if n_missing == 0:
                results['feature_status'][col] = {
                    'valid': True,
                    'n_missing': 0,
                    'missing_dates': []
                }
                if verbose:
                    print(f"  ✓ {col}: No missing values ({len(df_hmm)} obs)")
            else:
                missing_dates = df_hmm.index[missing_mask].strftime('%Y-%m-%d').tolist()
                results['feature_status'][col] = {
                    'valid': False,
                    'n_missing': n_missing,
                    'missing_dates': missing_dates
                }
                results['all_valid'] = False
                if verbose:
                    print(f"  ✗ {col}: {n_missing} missing values")
                    print(f"    Dates: {missing_dates[:10]}{'...' if n_missing > 10 else ''}")
        else:
            results['feature_status'][col] = {
                'valid': False,
                'n_missing': -1,
                'missing_dates': []
            }
            results['all_valid'] = False
            if verbose:
                print(f"  ✗ {col}: Column not found in df_hmm!")
    
    return results


def fit_expanding_hmm(df_hmm: pd.DataFrame, 
                      df_hmm_raw: pd.DataFrame,
                      min_train: int = 60,
                      refit_freq: int = 12,
                      n_components: int = 2,
                      random_state: int = None) -> pd.DataFrame:
    """
    Fit HMM using expanding window with TRUE FILTERED probabilities (no look-ahead).
    
    CRITICAL: Uses custom forward-only filter instead of hmmlearn's predict_proba().
    
    hmmlearn's predict_proba() computes SMOOTHED posteriors P(q_t | O_1:T) using
    forward-backward algorithm, which uses future observations (look-ahead bias).
    
    We compute FILTERED posteriors P(q_t | O_1:t) using forward algorithm only,
    ensuring no information from future observations leaks into regime labels.
    
    Parameters
    ----------
    df_hmm : pd.DataFrame
        Standardized HMM input features
    df_hmm_raw : pd.DataFrame
        Raw HMM input (for state sorting based on returns)
    min_train : int
        Minimum training periods before first HMM fit
    refit_freq : int
        Frequency of HMM refitting (in periods)
    n_components : int
        Number of HMM states (default 2 for Bear/Bull)
    random_state : int, optional
        Random seed for reproducibility. If None, uses centralized seed from config.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with regime labels and filtered probabilities
    """
    # Use centralized seed if not provided
    if random_state is None:
        random_state = get_random_state()
    
    logger.info("="*70)
    logger.info("EXPANDING-WINDOW HMM FITTING (TRUE FILTERED PROBABILITIES)")
    logger.info("="*70)
    logger.info("Using FORWARD-ONLY filter: P(q_t | O_1:t) - NO look-ahead bias")
    logger.info(f"Min training: {min_train} months, Refit frequency: {refit_freq} months")
    
    X_hmm = df_hmm.values
    dates_hmm = df_hmm.index.tolist()
    n_obs = len(X_hmm)
    
    # Storage for results
    filtered_probs = np.full((n_obs, 2), np.nan)
    filtered_states = np.full(n_obs, -1, dtype=int)
    
    # Track model state
    last_model = None
    last_fit_idx = -refit_freq
    state_swap_needed = False
    
    for t in tqdm(range(min_train, n_obs), desc="Expanding HMM (Forward Filter)", leave=False):
        # Refit HMM periodically
        if t - last_fit_idx >= refit_freq or last_model is None:
            X_train = X_hmm[:t]
            
            hmm_model = GaussianHMM(
                n_components=n_components,
                covariance_type='full',
                n_iter=1000,
                random_state=random_state,
                tol=1e-4
            )
            
            try:
                hmm_model.fit(X_train)
                last_model = hmm_model
                last_fit_idx = t
                
                # Determine state mapping based on training returns
                train_states = hmm_model.predict(X_train)
                train_returns = df_hmm_raw['log_return'].iloc[:t].values
                
                state_0_return = train_returns[train_states == 0].mean() if (train_states == 0).sum() > 0 else 0
                state_1_return = train_returns[train_states == 1].mean() if (train_states == 1).sum() > 0 else 0
                
                # If state 0 has higher return, swap (we want 0=Bear, 1=Bull)
                state_swap_needed = state_0_return > state_1_return
                
            except Exception as e:
                logger.warning(f"HMM fitting failed at t={t}: {e}")
        
        if last_model is None:
            continue
        
        # =========================================================================
        # TRUE FILTERED PROBABILITY: P(q_t | O_1:t) using forward algorithm ONLY
        # =========================================================================
        # This is the key fix: we use _forward_filter() instead of predict_proba()
        # predict_proba() uses forward-backward which includes future information
        X_seq = X_hmm[:t+1]
        
        try:
            # Compute filtered probabilities using forward pass only
            all_filtered_probs = _forward_filter(last_model, X_seq)
            filtered_prob_t = all_filtered_probs[-1]  # P(q_t | O_1:t)
            
            if state_swap_needed:
                filtered_prob_t = filtered_prob_t[::-1]
            
            filtered_probs[t] = filtered_prob_t
            filtered_states[t] = 0 if filtered_prob_t[0] >= 0.5 else 1
            
        except Exception as e:
            # Fallback: assume Bull regime
            filtered_probs[t] = [0.3, 0.7]
            filtered_states[t] = 1
    
    # Create regime DataFrame
    df_regimes = pd.DataFrame(index=df_hmm.index)
    df_regimes['regime'] = filtered_states
    df_regimes['prob_bear'] = filtered_probs[:, 0]
    df_regimes['prob_bull'] = filtered_probs[:, 1]
    df_regimes['log_return'] = df_hmm_raw['log_return']
    df_regimes['downside_dev'] = df_hmm_raw['downside_dev']
    
    # Drop burn-in period
    df_regimes = df_regimes[df_regimes['regime'] >= 0].copy()
    df_regimes['regime_label'] = df_regimes['regime'].map({0: 'Bear', 1: 'Bull'})
    
    # Log regime statistics
    n_bear = (df_regimes['regime'] == 0).sum()
    n_bull = (df_regimes['regime'] == 1).sum()
    logger.info(f"Regime labels: {len(df_regimes)} months ({n_bear} Bear, {n_bull} Bull)")
    
    return df_regimes


def compute_transition_matrix(regimes: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Compute empirical transition matrix from regime sequence.
    
    Parameters
    ----------
    regimes : np.ndarray
        Array of regime labels (0 or 1)
        
    Returns
    -------
    tuple[np.ndarray, pd.DataFrame]
        Raw transition matrix and formatted DataFrame
    """
    trans_counts = np.zeros((2, 2))
    for i in range(len(regimes) - 1):
        from_state = regimes[i]
        to_state = regimes[i + 1]
        trans_counts[from_state, to_state] += 1
    
    trans_matrix = trans_counts / trans_counts.sum(axis=1, keepdims=True)
    
    trans_df = pd.DataFrame(
        trans_matrix,
        index=['From Bear', 'From Bull'],
        columns=['To Bear', 'To Bull']
    )
    
    return trans_matrix, trans_df


def get_regime_statistics(df_regimes: pd.DataFrame) -> dict:
    """
    Compute statistics for each regime.
    
    Parameters
    ----------
    df_regimes : pd.DataFrame
        Regime DataFrame from fit_expanding_hmm
        
    Returns
    -------
    dict
        Statistics for Bear and Bull regimes
    """
    stats = {}
    
    # Check required columns exist
    required_cols = ['regime', 'log_return', 'downside_dev']
    missing_cols = [c for c in required_cols if c not in df_regimes.columns]
    if missing_cols:
        logger.warning(f"Missing columns in df_regimes: {missing_cols}")
        # Return empty stats with default structure
        for name in ['Bear', 'Bull']:
            stats[name] = {
                'months': 0, 'pct': 0.0, 'mean_log_return': np.nan,
                'ann_return': np.nan, 'mean_downside_dev': np.nan,
                'ann_downside_dev': np.nan, 'expected_duration': np.nan
            }
        stats['transition_matrix'] = pd.DataFrame(
            np.nan, index=['From Bear', 'From Bull'], columns=['To Bear', 'To Bull']
        )
        return stats
    
    for regime_val, regime_name in [(0, 'Bear'), (1, 'Bull')]:
        mask = df_regimes['regime'] == regime_val
        regime_data = df_regimes[mask]
        
        n_regime = len(regime_data)
        n_total = len(df_regimes)
        
        stats[regime_name] = {
            'months': n_regime,
            'pct': n_regime / n_total if n_total > 0 else 0.0,
            'mean_log_return': regime_data['log_return'].mean() if n_regime > 0 else np.nan,
            'ann_return': np.exp(regime_data['log_return'].mean() * 12) - 1 if n_regime > 0 else np.nan,
            'mean_downside_dev': regime_data['downside_dev'].mean() if n_regime > 0 else np.nan,
            'ann_downside_dev': regime_data['downside_dev'].mean() * np.sqrt(12) if n_regime > 0 else np.nan
        }
    
    # Add transition matrix
    trans_matrix, trans_df = compute_transition_matrix(df_regimes['regime'].values)
    stats['transition_matrix'] = trans_df
    
    # Expected durations (with safety check)
    for i, name in enumerate(['Bear', 'Bull']):
        if trans_matrix[i, i] < 1 and not np.isnan(trans_matrix[i, i]):
            expected_duration = 1 / (1 - trans_matrix[i, i])
        else:
            expected_duration = np.inf
        stats[name]['expected_duration'] = expected_duration
    
    return stats
