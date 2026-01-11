import pandas as pd
import numpy as np
import ta
from hrp_logger import setup_logger

logger = setup_logger()

# =============================================================================
# GPU Detection for CuPy
# =============================================================================
GPU_AVAILABLE = False
cp = None

def _detect_gpu():
    """Detect if CuPy GPU is available."""
    global GPU_AVAILABLE, cp
    try:
        import cupy as _cp
        # Test GPU availability
        _cp.array([1, 2, 3])
        cp = _cp
        GPU_AVAILABLE = True
        logger.info("✓ hrp_features: CuPy GPU available")
    except Exception as e:
        GPU_AVAILABLE = False
        cp = None
        logger.info(f"[!] hrp_features: CuPy not available ({type(e).__name__}), using CPU")

_detect_gpu()

def _zscore(series, window):
    """
    Compute rolling Z-score using strictly ex-ante data (shifted by 1).
    z_t = (x_t - mean(x_{t-window}...x_{t-1})) / std(x_{t-window}...x_{t-1})
    """
    mean = series.shift(1).rolling(window).mean()
    std = series.shift(1).rolling(window).std()
    return (series - mean) / (std + 1e-6)


def compute_hrp_momentum_features(hrp_returns: pd.Series, zscore_window: int = 24) -> pd.DataFrame:
    """
    Compute HRP strategy momentum features at multiple lookbacks.
    
    Features:
    - hrp_mom_1m_z: 1-month HRP return (Z-scored)
    - hrp_mom_3m_z: 3-month cumulative HRP return (Z-scored)
    - hrp_mom_12m_z: 12-month cumulative HRP return (Z-scored)
    
    These capture time-series momentum in the HRP strategy itself,
    which can predict future regime persistence or mean reversion.
    
    Parameters
    ----------
    hrp_returns : pd.Series
        Monthly HRP strategy returns
    zscore_window : int
        Rolling window for Z-score calculation (default 24 months, reduced from 60 
        to minimize data loss while still providing meaningful normalization)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with momentum features indexed by date
    """
    logger.info("  - Computing HRP Momentum Features (1M, 3M, 12M)")
    
    # Ensure index is datetime
    hrp_returns = hrp_returns.copy()
    hrp_returns.index = pd.to_datetime(hrp_returns.index)
    
    # 1-month momentum (just the return itself)
    mom_1m = hrp_returns.copy()
    
    # 3-month cumulative return (log returns summed, then converted back)
    log_ret = np.log1p(hrp_returns)
    mom_3m = log_ret.rolling(3, min_periods=3).sum()
    
    # 12-month cumulative return
    mom_12m = log_ret.rolling(12, min_periods=12).sum()
    
    # Z-score all momentum features (using shorter window to reduce data loss)
    # 24 months = 2 years, sufficient for normalizing momentum signals
    mom_1m_z = _zscore(mom_1m, zscore_window)
    mom_3m_z = _zscore(mom_3m, zscore_window)
    mom_12m_z = _zscore(mom_12m, zscore_window)
    
    # Create DataFrame
    features = pd.DataFrame({
        'hrp_mom_1m_z': mom_1m_z,
        'hrp_mom_3m_z': mom_3m_z,
        'hrp_mom_12m_z': mom_12m_z
    }, index=hrp_returns.index)
    
    features.index.name = 'DATE'
    
    logger.info(f"    ✓ HRP Momentum: {features.notna().sum().sum()} valid observations")
    
    return features


def compute_dispersion(df_universe):
    """
    Compute Cross-Sectional Dispersion of returns.
    """
    # Filter for in_universe_ML == 1
    if 'in_universe_ML' in df_universe.columns:
        df = df_universe[df_universe['in_universe_ML'] == 1].copy()
    else:
        df = df_universe.copy()
    
    # Dispersion: std of RET per date
    disp = df.groupby('DATE')['RET'].std().reset_index(name='dispersion')
    
    # Z-score
    disp['dispersion_z'] = _zscore(disp['dispersion'], 36)
    
    return disp[['DATE', 'dispersion_z']]

def compute_amihud(df_universe):
    """
    Compute Amihud Illiquidity Measure.
    """
    # Filter for in_universe_ML == 1
    if 'in_universe_ML' in df_universe.columns:
        df = df_universe[df_universe['in_universe_ML'] == 1].copy()
    else:
        df = df_universe.copy()
    
    # Amihud: |RET| / (PRC * VOL)
    # Ensure VOL and PRC are present.
    # df['LIQUIDITY'] was VOL * 100 * ABS_PRC. 
    # Dollar Vol = VOL * ABS_PRC (if VOL is shares). 
    # CRSP VOL is shares.
    
    df['dollar_vol'] = df['ABS_PRC'] * df['VOL']
    
    # Filter for meaningful volume
    df = df[df['dollar_vol'] > 10000]
    df = df[df['dollar_vol'] > 0]
    
    df['illiq'] = (df['RET'].abs() / df['dollar_vol']) * 1_000_000
    
    # Median across universe
    mkt = df.groupby('DATE')['illiq'].median().reset_index(name='amihud')
    
    # Log and Z-score
    mkt['amihud_log'] = np.log(mkt['amihud'].replace(0, np.nan))
    mkt['amihud_z'] = _zscore(mkt['amihud_log'], 36)
    
    return mkt[['DATE', 'amihud_z']]

def compute_valuation_spread(df_merged):
    """
    Compute Valuation Spread (Value vs Growth).
    Requires merged CRSP + Compustat data with BOOK_EQUITY.
    """
    # Check if BOOK_EQUITY exists
    if 'BOOK_EQUITY' not in df_merged.columns:
        return pd.DataFrame(columns=['DATE', 'valuation_spread_z'])
        
    # Filter for valid data
    # Use MKT_CAP from CRSP (already in df_merged)
    df = df_merged[['DATE', 'BOOK_EQUITY', 'MKT_CAP']].dropna()
    
    # Calculate Book-to-Market
    # MKT_CAP is in thousands, BOOK_EQUITY is usually in millions in Compustat
    # But let's check the user's code. 
    # User code: comp['BOOK_EQUITY'] = comp['seqq'] (millions)
    # CRSP MKT_CAP = ABS_PRC * SHROUT * 1000 (actual value)
    # Wait, CRSP SHROUT is in thousands. So PRC * SHROUT is in thousands.
    # So MKT_CAP = PRC * SHROUT * 1000 is in DOLLARS.
    # Compustat SEQQ is in MILLIONS.
    # So we need to multiply BOOK_EQUITY by 1,000,000 to match MKT_CAP.
    
    df['BM'] = (df['BOOK_EQUITY'] * 1_000_000) / df['MKT_CAP']
    
    # Filter positive BM
    df = df[df['BM'] > 0]
    
    if df.empty:
        return pd.DataFrame(columns=['DATE', 'valuation_spread_z'])
    
    def spread(g):
        if len(g) < 10: return np.nan
        try:
            g = g.copy()
            # Quintiles
            g['q'] = pd.qcut(g['BM'], 5, labels=False, duplicates='drop')
            # Log difference of median BM of top vs bottom quintile
            val = g[g['q']==4]['BM'].median()
            gro = g[g['q']==0]['BM'].median()
            return np.log(val) - np.log(gro)
        except:
            return np.nan
            
    res = df.groupby('DATE')[['BM']].apply(spread).reset_index(name='valuation_spread')
    
    # Z-Score
    res['valuation_spread_z'] = _zscore(res['valuation_spread'], 60)
    
    return res[['DATE', 'valuation_spread_z']]

def compute_macro_features(macro_df):
    """
    Compute Macroeconomic Features.
    
    NOTE: CPI, M2, and UNRATE are lagged by 1 month to account for 
    publication delay (data is released ~1 month after reference period).
    Interest rates (BAA, DGS10, TB3MS) are available in real-time.
    """
    df = macro_df.copy()
    
    # Credit Spread: BAA - 10Y Treasury (real-time, no lag needed)
    if 'BAA' in df.columns and 'DGS10' in df.columns:
        df['credit_spread'] = df['BAA'] - df['DGS10']
    else:
        df['credit_spread'] = np.nan

    # Term Spread: 10Y Treasury - 3M Treasury (real-time, no lag needed)
    if 'DGS10' in df.columns and 'TB3MS' in df.columns:
        df['term_spread'] = df['DGS10'] - df['TB3MS']
    else:
        df['term_spread'] = np.nan
        
    # CPI Volatility (Rolling 24m std of YoY CPI)
    # LAGGED BY 1 MONTH: CPI for month t is released in month t+1
    if 'CPI' in df.columns:
        cpi_lagged = df['CPI'].shift(1)  # Lag by 1 month for publication delay
        df['cpi_yoy'] = cpi_lagged.pct_change(12)
        df['cpi_vol'] = df['cpi_yoy'].rolling(24).std()
    else:
        df['cpi_vol'] = np.nan
        
    # M2 Growth (YoY)
    # LAGGED BY 1 MONTH: M2 for month t is released in month t+1
    if 'M2SL' in df.columns:
        m2_lagged = df['M2SL'].shift(1)  # Lag by 1 month for publication delay
        df['m2_growth'] = m2_lagged.pct_change(12)
    else:
        df['m2_growth'] = np.nan
        
    # Unemployment Trend (UNRATE - 12m MA)
    # LAGGED BY 1 MONTH: UNRATE for month t is released in month t+1
    if 'UNRATE' in df.columns:
        unrate_lagged = df['UNRATE'].shift(1)  # Lag by 1 month for publication delay
        df['unrate_trend'] = unrate_lagged - unrate_lagged.rolling(12).mean()
    else:
        df['unrate_trend'] = np.nan
        
    # Z-Score all features (60m window)
    cols = ['credit_spread', 'term_spread', 'cpi_vol', 'm2_growth', 'unrate_trend']
    res = pd.DataFrame(index=df.index)
    
    for col in cols:
        if col in df.columns:
            res[col] = _zscore(df[col], 60)
        else:
            res[col] = np.nan
            
    return res

def compute_vix_proxy(df_universe):
    """
    Compute VIX Proxy (Vol Regime).
    1. Calculate VW Market Return.
    2. Rolling 12m Std Dev.
    3. Rolling 60m Z-Score.
    """
    # Filter for in_universe_ML == 1
    if 'in_universe_ML' in df_universe.columns:
        df = df_universe[df_universe['in_universe_ML'] == 1].copy()
    else:
        df = df_universe.copy()
        
    # Calculate VW Return per date
    # Weighted Average: sum(RET * MKT_CAP) / sum(MKT_CAP)
    df['w_ret'] = df['RET'] * df['MKT_CAP']
    sums = df.groupby('DATE')[['w_ret', 'MKT_CAP']].sum()
    vw_ret = sums['w_ret'] / sums['MKT_CAP']
    vw_ret = vw_ret.reset_index(name='vw_ret')
    
    # Rolling 12m Vol
    vw_ret['vol_regime'] = vw_ret['vw_ret'].rolling(12).std()
    
    # Z-Score (60m)
    vw_ret['vix_proxy_z'] = _zscore(vw_ret['vol_regime'], 60)
    
    return vw_ret[['DATE', 'vix_proxy_z']]

def compute_bab_factor(df_universe, use_gpu: bool = None):
    """
    Compute Betting Against Beta (BAB) Factor with GPU acceleration.
    
    Methodology (Enhanced):
    1. Compute Market Return (VW).
    2. Compute Rolling Beta (36m, min 24) for all stocks.
    3. Clip Betas to [-5, 5] to remove outliers.
    4. Form BAB Portfolio: Long Low Beta (Bottom 20%), Short High Beta (Top 20%).
       - Uses LAGGED Beta (t-1) to sort, ensuring tradability.
    5. Compute Factor Momentum: Rolling 12-month cumulative return of the BAB factor.
    6. Return Z-scored Factor Momentum.
    
    Parameters
    ----------
    df_universe : pd.DataFrame
        Universe data with RET, MKT_CAP, PERMNO, DATE columns
    use_gpu : bool, optional
        Force GPU (True) or CPU (False). If None, auto-detect.
    """
    # Determine GPU usage
    if use_gpu is None:
        use_gpu = GPU_AVAILABLE
    
    device = "GPU" if use_gpu and GPU_AVAILABLE else "CPU"
    print(f"  - Computing BAB Factor (Vectorized) [{device}]...")
    
    if 'in_universe_ML' in df_universe.columns:
        df = df_universe[df_universe['in_universe_ML'] == 1].copy()
    else:
        df = df_universe.copy()
    
    if df.duplicated(subset=['DATE', 'PERMNO']).any():
        df = df.drop_duplicates(subset=['DATE', 'PERMNO'], keep='first')

    ret_wide = df.pivot(index='DATE', columns='PERMNO', values='RET')
    
    # Market Return
    df['mkt_val'] = df['MKT_CAP']
    df['w_ret'] = df['RET'] * df['mkt_val']
    daily_sums = df.groupby('DATE')[['w_ret', 'mkt_val']].sum()
    mkt_ret = daily_sums['w_ret'] / daily_sums['mkt_val']
    
    # =========================================================================
    # GPU-ACCELERATED ROLLING BETA CALCULATION
    # =========================================================================
    if use_gpu and GPU_AVAILABLE:
        betas = _compute_rolling_betas_gpu(ret_wide, mkt_ret, window=36, min_periods=24)
    else:
        betas = _compute_rolling_betas_cpu(ret_wide, mkt_ret, window=36, min_periods=24)
    
    # Clip Outliers
    betas = betas.clip(lower=-5, upper=5)
    
    # Shift betas forward by 2 to avoid look-ahead bias
    # Beta at time t uses returns from [t-35, t], so we need to lag by 2:
    # - At time t, we sort stocks using beta computed from [t-37, t-2]
    # - This ensures no overlap between beta estimation and portfolio return period
    betas_lag = betas.shift(2)
    
    # Rank stocks by beta at each time t
    ranks = betas_lag.rank(axis=1, pct=True)
    
    # Create masks
    low_beta_mask = ranks <= 0.2
    high_beta_mask = ranks >= 0.8
    
    # Calculate Portfolio Returns
    low_beta_ret = ret_wide[low_beta_mask].mean(axis=1)
    high_beta_ret = ret_wide[high_beta_mask].mean(axis=1)
    
    # BAB Factor Return (Long Low, Short High)
    bab_factor = low_beta_ret - high_beta_ret
    
    # Factor Momentum: 12-month cumulative return (Log sum)
    bab_log_ret = np.log1p(bab_factor.fillna(0))
    bab_12m_cum = bab_log_ret.rolling(12).sum()
    
    # Z-Score (60m)
    bab_z = _zscore(bab_12m_cum, 60)
    
    return pd.DataFrame({'DATE': bab_z.index, 'bab_z': bab_z})


def _compute_rolling_betas_cpu(ret_wide: pd.DataFrame, mkt_ret: pd.Series, 
                                window: int = 36, min_periods: int = 24) -> pd.DataFrame:
    """
    Compute rolling betas using vectorized Pandas (CPU).
    
    Beta = Cov(stock, market) / Var(market)
    Using: Cov(X,Y) = E[XY] - E[X]E[Y]
    """
    # Rolling Means
    mean_mkt = mkt_ret.rolling(window, min_periods=min_periods).mean()
    mean_stocks = ret_wide.rolling(window, min_periods=min_periods).mean()

    # Rolling Covariance = E[XY] - E[X]E[Y]
    ret_times_mkt = ret_wide.multiply(mkt_ret, axis=0)
    mean_product = ret_times_mkt.rolling(window, min_periods=min_periods).mean()
    cov_matrix = mean_product.sub(mean_stocks.multiply(mean_mkt, axis=0))

    # Rolling Variance of Market
    mkt_sq = mkt_ret ** 2
    mean_mkt_sq = mkt_sq.rolling(window, min_periods=min_periods).mean()
    var_mkt = mean_mkt_sq - (mean_mkt ** 2)

    # Beta = Cov / Var
    betas = cov_matrix.div(var_mkt, axis=0)
    
    return betas


def _compute_rolling_betas_gpu(ret_wide: pd.DataFrame, mkt_ret: pd.Series,
                                window: int = 36, min_periods: int = 24) -> pd.DataFrame:
    """
    Compute rolling betas using CuPy GPU acceleration.
    
    GPU parallelizes the rolling window computation across all stocks simultaneously.
    """
    import cupy as cp
    
    # Convert to numpy then CuPy arrays
    ret_arr = cp.asarray(ret_wide.values, dtype=cp.float32)  # (T, N) stocks
    mkt_arr = cp.asarray(mkt_ret.values, dtype=cp.float32)   # (T,)
    
    T, N = ret_arr.shape
    betas_gpu = cp.full((T, N), cp.nan, dtype=cp.float32)
    
    # Expand market returns for broadcasting: (T,) -> (T, 1)
    mkt_expanded = mkt_arr[:, cp.newaxis]
    
    # Compute rolling betas for each starting position
    for t in range(window - 1, T):
        start = t - window + 1
        
        # Get windows: (window, N) and (window,)
        ret_window = ret_arr[start:t+1, :]  # (window, N)
        mkt_window = mkt_arr[start:t+1]      # (window,)
        
        # Check minimum valid observations per stock
        valid_mask = ~cp.isnan(ret_window)
        n_valid = valid_mask.sum(axis=0)  # (N,)
        
        # Replace NaN with 0 for computation (will mask later)
        ret_clean = cp.where(cp.isnan(ret_window), 0, ret_window)
        mkt_clean = cp.where(cp.isnan(mkt_window), 0, mkt_window)
        
        # Means (using actual counts)
        mkt_mean = mkt_clean.sum() / window
        ret_mean = ret_clean.sum(axis=0) / cp.maximum(n_valid, 1)  # (N,)
        
        # Covariance: E[XY] - E[X]E[Y]
        xy = ret_clean * mkt_clean[:, cp.newaxis]  # (window, N)
        mean_xy = xy.sum(axis=0) / cp.maximum(n_valid, 1)
        cov = mean_xy - ret_mean * mkt_mean
        
        # Market variance
        var_mkt = (mkt_clean ** 2).sum() / window - mkt_mean ** 2
        
        # Beta
        beta_t = cov / (var_mkt + 1e-10)
        
        # Mask stocks with insufficient observations
        beta_t = cp.where(n_valid >= min_periods, beta_t, cp.nan)
        
        betas_gpu[t, :] = beta_t
    
    # Transfer back to CPU and create DataFrame
    betas_np = cp.asnumpy(betas_gpu)
    betas_df = pd.DataFrame(betas_np, index=ret_wide.index, columns=ret_wide.columns)
    
    return betas_df

def compute_rolling_mean_pairwise_corr(df_universe, window=12):
    """
    Compute Rolling Mean of Pairwise Correlations using Variance Ratio Proxy.
    Proxy = Var(Market) / Mean(Var(Stocks))
    This is a vectorized approximation that avoids the O(N^2) correlation matrix loop.
    """
    logger.info(f"  - Rolling Mean Pairwise Correlation (Window={window}) [Vectorized Proxy]")
    # Filter for in_universe_ML == 1
    if 'in_universe_ML' in df_universe.columns:
        df = df_universe[df_universe['in_universe_ML'] == 1].copy()
    else:
        df = df_universe.copy()
        
    # Pivot to wide format (Dates x Stocks)
    ret_wide = df.pivot(index='DATE', columns='PERMNO', values='RET')
    cap_wide = df.pivot(index='DATE', columns='PERMNO', values='MKT_CAP')
    
    # 1. Compute Market Return (Value Weighted)
    # Using VW portfolio of the universe as the "Market"
    # Fill NaNs in weights with 0 for the dot product, but be careful with division
    w_ret = ret_wide * cap_wide
    mkt_ret = w_ret.sum(axis=1) / cap_wide.sum(axis=1)
    
    # 2. Rolling Variance of Market
    mkt_var = mkt_ret.rolling(window=window).var()
    
    # 3. Rolling Variance of Individual Stocks
    # Vectorized rolling variance for all stocks
    stock_vars = ret_wide.rolling(window=window).var()
    
    # 4. Mean of Stock Variances (Cross-sectional mean of rolling variances)
    avg_stock_var = stock_vars.mean(axis=1)
    
    # 5. Variance Ratio Proxy
    # This ratio tracks the average correlation level.
    # When correlation is 1, Ratio = 1. When correlation is 0, Ratio = 1/N.
    avg_corr_proxy = mkt_var / avg_stock_var
    
    res = pd.DataFrame({'DATE': ret_wide.index, 'avg_pairwise_corr': avg_corr_proxy})
    
    # Z-Score
    res['avg_pairwise_corr_z'] = _zscore(res['avg_pairwise_corr'], 60)
    
    return res[['DATE', 'avg_pairwise_corr_z']]

def compute_all_features(df_universe, macro_df, df_merged_comp=None):
    """
    Compute all advanced features for the entire history.
    """
    logger.info("Computing advanced features...")
    
    # Global Duplicate Check
    if df_universe.duplicated(subset=['DATE', 'PERMNO']).any():
        logger.warning("  [!] Global: Found duplicates in universe. Dropping duplicates (keeping first)...")
        df_universe = df_universe.drop_duplicates(subset=['DATE', 'PERMNO'], keep='first')
    
    # 1. Dispersion
    logger.info("  - Dispersion (std of returns)")
    disp = compute_dispersion(df_universe)
    
    # 2. Amihud
    logger.info("  - Amihud Illiquidity")
    amihud = compute_amihud(df_universe)
    
    # 3. VIX Proxy
    logger.info("  - VIX Proxy (Vol Regime)")
    vix = compute_vix_proxy(df_universe)
    
    # 4. BAB Factor
    bab = compute_bab_factor(df_universe)
    
    # 5. Rolling Mean Pairwise Correlation
    avg_corr = compute_rolling_mean_pairwise_corr(df_universe)

    # 6. Macro
    logger.info("  - Macro Features (Credit Spread, CPI Vol, M2, Unemployment)")
    macro = compute_macro_features(macro_df)

    # 7. Valuation Spread (if Compustat data provided)
    val_spread = pd.DataFrame(columns=['DATE', 'valuation_spread_z'])
    if df_merged_comp is not None:
        logger.info("  - Valuation Spread (Value vs Growth)")
        val_spread = compute_valuation_spread(df_merged_comp)

    # Merge all
    # Create master index
    all_dates = macro.index.union(disp['DATE']).union(amihud['DATE']).union(vix['DATE']).union(bab['DATE']).union(avg_corr['DATE']).unique().sort_values()
    if not val_spread.empty:
        all_dates = all_dates.union(val_spread['DATE']).unique().sort_values()
        
    features = pd.DataFrame(index=all_dates)
    features.index.name = 'DATE'
    
    # Merge Dispersion
    disp.set_index('DATE', inplace=True)
    features = features.join(disp, how='left')
    
    # Merge Amihud
    amihud.set_index('DATE', inplace=True)
    features = features.join(amihud, how='left')
    
    # Merge VIX Proxy
    vix.set_index('DATE', inplace=True)
    features = features.join(vix, how='left')
    
    # Merge BAB
    bab.set_index('DATE', inplace=True)
    features = features.join(bab, how='left')
    
    # Merge Avg Corr
    avg_corr.set_index('DATE', inplace=True)
    features = features.join(avg_corr, how='left')

    # Merge Macro
    features = features.join(macro, how='left')

    # Merge Valuation Spread
    if not val_spread.empty:
        val_spread.set_index('DATE', inplace=True)
        features = features.join(val_spread, how='left')
    
    # Forward fill to handle slight date mismatches
    features = features.ffill()
    
    return features