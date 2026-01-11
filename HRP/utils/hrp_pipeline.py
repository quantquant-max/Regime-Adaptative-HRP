import pandas as pd
import numpy as np
import os
import joblib
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import internal modules
import hrp_functions
import hrp_analytics
import hrp_data
from hrp_logger import setup_logger

logger = setup_logger()


def _safe_save(save_func, filepath, max_retries=3, retry_delay=1.0):
    """
    Safely save a file with retry logic for OneDrive sync issues.
    
    Args:
        save_func: Callable that performs the save (e.g., lambda: joblib.dump(data, path))
        filepath: Path being saved (for logging)
        max_retries: Number of retry attempts
        retry_delay: Seconds to wait between retries
    """
    for attempt in range(max_retries):
        try:
            save_func()
            return True
        except OSError as e:
            if attempt < max_retries - 1:
                logger.warning(f"  Save failed (attempt {attempt+1}/{max_retries}): {filepath}")
                time.sleep(retry_delay)
            else:
                logger.error(f"  Failed to save after {max_retries} attempts: {filepath}")
                logger.error(f"  Error: {e}")
                # Try alternative: save to temp then rename
                try:
                    temp_path = filepath + '.tmp'
                    save_func_temp = lambda: save_func.__self__ if hasattr(save_func, '__self__') else save_func()
                    # For joblib.dump, we need to handle differently
                    logger.info(f"  Skipping pickle save due to OneDrive lock: {os.path.basename(filepath)}")
                    return False
                except:
                    return False
    return False


def run_hrp_computation(returns_all, valid_rebal_dates, window, min_stocks, output_dir, 
                        universe_flags=None, df_universe=None, use_industry_etfs=True,
                        min_stocks_per_industry=3, use_gpu=True):
    """
    Compute HRP weights and monthly returns using two-stage approach:
    
    Stage 1: Aggregate stocks into 12 Fama-French industry ETFs (value-weighted)
    Stage 2: Apply HRP to the 12 industry ETFs (N=12 < T=60, statistically valid)
    
    Final stock weights = industry_weight × within_industry_weight
    
    Args:
        returns_all: DataFrame of all returns (dates x assets)
        valid_rebal_dates: List of valid rebalancing dates
        window: Lookback window in months (e.g., 60)
        min_stocks: Minimum number of stocks required (for industry ETF mode: min industries)
        output_dir: Output directory path
        universe_flags: DataFrame indicating which stocks are in universe at each date
        df_universe: Full CRSP DataFrame with FF_12, MKT_CAP columns (required for industry ETFs)
        use_industry_etfs: If True, use two-stage industry ETF approach (recommended)
        min_stocks_per_industry: Minimum stocks per industry to include
        use_gpu: If True, attempt GPU acceleration via cuDF (falls back to CPU if unavailable)
        
    Returns:
        strategy_returns: Series of monthly HRP returns
    """
    logger.info(f"\n{'='*70}")
    if use_industry_etfs:
        logger.info(f"TWO-STAGE HRP COMPUTATION (Industry ETF Mode)")
        logger.info(f"  Stage 1: Value-weighted FF12 Industry ETFs")
        logger.info(f"  Stage 2: HRP allocation across 12 industries (N=12 < T={window})")
    else:
        logger.info(f"HRP COMPUTATION (Direct Stock Mode - WARNING: N>>T)")
    logger.info(f"{'='*70}")
    
    # ==========================================================================
    # INDUSTRY ETF MODE (Two-Stage HRP)
    # ==========================================================================
    if use_industry_etfs:
        if df_universe is None:
            raise ValueError("df_universe required for industry ETF mode. Pass the full CRSP DataFrame.")
        
        # Compute industry ETF returns (GPU-accelerated if available)
        industry_returns, industry_weights_dict, industry_counts = hrp_data.compute_industry_etf_returns(
            df_universe, returns_all, universe_flags, 
            min_stocks_per_industry=min_stocks_per_industry,
            use_gpu=use_gpu
        )
        
        # Save industry counts for analysis
        industry_counts.to_csv(os.path.join(output_dir, 'industry_stock_counts.csv'))
        
        # Save industry ETF returns for reuse (e.g., dendrogram visualization)
        industry_returns.to_csv(os.path.join(output_dir, 'industry_etf_returns.csv'))
        logger.info(f"✓ Saved industry ETF returns: {os.path.join(output_dir, 'industry_etf_returns.csv')}")
        
        # Save within-industry VW weights for transaction cost computation
        joblib.dump(industry_weights_dict, os.path.join(output_dir, 'within_industry_weights.pkl'))
        logger.info(f"✓ Saved within-industry weights: {os.path.join(output_dir, 'within_industry_weights.pkl')}")
        
        # Run HRP on industry ETFs
        strategy_returns, all_weights_dict, stock_weights_dict = _run_industry_hrp(
            industry_returns, industry_weights_dict, returns_all, 
            valid_rebal_dates, window, min_stocks, output_dir
        )
        
        return strategy_returns
    
    # ==========================================================================
    # LEGACY MODE (Direct Stock HRP - N>>T, statistically questionable)
    # ==========================================================================
    else:
        logger.warning("Using legacy direct stock mode. N>>T may cause covariance instability.")
        return _run_direct_stock_hrp(
            returns_all, valid_rebal_dates, window, min_stocks, output_dir, universe_flags
        )


def _run_industry_hrp(industry_returns, industry_weights_dict, returns_all,
                      valid_rebal_dates, window, min_industries, output_dir):
    """
    Run HRP on industry ETF returns (Stage 2 of two-stage HRP).
    
    Args:
        industry_returns: DataFrame of industry ETF returns (dates x 12 industries)
        industry_weights_dict: {date: {industry: {permno: weight}}} within-industry weights
        returns_all: Original stock returns for computing actual portfolio returns
        valid_rebal_dates: List of valid rebalancing dates
        window: Lookback window in months
        min_industries: Minimum number of industries required
        output_dir: Output directory path
        
    Returns:
        strategy_returns: Series of monthly HRP returns
        all_weights_dict: {date: Series of industry weights}
        stock_weights_dict: {date: Series of stock weights}
    """
    logger.info(f"\nRunning HRP on {industry_returns.shape[1]} Industry ETFs...")
    logger.info(f"Lookback window: {window} months")
    logger.info(f"Covariance matrix: {industry_returns.shape[1]}×{industry_returns.shape[1]} (FULL RANK)")
    
    strategy_returns = pd.Series(index=valid_rebal_dates, dtype=float, name=f'HRP_{window}m')
    all_weights_dict = {}      # Industry-level HRP weights
    stock_weights_dict = {}    # Final stock-level weights
    
    checkpoint_file = os.path.join(output_dir, 'hrp_checkpoint.pkl')
    start_idx = 0
    
    if os.path.exists(checkpoint_file):
        logger.info(f"Found checkpoint file. Loading previous progress...")
        checkpoint = joblib.load(checkpoint_file)
        strategy_returns = checkpoint['strategy_returns']
        all_weights_dict = checkpoint.get('all_weights_dict', {})
        stock_weights_dict = checkpoint.get('stock_weights_dict', {})
        start_idx = checkpoint['last_completed_idx'] + 1
        logger.info(f"Resuming from rebalance {start_idx}/{len(valid_rebal_dates)}")
    else:
        logger.info("Starting HRP computation from scratch...")

    logger.info(f"Total rebalance dates: {len(valid_rebal_dates)}")
    
    # Track missing returns for summary (instead of verbose per-date warnings)
    missing_returns_log = []  # List of (date, n_missing) tuples

    for idx, rebal_date in enumerate(tqdm(valid_rebal_dates[start_idx:], desc="Computing Industry HRP", 
                                          initial=start_idx, total=len(valid_rebal_dates))):
        
        # Get window of industry returns
        window_start = rebal_date - pd.DateOffset(months=window)
        
        # Filter industry returns to window
        mask = (industry_returns.index >= window_start) & (industry_returns.index <= rebal_date)
        window_ind_returns = industry_returns.loc[mask].copy()
        
        # Drop industries with any NaN in window (need complete history)
        window_ind_returns = window_ind_returns.dropna(axis=1, how='any')
        
        if window_ind_returns.shape[1] < min_industries:
            logger.warning(f"  {rebal_date.date()}: Only {window_ind_returns.shape[1]} industries with full data, skipping")
            continue
        
        if window_ind_returns.shape[0] < window * 0.8:  # Require at least 80% of window
            logger.warning(f"  {rebal_date.date()}: Only {window_ind_returns.shape[0]} months of data, skipping")
            continue
        
        try:
            # =================================================================
            # STAGE 2: HRP on Industry ETFs
            # =================================================================
            industry_weights = hrp_functions.compute_hrp_weights(window_ind_returns)
            
            # Check for identity risk
            risk = hrp_analytics.check_identity_risk(industry_weights)
            if risk['is_identity']:
                logger.warning(f"  [!] {rebal_date.date()}: Industry weights near equal weight (MAD={risk['mad_ew']:.6f})")
            
            all_weights_dict[rebal_date] = industry_weights
            
            # =================================================================
            # DECOMPOSE TO STOCK WEIGHTS
            # =================================================================
            # Get within-industry weights for this date
            if rebal_date not in industry_weights_dict:
                logger.warning(f"  {rebal_date.date()}: No within-industry weights, skipping")
                continue
            
            within_ind_weights = industry_weights_dict[rebal_date]
            
            # Final stock weight = industry_weight × within_industry_weight
            final_stock_weights = {}
            for industry_name, ind_weight in industry_weights.items():
                # Map industry name back to ID
                ind_id = [k for k, v in hrp_data.FF12_INDUSTRY_NAMES.items() if v == industry_name]
                if not ind_id:
                    continue
                ind_id = ind_id[0]
                
                if ind_id in within_ind_weights:
                    for permno, within_weight in within_ind_weights[ind_id].items():
                        final_stock_weights[permno] = ind_weight * within_weight
            
            if len(final_stock_weights) == 0:
                logger.warning(f"  {rebal_date.date()}: No stock weights computed, skipping")
                continue
            
            stock_weights = pd.Series(final_stock_weights)
            stock_weights = stock_weights / stock_weights.sum()  # Normalize to sum to 1
            stock_weights_dict[rebal_date] = stock_weights
            
            # =================================================================
            # COMPUTE NEXT MONTH RETURN
            # =================================================================
            next_start = rebal_date + pd.DateOffset(days=1)
            next_end = (rebal_date + pd.DateOffset(months=1)) + pd.tseries.offsets.MonthEnd(0)
            next_returns = returns_all.loc[next_start:next_end]
            
            if next_returns.empty:
                continue
            
            # Align weights with available returns
            valid_permnos = [p for p in stock_weights.index if p in next_returns.columns]
            if len(valid_permnos) == 0:
                continue
            
            aligned_weights = stock_weights[valid_permnos]
            aligned_weights = aligned_weights / aligned_weights.sum()  # Renormalize
            
            # Get returns and handle missing (treat as -100% loss)
            next_month_ret = next_returns[valid_permnos].copy()
            n_missing = next_month_ret.isna().sum().sum()
            if n_missing > 0:
                missing_returns_log.append((rebal_date, n_missing))
            next_month_ret = next_month_ret.fillna(-1.0)
            
            # Compute portfolio return
            port_ret_series = next_month_ret.dot(aligned_weights)
            
            if len(port_ret_series) > 0:
                month_ret = (1 + port_ret_series).prod() - 1
                strategy_returns.loc[rebal_date] = month_ret
                
        except Exception as e:
            logger.error(f"Error at {rebal_date}: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Save checkpoint periodically
        if (start_idx + idx) % 20 == 0:
            joblib.dump({
                'strategy_returns': strategy_returns,
                'all_weights_dict': all_weights_dict,
                'stock_weights_dict': stock_weights_dict,
                'last_completed_idx': start_idx + idx
            }, checkpoint_file)

    # ==========================================================================
    # MISSING RETURNS SUMMARY (delistings treated as -100% loss)
    # ==========================================================================
    if missing_returns_log:
        total_missing = sum(n for _, n in missing_returns_log)
        months_affected = len(missing_returns_log)
        mean_per_month = total_missing / months_affected if months_affected > 0 else 0
        logger.info(f"  Missing returns summary: {total_missing} total across {months_affected} months "
                    f"(mean: {mean_per_month:.1f}/month, treated as -100% delisting loss)")
    
    # ==========================================================================
    # SAVE OUTPUTS (CSV first - more reliable with OneDrive)
    # ==========================================================================
    
    # Save CSV files first (essential outputs)
    strategy_returns.to_csv(os.path.join(output_dir, 'hrp_strategy_returns.csv'))
    logger.info(f"  ✓ Saved: hrp_strategy_returns.csv")
    
    # Save industry weights to CSV
    industry_weights_df = pd.DataFrame(all_weights_dict).T.sort_index()
    industry_weights_df.to_csv(os.path.join(output_dir, 'hrp_industry_weights.csv'))
    logger.info(f"  ✓ Saved: hrp_industry_weights.csv")
    
    # Save stock weights to CSV
    stock_weights_df = pd.DataFrame(stock_weights_dict).T.sort_index()
    stock_weights_df.to_csv(os.path.join(output_dir, 'all_hrp_weights.csv'))
    logger.info(f"  ✓ Saved: all_hrp_weights.csv")
    
    # Save pickle files (optional - may fail due to OneDrive sync)
    # These are redundant since we have CSVs, but useful for faster reloading
    try:
        joblib.dump(all_weights_dict, os.path.join(output_dir, 'hrp_industry_weights_dict.pkl'))
        joblib.dump(stock_weights_dict, os.path.join(output_dir, 'hrp_weights_dict.pkl'))
        logger.info(f"  ✓ Saved: pickle files (optional cache)")
    except OSError as e:
        logger.warning(f"  ⚠ Pickle save skipped (OneDrive sync): {e}")
        logger.info(f"  ℹ CSV files saved successfully - pickle files are optional cache")

    # Clean up checkpoint
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    logger.info(f"\n✓ Two-Stage HRP Computation Completed.")
    logger.info(f"  - Strategy returns: {os.path.join(output_dir, 'hrp_strategy_returns.csv')}")
    logger.info(f"  - Industry weights: {os.path.join(output_dir, 'hrp_industry_weights.csv')}")
    logger.info(f"  - Stock weights: {os.path.join(output_dir, 'all_hrp_weights.csv')}")
    
    return strategy_returns, all_weights_dict, stock_weights_dict


# NOTE: _run_direct_stock_hrp (legacy ~600 stock mode) removed
# Two-stage industry ETF approach is now the only supported mode


def run_backtest(strategy_returns, rf_monthly_aligned, output_dir, market_returns=None):
    """
    Backtest the HRP strategy and generate performance metrics.
    
    Args:
        strategy_returns: Series of monthly HRP returns
        rf_monthly_aligned: Series of risk-free rates aligned to dates
        output_dir: Output directory path
        market_returns: Series of market benchmark returns (optional, e.g., CRSP VW)
        
    Returns:
        metrics_df: DataFrame with performance metrics
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"BACKTEST")
    logger.info(f"{'='*70}")

    strategy_returns = pd.to_numeric(strategy_returns, errors='coerce').dropna()
    
    logger.info(f"Backtest Period: {strategy_returns.index[0].date()} to {strategy_returns.index[-1].date()} ({len(strategy_returns)} months)")

    # Compute cumulative returns
    cum_returns = (1 + strategy_returns).cumprod()
    
    # Compute market benchmark cumulative returns if provided
    if market_returns is not None:
        market_aligned = market_returns.loc[strategy_returns.index].dropna()
        cum_market = (1 + market_aligned).cumprod()
    else:
        cum_market = None

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Linear Scale
    ax1.plot(cum_returns, label='HRP 60m', linewidth=2, color='blue')
    if cum_market is not None:
        ax1.plot(cum_market.index, cum_market.values, label='CRSP VW Index', linewidth=2, color='red', linestyle='--')
    ax1.set_title('HRP Strategy Equity Curve (Linear Scale)')
    ax1.set_ylabel('Cumulative Return')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Log Scale
    ax2.plot(cum_returns, label='HRP 60m', linewidth=2, color='blue')
    if cum_market is not None:
        ax2.plot(cum_market.index, cum_market.values, label='CRSP VW Index', linewidth=2, color='red', linestyle='--')
    ax2.set_yscale('log')
    ax2.set_title('HRP Strategy Equity Curve (Log Scale)')
    ax2.set_ylabel('Cumulative Return (Log)')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hrp_equity_curve.png'))
    plt.show()

    # Drawdown Plot
    drawdown = (cum_returns.cummax() - cum_returns) / cum_returns.cummax()
    
    plt.figure(figsize=(12, 4))
    plt.fill_between(drawdown.index, drawdown.values, alpha=0.7, color='red')
    plt.title('HRP Strategy Drawdown')
    plt.ylabel('Drawdown')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hrp_drawdown.png'))
    plt.show()

    # Compute Metrics
    metrics = hrp_analytics.compute_metrics(strategy_returns, rf_monthly_aligned, 'HRP 60m')
    metrics_df = pd.DataFrame([metrics]).set_index('Strategy')
    
    print("\n" + "="*70)
    print("PERFORMANCE METRICS")
    print("="*70)
    print(metrics_df.T)
    
    metrics_df.to_csv(os.path.join(output_dir, 'final_metrics.csv'))
    logger.info(f"✓ Metrics saved to {os.path.join(output_dir, 'final_metrics.csv')}")
    
    return metrics_df


def plot_monthly_returns_distribution(strategy_returns, output_dir):
    """
    Plot the distribution of monthly returns.
    """
    plt.figure(figsize=(12, 5))
    
    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(strategy_returns.dropna(), bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=1)
    plt.axvline(x=strategy_returns.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {strategy_returns.mean():.4f}')
    plt.title('Distribution of Monthly Returns')
    plt.xlabel('Monthly Return')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Rolling 12-month returns
    plt.subplot(1, 2, 2)
    rolling_12m = (1 + strategy_returns).rolling(12).apply(lambda x: x.prod() - 1, raw=True)
    plt.plot(rolling_12m, linewidth=1.5)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
    plt.title('Rolling 12-Month Returns')
    plt.xlabel('Date')
    plt.ylabel('12-Month Return')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'returns_distribution.png'))
    plt.show()


