import pandas as pd
import numpy as np
import os
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import internal modules
import hrp_functions
import hrp_analytics
from hrp_logger import setup_logger

logger = setup_logger()


def run_hrp_computation(returns_all, valid_rebal_dates, window, min_stocks, output_dir, universe_flags=None):
    """
    Compute HRP weights and monthly returns for a single window (60 months).
    
    Args:
        returns_all: DataFrame of all returns (dates x assets)
        valid_rebal_dates: List of valid rebalancing dates
        window: Lookback window in months (e.g., 60)
        min_stocks: Minimum number of stocks required
        output_dir: Output directory path
        universe_flags: DataFrame indicating which stocks are in universe at each date
        
    Returns:
        strategy_returns: Series of monthly HRP returns
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"HRP COMPUTATION (Window: {window} months)")
    logger.info(f"{'='*70}")
    
    strategy_returns = pd.Series(index=valid_rebal_dates, dtype=float, name=f'HRP_{window}m')
    all_weights_dict = {}
    
    checkpoint_file = os.path.join(output_dir, 'hrp_checkpoint.pkl')
    start_idx = 0
    
    if os.path.exists(checkpoint_file):
        logger.info(f"Found checkpoint file. Loading previous progress...")
        checkpoint = joblib.load(checkpoint_file)
        strategy_returns = checkpoint['strategy_returns']
        all_weights_dict = checkpoint.get('all_weights_dict', {})
        start_idx = checkpoint['last_completed_idx'] + 1
        logger.info(f"Resuming from rebalance {start_idx}/{len(valid_rebal_dates)}")
    else:
        logger.info("Starting HRP computation from scratch...")

    logger.info(f"Total rebalance dates: {len(valid_rebal_dates)}")
    logger.info(f"Window: {window} months")

    for idx, rebal_date in enumerate(tqdm(valid_rebal_dates[start_idx:], desc="Computing HRP", initial=start_idx, total=len(valid_rebal_dates))):
        next_start = rebal_date + pd.DateOffset(days=1)
        next_end = (rebal_date + pd.DateOffset(months=1)) + pd.tseries.offsets.MonthEnd(0)
        next_returns_df = returns_all.loc[next_start:next_end]
        
        if next_returns_df.empty:
            continue

        if universe_flags is not None:
            if rebal_date in universe_flags.index:
                current_flags = universe_flags.loc[rebal_date]
                potential_universe = current_flags[current_flags == 1].index
                
                # Get stocks with full data history for the window (NO future tradability check)
                window_start = rebal_date - pd.DateOffset(months=window)
                valid_universe = returns_all.loc[window_start:rebal_date, potential_universe].dropna(axis=1, how='any').columns
            else:
                valid_universe = []
        else:
            window_start = rebal_date - pd.DateOffset(months=window)
            valid_universe = returns_all.loc[window_start:rebal_date].dropna(axis=1, how='any').columns
        
        if len(valid_universe) < min_stocks:
            continue

        # Get returns for the window
        window_start = rebal_date - pd.DateOffset(months=window)
        window_returns = returns_all.loc[window_start:rebal_date][valid_universe]

        try:
            weights = hrp_functions.compute_hrp_weights(window_returns)
            
            # Check for Identity Risk (Shrinkage Saturation)
            risk = hrp_analytics.check_identity_risk(weights)
            if risk['is_identity']:
                logger.warning(f"  [!] Warning: Weights are identical to Equal Weight (MAD={risk['mad_ew']:.6f}). Shrinkage likely saturated.")

            all_weights_dict[rebal_date] = weights
            
            weights_aligned = weights[valid_universe]
            
            # Get next month returns and handle missing values (strict compliance)
            # Missing returns are treated as -1 (total loss) to avoid look-ahead bias
            next_month_returns = next_returns_df[valid_universe].copy()
            n_missing = next_month_returns.isna().sum().sum()
            if n_missing > 0:
                missing_stocks = next_month_returns.columns[next_month_returns.isna().any()].tolist()
                logger.warning(f"  [!] {rebal_date.date()}: {n_missing} missing returns treated as -100% loss. Stocks: {missing_stocks[:5]}{'...' if len(missing_stocks) > 5 else ''}")
            next_month_returns = next_month_returns.fillna(-1.0)  # Total loss for missing returns
            
            port_ret_series = next_month_returns.dot(weights_aligned)
            
            if len(port_ret_series) > 0:
                month_ret = (1 + port_ret_series).prod() - 1
                strategy_returns.loc[rebal_date] = month_ret
                
        except Exception as e:
            logger.error(f"Error at {rebal_date}: {e}")
            pass

        # Save checkpoint periodically
        if (start_idx + idx) % 20 == 0:
            joblib.dump({
                'strategy_returns': strategy_returns,
                'all_weights_dict': all_weights_dict,
                'last_completed_idx': start_idx + idx
            }, checkpoint_file)

    # Save final outputs
    strategy_returns.to_csv(os.path.join(output_dir, 'hrp_strategy_returns.csv'))
    joblib.dump(all_weights_dict, os.path.join(output_dir, 'hrp_weights_dict.pkl'))
    
    # Save weights to CSV for inspection
    weights_df = pd.DataFrame(all_weights_dict).T.sort_index()
    weights_df.to_csv(os.path.join(output_dir, 'all_hrp_weights.csv'))

    # Clean up checkpoint
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    logger.info(f"✓ HRP Computation Completed.")
    logger.info(f"  - Strategy returns saved to {os.path.join(output_dir, 'hrp_strategy_returns.csv')}")
    logger.info(f"  - Weights saved to {os.path.join(output_dir, 'all_hrp_weights.csv')}")
    
    return strategy_returns


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


