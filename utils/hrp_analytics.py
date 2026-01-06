import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from scipy import stats
from sklearn.covariance import LedoitWolf
try:
    from cov_shrinkage import compute_rmt
except ImportError:
    # Fallback if running as script
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from cov_shrinkage import compute_rmt


def print_distribution_comparison(strategy_returns, benchmark_returns, 
                                   strategy_name='HRP Strategy', 
                                   benchmark_name='CRSP VW Index'):
    """
    Print detailed distribution statistics comparing strategy vs benchmark.
    
    Args:
        strategy_returns: pd.Series of strategy monthly returns
        benchmark_returns: pd.Series of benchmark monthly returns
        strategy_name: Display name for strategy
        benchmark_name: Display name for benchmark
        
    Returns:
        dict: Dictionary with strategy and benchmark statistics
    """
    # Align to same period
    common_idx = strategy_returns.index.intersection(benchmark_returns.index)
    strat_aligned = strategy_returns.loc[common_idx]
    bench_aligned = benchmark_returns.loc[common_idx]
    
    def compute_stats(returns, name):
        """Compute mean, std, skewness, kurtosis for a return series."""
        return {
            'Strategy': name,
            'Mean (Monthly)': returns.mean(),
            'Mean (Ann.)': returns.mean() * 12,
            'Std (Monthly)': returns.std(),
            'Std (Ann.)': returns.std() * np.sqrt(12),
            'Skewness': stats.skew(returns.dropna()),
            'Kurtosis': stats.kurtosis(returns.dropna()),  # Excess kurtosis (normal = 0)
            'Min': returns.min(),
            'Max': returns.max(),
            'Observations': len(returns)
        }
    
    strat_stats = compute_stats(strat_aligned, strategy_name)
    bench_stats = compute_stats(bench_aligned, benchmark_name)
    
    # Display comparison table
    print("="*80)
    print("MONTHLY RETURNS DISTRIBUTION STATISTICS")
    print("="*80)
    print(f"Period: {common_idx.min():%Y-%m} to {common_idx.max():%Y-%m} ({len(common_idx)} months)")
    print(f"\n{'Metric':<20} {strategy_name:>18} {benchmark_name:>18}")
    print("-"*60)
    print(f"{'Mean (Monthly)':<20} {strat_stats['Mean (Monthly)']:>17.3%} {bench_stats['Mean (Monthly)']:>17.3%}")
    print(f"{'Mean (Annualized)':<20} {strat_stats['Mean (Ann.)']:>17.2%} {bench_stats['Mean (Ann.)']:>17.2%}")
    print(f"{'Std (Monthly)':<20} {strat_stats['Std (Monthly)']:>17.3%} {bench_stats['Std (Monthly)']:>17.3%}")
    print(f"{'Std (Annualized)':<20} {strat_stats['Std (Ann.)']:>17.2%} {bench_stats['Std (Ann.)']:>17.2%}")
    print(f"{'Skewness':<20} {strat_stats['Skewness']:>18.3f} {bench_stats['Skewness']:>18.3f}")
    print(f"{'Excess Kurtosis':<20} {strat_stats['Kurtosis']:>18.3f} {bench_stats['Kurtosis']:>18.3f}")
    print(f"{'Min (Monthly)':<20} {strat_stats['Min']:>17.2%} {bench_stats['Min']:>17.2%}")
    print(f"{'Max (Monthly)':<20} {strat_stats['Max']:>17.2%} {bench_stats['Max']:>17.2%}")
    print("-"*60)
    
    # Interpretation
    print("\nInterpretation:")
    if strat_stats['Skewness'] > bench_stats['Skewness']:
        print(f"  • {strategy_name} has MORE POSITIVE skew ({strat_stats['Skewness']:.3f} vs {bench_stats['Skewness']:.3f})")
        print("    → Fewer extreme negative returns, more upside potential")
    else:
        print(f"  • {strategy_name} has MORE NEGATIVE skew ({strat_stats['Skewness']:.3f} vs {bench_stats['Skewness']:.3f})")
        print("    → More exposure to tail risk")
    
    if strat_stats['Kurtosis'] < bench_stats['Kurtosis']:
        print(f"  • {strategy_name} has LOWER kurtosis ({strat_stats['Kurtosis']:.3f} vs {bench_stats['Kurtosis']:.3f})")
        print("    → Thinner tails, fewer extreme events (both positive and negative)")
    else:
        print(f"  • {strategy_name} has HIGHER kurtosis ({strat_stats['Kurtosis']:.3f} vs {bench_stats['Kurtosis']:.3f})")
        print("    → Fatter tails, more frequent extreme events")
    
    # Jarque-Bera test for normality
    jb_strat, jb_strat_p = stats.jarque_bera(strat_aligned.dropna())
    jb_bench, jb_bench_p = stats.jarque_bera(bench_aligned.dropna())
    print(f"\nJarque-Bera Normality Test:")
    print(f"  • {strategy_name}:  JB={jb_strat:.1f}, p-value={jb_strat_p:.4f} {'(Reject H0: Non-normal)' if jb_strat_p < 0.05 else '(Fail to reject H0)'}")
    print(f"  • {benchmark_name}: JB={jb_bench:.1f}, p-value={jb_bench_p:.4f} {'(Reject H0: Non-normal)' if jb_bench_p < 0.05 else '(Fail to reject H0)'}")
    
    return {strategy_name: strat_stats, benchmark_name: bench_stats}

def compute_sharpe(returns, rf_rates):
    """
    Compute QUARTERLY Sharpe ratio from MONTHLY returns
    Methodology:
    SR = (Cumulative 3-month Return - Cumulative 3-month RF) / 3-month Std
    """
    if returns.std() == 0 or len(returns) < 2:
        return np.nan
    
    # Cumulative return over the period
    cumulative_return = (1 + returns).prod() - 1
    
    # Cumulative RF over the period - ensure exact alignment
    rf_aligned = rf_rates.reindex(returns.index, method='ffill')
    # Check for any NaN values in RF
    if rf_aligned.isna().any():
        # Fill with 0 if RF data is missing (conservative assumption)
        rf_aligned = rf_aligned.fillna(0)
    cumulative_rf = (1 + rf_aligned).prod() - 1
    
    # Standard deviation over the period
    std_period = returns.std()
    
    # Quarterly Sharpe
    return (cumulative_return - cumulative_rf) / std_period

def compute_metrics(returns, rf_monthly_aligned, name):
    """
    Compute comprehensive performance metrics from MONTHLY returns
    Uses QUARTERLY Sharpe methodology:
    - SR = (Cumulative Return - Cumulative RF) / Std
    - Computed over rolling 3-month windows, then averaged
    """
    cum = (1 + returns).cumprod()
    
    # Get RF for this strategy's period (monthly)
    rf_period = rf_monthly_aligned.reindex(returns.index, fill_value=0)
    
    # Calculate number of months
    n_months = len(returns)
    
    # Cumulative and annualized return (Strategy)
    cumulative_return = cum.iloc[-1] - 1
    annualized_return = (1 + cumulative_return) ** (12 / n_months) - 1
    
    # Cumulative and annualized return (Risk Free)
    cumulative_rf = (1 + rf_period).prod() - 1
    annualized_rf = (1 + cumulative_rf) ** (12 / n_months) - 1
    
    # Excess Returns
    total_excess_return = cumulative_return - cumulative_rf
    annualized_excess_return = annualized_return - annualized_rf
    
    # Standard Annualized Sharpe Ratio
    # SR = (Mean Monthly Excess Return / Std Monthly Excess Return) * sqrt(12)
    monthly_excess = returns - rf_period
    if monthly_excess.std() > 0:
        sharpe = (monthly_excess.mean() / monthly_excess.std()) * np.sqrt(12)
    else:
        sharpe = 0
    
    annualized_std = returns.std() * np.sqrt(12)
    
    metrics = {
        'Strategy': name,
        'Total Excess Return': total_excess_return,
        'Annualized Excess Return': annualized_excess_return,
        'Sharpe Ratio': sharpe,
        'Volatility (Ann.)': annualized_std,
        'Max Drawdown': (cum.cummax() - cum).max() / cum.cummax().max(),
        'Calmar Ratio': annualized_return / ((cum.cummax() - cum).max() / cum.cummax().max()) if (cum.cummax() - cum).max() > 0 else 0,
        'Win Rate': (returns > 0).sum() / len(returns),
        'Avg Win': returns[returns > 0].mean() if (returns > 0).any() else 0,
        'Avg Loss': returns[returns < 0].mean() if (returns < 0).any() else 0,
        'Best Month': returns.max(),
        'Worst Month': returns.min(),
        'Total Months': n_months
    }
    return metrics

def check_identity_risk(weights, threshold=None):
    """
    Check if weights are dangerously close to Equal Weights,
    indicating that the covariance shrinkage might have saturated (Identity Matrix).
    
    Args:
        weights (pd.Series): Portfolio weights.
        threshold (float, optional): Mean Absolute Deviation threshold. 
                                     If None, uses dynamic threshold (1/N * 0.05).
        
    Returns:
        dict: {'is_identity': bool, 'mad_ew': float}
    """
    n = len(weights)
    if n == 0:
        return {'is_identity': False, 'mad_ew': 0.0}
        
    ew = 1.0 / n
    # Mean Absolute Deviation from Equal Weight
    mad_ew = (weights - ew).abs().mean()
    
    # Dynamic Threshold: 5% of the Equal Weight value
    # For N=3000, ew=0.00033, threshold=0.000016
    if threshold is None:
        threshold = ew * 0.05
    
    is_identity = mad_ew < threshold
    
    return {'is_identity': is_identity, 'mad_ew': mad_ew}

def plot_shrunk_covariance_matrices(returns_all, valid_rebal_dates, universe_flags, windows=[60], num_dates=3):
    """
    Visualize Empirical vs Shrunk Covariance Matrices for a few sample dates.
    """
    print(f"\n{'='*70}")
    print(f"VISUALIZATION: Covariance Shrinkage Analysis")
    print(f"{'='*70}")

    # Select sample dates (beginning, middle, end)
    if len(valid_rebal_dates) > num_dates:
        indices = np.linspace(0, len(valid_rebal_dates) - 1, num_dates, dtype=int)
        sample_dates = [valid_rebal_dates[i] for i in indices]
    else:
        sample_dates = valid_rebal_dates

    max_win = max(windows)

    for date in sample_dates:
        # Reconstruct universe logic
        if universe_flags is not None and date in universe_flags.index:
            current_flags = universe_flags.loc[date]
            potential_universe = current_flags[current_flags == 1].index
            max_window_start = date - pd.DateOffset(months=max_win)
            # Use the same strict filtering as pipeline
            valid_universe = returns_all.loc[max_window_start:date, potential_universe].dropna(axis=1, how='any').columns
        else:
            # Fallback if no flags
            max_window_start = date - pd.DateOffset(months=max_win)
            valid_universe = returns_all.loc[max_window_start:date].dropna(axis=1, how='any').columns

        if len(valid_universe) < 2:
            print(f"Skipping {date.date()}: Not enough stocks ({len(valid_universe)})")
            continue

        # Get returns for the max window
        window_returns = returns_all.loc[max_window_start:date][valid_universe]
        
        # Compute Empirical Covariance
        emp_cov = window_returns.cov()
        
        # Compute Shrunk Covariance (RMT Denoising)
        try:
            shrunk_cov = compute_rmt(window_returns.values)
            method_name = "RMT Denoising"
        except Exception as e:
            print(f"RMT failed ({e}), falling back to Linear Ledoit-Wolf")
            lw = LedoitWolf()
            shrunk_cov = lw.fit(window_returns).covariance_
            method_name = f"Linear Shrinkage (LW: {lw.shrinkage_:.4f})"
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # Empirical
        im0 = axes[0].imshow(emp_cov, cmap='viridis', interpolation='none', vmin=-0.02, vmax=0.02)
        axes[0].set_title(f"Empirical Covariance\n{date.date()} (N={len(valid_universe)})")
        fig.colorbar(im0, ax=axes[0])
        
        # Shrunk
        im1 = axes[1].imshow(shrunk_cov, cmap='viridis', interpolation='none', vmin=-0.02, vmax=0.02)
        axes[1].set_title(f"{method_name}")
        fig.colorbar(im1, ax=axes[1])
        
        # Difference
        # Force numpy arrays to avoid DataFrame/Series ambiguity
        shrunk_cov_np = np.array(shrunk_cov)
        emp_cov_np = np.array(emp_cov)
        diff = shrunk_cov_np - emp_cov_np
        
        # Use symmetric scale for difference
        max_diff = max(abs(np.min(diff)), abs(np.max(diff)))
        im2 = axes[2].imshow(diff, cmap='coolwarm', interpolation='none', vmin=-max_diff, vmax=max_diff)
        axes[2].set_title("Difference (Shrunk - Empirical)")
        fig.colorbar(im2, ax=axes[2])
        
        plt.tight_layout()
        plt.show()

def plot_universe_size(universe_flags, output_dir):
    """
    Plots the number of stocks in the HRP universe (in_universe_hrp == 1) over time.
    """
    # Sum across columns (stocks) to get count per date
    # universe_flags is a DataFrame (Dates x Stocks) with 1s and 0s
    counts = universe_flags.sum(axis=1)
    
    plt.figure(figsize=(12, 6))
    plt.plot(counts.index, counts.values, color='blue', linewidth=2)
    plt.title('Number of Stocks in HRP Universe (Liquidity & Price Filter)')
    plt.ylabel('Number of Stocks')
    plt.xlabel('Date')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'universe_size_hrp.png'))
    plt.show()


def plot_portfolio_size(weights_df, output_dir):
    """
    Plots the number of stocks actually in the HRP portfolio (non-zero weights) over time.
    
    Parameters:
    -----------
    weights_df : pd.DataFrame
        DataFrame with dates as index and PERMNOs as columns, containing portfolio weights.
    output_dir : str
        Directory to save the output plot.
    """
    # Count non-zero weights per date (stocks with weight > small threshold)
    threshold = 1e-6
    portfolio_counts = (weights_df.abs() > threshold).sum(axis=1)
    
    # Statistics
    mean_count = portfolio_counts.mean()
    min_count = portfolio_counts.min()
    max_count = portfolio_counts.max()
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Top plot: Time series of portfolio size
    ax1 = axes[0]
    ax1.plot(portfolio_counts.index, portfolio_counts.values, color='green', linewidth=1.5, alpha=0.8)
    ax1.axhline(y=mean_count, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_count:.0f}')
    ax1.fill_between(portfolio_counts.index, portfolio_counts.values, alpha=0.3, color='green')
    ax1.set_title('Number of Stocks in HRP Portfolio (Non-Zero Weights)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Stocks')
    ax1.set_xlabel('Date')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Distribution histogram
    ax2 = axes[1]
    ax2.hist(portfolio_counts.values, bins=30, color='green', alpha=0.7, edgecolor='black')
    ax2.axvline(x=mean_count, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_count:.0f}')
    ax2.axvline(x=min_count, color='orange', linestyle=':', linewidth=1.5, label=f'Min: {min_count:.0f}')
    ax2.axvline(x=max_count, color='purple', linestyle=':', linewidth=1.5, label=f'Max: {max_count:.0f}')
    ax2.set_title('Distribution of Portfolio Size', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Number of Stocks')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'portfolio_size_hrp.png'), dpi=150)
    plt.show()
    
    # Print summary statistics
    print(f"\nPortfolio Size Statistics:")
    print(f"  Mean:   {mean_count:.1f} stocks")
    print(f"  Median: {portfolio_counts.median():.1f} stocks")
    print(f"  Min:    {min_count:.0f} stocks")
    print(f"  Max:    {max_count:.0f} stocks")
    print(f"  Std:    {portfolio_counts.std():.1f} stocks")
    
    return portfolio_counts


def plot_hrp_dendrogram(returns_all, universe_flags, date, window, output_dir):
    """
    Plots the HRP dendrogram (clustering tree) for a specific date and window.
    Replicates the HRP clustering logic (Correlation -> Distance -> Ward Linkage).
    """
    print(f"\n{'='*70}")
    print(f"PLOTTING DENDROGRAM: {date.date()} (Window: {window}m)")
    print(f"{'='*70}")
    
    # 1. Filter Data
    if date not in returns_all.index:
        # Find closest previous date
        available_dates = returns_all.index[returns_all.index <= date]
        if len(available_dates) == 0:
            print(f"No data available before {date}")
            return
        date = available_dates[-1]
        print(f"Date {date} not found, using closest: {date.date()}")

    # Get universe for this date/window
    if window not in universe_flags.columns and isinstance(universe_flags, pd.DataFrame):
        # If universe_flags is just the raw dataframe (Dates x Stocks)
        valid_universe = universe_flags.loc[date]
        valid_universe = valid_universe[valid_universe == 1].index.tolist()
    elif isinstance(universe_flags, dict) and window in universe_flags:
        # If it's the dictionary format
        valid_universe = universe_flags[window].loc[date]
        valid_universe = valid_universe[valid_universe == 1].index.tolist()
    else:
        # Fallback: use all stocks with valid data
        valid_universe = returns_all.columns.tolist()

    # Get returns window
    end_idx = returns_all.index.get_loc(date)
    start_idx = max(0, end_idx - window + 1)
    window_returns = returns_all.iloc[start_idx : end_idx+1][valid_universe]
    
    # Filter valid stocks (non-zero variance)
    window_returns = window_returns.loc[:, window_returns.var() > 1e-8]
    
    # Drop any columns with NaNs (Ledoit-Wolf cannot handle NaNs)
    window_returns = window_returns.dropna(axis=1)
    
    valid_assets = window_returns.columns.tolist()
    
    print(f"Universe Size: {len(valid_assets)} stocks")
    
    if len(valid_assets) < 2:
        print("Insufficient stocks to plot dendrogram.")
        return

    # 2. Compute Covariance & Correlation (Ledoit-Wolf)
    lw = LedoitWolf()
    lw.fit(window_returns)
    cov = lw.covariance_
    
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr = np.clip(corr, -1.0, 1.0)
    
    # 3. Compute Distance
    dist = np.sqrt(np.clip((1 - corr) / 2.0, 0.0, None))
    
    # Euclidean distance on distance matrix
    squared_norms = np.sum(dist ** 2, axis=1, keepdims=True)
    eucl_dist = np.sqrt(np.clip(squared_norms + squared_norms.T - 2 * np.dot(dist, dist.T), 0.0, None))
    eucl_dist_condensed = squareform(eucl_dist, checks=False)
    
    # 4. Clustering (Ward Linkage)
    link = sch.linkage(eucl_dist_condensed, method='ward')
    
    # 5. Plot
    plt.figure(figsize=(15, 8))
    sch.dendrogram(
        link,
        labels=valid_assets,
        leaf_rotation=90.,
        leaf_font_size=8.,
        no_plot=False,
        truncate_mode='lastp',  # Show only last p merged clusters
        p=50,                   # Show top 50 clusters to keep it readable
        show_contracted=True
    )
    plt.title(f'HRP Dendrogram (Ward Linkage)\nDate: {date.date()} | Window: {window}m | Stocks: {len(valid_assets)}')
    plt.xlabel('Stocks / Clusters')
    plt.ylabel('Distance')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'dendrogram_{date.date()}_{window}m.png'))
    plt.show()


# Fama-French 12 Industry Names
FF12_NAMES = {
    1: 'Consumer NonDur', 2: 'Consumer Dur', 3: 'Manufacturing', 4: 'Energy',
    5: 'Chemicals', 6: 'Business Equip', 7: 'Telecom', 8: 'Utilities',
    9: 'Shops', 10: 'Healthcare', 11: 'Finance', 12: 'Other'
}


def load_permno_to_ff12_mapping(project_root):
    """
    Load PERMNO to FF12 industry mapping from CRSP and SIC files.
    
    Parameters:
    -----------
    project_root : str
        Root directory of the project.
        
    Returns:
    --------
    dict : Mapping of PERMNO (str) to FF_12 industry code (int)
    """
    crsp_path = os.path.join(project_root, 'DATA', 'CRSP', 'CRSP_selected_columns.csv')
    sic_path = os.path.join(project_root, 'DATA', 'CRSP', 'SIC_to_Fama_French_industry.csv')
    
    df_crsp = pd.read_csv(crsp_path, usecols=['PERMNO', 'date', 'SICCD'], parse_dates=['date'],
                          dtype={'SICCD': str}, low_memory=False)
    df_crsp['PERMNO'] = df_crsp['PERMNO'].astype(str)
    sic_df = pd.read_csv(sic_path)
    
    # Use latest available SIC for each PERMNO
    df_crsp = df_crsp.sort_values('date').drop_duplicates(subset='PERMNO', keep='last')
    df_crsp['SICCD'] = pd.to_numeric(df_crsp['SICCD'], errors='coerce')
    df_crsp = df_crsp.merge(sic_df[['SIC', 'FF_12']], left_on='SICCD', right_on='SIC', how='left')
    
    return df_crsp.set_index('PERMNO')['FF_12'].to_dict()


def plot_weight_distribution(hrp_weights_df, permno_to_ff12, output_dir, 
                             sample_dates=['2000-12-31', '2017-01-31', '2023-05-31']):
    """
    Plot weight distribution for sample months showing both stock-level and industry-level allocations.
    
    Parameters:
    -----------
    hrp_weights_df : pd.DataFrame
        DataFrame with dates as index and PERMNOs as columns, containing portfolio weights.
    permno_to_ff12 : dict
        Mapping of PERMNO (str) to FF_12 industry code.
    output_dir : str
        Directory to save output plots.
    sample_dates : list
        List of date strings to plot (format: 'YYYY-MM-DD').
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    for i, date_str in enumerate(sample_dates):
        # Find closest available date
        target_date = pd.Timestamp(date_str)
        available_dates = hrp_weights_df.index[hrp_weights_df.index <= target_date]
        if len(available_dates) == 0:
            available_dates = hrp_weights_df.index
        closest_date = available_dates[-1]
        
        # Get weights for this date (non-zero only)
        weights = hrp_weights_df.loc[closest_date]
        weights = weights[weights > 1e-6].sort_values(ascending=False)
        n_stocks = len(weights)
        
        # --- TOP ROW: All Stock Weights ---
        ax1 = axes[0, i]
        ax1.bar(range(n_stocks), weights.values * 100, color='steelblue', alpha=0.8, width=1.0)
        ax1.set_title(f'{closest_date.strftime("%b %Y")} ({n_stocks} stocks)', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Stock Rank')
        ax1.set_ylabel('Weight (%)')
        ax1.set_xlim(-0.5, n_stocks - 0.5)
        ax1.grid(axis='y', alpha=0.3)
        
        # Stats annotation
        max_w = weights.max() * 100
        top5_w = weights.nlargest(5).sum() * 100
        hhi = (weights ** 2).sum() * 10000
        ax1.text(0.95, 0.95, f'Max: {max_w:.2f}%\nTop5: {top5_w:.1f}%\nHHI: {hhi:.0f}', 
                 transform=ax1.transAxes, fontsize=9, verticalalignment='top', 
                 horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # --- BOTTOM ROW: Industry Weights ---
        ax2 = axes[1, i]
        
        # Map PERMNOs to FF_12 and aggregate weights
        industry_weights = {}
        for permno, w in weights.items():
            ff12 = permno_to_ff12.get(str(permno), 12)
            if pd.isna(ff12):
                ff12 = 12
            industry_weights[int(ff12)] = industry_weights.get(int(ff12), 0) + w
        
        # Sort by industry number
        industries = sorted(industry_weights.keys())
        ind_w = [industry_weights[ind] * 100 for ind in industries]
        ind_labels = [FF12_NAMES.get(ind, f'Ind {ind}') for ind in industries]
        
        colors = plt.cm.tab20(range(len(industries)))
        bars2 = ax2.barh(range(len(industries)), ind_w, color=colors, alpha=0.8)
        ax2.set_yticks(range(len(industries)))
        ax2.set_yticklabels(ind_labels, fontsize=9)
        ax2.set_xlabel('Weight (%)')
        ax2.set_title('FF12 Industry Allocation', fontsize=10)
        ax2.grid(axis='x', alpha=0.3)
        ax2.invert_yaxis()
        
        # Add percentage labels on bars
        for j, (bar, w) in enumerate(zip(bars2, ind_w)):
            if w > 2:
                ax2.text(w + 0.5, j, f'{w:.1f}%', va='center', fontsize=8)
    
    plt.suptitle('HRP Portfolio Weight Distribution', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'weight_distribution_samples.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print(f"\nWeight Distribution Summary ({len(sample_dates)} Sample Months):")
    for date_str in sample_dates:
        target_date = pd.Timestamp(date_str)
        available_dates = hrp_weights_df.index[hrp_weights_df.index <= target_date]
        if len(available_dates) == 0:
            continue
        closest_date = available_dates[-1]
        weights = hrp_weights_df.loc[closest_date]
        weights = weights[weights > 1e-6]
        print(f"\n  {closest_date.strftime('%b %Y')}: {len(weights)} stocks")
        print(f"    Max: {weights.max():.2%}, Top5: {weights.nlargest(5).sum():.2%}")
        
        # Industry breakdown
        industry_weights = {}
        for permno, w in weights.items():
            ff12 = permno_to_ff12.get(str(permno), 12)
            if pd.isna(ff12):
                ff12 = 12
            industry_weights[int(ff12)] = industry_weights.get(int(ff12), 0) + w
        
        top3_ind = sorted(industry_weights.items(), key=lambda x: -x[1])[:3]
        print(f"    Top 3 Industries: {', '.join([f'{FF12_NAMES[ind]}={w:.1%}' for ind, w in top3_ind])}")


def plot_industry_exposure_over_time(hrp_weights_df, permno_to_ff12, output_dir):
    """
    Plot industry exposure over time as a stacked area chart.
    
    Parameters:
    -----------
    hrp_weights_df : pd.DataFrame
        DataFrame with dates as index and PERMNOs as columns, containing portfolio weights.
    permno_to_ff12 : dict
        Mapping of PERMNO (str) to FF_12 industry code.
    output_dir : str
        Directory to save output plots.
        
    Returns:
    --------
    pd.DataFrame : Industry weights time series (columns = industry names)
    """
    print("\nComputing industry exposures over time...")
    
    # Build industry weights time series
    industry_ts = {ind: [] for ind in range(1, 13)}
    dates_list = []
    
    for date in hrp_weights_df.index:
        weights = hrp_weights_df.loc[date]
        weights = weights[weights > 1e-6]
        
        # Aggregate by industry
        ind_weights = {ind: 0.0 for ind in range(1, 13)}
        for permno, w in weights.items():
            ff12 = permno_to_ff12.get(str(permno), 12)
            if pd.isna(ff12):
                ff12 = 12
            ind_weights[int(ff12)] += w
        
        dates_list.append(date)
        for ind in range(1, 13):
            industry_ts[ind].append(ind_weights[ind] * 100)
    
    # Create DataFrame
    industry_df = pd.DataFrame(industry_ts, index=dates_list)
    industry_df.columns = [FF12_NAMES[i] for i in range(1, 13)]
    
    # Plot stacked area chart
    fig, ax = plt.subplots(figsize=(16, 8))
    
    colors = plt.cm.tab20(range(12))
    ax.stackplot(industry_df.index, industry_df.T.values, labels=industry_df.columns, 
                 colors=colors, alpha=0.85)
    
    ax.set_xlim(industry_df.index.min(), industry_df.index.max())
    ax.set_ylim(0, 100)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Portfolio Weight (%)', fontsize=11)
    ax.set_title('HRP Portfolio Industry Exposure Over Time (FF12)', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Legend outside plot
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=9, title='Industry')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'industry_exposure_time_series.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print average industry allocation
    print(f"\nAverage Industry Allocation (Full Period):")
    avg_allocation = industry_df.mean().sort_values(ascending=False)
    for ind, alloc in avg_allocation.items():
        print(f"  {ind:<18}: {alloc:.1f}%")
    
    return industry_df


def plot_rmt_comparison(returns_all, universe_flags, target_date, window, save_path, 
                        denoise_func=None):
    """
    Plot correlation matrix before and after RMT denoising side-by-side.
    Uses quasi-diagonalization (seriation) for better visualization.
    
    Args:
        returns_all: DataFrame of all stock returns (date x PERMNO)
        universe_flags: DataFrame of boolean flags for universe membership
        target_date: str or Timestamp for the rebalancing date
        window: int, lookback window in months
        save_path: str, path to save the output figure
        denoise_func: callable, RMT denoising function (default: hrp_functions.denoise_covariance_rmt)
        
    Returns:
        tuple: (corr_raw, corr_denoised) numpy arrays
    """
    from scipy.cluster.hierarchy import linkage, leaves_list
    
    # Import denoising function if not provided
    if denoise_func is None:
        try:
            from cov_shrinkage import compute_rmt
            denoise_func = compute_rmt
        except ImportError:
            raise ImportError("cov_shrinkage module required for RMT denoising")
    
    # Get stocks in universe at target date
    target_date = pd.Timestamp(target_date)
    if target_date not in universe_flags.index:
        # Find closest date
        closest_idx = universe_flags.index.get_indexer([target_date], method='nearest')[0]
        target_date = universe_flags.index[closest_idx]
        print(f"  (Date adjusted to nearest: {target_date.strftime('%Y-%m-%d')})")
    
    in_universe = universe_flags.loc[target_date]
    valid_permnos = in_universe[in_universe == True].index.tolist()
    
    if len(valid_permnos) == 0:
        raise ValueError(f"No stocks in universe for date {target_date.strftime('%Y-%m')}. "
                        f"Check if universe_flags covers this date range.")
    
    # Get lookback returns
    start_date = target_date - pd.DateOffset(months=window)
    mask = (returns_all.index >= start_date) & (returns_all.index <= target_date)
    returns_window = returns_all.loc[mask, valid_permnos].dropna(axis=1, how='any')
    
    print(f"Date: {target_date.strftime('%Y-%m')}, Universe: {len(valid_permnos)}, "
          f"Before filter: {returns_window.shape[1]} stocks, {returns_window.shape[0]} months")
    
    # Require at least `window` months of data (use >= to handle slight date variations)
    returns_window = returns_window.loc[:, returns_window.count() >= window]
    
    # Validate we have enough data
    if returns_window.shape[1] < 2:
        raise ValueError(f"Insufficient stocks ({returns_window.shape[1]}) for date {target_date.strftime('%Y-%m')}. "
                        f"Need at least 2 stocks with {window} months of returns.")
    
    print(f"  After filter: {returns_window.shape[1]} stocks with {window}+ months")
    
    # Compute raw correlation matrix
    corr_raw = returns_window.corr().values
    
    # Compute eigenvalues of raw correlation to show MP threshold info
    T, N = returns_window.shape
    eVal_raw = np.linalg.eigvalsh(corr_raw)[::-1]  # Sorted descending
    q = N / T  # Aspect ratio
    sigma_sq = 1.0  # Theoretical variance for standardized data
    lambda_max = sigma_sq * (1 + np.sqrt(q))**2
    n_signal = int(np.sum(eVal_raw > lambda_max))
    effective_rank = min(T - 1, N)
    print(f"  MP threshold: λ_max={lambda_max:.3f} (q=N/T={q:.2f}), Signal: {n_signal}/{effective_rank} non-zero eigenvalues")
    
    # Apply RMT denoising (returns covariance matrix)
    cov_denoised = denoise_func(returns_window.values)
    
    # Convert denoised covariance to correlation
    std_denoised = np.sqrt(np.diag(cov_denoised))
    std_denoised = np.maximum(std_denoised, 1e-10)  # Avoid division by zero
    corr_denoised = cov_denoised / np.outer(std_denoised, std_denoised)
    np.fill_diagonal(corr_denoised, 1.0)  # Ensure diagonal is exactly 1
    
    # Quasi-diagonalize for better visualization (reorder by hierarchical clustering)
    # Use correlation distance for clustering
    dist = np.sqrt(0.5 * (1 - corr_denoised))
    np.fill_diagonal(dist, 0)
    
    # Hierarchical clustering
    condensed_dist = dist[np.triu_indices(len(dist), k=1)]
    Z = linkage(condensed_dist, method='ward')
    order = leaves_list(Z)
    
    # Reorder both matrices
    corr_raw_ordered = corr_raw[np.ix_(order, order)]
    corr_denoised_ordered = corr_denoised[np.ix_(order, order)]
    
    # Compute difference matrix (Raw - Denoised)
    corr_diff_ordered = corr_raw_ordered - corr_denoised_ordered
    
    # Create figure with 3 panels: Raw, Denoised, Difference
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Common colorbar settings for correlation matrices
    vmin, vmax = -0.5, 1.0
    cmap = 'RdBu_r'
    
    # Raw correlation matrix
    im1 = axes[0].imshow(corr_raw_ordered, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    axes[0].set_title(f'Raw Correlation\n{target_date.strftime("%Y-%m")} (N={len(order)})', fontsize=11)
    axes[0].set_xlabel('Asset (quasi-diagonalized)')
    axes[0].set_ylabel('Asset (quasi-diagonalized)')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    cbar1 = fig.colorbar(im1, ax=axes[0], orientation='vertical', fraction=0.046, pad=0.04)
    cbar1.set_label('ρ', fontsize=10)
    
    # Denoised correlation matrix
    im2 = axes[1].imshow(corr_denoised_ordered, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    axes[1].set_title(f'RMT Denoised Correlation\n{target_date.strftime("%Y-%m")} (N={len(order)})', fontsize=11)
    axes[1].set_xlabel('Asset (quasi-diagonalized)')
    axes[1].set_ylabel('Asset (quasi-diagonalized)')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    cbar2 = fig.colorbar(im2, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)
    cbar2.set_label('ρ', fontsize=10)
    
    # Difference matrix with ADAPTIVE scale (centered at 0)
    diff_off_diag = corr_diff_ordered[np.triu_indices(len(corr_diff_ordered), k=1)]
    diff_max = max(abs(diff_off_diag.min()), abs(diff_off_diag.max()))
    diff_max = max(diff_max, 0.01)  # Ensure minimum scale
    
    im3 = axes[2].imshow(corr_diff_ordered, cmap='RdBu_r', vmin=-diff_max, vmax=diff_max, aspect='auto')
    axes[2].set_title(f'Difference (Raw − Denoised)\nAdaptive Scale ±{diff_max:.3f}', fontsize=11)
    axes[2].set_xlabel('Asset (quasi-diagonalized)')
    axes[2].set_ylabel('Asset (quasi-diagonalized)')
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    cbar3 = fig.colorbar(im3, ax=axes[2], orientation='vertical', fraction=0.046, pad=0.04)
    cbar3.set_label('Δρ', fontsize=10)
    
    # Add statistics annotation
    raw_off_diag = corr_raw[np.triu_indices(len(corr_raw), k=1)]
    denoised_off_diag = corr_denoised[np.triu_indices(len(corr_denoised), k=1)]
    
    stats_text = (
        f"Raw: mean={raw_off_diag.mean():.3f}, std={raw_off_diag.std():.3f}  |  "
        f"Denoised: mean={denoised_off_diag.mean():.3f}, std={denoised_off_diag.std():.3f}  |  "
        f"Diff: mean={diff_off_diag.mean():.4f}, std={diff_off_diag.std():.4f}"
    )
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.06, 1.0, 1])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return corr_raw, corr_denoised


def plot_industry_hrp_weights(industry_weights_path, output_dir):
    """
    Plot HRP industry weights over time (stacked area chart + heatmap).
    
    Args:
        industry_weights_path: Path to hrp_industry_weights.csv
        output_dir: Directory to save plots
    """
    # Load industry weights
    df = pd.read_csv(industry_weights_path, index_col=0, parse_dates=True)
    
    # Define colors for industries
    colors = plt.cm.tab20(np.linspace(0, 1, len(df.columns)))
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # =========================================================================
    # Plot 1: Stacked Area Chart
    # =========================================================================
    ax1 = axes[0]
    ax1.stackplot(df.index, df.T.values, labels=df.columns, colors=colors, alpha=0.8)
    ax1.set_ylabel('Weight')
    ax1.set_xlabel('Date')
    ax1.set_title('HRP Industry Weights Over Time (Stacked Area)', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # =========================================================================
    # Plot 2: Heatmap
    # =========================================================================
    ax2 = axes[1]
    
    # Resample to quarterly for readability if too many dates
    if len(df) > 100:
        df_plot = df.resample('QE').mean()
        title_suffix = "(Quarterly Average)"
    else:
        df_plot = df
        title_suffix = "(Monthly)"
    
    im = ax2.imshow(df_plot.T.values, aspect='auto', cmap='YlOrRd', vmin=0, vmax=df_plot.max().max())
    ax2.set_yticks(range(len(df_plot.columns)))
    ax2.set_yticklabels(df_plot.columns)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Industry')
    ax2.set_title(f'HRP Industry Weights Heatmap {title_suffix}', fontsize=12, fontweight='bold')
    
    # Set x-axis ticks (show every Nth date)
    n_ticks = 10
    tick_positions = np.linspace(0, len(df_plot)-1, n_ticks, dtype=int)
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels([df_plot.index[i].strftime('%Y-%m') for i in tick_positions], rotation=45, ha='right')
    
    cbar = fig.colorbar(im, ax=ax2, orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label('Weight')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hrp_industry_weights_over_time.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*70)
    print("HRP INDUSTRY WEIGHTS SUMMARY")
    print("="*70)
    print(f"\n{'Industry':<12} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-"*50)
    for col in df.columns:
        print(f"{col:<12} {df[col].mean():>8.1%} {df[col].std():>8.1%} {df[col].min():>8.1%} {df[col].max():>8.1%}")
    
    return df


def plot_industry_dendrogram(industry_returns, date, window, output_dir):
    """
    Plot dendrogram for industry ETF clustering (much cleaner than 600-stock dendrogram).
    
    Args:
        industry_returns: DataFrame of industry ETF returns
        date: Target rebalancing date
        window: Lookback window in months
        output_dir: Directory to save plot
    """
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform
    from sklearn.covariance import LedoitWolf
    
    # Get window data
    window_start = date - pd.DateOffset(months=window)
    mask = (industry_returns.index >= window_start) & (industry_returns.index <= date)
    window_data = industry_returns.loc[mask].dropna(axis=1, how='any')
    
    if window_data.shape[1] < 2:
        print(f"Not enough industries with valid data for {date.date()}")
        return
    
    # Compute correlation matrix
    corr = window_data.corr()
    
    # Compute distance matrix
    dist = np.sqrt((1 - corr) / 2.0)
    dist_condensed = squareform(dist.values, checks=False)
    
    # Hierarchical clustering
    link = linkage(dist_condensed, method='ward')
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    dendro = dendrogram(
        link,
        labels=window_data.columns.tolist(),
        leaf_rotation=45,
        leaf_font_size=11,
        ax=ax
    )
    
    ax.set_title(f'HRP Industry Dendrogram\nDate: {date.date()} | Window: {window}m | Industries: {window_data.shape[1]}',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Industry')
    ax.set_ylabel('Distance')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'industry_dendrogram_{date.date()}.png'), dpi=150)
    plt.show()
    
    return dendro
    print(f"✓ Saved: {save_path}")
    return corr_raw, corr_denoised