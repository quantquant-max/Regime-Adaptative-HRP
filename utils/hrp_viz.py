"""
HRP Visualization Module - Plotting Functions
=============================================
Provides functions for:
- HMM regime visualization
- XGBoost prediction analysis plots
- Strategy performance visualization
- Volatility targeting plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import roc_curve
from hrp_logger import setup_logger

logger = setup_logger()


def plot_hmm_input(df_hmm: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """
    Plot HMM input features (all standardized features dynamically).
    
    Parameters
    ----------
    df_hmm : pd.DataFrame
        Standardized HMM input data (can have 2 or 3+ features)
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    n_features = len(df_hmm.columns)
    fig, axes = plt.subplots(n_features, 1, figsize=(14, 3 * n_features), sharex=True)
    
    # Handle single feature case
    if n_features == 1:
        axes = [axes]
    
    # Color mapping for features
    colors = {
        'log_return_z': 'blue',
        'downside_dev_z': 'orange',
        'sent_change_z': 'green'
    }
    
    # Labels for features
    labels = {
        'log_return_z': 'Log Return (Z)',
        'downside_dev_z': 'Downside Dev (Z)',
        'sent_change_z': 'Sentiment YoY Change (Z)'
    }
    
    for i, col in enumerate(df_hmm.columns):
        ax = axes[i]
        color = colors.get(col, 'purple')
        label = labels.get(col, col)
        
        ax.plot(df_hmm.index, df_hmm[col], color=color, linewidth=0.8)
        ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
        ax.axhline(2, color='red', linestyle=':', linewidth=0.5, alpha=0.7)
        ax.axhline(-2, color='red', linestyle=':', linewidth=0.5, alpha=0.7)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.set_title(f'HMM Input Features ({n_features} features, Expanding Z-Scored)')
    
    axes[-1].set_xlabel('Date')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


def plot_hmm_regimes(df_regimes: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """
    Plot HMM regime detection results.
    
    Parameters
    ----------
    df_regimes : pd.DataFrame
        Regime labels and probabilities
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Plot 1: Cumulative Returns with Regime Shading
    ax1 = axes[0]
    cum_log_ret = df_regimes['log_return'].cumsum()
    ax1.plot(cum_log_ret.index, cum_log_ret.values, color='black', linewidth=1.5)
    
    # Shade Bear periods
    bear_mask = df_regimes['regime'] == 0
    starts, ends = _get_regime_periods(bear_mask)
    for start, end in zip(starts, ends):
        ax1.axvspan(start, end, alpha=0.3, color='red')
    
    ax1.set_ylabel('Cumulative Log Return')
    ax1.set_title('Market Returns with Bear Regime Shading (FILTERED - No Look-Ahead)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Regime Probabilities
    ax2 = axes[1]
    ax2.fill_between(df_regimes.index, 0, df_regimes['prob_bear'], 
                     alpha=0.7, color='red', label='P(Bear)')
    ax2.fill_between(df_regimes.index, df_regimes['prob_bear'], 1, 
                     alpha=0.7, color='green', label='P(Bull)')
    ax2.axhline(0.5, color='black', linestyle='--', linewidth=1)
    ax2.set_ylabel('Probability')
    ax2.set_title('Filtered Regime Probabilities (Forward Pass Only)')
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Downside Deviation by Regime
    ax3 = axes[2]
    colors = df_regimes['regime'].map({0: 'red', 1: 'green'})
    ax3.scatter(df_regimes.index, df_regimes['downside_dev'], c=colors, s=10, alpha=0.6)
    ax3.set_ylabel('Downside Deviation (12M)')
    ax3.set_xlabel('Date')
    ax3.set_title('Downside Deviation by Regime (Red=Bear, Green=Bull)')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


def plot_xgb_analysis(wf_df: pd.DataFrame, feature_importances: list = None,
                      save_path: str = None) -> plt.Figure:
    """
    Plot XGBoost prediction analysis.
    
    Parameters
    ----------
    wf_df : pd.DataFrame
        Walk-forward prediction results
    feature_importances : list, optional
        Feature importances from training
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    y_true = wf_df['actual_regime'].values
    y_pred = wf_df['predicted_regime'].values
    y_prob = wf_df['prob_bull'].values
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    # Plot 1: ROC Curve
    ax1 = axes[0, 0]
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    from sklearn.metrics import roc_auc_score
    roc_auc = roc_auc_score(y_true, y_prob)
    ax1.plot(fpr, tpr, color='blue', linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Probability Distribution
    ax2 = axes[0, 1]
    bear_probs = wf_df[wf_df['actual_regime'] == 0]['prob_bull']
    bull_probs = wf_df[wf_df['actual_regime'] == 1]['prob_bull']
    ax2.hist(bear_probs, bins=30, alpha=0.6, color='red', label='Actual Bear', density=True)
    ax2.hist(bull_probs, bins=30, alpha=0.6, color='green', label='Actual Bull', density=True)
    ax2.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
    ax2.set_xlabel('P(Bull)')
    ax2.set_ylabel('Density')
    ax2.set_title('Prediction Probability Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Rolling Accuracy
    ax3 = axes[1, 0]
    wf_df_temp = wf_df.copy()
    wf_df_temp['correct'] = (wf_df_temp['predicted_regime'] == wf_df_temp['actual_regime']).astype(int)
    rolling_acc = wf_df_temp['correct'].rolling(window=60, min_periods=30).mean()
    accuracy = wf_df_temp['correct'].mean()
    ax3.plot(rolling_acc.index, rolling_acc.values, color='blue', linewidth=1.5)
    ax3.axhline(0.5, color='red', linestyle='--', linewidth=1, label='Random (50%)')
    ax3.axhline(accuracy, color='green', linestyle='--', linewidth=1, label=f'Overall ({accuracy:.1%})')
    ax3.set_ylabel('5-Year Rolling Accuracy')
    ax3.set_title('Prediction Accuracy Over Time')
    ax3.set_ylim(0, 1)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Feature Importance
    ax4 = axes[1, 1]
    if feature_importances:
        fi_df = pd.DataFrame(feature_importances).set_index('date')
        avg_imp = fi_df.mean().sort_values(ascending=True)
        ax4.barh(range(len(avg_imp)), avg_imp.values, color='steelblue')
        ax4.set_yticks(range(len(avg_imp)))
        ax4.set_yticklabels(avg_imp.index)
        ax4.set_xlabel('Average Gain')
        ax4.set_title('Feature Importance (XGBoost)')
        ax4.grid(True, alpha=0.3, axis='x')
    
    # Plot 5: Predicted vs Actual
    ax5 = axes[2, 0]
    ax5.fill_between(wf_df.index, 0, wf_df['actual_regime'], 
                     alpha=0.4, color='blue', label='Actual', step='mid')
    ax5.plot(wf_df.index, wf_df['predicted_regime'], 
             color='red', linewidth=0.8, label='Predicted', alpha=0.8)
    ax5.set_ylabel('Regime')
    ax5.set_yticks([0, 1])
    ax5.set_yticklabels(['Bear', 'Bull'])
    ax5.set_title('Predicted vs Actual Regimes')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: P(Bull) Time Series
    ax6 = axes[2, 1]
    colors = wf_df['actual_regime'].map({0: 'red', 1: 'green'})
    ax6.scatter(wf_df.index, wf_df['prob_bull'], c=colors, s=10, alpha=0.6)
    ax6.axhline(0.5, color='black', linestyle='--', linewidth=1)
    ax6.set_ylabel('P(Bull)')
    ax6.set_xlabel('Date')
    ax6.set_title('Predicted P(Bull) by Actual Regime')
    ax6.set_ylim(0, 1)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


def plot_strategy_equity_curves(strategy_results: pd.DataFrame, 
                                strategies: dict,
                                market_returns: pd.Series = None,
                                save_path: str = None) -> plt.Figure:
    """
    Plot strategy equity curves and drawdowns.
    
    Parameters
    ----------
    strategy_results : pd.DataFrame
        Strategy returns
    strategies : dict
        Strategy name -> column mapping
    market_returns : pd.Series, optional
        Market benchmark returns
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, axes = plt.subplots(5, 1, figsize=(14, 20), sharex=True)
    
    # Pre-compute equity curves for reuse
    equity_curves = {}
    for name, col in strategies.items():
        if col in strategy_results.columns:
            equity_curves[name] = (1 + strategy_results[col]).cumprod()
    
    # Pre-compute market equity if available
    market_equity = None
    if market_returns is not None:
        market_norm = market_returns.copy()
        market_norm.index = market_norm.index.to_period('M').to_timestamp('M')
        common_idx = strategy_results.index.intersection(market_norm.index)
        market_subset = market_norm.loc[common_idx].dropna()
        market_equity = (1 + market_subset).cumprod()
    
    # Plot 1: Equity Curves (Linear Scale)
    ax1 = axes[0]
    for name, equity in equity_curves.items():
        ax1.plot(equity.index, equity.values, linewidth=1.5, label=name)
    
    if market_equity is not None:
        ax1.plot(market_equity.index, market_equity.values, 
                 linewidth=1.5, label='CRSP VW Index', color='red', linestyle='--')
    
    ax1.set_ylabel('Growth of $1')
    ax1.set_title('Equity Curves (Linear Scale)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Equity Curves (Log Scale)
    ax2 = axes[1]
    for name, equity in equity_curves.items():
        ax2.plot(equity.index, equity.values, linewidth=1.5, label=name)
    
    if market_equity is not None:
        ax2.plot(market_equity.index, market_equity.values, 
                 linewidth=1.5, label='CRSP VW Index', color='red', linestyle='--')
    
    ax2.set_ylabel('Growth of $1 (log)')
    ax2.set_title('Equity Curves (Log Scale)')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Plot 3: P(Bull) with Bear Shading (moved up from position 5)
    ax3 = axes[2]
    bear_mask = strategy_results['actual_regime'] == 0
    starts, ends = _get_regime_periods(bear_mask)
    for start, end in zip(starts, ends):
        ax3.axvspan(start, end, alpha=0.3, color='red')
    
    ax3.plot(strategy_results.index, strategy_results['prob_bull'], 
             color='blue', linewidth=1)
    ax3.axhline(0.5, color='black', linestyle='--', linewidth=1)
    ax3.set_ylabel('P(Bull)')
    ax3.set_title('P(Bull) with Bear Regimes (Red)')
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Drawdowns
    ax4 = axes[3]
    for name, col in strategies.items():
        if col in strategy_results.columns:
            cum_ret = (1 + strategy_results[col]).cumprod()
            rolling_max = cum_ret.expanding().max()
            drawdown = cum_ret / rolling_max - 1
            ax4.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, label=name)
    
    ax4.set_ylabel('Drawdown')
    ax4.set_title('Drawdown Comparison')
    ax4.legend(loc='lower left')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Leverage (P(Bull) Scaled Allocation)
    ax5 = axes[4]
    if 'allocation_prob_scaled' in strategy_results.columns:
        alloc = strategy_results['allocation_prob_scaled']
        ax5.plot(strategy_results.index, alloc, color='purple', linewidth=1, label='P(Bull) Scaled')
        ax5.axhline(1.0, color='black', linestyle='--', linewidth=1, label='No Leverage')
        # Dynamic y-axis based on actual leverage range
        y_min = max(0, alloc.min() - 0.1)
        y_max = alloc.max() + 0.1
        ax5.set_ylim(y_min, y_max)
    ax5.set_ylabel('Leverage (L)')
    ax5.set_xlabel('Date')
    ax5.set_title('Leverage Over Time (L > 1 = Borrowing, L < 1 = Defensive)')
    ax5.legend(loc='upper right')
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


def _get_regime_periods(bear_mask: pd.Series) -> tuple[list, list]:
    """
    Get start and end dates for regime periods.
    
    Parameters
    ----------
    bear_mask : pd.Series
        Boolean mask for bear regime
        
    Returns
    -------
    tuple[list, list]
        Lists of start and end dates
    """
    starts, ends = [], []
    in_bear = False
    for date, is_bear in bear_mask.items():
        if is_bear and not in_bear:
            starts.append(date)
            in_bear = True
        elif not is_bear and in_bear:
            ends.append(date)
            in_bear = False
    if in_bear:
        ends.append(bear_mask.index[-1])
    return starts, ends


def plot_gross_vs_net_equity(
    strategy_results: pd.DataFrame,
    strategy_results_net: pd.DataFrame,
    tx_cost_bps: int = 10,
    save_path: str = None
) -> plt.Figure:
    """
    Plot GROSS vs NET equity curves for HRP and P(Bull) Scaled strategies.
    
    Parameters
    ----------
    strategy_results : pd.DataFrame
        GROSS strategy results with 'hrp_return' and 'regime_prob_scaled_return'
    strategy_results_net : pd.DataFrame
        NET strategy results with 'hrp_return_net' and 'regime_prob_scaled_return_net'
    tx_cost_bps : int
        Transaction cost in basis points (for title)
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    # Compute cumulative returns
    cum_hrp_gross = (1 + strategy_results['hrp_return']).cumprod()
    cum_hrp_net = (1 + strategy_results_net['hrp_return_net']).cumprod()
    cum_scaled_gross = (1 + strategy_results['regime_prob_scaled_return']).cumprod()
    cum_scaled_net = (1 + strategy_results_net['regime_prob_scaled_return_net']).cumprod()

    # Create figure with 2 subplots (linear and log scale)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Linear Scale
    ax1 = axes[0]
    ax1.plot(cum_hrp_gross.index, cum_hrp_gross.values, label='HRP (GROSS)', 
             color='blue', linestyle='--', alpha=0.7, linewidth=0.8)
    ax1.plot(cum_hrp_net.index, cum_hrp_net.values, label='HRP (NET)', 
             color='blue', linewidth=1)
    ax1.plot(cum_scaled_gross.index, cum_scaled_gross.values, label='P(Bull) Scaled (GROSS)', 
             color='green', linestyle='--', alpha=0.7, linewidth=0.8)
    ax1.plot(cum_scaled_net.index, cum_scaled_net.values, label='P(Bull) Scaled (NET)', 
             color='green', linewidth=1)
    ax1.set_title(f'Equity Curves: GROSS vs NET ({tx_cost_bps} bps) - Linear Scale')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative Return ($1 invested)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Right: Log Scale
    ax2 = axes[1]
    ax2.plot(cum_hrp_gross.index, cum_hrp_gross.values, label='HRP (GROSS)', 
             color='blue', linestyle='--', alpha=0.7, linewidth=0.8)
    ax2.plot(cum_hrp_net.index, cum_hrp_net.values, label='HRP (NET)', 
             color='blue', linewidth=1)
    ax2.plot(cum_scaled_gross.index, cum_scaled_gross.values, label='P(Bull) Scaled (GROSS)', 
             color='green', linestyle='--', alpha=0.7, linewidth=0.8)
    ax2.plot(cum_scaled_net.index, cum_scaled_net.values, label='P(Bull) Scaled (NET)', 
             color='green', linewidth=1)
    ax2.set_yscale('log')
    ax2.set_title(f'Equity Curves: GROSS vs NET ({tx_cost_bps} bps) - Log Scale')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Cumulative Return (log scale)')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved GROSS vs NET equity plot to {save_path}")
    
    # Print terminal wealth
    print(f"\nTerminal Wealth ($1 invested):")
    print(f"  HRP GROSS:            ${cum_hrp_gross.iloc[-1]:,.2f}")
    print(f"  HRP NET:              ${cum_hrp_net.iloc[-1]:,.2f} (drag: ${cum_hrp_gross.iloc[-1] - cum_hrp_net.iloc[-1]:,.2f})")
    print(f"  P(Bull) Scaled GROSS: ${cum_scaled_gross.iloc[-1]:,.2f}")
    print(f"  P(Bull) Scaled NET:   ${cum_scaled_net.iloc[-1]:,.2f} (drag: ${cum_scaled_gross.iloc[-1] - cum_scaled_net.iloc[-1]:,.2f})")
    
    return fig
