"""
HRP Strategy Module - Strategy Execution and Performance Analysis
================================================================
Provides functions for:
- Transaction cost calculation
- Strategy return computation (Buy & Hold, Probability-weighted)
- Prediction quality metrics
"""

import numpy as np
import pandas as pd
from hrp_logger import setup_logger

logger = setup_logger()


def calculate_turnover_naive(weights: pd.DataFrame) -> pd.Series:
    """
    Calculate NAIVE turnover (target-to-target, ignores price drift).
    
    WARNING: This underestimates true turnover because it compares
    target weights at t vs t-1, ignoring the drift that occurs during
    the month due to differential stock returns.
    
    Parameters
    ----------
    weights : pd.DataFrame
        Portfolio weights over time (target weights)
        
    Returns
    -------
    pd.Series
        Monthly turnover (naive estimate)
    """
    weights_filled = weights.fillna(0)
    weight_changes = weights_filled.diff()
    monthly_turnover = weight_changes.abs().sum(axis=1)
    monthly_turnover.iloc[0] = weights_filled.iloc[0].sum()
    return monthly_turnover


def calculate_turnover_with_drift(weights: pd.DataFrame, 
                                   stock_returns: pd.DataFrame,
                                   portfolio_returns: pd.Series) -> pd.Series:
    """
    Calculate TRUE turnover accounting for price drift during the holding period.
    
    Turnover_t = Σ|w_{i,t}^{Target} - w_{i,t}^{Drifted}|
    
    Where: w_{i,t}^{Drifted} = w_{i,t-1}^{Target} × (1 + r_{i,t}) / (1 + r_{portfolio,t})
    
    This captures the actual trading required: from where the portfolio drifted
    TO the new target weights.
    
    Parameters
    ----------
    weights : pd.DataFrame
        Target portfolio weights over time (dates × assets)
    stock_returns : pd.DataFrame
        Individual stock returns (dates × assets)
    portfolio_returns : pd.Series
        Portfolio-level returns (dates)
        
    Returns
    -------
    pd.Series
        Monthly turnover accounting for drift
    """
    weights_filled = weights.fillna(0)
    turnover = pd.Series(index=weights.index, dtype=float)
    
    # First month: full investment from cash
    turnover.iloc[0] = weights_filled.iloc[0].sum()
    
    for i in range(1, len(weights)):
        date_t = weights.index[i]
        date_t_minus_1 = weights.index[i - 1]
        
        # Target weights at t-1
        w_target_prev = weights_filled.loc[date_t_minus_1]
        
        # Target weights at t  
        w_target_new = weights_filled.loc[date_t]
        
        # Get returns for the period (from t-1 to t)
        # Stock returns at date_t represent the return earned during month t
        if date_t in stock_returns.index:
            r_stocks = stock_returns.loc[date_t].reindex(w_target_prev.index).fillna(0)
        else:
            r_stocks = pd.Series(0, index=w_target_prev.index)
        
        if date_t in portfolio_returns.index:
            r_portfolio = portfolio_returns.loc[date_t]
        else:
            r_portfolio = 0
        
        # Compute drifted weights: how weights evolved due to price changes
        # w_drifted = w_prev × (1 + r_stock) / (1 + r_portfolio)
        if abs(1 + r_portfolio) > 1e-8:
            w_drifted = w_target_prev * (1 + r_stocks) / (1 + r_portfolio)
        else:
            w_drifted = w_target_prev
        
        # True turnover: distance from drifted weights to new target
        turnover.loc[date_t] = (w_target_new - w_drifted).abs().sum()
    
    return turnover


def compute_strategy_returns(strategy_returns: pd.Series,
                             predictions: pd.DataFrame,
                             hrp_weights: pd.DataFrame,
                             tx_cost_bps: int = 0,
                             bear_allocation: float = 0.5,
                             rf_rate: pd.Series = None,
                             stock_returns: pd.DataFrame = None,
                             financing_spread_bps: int = 50) -> pd.DataFrame:
    """
    Compute GROSS returns for all strategy variants (no transaction costs).
    
    Transaction costs should be applied separately using compute_net_returns()
    to properly separate inner (HRP turnover) and outer (leverage overlay) costs.
    
    Uses two-fund separation theorem: portfolio = w × HRP + (1-w) × T-Bill
    - When w < 1: Earn risk-free rate on cash portion (long T-Bills)
    - When w > 1: Pay r_f + spread on borrowed amount (short T-Bills)
    
    Parameters
    ----------
    strategy_returns : pd.Series
        Raw HRP strategy returns
    predictions : pd.DataFrame
        Walk-forward predictions (prob_bull, predicted_regime, actual_regime)
    hrp_weights : pd.DataFrame
        HRP portfolio weights
    tx_cost_bps : int
        Transaction cost in basis points (default 0 for gross returns)
    bear_allocation : float
        Allocation during Bear regime for binary strategy
    rf_rate : pd.Series, optional
        Monthly risk-free rate (1M T-Bill). If None, cash earns 0%.
    stock_returns : pd.DataFrame, optional
        Individual stock returns for drift-adjusted turnover calculation.
        If None, uses naive target-to-target turnover (underestimates true turnover).
    financing_spread_bps : int
        Borrowing spread over risk-free rate in basis points (default 50).
        Applied only when leverage > 1 (borrowing to buy more HRP).
        
    Returns
    -------
    pd.DataFrame
        Strategy results with GROSS return variants
    """
    # Normalize all dates to month-end to handle mismatched day-of-month
    # (HRP uses last trading day, predictions use calendar month-end)
    strategy_returns_norm = strategy_returns.copy()
    strategy_returns_norm.index = strategy_returns_norm.index.to_period('M').to_timestamp('M')
    
    predictions_norm = predictions.copy()
    predictions_norm.index = predictions_norm.index.to_period('M').to_timestamp('M')
    
    hrp_weights_norm = hrp_weights.copy()
    hrp_weights_norm.index = hrp_weights_norm.index.to_period('M').to_timestamp('M')
    
    # Normalize and align risk-free rate
    if rf_rate is not None:
        rf_norm = rf_rate.copy()
        rf_norm.index = rf_norm.index.to_period('M').to_timestamp('M')
    else:
        rf_norm = None
    
    # Normalize stock returns if provided
    if stock_returns is not None:
        stock_returns_norm = stock_returns.copy()
        stock_returns_norm.index = stock_returns_norm.index.to_period('M').to_timestamp('M')
    else:
        stock_returns_norm = None
    
    # Align dates using normalized indices
    common_dates = strategy_returns_norm.index.intersection(predictions_norm.index)
    
    # Calculate turnover (stored for later tx cost computation)
    # Use drift-adjusted turnover if stock returns available, otherwise naive
    if stock_returns_norm is not None:
        logger.info("Computing drift-adjusted turnover (accounts for price drift during holding period)")
        turnover = calculate_turnover_with_drift(
            hrp_weights_norm, 
            stock_returns_norm, 
            strategy_returns_norm
        )
    else:
        logger.warning("No stock returns provided - using naive turnover (underestimates true turnover)")
        turnover = calculate_turnover_naive(hrp_weights_norm)
    turnover_aligned = turnover.reindex(common_dates).fillna(turnover.mean())
    
    # Get aligned risk-free rate (0 if not provided)
    if rf_norm is not None:
        rf_aligned = rf_norm.reindex(common_dates).fillna(rf_norm.mean())
    else:
        rf_aligned = pd.Series(0.0, index=common_dates)
    
    # Financing spread for leverage (borrowing cost above risk-free rate)
    financing_spread = financing_spread_bps / 10000  # Convert to decimal (monthly)
    
    # Create results DataFrame
    results = pd.DataFrame(index=common_dates)
    results['hrp_turnover'] = turnover_aligned  # Store for tx cost computation
    results['rf_rate'] = rf_aligned
    results['hrp_return'] = strategy_returns_norm.loc[common_dates]  # GROSS HRP return
    results['predicted_regime'] = predictions_norm.loc[common_dates, 'predicted_regime']
    results['prob_bull'] = predictions_norm.loc[common_dates, 'prob_bull']
    results['actual_regime'] = predictions_norm.loc[common_dates, 'actual_regime']
    
    # ═══════════════════════════════════════════════════════════════════════════
    # GROSS RETURNS: No transaction costs, no financing spread
    # - When L < 1: Long T-Bills → Earn r_f on (1-L) cash
    # - When L > 1: Short T-Bills → Borrow at r_f (spread applied in NET)
    # 
    # Formula: r_gross = L × r_HRP + (1-L) × r_f  (regardless of L)
    # Note: (1-L) is negative when L>1, so we pay r_f on borrowed amount
    # The financing spread is applied in compute_net_returns()
    # ═══════════════════════════════════════════════════════════════════════════
    
    def compute_gross_return(w: pd.Series, r_hrp: pd.Series, r_f: pd.Series) -> pd.Series:
        """Compute GROSS return (no financing spread, no tx costs)."""
        # Simple two-fund separation: w × HRP + (1-w) × T-Bill
        # When w > 1: (1-w) is negative → paying r_f on borrowed amount
        return w * r_hrp + (1 - w) * r_f
    
    # Store financing spread for NET computation
    results['financing_spread'] = financing_spread / 12  # Monthly spread (already in decimal)
    
    # Strategy 2: Binary Regime Switch (GROSS - no costs)
    results['allocation_binary'] = np.where(
        results['predicted_regime'] == 0, bear_allocation, 1.0
    )
    w_binary = results['allocation_binary']
    results['regime_binary_return'] = compute_gross_return(
        w_binary, results['hrp_return'], rf_aligned
    )
    
    # Strategy 3: Probability-Weighted (GROSS - no costs)
    results['allocation_prob'] = results['prob_bull']
    w_prob = results['allocation_prob']
    results['regime_prob_return'] = compute_gross_return(
        w_prob, results['hrp_return'], rf_aligned
    )
    
    # Strategy 4: Scaled Probability-Weighted (GROSS - no costs)
    # Mean leverage = 1.0 for fair comparison
    mean_prob_bull = results['prob_bull'].mean()
    scale_factor = 1.0 / mean_prob_bull if mean_prob_bull > 0 else 1.0
    results['allocation_prob_scaled'] = results['prob_bull'] * scale_factor
    w_scaled = results['allocation_prob_scaled']
    results['regime_prob_scaled_return'] = compute_gross_return(
        w_scaled, results['hrp_return'], rf_aligned
    )
    results['prob_scale_factor'] = scale_factor
    
    # Track leverage statistics for reporting
    levered_months = (results['allocation_prob_scaled'] > 1).sum()
    avg_leverage_when_levered = results.loc[results['allocation_prob_scaled'] > 1, 'allocation_prob_scaled'].mean() if levered_months > 0 else 0
    
    logger.info(f"Strategy GROSS returns computed for {len(results)} months")
    logger.info(f"Prob-Weighted scale factor: {scale_factor:.3f}x (mean P(Bull)={mean_prob_bull:.3f})")
    logger.info(f"Mean Rf rate: {results['rf_rate'].mean()*12:.2%} annualized (over strategy period)")
    logger.info(f"Financing spread: {financing_spread_bps} bps (applied in NET only)")
    logger.info(f"Levered months: {levered_months}/{len(results)} ({100*levered_months/len(results):.1f}%), avg leverage when levered: {avg_leverage_when_levered:.2f}x")
    
    return results


def compute_net_returns(strategy_results: pd.DataFrame,
                        tx_cost_bps: int = 10) -> pd.DataFrame:
    """
    Apply transaction costs to strategy returns.
    
    Two-layer transaction cost model:
    
    1. Inner cost (HRP rebalancing): Cost of trading stocks to reach new target weights
       - Uses drift-adjusted turnover: |w_target - w_drifted|
       - Where w_drifted = w_prev × (1 + r_stock) / (1 + r_portfolio)
       - Scales with leverage: L × turnover × bps
       
    2. Outer cost (leverage overlay): Cost of changing allocation between HRP and T-Bills
       - Charges BOTH sides of the trade (HRP ↔ T-Bill)
       - outer_cost = 2 × |L_t - L_{t-1}| × C_bps
       
    3. Financing cost (asymmetric borrowing): Spread paid when leveraged > 1
       - When L > 1: pay additional spread on (L-1) borrowed amount
       - financing_cost = max(L-1, 0) × spread
    
    Parameters
    ----------
    strategy_results : pd.DataFrame
        Strategy results from compute_strategy_returns (GROSS returns)
    tx_cost_bps : int
        Transaction cost in basis points (default 10)
        
    Returns
    -------
    pd.DataFrame
        Strategy results with NET returns and cost breakdown
    """
    results = strategy_results.copy()
    tx_cost = tx_cost_bps / 10000
    
    # Get financing spread (stored in strategy_results from compute_strategy_returns)
    financing_spread_monthly = results['financing_spread'].iloc[0] if 'financing_spread' in results.columns else 0.0
    
    # ═══════════════════════════════════════════════════════════════════════════
    # INNER COST: HRP rebalancing (drift-adjusted turnover × cost)
    # Turnover = Σ|w_target,t - w_drifted,t| where drift accounts for price changes
    # ═══════════════════════════════════════════════════════════════════════════
    results['inner_cost'] = results['hrp_turnover'] * tx_cost
    
    # ═══════════════════════════════════════════════════════════════════════════
    # NET RETURNS = GROSS RETURNS - COSTS
    # 
    # Three cost components:
    # 1. Inner cost = turnover × bps × leverage (HRP rebalancing)
    # 2. Outer cost = 2 × |ΔL| × bps (trading HRP ↔ T-Bill)
    # 3. Financing cost = max(L-1, 0) × spread (borrowing cost when L > 1)
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Buy & Hold HRP: Inner cost only, no outer cost, no financing (L=1 always)
    results['outer_cost_buyhold'] = 0.0
    results['financing_cost_buyhold'] = 0.0
    results['hrp_return_net'] = results['hrp_return'] - results['inner_cost']
    
    # Binary strategy
    w_binary = results['allocation_binary']
    alloc_change_binary = w_binary.diff().abs().fillna(0)
    results['outer_cost_binary'] = 2 * alloc_change_binary * tx_cost
    results['financing_cost_binary'] = np.maximum(w_binary - 1, 0) * financing_spread_monthly
    results['regime_binary_return_net'] = (
        results['regime_binary_return'] 
        - w_binary * results['inner_cost']  # Inner cost scales with leverage
        - results['outer_cost_binary']
        - results['financing_cost_binary']  # Financing spread when levered
    )
    
    # Probability-weighted strategy
    w_prob = results['allocation_prob']
    alloc_change_prob = w_prob.diff().abs().fillna(0)
    results['outer_cost_prob'] = 2 * alloc_change_prob * tx_cost
    results['financing_cost_prob'] = np.maximum(w_prob - 1, 0) * financing_spread_monthly
    results['regime_prob_return_net'] = (
        results['regime_prob_return']
        - w_prob * results['inner_cost']
        - results['outer_cost_prob']
        - results['financing_cost_prob']
    )
    
    # Scaled probability-weighted strategy
    w_scaled = results['allocation_prob_scaled']
    alloc_change_scaled = w_scaled.diff().abs().fillna(0)
    results['outer_cost_scaled'] = 2 * alloc_change_scaled * tx_cost
    results['financing_cost_scaled'] = np.maximum(w_scaled - 1, 0) * financing_spread_monthly
    results['regime_prob_scaled_return_net'] = (
        results['regime_prob_scaled_return']
        - w_scaled * results['inner_cost']
        - results['outer_cost_scaled']
        - results['financing_cost_scaled']
    )
    
    # Summary statistics
    logger.info(f"Transaction costs applied ({tx_cost_bps} bps)")
    logger.info(f"  Inner cost (HRP turnover × L): {(results['inner_cost'] * w_scaled).mean()*12:.2%} annualized (scaled)")
    logger.info(f"  Outer cost (2 × |ΔL| × bps): {results['outer_cost_scaled'].mean()*12:.2%} annualized")
    logger.info(f"  Financing cost (spread when L>1): {results['financing_cost_scaled'].mean()*12:.2%} annualized")
    
    return results


def print_tx_cost_summary(results: pd.DataFrame, tx_cost_bps: int = 10):
    """
    Print detailed transaction cost breakdown.
    
    Parameters
    ----------
    results : pd.DataFrame
        Strategy results with NET returns from compute_net_returns()
    tx_cost_bps : int
        Transaction cost used
    """
    print("\n" + "="*80)
    print(f"TRANSACTION COST ANALYSIS ({tx_cost_bps} bps)")
    print("="*80)
    
    # Inner costs (HRP rebalancing)
    inner_cost_ann = results['inner_cost'].mean() * 12
    hrp_turnover_mean = results['hrp_turnover'].mean()
    print(f"\n📊 INNER COST (HRP Rebalancing - Drift-Adjusted Turnover):")
    print(f"   Formula: inner_cost = L × turnover × {tx_cost_bps}bps")
    print(f"   Where:   turnover = Σ|w_target,t - w_drifted,t|")
    print(f"   Mean Monthly Turnover: {hrp_turnover_mean:.1%}")
    print(f"   Base Annualized Inner Cost (L=1): {inner_cost_ann:.2%}")
    
    # Outer costs (leverage overlay)
    print(f"\n📊 OUTER COST (Leverage Overlay - Both Sides):")
    print(f"   Formula: outer_cost = 2 × |L_t - L_{{t-1}}| × {tx_cost_bps}bps")
    print(f"   Rationale: Changing leverage trades BOTH HRP portfolio AND T-Bills")
    
    # Financing costs
    financing_spread_monthly = results['financing_spread'].iloc[0] if 'financing_spread' in results.columns else 0.0
    financing_spread_bps = financing_spread_monthly * 12 * 10000  # Convert to annual bps
    print(f"\n📊 FINANCING COST (Asymmetric Borrowing Spread):")
    print(f"   Formula: financing_cost = max(L-1, 0) × {financing_spread_bps:.0f}bps/year")
    print(f"   Rationale: Pay spread over r_f when leveraged (L > 1)")
    
    strategies = [
        ('Buy & Hold HRP', 'outer_cost_buyhold', 'financing_cost_buyhold', 'allocation_binary'),
        ('P(Bull) Scaled', 'outer_cost_scaled', 'financing_cost_scaled', 'allocation_prob_scaled'),
    ]
    
    print(f"\n📊 COST BREAKDOWN BY STRATEGY:")
    for name, outer_col, fin_col, alloc_col in strategies:
        w = results[alloc_col]
        inner_scaled_ann = (results['inner_cost'] * w).mean() * 12
        outer_ann = results[outer_col].mean() * 12
        financing_ann = results[fin_col].mean() * 12 if fin_col in results.columns else 0.0
        total_cost_ann = inner_scaled_ann + outer_ann + financing_ann
        alloc_change = w.diff().abs().mean()
        levered_pct = (w > 1).mean()
        
        print(f"\n   {name}:")
        print(f"     Mean Allocation: {w.mean():.1%}, % Time Leveraged: {levered_pct:.1%}")
        print(f"     Inner Cost (L × turnover):  {inner_scaled_ann:.2%} ann")
        print(f"     Outer Cost (2× |ΔL|):       {outer_ann:.2%} ann")
        print(f"     Financing Cost:             {financing_ann:.2%} ann")
        print(f"     TOTAL COST:                 {total_cost_ann:.2%} ann")
    
    # Total impact
    print(f"\n📊 GROSS vs NET PERFORMANCE:")
    print(f"   {'Strategy':<20} {'Gross':>10} {'Net':>10} {'Drag':>10}")
    print(f"   {'-'*50}")
    
    gross_hrp = results['hrp_return'].mean() * 12
    net_hrp = results['hrp_return_net'].mean() * 12
    print(f"   {'Buy & Hold HRP':<20} {gross_hrp:>9.1%} {net_hrp:>9.1%} {gross_hrp - net_hrp:>9.2%}")
    
    gross_scaled = results['regime_prob_scaled_return'].mean() * 12
    net_scaled = results['regime_prob_scaled_return_net'].mean() * 12
    print(f"   {'P(Bull) Scaled':<20} {gross_scaled:>9.1%} {net_scaled:>9.1%} {gross_scaled - net_scaled:>9.2%}")


def compute_prediction_quality(strategy_results: pd.DataFrame) -> dict:
    """
    Compute prediction quality metrics.
    
    Parameters
    ----------
    strategy_results : pd.DataFrame
        Strategy results DataFrame
        
    Returns
    -------
    dict
        Prediction quality metrics
    """
    actual_bear = strategy_results['actual_regime'] == 0
    predicted_bear = strategy_results['predicted_regime'] == 0
    actual_bull = strategy_results['actual_regime'] == 1
    predicted_bull = strategy_results['predicted_regime'] == 1
    
    # Bear detection
    bear_detected = (predicted_bear & actual_bear).sum() / actual_bear.sum() if actual_bear.sum() > 0 else 0
    
    # False bear rate
    false_bear = (predicted_bear & actual_bull).sum() / actual_bull.sum() if actual_bull.sum() > 0 else 0
    
    return {
        'bear_detection_rate': bear_detected,
        'false_bear_rate': false_bear,
        'actual_bear_months': actual_bear.sum(),
        'predicted_bear_months': predicted_bear.sum(),
        'correctly_predicted_bear': (predicted_bear & actual_bear).sum()
    }
