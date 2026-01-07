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


# =============================================================================
# TWO-STAGE TRANSACTION COST MODEL (Industry ETF Framework)
# =============================================================================

def calculate_within_industry_turnover(within_industry_weights: dict,
                                       stock_returns: pd.DataFrame,
                                       industry_returns: pd.DataFrame,
                                       dates: list) -> pd.DataFrame:
    """
    Calculate drift-adjusted turnover WITHIN each industry (Stage 1 cost).
    
    For each industry k:
        turnover_k,t = Σ_{i ∈ k} |w_i,t^target - w_i,t^drifted|
    
    Where w_i,t^drifted = w_i,t-1 × (1 + r_i,t) / (1 + r_k,t)
    (drift normalized by industry return, not portfolio return)
    
    Parameters
    ----------
    within_industry_weights : dict
        {date: {industry_id: {permno: weight}}} - VW weights within each industry
    stock_returns : pd.DataFrame
        Individual stock returns (dates × PERMNOs)
    industry_returns : pd.DataFrame
        Industry ETF returns (dates × industry names)
    dates : list
        Ordered list of dates to compute turnover for
        
    Returns
    -------
    pd.DataFrame
        Within-industry turnover per industry (dates × industries)
    """
    from hrp_data import FF12_INDUSTRY_NAMES
    
    # Map industry names back to IDs
    name_to_id = {v: k for k, v in FF12_INDUSTRY_NAMES.items()}
    
    # Initialize output
    turnover_df = pd.DataFrame(index=dates, columns=industry_returns.columns, dtype=float)
    
    for i, date_t in enumerate(dates):
        if i == 0:
            # First period: full investment (all stocks bought)
            if date_t in within_industry_weights:
                for ind_name in industry_returns.columns:
                    ind_id = name_to_id.get(ind_name)
                    if ind_id and ind_id in within_industry_weights[date_t]:
                        weights_t = within_industry_weights[date_t][ind_id]
                        turnover_df.loc[date_t, ind_name] = sum(abs(w) for w in weights_t.values())
            continue
        
        date_t_minus_1 = dates[i - 1]
        
        # Skip if missing data
        if date_t not in within_industry_weights or date_t_minus_1 not in within_industry_weights:
            continue
        
        for ind_name in industry_returns.columns:
            ind_id = name_to_id.get(ind_name)
            if not ind_id:
                continue
            
            # Get within-industry weights
            weights_prev = within_industry_weights.get(date_t_minus_1, {}).get(ind_id, {})
            weights_new = within_industry_weights.get(date_t, {}).get(ind_id, {})
            
            if not weights_prev and not weights_new:
                turnover_df.loc[date_t, ind_name] = 0.0
                continue
            
            # Get industry return for drift normalization
            r_industry = industry_returns.loc[date_t, ind_name] if date_t in industry_returns.index else 0.0
            if pd.isna(r_industry):
                r_industry = 0.0
            
            # Compute drifted weights
            all_permnos = set(weights_prev.keys()) | set(weights_new.keys())
            turnover = 0.0
            
            for permno in all_permnos:
                w_prev = weights_prev.get(permno, 0.0)
                w_new = weights_new.get(permno, 0.0)
                
                # Get stock return
                if permno in stock_returns.columns and date_t in stock_returns.index:
                    r_stock = stock_returns.loc[date_t, permno]
                    if pd.isna(r_stock):
                        r_stock = 0.0
                else:
                    r_stock = 0.0
                
                # Drifted weight (normalized by industry return)
                if abs(1 + r_industry) > 1e-8:
                    w_drifted = w_prev * (1 + r_stock) / (1 + r_industry)
                else:
                    w_drifted = w_prev
                
                turnover += abs(w_new - w_drifted)
            
            turnover_df.loc[date_t, ind_name] = turnover
    
    return turnover_df


def calculate_industry_hrp_turnover(industry_weights: pd.DataFrame,
                                    industry_returns: pd.DataFrame,
                                    portfolio_returns: pd.Series) -> pd.Series:
    """
    Calculate drift-adjusted turnover ACROSS industries (Stage 2 / HRP cost).
    
    turnover_t = Σ_k |Ω_k,t^target - Ω_k,t^drifted|
    
    Where Ω_k,t^drifted = Ω_k,t-1 × (1 + r_k,t) / (1 + r_portfolio,t)
    
    Parameters
    ----------
    industry_weights : pd.DataFrame
        HRP-optimized industry weights (dates × industries) - Ω_k,t
    industry_returns : pd.DataFrame
        Industry ETF returns (dates × industries)
    portfolio_returns : pd.Series
        Portfolio-level returns
        
    Returns
    -------
    pd.Series
        HRP rebalancing turnover across industries
    """
    turnover = pd.Series(index=industry_weights.index, dtype=float)
    
    # First period: full investment
    turnover.iloc[0] = industry_weights.iloc[0].fillna(0).sum()
    
    for i in range(1, len(industry_weights)):
        date_t = industry_weights.index[i]
        date_t_minus_1 = industry_weights.index[i - 1]
        
        # Target weights
        omega_prev = industry_weights.loc[date_t_minus_1].fillna(0)
        omega_new = industry_weights.loc[date_t].fillna(0)
        
        # Industry returns
        if date_t in industry_returns.index:
            r_industries = industry_returns.loc[date_t].reindex(omega_prev.index).fillna(0)
        else:
            r_industries = pd.Series(0, index=omega_prev.index)
        
        # Portfolio return
        if date_t in portfolio_returns.index:
            r_portfolio = portfolio_returns.loc[date_t]
        else:
            r_portfolio = 0
        
        # Drifted industry weights
        if abs(1 + r_portfolio) > 1e-8:
            omega_drifted = omega_prev * (1 + r_industries) / (1 + r_portfolio)
        else:
            omega_drifted = omega_prev
        
        # HRP turnover
        turnover.loc[date_t] = (omega_new - omega_drifted).abs().sum()
    
    return turnover


def compute_two_stage_costs(strategy_returns: pd.Series,
                            industry_weights: pd.DataFrame,
                            within_industry_weights: dict,
                            industry_returns: pd.DataFrame,
                            stock_returns: pd.DataFrame,
                            inner_cost_bps: int = 10,
                            outer_cost_bps: int = 10) -> dict:
    """
    Compute two-stage transaction costs for Industry ETF HRP framework.
    
    Stage 1 (Inner): Within-industry rebalancing
        r_k,t^net = r_k,t^gross - (within_industry_turnover_k,t) × C_inner
        
    Stage 2 (Outer): HRP rebalancing across industries
        r_portfolio,t = Σ_k (Ω_k,t-1 × r_k,t^net) - (hrp_turnover_t) × C_outer
    
    Parameters
    ----------
    strategy_returns : pd.Series
        Raw HRP strategy returns (gross)
    industry_weights : pd.DataFrame
        HRP-optimized industry weights (Ω_k,t)
    within_industry_weights : dict
        {date: {industry_id: {permno: vw_weight}}}
    industry_returns : pd.DataFrame
        Industry ETF returns (gross)
    stock_returns : pd.DataFrame
        Individual stock returns
    inner_cost_bps : int
        Transaction cost for within-industry rebalancing
    outer_cost_bps : int
        Transaction cost for HRP rebalancing
        
    Returns
    -------
    dict
        'inner_turnover': within-industry turnover (dates × industries)
        'outer_turnover': HRP turnover (dates)
        'industry_returns_net': net industry returns after inner cost
        'inner_cost_total': total inner cost per date
        'outer_cost': HRP rebalancing cost per date
        'hrp_return_net': net HRP return after two-stage costs
    """
    inner_cost = inner_cost_bps / 10000
    outer_cost = outer_cost_bps / 10000
    
    # Normalize dates to month-end
    strategy_returns_norm = strategy_returns.copy()
    strategy_returns_norm.index = strategy_returns_norm.index.to_period('M').to_timestamp('M')
    
    industry_weights_norm = industry_weights.copy()
    industry_weights_norm.index = industry_weights_norm.index.to_period('M').to_timestamp('M')
    
    industry_returns_norm = industry_returns.copy()
    industry_returns_norm.index = industry_returns_norm.index.to_period('M').to_timestamp('M')
    
    stock_returns_norm = stock_returns.copy()
    stock_returns_norm.index = stock_returns_norm.index.to_period('M').to_timestamp('M')
    
    # Normalize within_industry_weights dates
    within_weights_norm = {}
    for date, ind_dict in within_industry_weights.items():
        date_norm = pd.Timestamp(date).to_period('M').to_timestamp('M')
        within_weights_norm[date_norm] = ind_dict
    
    # Get common dates
    common_dates = sorted(set(strategy_returns_norm.index) & set(industry_weights_norm.index))
    
    # ==========================================================================
    # STAGE 1: Within-Industry Turnover (Inner Cost)
    # ==========================================================================
    inner_turnover = calculate_within_industry_turnover(
        within_weights_norm, stock_returns_norm, industry_returns_norm, common_dates
    )
    
    # Compute net industry returns: r_k,t^net = r_k,t^gross - turnover_k,t × C_inner
    industry_returns_net = industry_returns_norm.copy()
    for ind_name in industry_returns_norm.columns:
        if ind_name in inner_turnover.columns:
            industry_returns_net[ind_name] = (
                industry_returns_norm[ind_name] - inner_turnover[ind_name] * inner_cost
            )
    
    # Total inner cost (weighted by industry allocation)
    inner_cost_total = pd.Series(index=common_dates, dtype=float)
    for date in common_dates:
        if date in industry_weights_norm.index and date in inner_turnover.index:
            omega = industry_weights_norm.loc[date].fillna(0)
            turnover = inner_turnover.loc[date].fillna(0)
            # Inner cost = Σ_k (Ω_k × turnover_k × C_inner)
            inner_cost_total.loc[date] = (omega * turnover * inner_cost).sum()
        else:
            inner_cost_total.loc[date] = 0.0
    
    # ==========================================================================
    # STAGE 2: HRP Turnover Across Industries (Outer Cost)
    # ==========================================================================
    outer_turnover = calculate_industry_hrp_turnover(
        industry_weights_norm.loc[common_dates],
        industry_returns_norm,
        strategy_returns_norm
    )
    
    outer_cost_series = outer_turnover * outer_cost
    
    # ==========================================================================
    # NET HRP RETURN (after two-stage costs)
    # ==========================================================================
    # r_portfolio,t^net = Σ_k (Ω_k,t-1 × r_k,t^net) - outer_cost_t
    # But since strategy_returns already has the gross return computed,
    # we subtract both cost components:
    hrp_return_net = strategy_returns_norm.loc[common_dates] - inner_cost_total - outer_cost_series
    
    logger.info(f"Two-Stage Transaction Costs Computed:")
    logger.info(f"  Inner cost (within-industry): mean {inner_cost_total.mean()*12:.2%} ann")
    logger.info(f"  Outer cost (HRP rebalancing): mean {outer_cost_series.mean()*12:.2%} ann")
    
    return {
        'inner_turnover': inner_turnover,
        'outer_turnover': outer_turnover,
        'industry_returns_net': industry_returns_net,
        'inner_cost_total': inner_cost_total,
        'outer_cost': outer_cost_series,
        'hrp_return_net': hrp_return_net,
        'common_dates': common_dates
    }


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


def compute_net_returns_two_stage(strategy_results: pd.DataFrame,
                                   industry_weights: pd.DataFrame,
                                   within_industry_weights: dict,
                                   industry_returns: pd.DataFrame,
                                   stock_returns: pd.DataFrame,
                                   inner_cost_bps: int = 10,
                                   outer_cost_bps: int = 10) -> pd.DataFrame:
    """
    Apply TWO-STAGE transaction costs for Industry ETF HRP framework.
    
    This is the proper transaction cost model for the two-stage HRP approach:
    
    Stage 1 (Inner): Within-industry rebalancing cost
        For each sector k:
        r_k,t^net = r_k,t^gross - (Σ_{i∈k} |w_i,t^target - w_i,t^drifted|) × C_inner
        
    Stage 2 (Outer): HRP rebalancing cost across industries
        r_portfolio,t = Σ_k (Ω_k,t-1 × r_k,t^net) - (Σ_k |Ω_k,t^target - Ω_k,t^drifted|) × C_outer
    
    Leverage overlay costs (unchanged):
        - Leverage change cost: 2 × |L_t - L_{t-1}| × C_bps
        - Financing cost: max(L-1, 0) × spread (when leveraged)
    
    Parameters
    ----------
    strategy_results : pd.DataFrame
        Strategy results from compute_strategy_returns (GROSS returns)
    industry_weights : pd.DataFrame
        HRP-optimized industry weights (Ω_k,t)
    within_industry_weights : dict
        {date: {industry_id: {permno: vw_weight}}}
    industry_returns : pd.DataFrame
        Industry ETF returns (gross)
    stock_returns : pd.DataFrame
        Individual stock returns
    inner_cost_bps : int
        Transaction cost for within-industry rebalancing (default 10)
    outer_cost_bps : int
        Transaction cost for HRP rebalancing (default 10)
        
    Returns
    -------
    pd.DataFrame
        Strategy results with NET returns and two-stage cost breakdown
    """
    results = strategy_results.copy()
    inner_cost = inner_cost_bps / 10000
    outer_cost = outer_cost_bps / 10000
    
    # Get financing spread
    financing_spread_monthly = results['financing_spread'].iloc[0] if 'financing_spread' in results.columns else 0.0
    
    # ==========================================================================
    # COMPUTE TWO-STAGE COSTS
    # ==========================================================================
    two_stage = compute_two_stage_costs(
        strategy_returns=results['hrp_return'],
        industry_weights=industry_weights,
        within_industry_weights=within_industry_weights,
        industry_returns=industry_returns,
        stock_returns=stock_returns,
        inner_cost_bps=inner_cost_bps,
        outer_cost_bps=outer_cost_bps
    )
    
    # Store cost components
    results['inner_cost_stage1'] = two_stage['inner_cost_total'].reindex(results.index).fillna(0)
    results['outer_cost_stage2'] = two_stage['outer_cost'].reindex(results.index).fillna(0)
    
    # Combined HRP cost (for Buy & Hold)
    results['inner_cost'] = results['inner_cost_stage1'] + results['outer_cost_stage2']
    
    # ==========================================================================
    # NET RETURNS: Buy & Hold HRP
    # ==========================================================================
    results['outer_cost_buyhold'] = 0.0  # No leverage overlay
    results['financing_cost_buyhold'] = 0.0  # No financing
    results['hrp_return_net'] = (
        results['hrp_return'] 
        - results['inner_cost_stage1']  # Within-industry cost
        - results['outer_cost_stage2']  # HRP rebalancing cost
    )
    
    # ==========================================================================
    # NET RETURNS: Levered Strategies
    # Additional costs: leverage change + financing
    # ==========================================================================
    leverage_cost = outer_cost  # Use same cost for leverage overlay
    
    # Binary strategy
    w_binary = results['allocation_binary']
    alloc_change_binary = w_binary.diff().abs().fillna(0)
    results['outer_cost_binary'] = 2 * alloc_change_binary * leverage_cost
    results['financing_cost_binary'] = np.maximum(w_binary - 1, 0) * financing_spread_monthly
    results['regime_binary_return_net'] = (
        results['regime_binary_return']
        - w_binary * (results['inner_cost_stage1'] + results['outer_cost_stage2'])
        - results['outer_cost_binary']
        - results['financing_cost_binary']
    )
    
    # Probability-weighted strategy
    w_prob = results['allocation_prob']
    alloc_change_prob = w_prob.diff().abs().fillna(0)
    results['outer_cost_prob'] = 2 * alloc_change_prob * leverage_cost
    results['financing_cost_prob'] = np.maximum(w_prob - 1, 0) * financing_spread_monthly
    results['regime_prob_return_net'] = (
        results['regime_prob_return']
        - w_prob * (results['inner_cost_stage1'] + results['outer_cost_stage2'])
        - results['outer_cost_prob']
        - results['financing_cost_prob']
    )
    
    # Scaled probability-weighted strategy
    w_scaled = results['allocation_prob_scaled']
    alloc_change_scaled = w_scaled.diff().abs().fillna(0)
    results['outer_cost_scaled'] = 2 * alloc_change_scaled * leverage_cost
    results['financing_cost_scaled'] = np.maximum(w_scaled - 1, 0) * financing_spread_monthly
    results['regime_prob_scaled_return_net'] = (
        results['regime_prob_scaled_return']
        - w_scaled * (results['inner_cost_stage1'] + results['outer_cost_stage2'])
        - results['outer_cost_scaled']
        - results['financing_cost_scaled']
    )
    
    # Summary statistics
    logger.info(f"Two-Stage Transaction Costs Applied:")
    logger.info(f"  Stage 1 (Within-Industry): {inner_cost_bps} bps → {results['inner_cost_stage1'].mean()*12:.2%} ann")
    logger.info(f"  Stage 2 (HRP Rebalancing): {outer_cost_bps} bps → {results['outer_cost_stage2'].mean()*12:.2%} ann")
    logger.info(f"  Leverage Overlay (Scaled): {results['outer_cost_scaled'].mean()*12:.2%} ann")
    logger.info(f"  Financing Cost (Scaled): {results['financing_cost_scaled'].mean()*12:.2%} ann")
    
    return results


def print_tx_cost_summary_two_stage(results: pd.DataFrame, 
                                     inner_cost_bps: int = 10,
                                     outer_cost_bps: int = 10):
    """
    Print detailed two-stage transaction cost breakdown.
    """
    print("\n" + "="*80)
    print(f"TWO-STAGE TRANSACTION COST ANALYSIS")
    print("="*80)
    
    # Stage 1: Within-Industry
    inner_ann = results['inner_cost_stage1'].mean() * 12
    print(f"\n📊 STAGE 1: WITHIN-INDUSTRY REBALANCING ({inner_cost_bps} bps)")
    print(f"   Formula: inner_k,t = Σ_{'{i∈k}'} |w_i,t^target - w_i,t^drifted| × C_inner")
    print(f"   Annualized Cost: {inner_ann:.2%}")
    
    # Stage 2: HRP Rebalancing
    outer_ann = results['outer_cost_stage2'].mean() * 12
    print(f"\n📊 STAGE 2: HRP REBALANCING ACROSS INDUSTRIES ({outer_cost_bps} bps)")
    print(f"   Formula: outer_t = Σ_k |Ω_k,t^target - Ω_k,t^drifted| × C_outer")
    print(f"   Annualized Cost: {outer_ann:.2%}")
    
    # Total HRP Cost
    total_hrp_cost = inner_ann + outer_ann
    print(f"\n📊 TOTAL HRP COST (Stage 1 + Stage 2): {total_hrp_cost:.2%} ann")
    
    # Financing
    financing_spread_monthly = results['financing_spread'].iloc[0] if 'financing_spread' in results.columns else 0.0
    financing_spread_bps = financing_spread_monthly * 12 * 10000
    print(f"\n📊 FINANCING COST (Asymmetric Borrowing):")
    print(f"   Spread: {financing_spread_bps:.0f} bps/year when L > 1")
    
    # By strategy
    strategies = [
        ('Buy & Hold HRP', 'allocation_binary', 'outer_cost_buyhold', 'financing_cost_buyhold'),
        ('P(Bull) Scaled', 'allocation_prob_scaled', 'outer_cost_scaled', 'financing_cost_scaled'),
    ]
    
    print(f"\n📊 COST BREAKDOWN BY STRATEGY:")
    for name, alloc_col, outer_col, fin_col in strategies:
        w = results[alloc_col] if alloc_col != 'allocation_binary' or name == 'Buy & Hold HRP' else pd.Series(1.0, index=results.index)
        if name == 'Buy & Hold HRP':
            w = pd.Series(1.0, index=results.index)
        
        stage1_ann = (results['inner_cost_stage1'] * w).mean() * 12
        stage2_ann = (results['outer_cost_stage2'] * w).mean() * 12
        leverage_ann = results[outer_col].mean() * 12
        financing_ann = results[fin_col].mean() * 12
        total_ann = stage1_ann + stage2_ann + leverage_ann + financing_ann
        
        print(f"\n   {name}:")
        print(f"     Stage 1 (Within-Industry): {stage1_ann:.2%} ann")
        print(f"     Stage 2 (HRP Rebalancing): {stage2_ann:.2%} ann")
        print(f"     Leverage Overlay:          {leverage_ann:.2%} ann")
        print(f"     Financing Cost:            {financing_ann:.2%} ann")
        print(f"     TOTAL COST:                {total_ann:.2%} ann")


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
