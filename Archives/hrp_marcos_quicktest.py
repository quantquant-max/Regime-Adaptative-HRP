# CUDA-Accelerated HRP - Marcos Lopez de Prado's Algorithm
# Quick Test Version - 5 Random Years

import pandas as pd
import numpy as np
import os
import time
import random
from tqdm import tqdm
import scipy.cluster.hierarchy as sch
from sklearn.covariance import LedoitWolf

# Try to import GPU libraries
GPU_AVAILABLE = False
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("✓ GPU (CUDA) libraries loaded successfully")
    print(f"  CuPy version: {cp.__version__}")
except ImportError as e:
    print("⚠ GPU libraries not available, using CPU fallback")
    print(f"  Error: {e}")

print(f"\nMode: {'GPU (CUDA)' if GPU_AVAILABLE else 'CPU'}")

# ============================================================================
# HRP Functions - Marcos Lopez de Prado's Original Algorithm
# ============================================================================

def getIVP(cov):
    """Compute the inverse-variance portfolio"""
    ivp = 1. / np.diag(cov)
    ivp /= ivp.sum()
    return ivp

def getClusterVar(cov, cItems):
    """Compute variance per cluster - EXACT Marcos implementation"""
    cov_ = cov.loc[cItems, cItems]  # matrix slice
    w_ = getIVP(cov_).reshape(-1, 1)  # CRITICAL: reshape to column vector
    cVar = np.dot(np.dot(w_.T, cov_), w_)[0, 0]  # CRITICAL: [0,0] indexing
    return cVar

def getQuasiDiag(link):
    """Sort clustered items by distance - Marcos implementation"""
    link = link.astype(int)
    sortIx = pd.Series([link[-1, 0], link[-1, 1]])
    numItems = link[-1, 3]  # number of original items
    while sortIx.max() >= numItems:
        sortIx.index = range(0, sortIx.shape[0] * 2, 2)  # make space
        df0 = sortIx[sortIx >= numItems]  # find clusters
        i = df0.index
        j = df0.values - numItems
        sortIx[i] = link[j, 0]  # item 1
        df0 = pd.Series(link[j, 1], index=i + 1)
        sortIx = pd.concat([sortIx, df0])  # item 2
        sortIx = sortIx.sort_index()  # re-sort
        sortIx.index = range(sortIx.shape[0])  # re-index
    return sortIx.tolist()

def getRecBipart(cov, sortIx):
    """Compute HRP allocation - EXACT Marcos implementation"""
    w = pd.Series(1, index=sortIx)
    cItems = [sortIx]  # initialize all items in one cluster
    while len(cItems) > 0:
        # Bi-section - CRITICAL: use integer division //
        cItems = [i[j:k] for i in cItems for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
        # Parse in pairs
        for i in range(0, len(cItems), 2):
            cItems0 = cItems[i]  # cluster 1
            cItems1 = cItems[i + 1]  # cluster 2
            cVar0 = getClusterVar(cov, cItems0)
            cVar1 = getClusterVar(cov, cItems1)
            alpha = 1 - cVar0 / (cVar0 + cVar1)
            w[cItems0] *= alpha  # weight 1
            w[cItems1] *= 1 - alpha  # weight 2
    return w

def correlDist(corr):
    """A distance matrix based on correlation - Marcos implementation"""
    dist = ((1 - corr) / 2.) ** 0.5  # distance matrix
    return dist

def compute_covariance_gpu(returns_np, returns_gpu=None):
    """Compute shrunk covariance using GPU (with Ledoit-Wolf shrinkage)"""
    if GPU_AVAILABLE:
        if returns_gpu is None:
            returns_gpu = cp.asarray(returns_np)
        mean = cp.mean(returns_gpu, axis=0, keepdims=True)
        centered = returns_gpu - mean
        n_samples = returns_gpu.shape[0]
        cov_sample = (centered.T @ centered) / (n_samples - 1)
        mu = cp.trace(cov_sample) / cov_sample.shape[0]
        delta = cp.sum((cov_sample - mu * cp.eye(cov_sample.shape[0])) ** 2)
        X2 = centered ** 2
        gamma = cp.sum((X2.T @ X2) / n_samples - cov_sample ** 2)
        kappa = gamma / delta if delta > 0 else 1.0
        shrinkage = max(0.0, min(1.0, float(cp.asnumpy(kappa))))
        target = mu * cp.eye(cov_sample.shape[0])
        cov_shrunk_gpu = shrinkage * target + (1 - shrinkage) * cov_sample
        return cp.asnumpy(cov_shrunk_gpu), shrinkage
    else:
        lw = LedoitWolf().fit(returns_np)
        return lw.covariance_, lw.shrinkage_

print("✓ HRP functions defined (Marcos Lopez de Prado's algorithm)")

# ============================================================================
# Load Data and Select Random Years
# ============================================================================

data_path = r'DATA (CRSP)\PREPROCESSED DATA\ADA-HRP-Preprocessed-DATA.csv'
df = pd.read_csv(data_path)

stocks_df = df[df['PERMNO'].notna()].copy()
date_cols = [col for col in stocks_df.columns if col not in ['PERMNO', 'Company_Ticker']]
dates = pd.to_datetime(date_cols)
date_strs = [d.strftime('%Y-%m-%d') for d in dates]

# Select 5 random years
random.seed(42)
all_years = sorted(set(dates.year))
selected_years = sorted(random.sample(all_years, 5))

# Get quarterly rebalance dates (last day of Mar, Jun, Sep, Dec) for selected years
quarterly_rebalance_dates = []
for date in dates:
    if date.year in selected_years and date.month in [3, 6, 9, 12]:
        if date == dates[dates.to_period('M') == date.to_period('M')].max():
            quarterly_rebalance_dates.append(date)

quarterly_rebalance_dates = sorted(set(quarterly_rebalance_dates))

print(f"\nSelected years: {selected_years}")
print(f"Quarterly rebalance dates: {len(quarterly_rebalance_dates)}")
print(f"Total stocks in dataset: {len(stocks_df)}")

# ============================================================================
# Process Quarterly Rebalances
# ============================================================================

weights_list = []
timing_stats = {'cov': [], 'corr': [], 'dist': [], 'cluster': [], 'weights': [], 'total': []}
skipped_count = 0

for rebal_date in tqdm(quarterly_rebalance_dates, desc="Processing rebalance dates"):
    t_start = time.time()
    rebal_str = rebal_date.strftime('%Y-%m-%d')
    
    try:
        rebal_idx = date_strs.index(rebal_str)
    except ValueError:
        skipped_count += 1
        continue
    
    if rebal_idx < 11:
        skipped_count += 1
        continue
    
    window_indices = list(range(rebal_idx - 11, rebal_idx + 1))
    actual_window_cols = [date_cols[i] for i in window_indices]
    
    if len(actual_window_cols) != 12:
        skipped_count += 1
        continue
    
    window_df = stocks_df[['PERMNO', 'Company_Ticker'] + actual_window_cols].copy()
    window_df = window_df[window_df['Company_Ticker'].notna()]
    
    valid_mask = window_df[actual_window_cols].notna().sum(axis=1) == 12
    window_df = window_df[valid_mask]
    
    if len(window_df) < 20:
        skipped_count += 1
        continue
    
    # Prepare returns matrix
    returns = window_df[actual_window_cols].T  # Time x Assets
    returns.columns = window_df['PERMNO'].astype(str)
    
    # Filter out stocks with zero or near-zero variance
    stock_variance = returns.values.var(axis=0, ddof=1)
    min_variance = 1e-10
    valid_variance_mask = (stock_variance > min_variance) & np.isfinite(stock_variance)

    if valid_variance_mask.sum() < 2:
        skipped_count += 1
        continue

    # Filter returns to keep only valid stocks
    valid_permnos = returns.columns[valid_variance_mask].tolist()
    returns = returns[valid_permnos]
    returns_np = returns.values
    
    # === GPU-ACCELERATED COVARIANCE ===
    t0 = time.time()
    if GPU_AVAILABLE:
        returns_gpu = cp.asarray(returns_np)
        cov_array, shrinkage = compute_covariance_gpu(returns_np, returns_gpu)
    else:
        cov_array, shrinkage = compute_covariance_gpu(returns_np)
    timing_stats['cov'].append(time.time() - t0)
    
    # Create covariance DataFrame with PERMNO labels
    cov = pd.DataFrame(cov_array, index=returns.columns, columns=returns.columns)
    
    # === GPU-ACCELERATED CORRELATION ===
    t0 = time.time()
    if GPU_AVAILABLE:
        cov_gpu = cp.asarray(cov_array)
        std_gpu = cp.sqrt(cp.diag(cov_gpu))
        std_gpu = cp.where(std_gpu < 1e-10, 1e-10, std_gpu)
        corr_gpu = cov_gpu / cp.outer(std_gpu, std_gpu)
        corr_gpu = cp.clip(corr_gpu, -1.0, 1.0)
        corr_array = cp.asnumpy(corr_gpu)
    else:
        std = np.sqrt(np.diag(cov_array))
        std = np.where(std < 1e-10, 1e-10, std)
        corr_array = cov_array / np.outer(std, std)
        corr_array = np.clip(corr_array, -1.0, 1.0)
    timing_stats['corr'].append(time.time() - t0)
    
    # Create correlation DataFrame with PERMNO labels
    corr = pd.DataFrame(corr_array, index=returns.columns, columns=returns.columns)
    
    # === CORRELATION DISTANCE ===
    t0 = time.time()
    dist = correlDist(corr)
    timing_stats['dist'].append(time.time() - t0)
    
    # === CLUSTERING ===
    t0 = time.time()
    try:
        link = sch.linkage(dist, method='single')
        sortIx = getQuasiDiag(link)
        # CRITICAL: sortIx contains INTEGER indices, convert to PERMNO labels
        sortIx = corr.index[sortIx].tolist()
    except Exception as e:
        print(f"⚠ Clustering failed for {rebal_str}: {e}, skipping")
        skipped_count += 1
        continue
    timing_stats['cluster'].append(time.time() - t0)
    
    # === COMPUTE HRP WEIGHTS ===
    t0 = time.time()
    try:
        hrp_weights = getRecBipart(cov, sortIx)
        
        weight_sum = hrp_weights.sum()
        if abs(weight_sum - 1.0) > 1e-6:
            print(f"⚠ WARNING {rebal_str}: Weights sum to {weight_sum:.10f}, renormalizing...")
            hrp_weights = hrp_weights / weight_sum
        
    except Exception as e:
        print(f"⚠ Weight computation failed for {rebal_str}: {e}")
        import traceback
        traceback.print_exc()
        skipped_count += 1
        continue
    
    timing_stats['weights'].append(time.time() - t0)
    timing_stats['total'].append(time.time() - t_start)
    
    # Store weights
    weight_series = pd.Series(0.0, index=stocks_df['PERMNO'].astype(str))
    weight_series.update(hrp_weights)
    weights_list.append({
        'date': rebal_date,
        'weights': weight_series
    })

# ============================================================================
# Save Results
# ============================================================================

if len(weights_list) > 0:
    all_weights = stocks_df[['PERMNO', 'Company_Ticker']].copy()
    for w_dict in weights_list:
        col_name = w_dict['date'].strftime('%Y-%m-%d')
        all_weights[col_name] = w_dict['weights'].values
    
    # Save to CSV
    rolling_dir = os.path.join('Rolling Windows Test')
    os.makedirs(rolling_dir, exist_ok=True)
    output_path = os.path.join(rolling_dir, 'hrp_weights_quicktest_marcos.csv')
    all_weights.to_csv(output_path, index=False)
else:
    print("⚠ No weights computed!")
    all_weights = pd.DataFrame()

# Print timing statistics
print("\n" + "="*60)
print("QUICK TEST PERFORMANCE SUMMARY (Marcos Algorithm)")
print("="*60)
print(f"Mode: {'GPU (CUDA)' if GPU_AVAILABLE else 'CPU'}")
print(f"Selected years: {selected_years}")
print(f"Total quarterly dates processed: {len(weights_list)}")
print(f"Skipped (insufficient history): {skipped_count}")
if len(timing_stats['total']) > 0:
    print(f"\nAverage timing per rebalance:")
    print(f"  Covariance:      {np.mean(timing_stats['cov'])*1000:.2f} ms")
    print(f"  Correlation:     {np.mean(timing_stats['corr'])*1000:.2f} ms")
    print(f"  Distances:       {np.mean(timing_stats['dist'])*1000:.2f} ms")
    print(f"  Clustering:      {np.mean(timing_stats['cluster'])*1000:.2f} ms")
    print(f"  Weight Calc:     {np.mean(timing_stats['weights'])*1000:.2f} ms")
    print(f"  Total:           {np.mean(timing_stats['total'])*1000:.2f} ms")
    print(f"\nTotal runtime:    {np.sum(timing_stats['total']):.2f} seconds")
print("\n✓ Saved test weights to hrp_weights_quicktest_marcos.csv")

# ============================================================================
# Validate Results
# ============================================================================

print("\n" + "="*80)
print("VALIDATION: CHECKING FOR EQUAL WEIGHTS BUG")
print("="*80)

weights_file = os.path.join('Rolling Windows Test', 'hrp_weights_quicktest_marcos.csv')
df_weights = pd.read_csv(weights_file)

date_cols_check = [col for col in df_weights.columns if col not in ['PERMNO', 'Company_Ticker']]

equal_dates = []
proper_dates = []

for date_col in date_cols_check:
    weights = df_weights[date_col].dropna()
    if len(weights) == 0:
        continue
    
    equal_weight = 1.0 / len(weights)
    all_equal = np.allclose(weights.values, equal_weight, rtol=1e-10)
    
    if all_equal:
        equal_dates.append(date_col)
        print(f"❌ {date_col}: ALL EQUAL (N={len(weights)}, w={weights.iloc[0]:.10f})")
    else:
        proper_dates.append(date_col)
        print(f"✅ {date_col}: PROPER HRP (N={len(weights)}, min={weights.min():.10f}, max={weights.max():.10f}, ratio={weights.max()/weights.min():.2f}x)")

print(f"\n{'='*80}")
print(f"SUMMARY:")
print(f"  Equal weight dates: {len(equal_dates)}/{len(date_cols_check)}")
print(f"  Proper HRP dates:   {len(proper_dates)}/{len(date_cols_check)}")

if len(equal_dates) == 0:
    print(f"\n✅ ✅ ✅ SUCCESS! Bug is FIXED! All weights show proper HRP dispersion!")
elif len(equal_dates) == len(date_cols_check):
    print(f"\n❌ ❌ ❌ PROBLEM! Bug still present - all dates show equal weights.")
else:
    print(f"\n⚠ PARTIAL: {len(proper_dates)} dates fixed, {len(equal_dates)} still equal.")
