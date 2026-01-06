# hrp_functions.py
# GPU-Accelerated Hierarchical Risk Parity (HRP) weight computation
# Optimized for CUDA 12.8+ (RTX 5090)
# Main function: compute_hrp_weights(returns_df)
# Batch function: compute_hrp_weights_batch(returns_dict)
# Input: pandas DataFrame where rows are time periods, columns are assets (e.g., PERMNO or tickers).
# Output: pd.Series of weights, indexed by asset names.

import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as sch
from sklearn.covariance import LedoitWolf
import time
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Import centralized seed management
try:
    from hrp_setup import get_random_state
except ImportError:
    def get_random_state():
        return 42

# Import RMT (Random Matrix Theory Denoising)
try:
    # Try importing your new local script
    from cov_shrinkage import compute_rmt
    print("✓ SOTA Covariance (RMT Denoising) loaded successfully.")
except ImportError:
    # Fallback only if file is missing
    from sklearn.covariance import LedoitWolf
    print("[!] cov_shrinkage.py not found. Using Linear Shrinkage (Inferior).")
    
    def compute_rmt(X):
        lw = LedoitWolf()
        return lw.fit(X).covariance_

# Initialize GPU_AVAILABLE at module level
GPU_AVAILABLE = False

try:
    import cupy as cp
    # Test that CuPy actually works (not just imports)
    _ = cp.array([1.0])
    GPU_AVAILABLE = True
    # Don't print here - let the notebook handle GPU status messages
except (ImportError, Exception):
    cp = np  # Fallback to NumPy if CuPy not available
    GPU_AVAILABLE = False
    # Don't print here - let the notebook handle GPU status messages

def get_correlation_distance_gpu(corr_gpu):
    """Compute correlation distance matrix on GPU"""
    corr_gpu = cp.clip(corr_gpu, -1.0, 1.0)
    dist = cp.sqrt(cp.clip((1 - corr_gpu) / 2.0, 0.0, None))
    dist = cp.nan_to_num(dist, nan=0.5, posinf=0.5, neginf=0.5)
    return dist

def get_euclidean_distance_gpu(dist_gpu):
    """Compute pairwise Euclidean distances on GPU with optimized memory usage"""
    n = dist_gpu.shape[0]
    
    # Compute squared norms efficiently
    squared_norms = cp.sum(dist_gpu ** 2, axis=1, keepdims=True)
    
    # Compute Euclidean distance matrix
    # d(i,j)^2 = ||x_i||^2 + ||x_j||^2 - 2*<x_i, x_j>
    eucl_dist_sq = squared_norms + squared_norms.T - 2.0 * cp.dot(dist_gpu, dist_gpu.T)
    eucl_dist = cp.sqrt(cp.clip(eucl_dist_sq, 0.0, None))
    
    # Handle numerical issues
    eucl_dist = cp.nan_to_num(eucl_dist, nan=1e-4, posinf=1e-4, neginf=1e-4)
    
    return eucl_dist

def compute_covariance_gpu(returns_np, use_gpu=True):
    """
    Compute shrunk covariance using RMT Denoising on CPU,
    then transfer to GPU if available.
    
    Returns:
        cov_array (np.ndarray): Covariance matrix on CPU
        shrinkage (float): Shrinkage coefficient (Dummy 0.0 for RMT)
        cov_gpu (cp.ndarray or None): Covariance matrix on GPU if available
    """
    # RMT Denoising (Random Matrix Theory)
    try:
        cov_array = compute_rmt(returns_np)
        shrinkage = 0.0 # RMT is analytical, no single shrinkage coefficient
    except Exception as e:
        print(f"[!] RMT Failed ({e}). Fallback to Linear Shrinkage.")
        lw = LedoitWolf()
        lw.fit(returns_np)
        cov_array = lw.covariance_
        shrinkage = lw.shrinkage_
    
    # Transfer to GPU if available
    if GPU_AVAILABLE and use_gpu:
        cov_gpu = cp.asarray(cov_array, dtype=cp.float32)  # Use float32 for faster GPU ops
        return cov_array, shrinkage, cov_gpu
    else:
        return cov_array, shrinkage, None

def get_quasi_diag(link):
    """
    Seriation from hierarchical clustering linkage.
    Reorders assets based on hierarchical clustering tree structure.
    """
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df0])
        sort_ix = sort_ix.sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    
    return sort_ix.tolist()

def perform_clustering(eucl_dist_condensed, method='ward'):
    """
    Hierarchical clustering using SciPy.
    Default is 'ward' linkage (minimized variance), which often yields 
    more balanced clusters than 'single' linkage.
    """
    return sch.linkage(eucl_dist_condensed, method=method)

def get_cluster_var(cov, c_items):
    """
    Compute cluster variance using inverse-variance portfolio.
    
    Args:
        cov (pd.DataFrame): Covariance matrix
        c_items (list): List of asset names in cluster
    
    Returns:
        float: Cluster variance
    """
    cov_ = cov.loc[c_items, c_items].values
    
    # Single asset cluster
    if len(c_items) == 1:
        return cov_[0, 0]
    
    # Inverse-variance weighting
    variances = np.diag(cov_)
    variances = np.maximum(variances, 1e-10)  # Avoid division by zero
    
    ivp = 1.0 / variances
    ivp /= ivp.sum()
    
    # Compute weighted variance
    w_ = ivp.reshape(-1, 1)
    cVar = (w_.T @ cov_ @ w_)[0, 0]
    
    return cVar

def get_recursive_bisection(cov, sort_ix, debug=False):
    """
    Recursive bisection for HRP weight allocation.
    
    Args:
        cov (pd.DataFrame): Covariance matrix
        sort_ix (list): Ordered list of asset names from clustering
        debug (bool): Print debug information
    
    Returns:
        pd.Series: HRP weights indexed by asset names
    """
    w = pd.Series(1.0, index=sort_ix)
    c_items = [sort_ix]
    
    iteration = 0
    while len(c_items) > 0:
        # Split each cluster into two sub-clusters
        c_items = [i[j:k] for i in c_items 
                   for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) 
                   if len(i) > 1]
        
        if len(c_items) == 0:
            break
        
        # Allocate weight between adjacent clusters
        for i in range(0, len(c_items), 2):
            if i + 1 >= len(c_items):
                break
            
            c_items0 = c_items[i]
            c_items1 = c_items[i + 1]
            
            # Compute cluster variances
            c_var0 = get_cluster_var(cov, c_items0)
            c_var1 = get_cluster_var(cov, c_items1)
            
            # Allocate inversely proportional to variance
            alpha = 1.0 - c_var0 / (c_var0 + c_var1)
            
            w[c_items0] *= alpha
            w[c_items1] *= (1.0 - alpha)
        
        iteration += 1
    
    # Normalize weights
    w = w / w.sum()
    
    return w

def compute_hrp_weights(returns_df, variance_window=None, debug=False, use_gpu=True):
    """
    Main function to compute HRP weights with GPU acceleration.
    
    Args:
        returns_df (pd.DataFrame): Returns data (rows: dates, columns: assets). 
                                   Used for Correlation/Clustering (should be long window, e.g. 60m).
        variance_window (int, optional): Number of recent observations to use for Variance/Allocation.
                                         If None, uses the full returns_df.
        debug (bool): Print debug information
        use_gpu (bool): Use GPU acceleration if available
    
    Returns:
        pd.Series: HRP weights indexed by asset names
    
    Raises:
        ValueError: If input is invalid or insufficient valid stocks
        RuntimeError: If clustering fails
    """
    if not isinstance(returns_df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame with returns.")

    returns_np = returns_df.values
    asset_names = returns_df.columns.tolist()

    # Filter valid stocks (non-zero variance, no NaN)
    stock_variance = np.var(returns_np, axis=0, ddof=1)
    min_variance = 1e-10
    valid_variance_mask = (stock_variance > min_variance) & np.isfinite(stock_variance)
    
    if valid_variance_mask.sum() < 2:
        raise ValueError(f"Insufficient valid stocks for HRP computation (found {valid_variance_mask.sum()}, need ≥2).")
    
    valid_assets = [asset_names[i] for i in range(len(asset_names)) if valid_variance_mask[i]]
    returns_np = returns_np[:, valid_variance_mask]

    # Step 1: Compute covariance matrix for CORRELATION (Clustering)
    # Uses the full provided history (e.g., 60 months)
    # Use RMT Denoising (Random Matrix Theory)
    try:
        cov_corr_array = compute_rmt(returns_np)
    except Exception as e:
        raise RuntimeError(f"RMT Covariance Estimation failed for input shape {returns_np.shape}. Error: {str(e)}")
        
    shrinkage = 0.0 # RMT doesn't return a single shrinkage coefficient
    if GPU_AVAILABLE and use_gpu:
        cov_gpu = cp.asarray(cov_corr_array, dtype=cp.float32)
    else:
        cov_gpu = None
    
    # Step 2: Compute correlation matrix (GPU-accelerated if available)
    if GPU_AVAILABLE and use_gpu and cov_gpu is not None:
        # GPU path
        std_gpu = cp.sqrt(cp.diag(cov_gpu))
        std_gpu = cp.where(std_gpu < 1e-10, 1e-10, std_gpu)
        corr_gpu = cov_gpu / cp.outer(std_gpu, std_gpu)
        corr_gpu = cp.clip(corr_gpu, -1.0, 1.0)
        corr_array = cp.asnumpy(corr_gpu)
        
        # Step 3: Compute distance matrices on GPU
        dist_gpu = get_correlation_distance_gpu(corr_gpu)
        eucl_dist_gpu = get_euclidean_distance_gpu(dist_gpu)
        eucl_dist_np = cp.asnumpy(eucl_dist_gpu)
    else:
        # CPU path
        std = np.sqrt(np.diag(cov_corr_array))
        std = np.where(std < 1e-10, 1e-10, std)
        corr_array = cov_corr_array / np.outer(std, std)
        corr_array = np.clip(corr_array, -1.0, 1.0)
        
        # Compute distances on CPU
        dist_np = np.sqrt(np.clip((1 - corr_array) / 2.0, 0.0, None))
        dist_np = np.nan_to_num(dist_np, nan=0.0, posinf=0.0, neginf=0.0)
        
        n = dist_np.shape[0]
        squared_norms = np.sum(dist_np ** 2, axis=1, keepdims=True)
        eucl_dist_np = np.sqrt(np.clip(squared_norms + squared_norms.T - 2 * np.dot(dist_np, dist_np.T), 0.0, None))
        eucl_dist_np = np.nan_to_num(eucl_dist_np, nan=1e-8, posinf=1e-8, neginf=1e-8)

    # Step 4: Convert to condensed distance matrix for scipy
    from scipy.spatial.distance import squareform
    eucl_dist_condensed = squareform(eucl_dist_np, checks=False)
    eucl_dist_condensed = np.nan_to_num(eucl_dist_condensed, nan=1e-8, posinf=1e-8, neginf=1e-8)

    # Step 5: Hierarchical clustering (CPU only - scipy)
    # Using 'single' linkage (nearest neighbor) - better suited for industry ETF framework
    try:
        link = perform_clustering(eucl_dist_condensed, method='single')
    except Exception as e:
        raise RuntimeError(f"Clustering failed: {e}")

    # Step 6: Quasi-diagonalization (seriation)
    sort_ix = get_quasi_diag(link)
    sort_ix = [valid_assets[i] for i in sort_ix]

    # Step 7: Recursive Bisection (Allocation)
    # Determine which covariance matrix to use for allocation
    if variance_window is not None and variance_window < len(returns_df):
        # Use only the most recent 'variance_window' observations for allocation variance
        returns_var_np = returns_np[-variance_window:, :]
        cov_var_array, _, _ = compute_covariance_gpu(returns_var_np, use_gpu=False) # CPU sufficient for allocation cov
        cov_allocation = pd.DataFrame(cov_var_array, index=valid_assets, columns=valid_assets)
    else:
        # Use the same covariance matrix as clustering
        cov_allocation = pd.DataFrame(cov_corr_array, index=valid_assets, columns=valid_assets)

    hrp_weights = get_recursive_bisection(cov_allocation, sort_ix, debug=debug)
    
    # Final normalization check
    weight_sum = hrp_weights.sum()
    if abs(weight_sum - 1.0) > 1e-6:
        hrp_weights /= weight_sum

    return hrp_weights


def compute_hrp_weights_batch(returns_dict, debug=False, use_gpu=True, max_batch_size=5):
    """
    Batch computation of HRP weights for multiple windows/dates.
    Optimized for GPU with memory-efficient processing.
    
    Args:
        returns_dict (dict): Dictionary of {key: returns_df} where key is (date, window) or similar
        debug (bool): Print debug information
        use_gpu (bool): Use GPU acceleration if available
        max_batch_size (int): Maximum number of portfolios to process in GPU memory at once
    
    Returns:
        dict: Dictionary of {key: weights_series}
    
    Example:
        returns_dict = {
            (date1, 6): returns_df_6m,
            (date1, 12): returns_df_12m,
            ...
        }
        weights_dict = compute_hrp_weights_batch(returns_dict)
    """
    results = {}
    keys = list(returns_dict.keys())
    
    print(f"Processing {len(keys)} HRP computations in batches of {max_batch_size}...")
    
    # Process in batches to avoid GPU memory overflow
    for batch_start in range(0, len(keys), max_batch_size):
        batch_end = min(batch_start + max_batch_size, len(keys))
        batch_keys = keys[batch_start:batch_end]
        
        if debug:
            print(f"  Batch {batch_start//max_batch_size + 1}: Processing {len(batch_keys)} portfolios...")
        
        # Process each item in batch
        for key in batch_keys:
            returns_df = returns_dict[key]
            try:
                weights = compute_hrp_weights(returns_df, debug=False, use_gpu=use_gpu)
                results[key] = weights
            except (ValueError, RuntimeError) as e:
                if debug:
                    print(f"    Skipped {key}: {str(e)[:60]}")
                results[key] = None  # Mark as failed
        
        # Clear GPU memory between batches
        if GPU_AVAILABLE and use_gpu:
            cp.get_default_memory_pool().free_all_blocks()
    
    # Filter out failed computations
    results = {k: v for k, v in results.items() if v is not None}
    
    print(f"✓ Successfully computed {len(results)}/{len(keys)} portfolios")
    
    return results