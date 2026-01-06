import numpy as np

# GPU Support - try to import CuPy for CUDA acceleration
GPU_AVAILABLE = False
try:
    import cupy as cp
    # Test that CuPy actually works
    _ = cp.array([1.0])
    GPU_AVAILABLE = True
except (ImportError, Exception):
    cp = None
    GPU_AVAILABLE = False


def compute_rmt_shrinkage(X, use_gpu=True):
    """
    Robust RMT Denoising (Constant Residual Eigenvalues).
    GPU-accelerated with automatic fallback to CPU.
    
    This method uses Random Matrix Theory (RMT) to filter the correlation matrix.
    It identifies "signal" eigenvalues (those exceeding the Marchenko-Pastur threshold)
    and "noise" eigenvalues (those below). The noise eigenvalues are replaced by their
    average value, preserving the trace (sum of variances).
    
    Methodology:
    1. Decompose Covariance into Correlation and Volatility.
    2. Compute Eigenvalues of Correlation Matrix (GPU if available).
    3. Determine Marchenko-Pastur Threshold (lambda_max).
    4. Separate Signal (eVal > lambda_max) from Noise.
    5. Replace Noise Eigenvalues with their mean.
    6. Reconstruct Correlation and then Covariance.
    
    Args:
        X (np.ndarray): Raw returns matrix of shape (T_samples, N_assets).
                        Rows = Time, Columns = Stocks.
        use_gpu (bool): Whether to use GPU acceleration if available.
    
    Returns:
        np.ndarray: Shrunk covariance matrix (N, N).
    """
    T, N = X.shape
    
    # Determine whether to use GPU
    use_cuda = GPU_AVAILABLE and use_gpu and cp is not None
    
    if use_cuda:
        return _compute_rmt_gpu(X, T, N)
    else:
        return _compute_rmt_cpu(X, T, N)


def _compute_rmt_cpu(X, T, N):
    """CPU implementation of RMT denoising using NumPy."""
    # 1. Decompose: Covariance -> Correlation + Volatility
    std = np.std(X, axis=0, ddof=1)
    std = np.maximum(std, 1e-10)
    
    # Standardize data
    X_centered = X - np.mean(X, axis=0)
    X_norm = X_centered / std
    
    # Compute Empirical Correlation Matrix
    corr = np.dot(X_norm.T, X_norm) / (T - 1)
    
    # 2. Eigen Decomposition (CPU)
    eVal, eVec = np.linalg.eigh(corr)
    
    # Sort descending
    idx = eVal.argsort()[::-1]
    eVal = eVal[idx]
    eVec = eVec[:, idx]
    
    # 3. Marchenko-Pastur threshold
    # For standardized data, σ² = 1 is the theoretical value
    # When N > T, many eigenvalues are ~0 due to rank deficiency
    # Using median(eVal) fails when N >> T, so we use σ² = 1
    q = N / T  # Aspect ratio
    sigma_sq = 1.0  # Theoretical variance for standardized data
    lambda_max = sigma_sq * (1 + np.sqrt(q))**2
    lambda_min = sigma_sq * (1 - np.sqrt(q))**2 if q < 1 else 0
    
    # 4. Filter: Signal vs Noise
    # Eigenvalues exceeding lambda_max are "signal"
    nFacts = max(np.sum(eVal > lambda_max), 1)
    
    # 5. Denoising: Replace noise eigenvalues with their mean
    eVal_denoised = eVal.copy()
    # Only consider non-zero eigenvalues for noise mean (rank = min(T-1, N))
    effective_rank = min(T - 1, N)
    noise_evals = eVal[nFacts:effective_rank]
    if len(noise_evals) > 0:
        noise_mean = np.mean(noise_evals)
    else:
        noise_mean = lambda_min  # Fallback to MP lower bound
    eVal_denoised[nFacts:] = noise_mean
    
    # 6. Reconstruct Clean Correlation
    corr_denoised = np.dot(eVec, np.dot(np.diag(eVal_denoised), eVec.T))
    np.fill_diagonal(corr_denoised, 1.0)
    
    # 7. Restore Clean Covariance
    cov_denoised = corr_denoised * np.outer(std, std)
    
    return cov_denoised


def _compute_rmt_gpu(X, T, N):
    """GPU-accelerated implementation of RMT denoising using CuPy."""
    # Transfer to GPU
    X_gpu = cp.asarray(X, dtype=cp.float64)
    
    # 1. Decompose: Covariance -> Correlation + Volatility
    std_gpu = cp.std(X_gpu, axis=0, ddof=1)
    std_gpu = cp.maximum(std_gpu, 1e-10)
    
    # Standardize data
    X_centered = X_gpu - cp.mean(X_gpu, axis=0)
    X_norm = X_centered / std_gpu
    
    # Compute Empirical Correlation Matrix
    corr_gpu = cp.dot(X_norm.T, X_norm) / (T - 1)
    
    # 2. Eigen Decomposition (GPU-accelerated)
    eVal_gpu, eVec_gpu = cp.linalg.eigh(corr_gpu)
    
    # Sort descending
    idx = cp.argsort(eVal_gpu)[::-1]
    eVal_gpu = eVal_gpu[idx]
    eVec_gpu = eVec_gpu[:, idx]
    
    # 3. Marchenko-Pastur threshold
    # For standardized data, σ² = 1 is the theoretical value
    # When N > T, many eigenvalues are ~0 due to rank deficiency
    q = N / T  # Aspect ratio
    sigma_sq = 1.0  # Theoretical variance for standardized data
    lambda_max = sigma_sq * (1 + cp.sqrt(q))**2
    lambda_min = sigma_sq * (1 - cp.sqrt(q))**2 if q < 1 else 0
    
    # 4. Filter: Signal vs Noise
    nFacts = int(max(cp.sum(eVal_gpu > lambda_max).get(), 1))
    
    # 5. Denoising: Replace noise eigenvalues with their mean
    eVal_denoised = eVal_gpu.copy()
    effective_rank = min(T - 1, N)
    noise_evals = eVal_gpu[nFacts:effective_rank]
    if len(noise_evals) > 0:
        noise_mean = cp.mean(noise_evals)
    else:
        noise_mean = lambda_min
    eVal_denoised[nFacts:] = noise_mean
    
    # 6. Reconstruct Clean Correlation
    corr_denoised = cp.dot(eVec_gpu, cp.dot(cp.diag(eVal_denoised), eVec_gpu.T))
    cp.fill_diagonal(corr_denoised, 1.0)
    
    # 7. Restore Clean Covariance
    cov_denoised = corr_denoised * cp.outer(std_gpu, std_gpu)
    
    # Transfer back to CPU
    return cp.asnumpy(cov_denoised)

# Alias for compatibility with existing pipeline imports
compute_rmt = compute_rmt_shrinkage
