import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add utils to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

from hrp_functions import get_quasi_diag, get_recursive_bisection

def test_hrp_weights_sum_to_one():
    """
    Test that HRP weights sum to 1.0
    """
    # Mock covariance matrix (3x3)
    cov_data = np.array([
        [1.0, 0.5, 0.2],
        [0.5, 1.0, 0.3],
        [0.2, 0.3, 1.0]
    ])
    cov = pd.DataFrame(cov_data, index=[0, 1, 2], columns=[0, 1, 2])
    
    # Mock linkage matrix (Ward)
    # 3 items: 0, 1, 2
    # Link 0 and 1 -> Cluster 3
    # Link 3 and 2 -> Cluster 4
    linkage = np.array([
        [0, 1, 0.5, 2],
        [2, 3, 0.8, 3]
    ])
    
    sort_ix = get_quasi_diag(linkage)
    weights = get_recursive_bisection(cov, sort_ix)
    
    assert np.isclose(weights.sum(), 1.0), f"Weights sum to {weights.sum()}, expected 1.0"
    assert len(weights) == 3, "Should have 3 weights"
    assert (weights >= 0).all(), "Weights should be non-negative"

def test_quasi_diag_ordering():
    """
    Test that quasi-diagonalization returns a valid permutation.
    """
    linkage = np.array([
        [0, 1, 0.1, 2],
        [2, 3, 0.2, 3],
        [4, 5, 0.3, 4]
    ])
    # This linkage is incomplete for 5 items, but let's use a simpler valid one
    # 4 items: 0, 1, 2, 3
    # 0+1 -> 4
    # 2+3 -> 5
    # 4+5 -> 6
    linkage = np.array([
        [0, 1, 0.1, 2],
        [2, 3, 0.1, 2],
        [4, 5, 0.2, 4]
    ])
    
    sort_ix = get_quasi_diag(linkage)
    
    assert len(sort_ix) == 4
    assert set(sort_ix) == {0, 1, 2, 3}
