import os
import sys
import shutil
import numpy as np
import random
import yaml
from importlib import reload

# Global random seed (default, overridden by config)
GLOBAL_SEED = 42
GLOBAL_RNG = None

def load_config(project_root):
    """
    Load configuration from config.yaml and set global random seed.
    """
    global GLOBAL_SEED, GLOBAL_RNG
    
    config_path = os.path.join(project_root, 'config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✓ Loaded configuration from {config_path}")
        
        # Set global seed from config
        GLOBAL_SEED = config.get('project', {}).get('random_seed', 42)
        GLOBAL_RNG = setup_seeds(GLOBAL_SEED)
        
        return config
    else:
        print(f"[!] Config file not found at {config_path}")
        return None


def setup_seeds(seed: int = 42):
    """
    Centralized random seed setup for reproducibility.
    
    Sets seeds for:
    - numpy.random (legacy API)
    - random module
    - Returns a numpy Generator for modern API usage
    
    Parameters
    ----------
    seed : int
        Random seed value
        
    Returns
    -------
    np.random.Generator
        Modern numpy random generator for use in functions
    """
    global GLOBAL_SEED, GLOBAL_RNG
    
    GLOBAL_SEED = seed
    np.random.seed(seed)
    random.seed(seed)
    GLOBAL_RNG = np.random.default_rng(seed)
    
    print(f"✓ Random seeds set to {seed}")
    return GLOBAL_RNG


def get_random_state() -> int:
    """Get the global random seed for use in functions that need a seed parameter."""
    return GLOBAL_SEED


def get_rng() -> np.random.Generator:
    """Get the global random number generator for modern numpy API."""
    global GLOBAL_RNG
    if GLOBAL_RNG is None:
        GLOBAL_RNG = np.random.default_rng(GLOBAL_SEED)
    return GLOBAL_RNG


def find_file(filename, search_paths):
    """
    Search for a file in a list of directories.
    Independent implementation to avoid circular dependencies during setup.
    """
    for path in search_paths:
        if not os.path.exists(path):
            continue
            
        # Check direct path
        full_path = os.path.join(path, filename)
        if os.path.exists(full_path):
            return full_path
        
        # Recursive search
        for root, dirs, files in os.walk(path):
            if filename in files:
                return os.path.join(root, filename)
    return None


def setup_environment(current_dir):
    """
    Sets up the environment:
    1. Adds utils to sys.path
    2. Clears pycache
    3. Reloads modules
    4. Checks GPU
    5. Sets random seeds
    """
    # 1. Path Setup
    if os.path.basename(current_dir) == 'Codes':
        project_root = os.path.abspath(os.path.join(current_dir, '..'))
    else:
        project_root = current_dir

    utils_path = os.path.join(project_root, 'utils')
    if utils_path not in sys.path:
        sys.path.append(utils_path)
        print(f"✓ Added {utils_path} to sys.path")

    # 2. Clear Cache
    pycache_path = os.path.join(utils_path, '__pycache__')
    if os.path.exists(pycache_path):
        try:
            shutil.rmtree(pycache_path)
            print("✓ Cleared module cache")
        except:
            pass

    # 3. Import and Reload
    import hrp_functions
    import hrp_data
    import hrp_analytics
    import hrp_pipeline

    reload(hrp_functions)
    reload(hrp_data)
    reload(hrp_analytics)
    reload(hrp_pipeline)

    # 4. GPU Check
    gpu_available = False
    try:
        import cupy as cp
        gpu_available = True
        print(f"✓ CuPy version: {cp.__version__}")
        print(f"✓ CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")
    except ImportError:
        print("[!] CuPy not available. Running on CPU.")

    # 5. Seeds - use centralized seed from config (loaded later) or default
    # Note: Seeds are properly set when load_config() is called
    setup_seeds(GLOBAL_SEED)
    
    return project_root, gpu_available, hrp_pipeline

def get_file_paths(project_root, hrp_data):
    """
    Locates necessary data files.
    """
    def get_file_path(filename):
        search_dirs = [
            os.path.join(project_root, 'DATA (CRSP)'), 
            os.path.join(project_root, 'DATA'),
            project_root
        ]
        # Use local find_file instead of relying on passed module
        path = find_file(filename, search_paths=search_dirs)
        if path is None:
            # Fallback to module if local failed (unlikely, but safe)
            if hasattr(hrp_data, 'find_file'):
                path = hrp_data.find_file(filename, search_paths=search_dirs)
            
        if path is None:
            raise FileNotFoundError(f"Could not find {filename}")
        return path

    try:
        data_path = get_file_path('CRSP_selected_columns.csv')
        benchmark_path = get_file_path('CRSP_value_weighted_returns.csv')
        ff_path = get_file_path('F-F_Research_Data_Factors.csv')
        prep_path = os.path.dirname(ff_path)
        comp_path = get_file_path('compustat_selected_columns.csv')
        
        print(f"✓ Found data at: {data_path}")
        return data_path, benchmark_path, prep_path, comp_path
    except Exception as e:
        print(f"❌ Error locating files: {e}")
        return None, None, None, None
