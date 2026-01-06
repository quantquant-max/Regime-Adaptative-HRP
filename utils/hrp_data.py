import pandas as pd
import numpy as np
import os


def find_file(filename, search_paths=['DATA', '.']):
    """Recursively find a file starting from search paths."""
    if isinstance(search_paths, str):
        search_paths = [search_paths]
        
    for path in search_paths:
        if os.path.exists(path):
            for root, dirs, files in os.walk(path):
                if filename in files:
                    return os.path.join(root, filename)
    return None

def get_quarterly_dates(dates):
    df_dates = pd.DataFrame({'date': dates})
    df_dates['year'] = df_dates['date'].dt.year
    df_dates['quarter'] = df_dates['date'].dt.quarter
    quarterly_ends = df_dates.groupby(['year', 'quarter'])['date'].max()
    return quarterly_ends.tolist()

def get_monthly_dates(dates):
    df_dates = pd.DataFrame({'date': dates})
    df_dates['year'] = df_dates['date'].dt.year
    df_dates['month'] = df_dates['date'].dt.month
    monthly_ends = df_dates.groupby(['year', 'month'])['date'].max()
    return monthly_ends.tolist()

def load_market_data(data_path, benchmark_path, start_date_str='1960-01-01', market_index='VWRETD', windows=[12, 24, 36, 60], freq='3M'):
    print(f"Loading stock data from {data_path}...")
    # Read raw CRSP data (Long format expected)
    # Use low_memory=False to handle mixed types in columns (e.g. RET with error codes)
    df = pd.read_csv(data_path, low_memory=False)
    print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")
    if 'PERMNO' in df.columns:
        print(f"Unique PERMNOs (Raw): {df['PERMNO'].nunique()}")
    elif 'permno' in df.columns:
        print(f"Unique PERMNOs (Raw): {df['permno'].nunique()}")
    
    # Standardize column names to uppercase
    df.columns = df.columns.str.upper()
    
    # Ensure required columns exist
    required_cols = ['PERMNO', 'DATE', 'SHRCD', 'EXCHCD', 'PRC', 'VOL', 'RET', 'SHROUT']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in CRSP data: {missing_cols}")

    # Parse Dates
    df['DATE'] = pd.to_datetime(df['DATE'])
    
    # FILTER: Keep only data from start_date onwards (Optimization)
    start_date = pd.Timestamp(start_date_str)
    print(f"Filtering raw data to start from {start_date.date()}...")
    df = df[df['DATE'] >= start_date]
    print(f"Unique PERMNOs (After Date Filter): {df['PERMNO'].nunique()}")
    
    # 1. Filter SHRCD (10, 11) and EXCHCD (1, 2, 3)
    print("Filtering by SHRCD (10, 11) and EXCHCD (1, 2, 3)...")
    df = df[df['SHRCD'].isin([10, 11])]
    df = df[df['EXCHCD'].isin([1, 2, 3])]
    print(f"Unique PERMNOs (After SHRCD/EXCHCD Filter): {df['PERMNO'].nunique()}")
    
    # 2. Calculate Liquidity and Universe Flag
    # User formula: Liquidity = VOL * 100 * abs(PRC)
    # Note: CRSP RET/PRC/VOL can be non-numeric strings (e.g. 'C'). Coerce errors.
    df['RET'] = pd.to_numeric(df['RET'], errors='coerce')
    
    # Merge RET and DLRET if DLRET exists
    if 'DLRET' in df.columns:
        print("Merging RET and DLRET columns...")
        df['DLRET'] = pd.to_numeric(df['DLRET'], errors='coerce').fillna(0)
        
        # Track where we have NO return info (RET is NaN and DLRET is 0)
        # If RET is NaN, we assume 0 return for compounding IF DLRET is non-zero.
        # If both are effectively missing/zero, result should be NaN.
        missing_ret_mask = df['RET'].isna() & (df['DLRET'] == 0)
        
        # Calculate total return: (1 + RET) * (1 + DLRET) - 1
        # fillna(0) on RET allows the formula to work when RET is missing but DLRET is present
        df['RET'] = (1 + df['RET'].fillna(0)) * (1 + df['DLRET']) - 1
        
        # Restore NaNs where we truly had no data
        df.loc[missing_ret_mask, 'RET'] = np.nan

    df['PRC'] = pd.to_numeric(df['PRC'], errors='coerce')
    df['VOL'] = pd.to_numeric(df['VOL'], errors='coerce')
    df['SHROUT'] = pd.to_numeric(df['SHROUT'], errors='coerce')
    
    df['ABS_PRC'] = df['PRC'].abs()
    df['LIQUIDITY'] = df['VOL'] * 100 * df['ABS_PRC']
    df['MKT_CAP'] = df['ABS_PRC'] * df['SHROUT'] * 1000
    
    print("Calculating universe flags...")
    
    # --- NEW LOGIC: Fama-French 12 Industry Filter ---
    crsp_dir = os.path.dirname(data_path)
    sic_path = os.path.join(crsp_dir, 'SIC_to_Fama_French_industry.csv')
    
    # Check if SIC mapping exists and SICCD column exists
    use_ff_filter = False
    if os.path.exists(sic_path) and 'SICCD' in df.columns:
        print(f"✓ Found SIC mapping file: {sic_path}")
        try:
            sic_df = pd.read_csv(sic_path)
            # Ensure SIC is integer
            sic_df['SIC'] = pd.to_numeric(sic_df['SIC'], errors='coerce')
            df['SICCD'] = pd.to_numeric(df['SICCD'], errors='coerce')
            
            # Merge FF_12 industry
            print("Mapping stocks to Fama-French 12 Industries...")
            df = df.merge(sic_df[['SIC', 'FF_12']], left_on='SICCD', right_on='SIC', how='left')
            
            print("Calculating Rolling 12-Month Median Liquidity...")
            df = df.sort_values(['PERMNO', 'DATE'])
            # Calculate rolling median (min_periods=6 to allow some missing data at start)
            df['ROLLING_MEDIAN_LIQ'] = df.groupby('PERMNO')['LIQUIDITY'].transform(
                lambda x: x.rolling(12, min_periods=6).median()
            )
            
            print("Filtering Top 20% Liquidity per Industry/Date...")
            # Calculate 80th percentile threshold per Date/Industry
            industry_thresholds = df.groupby(['DATE', 'FF_12'])['ROLLING_MEDIAN_LIQ'].transform(
                lambda x: x.quantile(0.8)
            )
            
            cond_industry_liq = (df['ROLLING_MEDIAN_LIQ'] >= industry_thresholds) & (df['FF_12'].notna())
            cond_price = df['ABS_PRC'] > 3
            
            df['in_universe_hrp'] = (cond_industry_liq & cond_price).astype(int)
            use_ff_filter = True
            print("  - Applied FF12 Industry Filter (Top 20% Rolling Median Liquidity)")
            
        except Exception as e:
            print(f"[!] Error applying FF filter: {e}. Reverting to simple filter.")
            use_ff_filter = False
    
    if not use_ff_filter:
        print("  - Using Standard Filter: Top 50% Global Liquidity")
        print("  - Price: > $3")
        
        # Calculate median liquidity per date
        median_liquidity = df.groupby('DATE')['LIQUIDITY'].transform('median')
        
        # Define Universe Conditions
        cond_liquidity = df['LIQUIDITY'] >= median_liquidity
        cond_price = df['ABS_PRC'] > 3
        
        df['in_universe_hrp'] = (cond_liquidity & cond_price).astype(int)

    # Calculate 20th percentile of Market Cap per date (to keep top 80%)
    # This is for the ML Universe (Features)
    print("  - Market Cap: Top 80% (for ML Universe)")
    mkt_cap_20pct = df.groupby('DATE')['MKT_CAP'].transform(lambda x: x.quantile(0.2))
    cond_mkt_cap = df['MKT_CAP'] >= mkt_cap_20pct
    
    df['in_universe_ML'] = cond_mkt_cap.astype(int)
    
    # Create specific universe for ML features (Top 80% Market Cap)
    # This is used for calculating market-wide metrics like Dispersion and Amihud
    df_ml_universe = df[df['in_universe_ML'] == 1].copy()
    print(f"Unique PERMNOs (ML Universe - Top 80% Mkt Cap): {df_ml_universe['PERMNO'].nunique()}")
    
    # Keep the FULL dataset (filtered only by SHRCD/EXCHCD) for returns_all
    # We will filter by 'in_universe_hrp' dynamically during HRP computation
    df_universe = df.copy()
    
    # Count unique PERMNOs that are ever in HRP universe
    hrp_permnos = df[df['in_universe_hrp'] == 1]['PERMNO'].nunique()
    print(f"Unique PERMNOs (HRP Universe - Liquidity & Price): {hrp_permnos}")
    print(f"Universe (Full): {len(df_universe)} rows")
    print(f"Universe (ML Features): {len(df_ml_universe)} rows")

    # Handle duplicates: keep the first occurrence for each DATE/PERMNO pair
    if df_universe.duplicated(subset=['DATE', 'PERMNO']).any():
        print("[!] Found duplicate DATE/PERMNO entries in universe. Keeping first occurrence.")
        df_universe = df_universe.drop_duplicates(subset=['DATE', 'PERMNO'], keep='first')

    # Pivot to create returns matrix (Dates x PERMNOs)
    # This matrix now contains ALL stocks (even those not currently in HRP universe)
    # We also need to pivot the 'in_universe_hrp' flag to know who is tradable when
    print("Pivoting to returns matrix...")
    returns_all = df_universe.pivot(index='DATE', columns='PERMNO', values='RET')
    
    # Pivot the HRP Universe Flag
    print("Pivoting HRP universe flags...")
    universe_flags = df_universe.pivot(index='DATE', columns='PERMNO', values='in_universe_hrp').fillna(0)
    
    # Load benchmark data
    print(f"Loading benchmark data from {benchmark_path}...")
    df_bench = pd.read_csv(benchmark_path)
    
    # Simplify: Expect 'date' and 'vwretd' (case insensitive)
    df_bench.columns = df_bench.columns.str.lower()
    
    if 'date' in df_bench.columns and 'vwretd' in df_bench.columns:
        df_bench['date'] = pd.to_datetime(df_bench['date'])
        df_bench.set_index('date', inplace=True)
        
        # Filter by start date
        df_bench = df_bench[df_bench.index >= start_date]
        
        # Assign to returns_all
        returns_all[market_index] = pd.to_numeric(df_bench['vwretd'], errors='coerce')
        print(f"✓ Market index loaded from {benchmark_path}")
    else:
        print(f"WARNING: Benchmark file must contain 'date' and 'vwretd' columns. Found: {list(df_bench.columns)}")

    # Handle Dates and Filtering
    dates = returns_all.index.sort_values()
    
    # FILTER: Ensure data starts from start_date onwards
    start_date = pd.Timestamp(start_date_str)
    dates_filtered = dates[dates >= start_date]
    
    print(f"\n{'='*70}")
    print(f"DATA FILTERING: Starting from {start_date.date()}")
    print(f"{'='*70}")
    print(f"  Original date range: {dates.min().date()} to {dates.max().date()} ({len(dates)} periods)")
    print(f"  Filtered date range: {dates_filtered.min().date()} to {dates_filtered.max().date()} ({len(dates_filtered)} periods)")

    if freq == '1M' or freq == 'M':
        rebalance_dates = get_monthly_dates(dates_filtered)
        print(f"  Monthly rebalance dates (from {start_date.year}): {len(rebalance_dates)}")
    else:
        rebalance_dates = get_quarterly_dates(dates_filtered)
        print(f"  Quarterly rebalance dates (from {start_date.year}): {len(rebalance_dates)}")

    # Filter returns matrix to date range
    returns_all = returns_all.loc[dates_filtered]
    universe_flags = universe_flags.loc[dates_filtered]
    
    # Ensure columns are strings
    returns_all.columns = returns_all.columns.astype(str)
    universe_flags.columns = universe_flags.columns.astype(str)

    # SAFETY CHECK: Verify market index exists
    if market_index not in returns_all.columns:
        print(f"WARNING: Market index {market_index} not found!")
    else:
        print(f"✓ Market index {market_index} verified in combined data")

    # Filter valid rebal dates with enough history
    max_win = max(windows)
    valid_rebal_dates = [d for d in rebalance_dates if (d - dates_filtered.min()).days / 30 >= max_win]
    print(f"Valid rebal dates for all windows: {len(valid_rebal_dates)}")
    
    # Return df_universe for industry ETF computation (contains FF_12, MKT_CAP, etc.)
    # Filter df_universe to same date range
    df_universe = df_universe[df_universe['DATE'].isin(dates_filtered)]
    
    return returns_all, dates_filtered, valid_rebal_dates, df_ml_universe, universe_flags, df_universe


# =============================================================================
# FAMA-FRENCH 12 INDUSTRY ETF CONSTRUCTION
# =============================================================================

# Fama-French 12 Industry Names (for reference and plotting)
FF12_INDUSTRY_NAMES = {
    1: 'NoDur',   # Consumer Non-Durables
    2: 'Durbl',   # Consumer Durables
    3: 'Manuf',   # Manufacturing
    4: 'Enrgy',   # Energy (Oil, Gas, Coal)
    5: 'Chems',   # Chemicals
    6: 'BusEq',   # Business Equipment (Computers, Software)
    7: 'Telcm',   # Telecommunications
    8: 'Utils',   # Utilities
    9: 'Shops',   # Wholesale/Retail
    10: 'Hlth',   # Healthcare
    11: 'Money',  # Finance
    12: 'Other'   # Other
}


def compute_industry_etf_returns(df_universe, returns_all, universe_flags, min_stocks_per_industry=3):
    """
    Compute value-weighted industry ETF returns for Fama-French 12 industries.
    
    This creates 12 synthetic ETFs from the HRP-filtered universe, enabling
    statistically valid HRP allocation (N=12 < T=60).
    
    Parameters
    ----------
    df_universe : pd.DataFrame
        Full CRSP dataset with columns: DATE, PERMNO, RET, MKT_CAP, FF_12, in_universe_hrp
    returns_all : pd.DataFrame
        Pivoted returns matrix (dates x PERMNOs)
    universe_flags : pd.DataFrame
        Binary flags (dates x PERMNOs) indicating HRP universe membership
    min_stocks_per_industry : int
        Minimum stocks required per industry (industries with fewer are excluded)
    
    Returns
    -------
    industry_returns : pd.DataFrame
        Value-weighted industry returns (dates x 12 industries)
    industry_weights : dict
        {date: {industry: {permno: weight}}} for decomposing HRP weights to stock level
    industry_counts : pd.DataFrame
        Number of stocks per industry per date (for diagnostics)
    """
    print(f"\n{'='*70}")
    print("COMPUTING VALUE-WEIGHTED INDUSTRY ETF RETURNS")
    print(f"{'='*70}")
    
    # Ensure FF_12 column exists
    if 'FF_12' not in df_universe.columns:
        raise ValueError("FF_12 column not found in df_universe. Run load_market_data with SIC mapping first.")
    
    # Filter to HRP universe only
    df_hrp = df_universe[df_universe['in_universe_hrp'] == 1].copy()
    print(f"HRP Universe: {len(df_hrp)} stock-month observations")
    print(f"Unique stocks: {df_hrp['PERMNO'].nunique()}")
    
    # Get unique dates from returns_all
    dates = returns_all.index
    
    # Initialize output containers
    industry_returns = pd.DataFrame(index=dates, columns=list(FF12_INDUSTRY_NAMES.keys()), dtype=float)
    industry_weights = {}  # {date: {industry: {permno: weight_within_industry}}}
    industry_counts = pd.DataFrame(index=dates, columns=list(FF12_INDUSTRY_NAMES.keys()), dtype=float)
    
    # Convert PERMNO to string for matching with returns_all columns
    df_hrp['PERMNO_STR'] = df_hrp['PERMNO'].astype(str)
    
    for date in dates:
        # Get stocks in universe on this date
        date_data = df_hrp[df_hrp['DATE'] == date].copy()
        
        if len(date_data) == 0:
            continue
        
        industry_weights[date] = {}
        
        for ind_id in FF12_INDUSTRY_NAMES.keys():
            # Get stocks in this industry
            ind_stocks = date_data[date_data['FF_12'] == ind_id]
            
            # Record count
            industry_counts.loc[date, ind_id] = len(ind_stocks)
            
            if len(ind_stocks) < min_stocks_per_industry:
                # Not enough stocks - leave as NaN
                industry_weights[date][ind_id] = {}
                continue
            
            # Calculate value weights within industry (market cap weighted)
            mkt_caps = ind_stocks.set_index('PERMNO_STR')['MKT_CAP']
            total_cap = mkt_caps.sum()
            
            if total_cap <= 0:
                industry_weights[date][ind_id] = {}
                continue
            
            # Value weights (sum to 1 within industry)
            vw_weights = mkt_caps / total_cap
            industry_weights[date][ind_id] = vw_weights.to_dict()
            
            # Get returns for these stocks
            permnos = ind_stocks['PERMNO_STR'].tolist()
            valid_permnos = [p for p in permnos if p in returns_all.columns]
            
            if len(valid_permnos) == 0:
                continue
            
            # Get return for this date
            stock_returns = returns_all.loc[date, valid_permnos]
            
            # Align weights with available returns
            aligned_weights = vw_weights.reindex(valid_permnos).fillna(0)
            aligned_weights = aligned_weights / aligned_weights.sum()  # Renormalize
            
            # Compute value-weighted return (handle NaN returns as 0)
            stock_returns_clean = stock_returns.fillna(0)
            industry_ret = (stock_returns_clean * aligned_weights).sum()
            
            industry_returns.loc[date, ind_id] = industry_ret
    
    # Rename columns to industry names
    industry_returns.columns = [FF12_INDUSTRY_NAMES[c] for c in industry_returns.columns]
    industry_counts.columns = [FF12_INDUSTRY_NAMES[c] for c in industry_counts.columns]
    
    # Drop rows with all NaN (before data starts)
    industry_returns = industry_returns.dropna(how='all')
    
    # Summary statistics
    print(f"\nIndustry ETF Returns: {industry_returns.shape[0]} dates × {industry_returns.shape[1]} industries")
    print(f"Date range: {industry_returns.index.min().date()} to {industry_returns.index.max().date()}")
    
    # Count valid industries per date
    valid_ind_counts = industry_returns.notna().sum(axis=1)
    print(f"\nValid industries per date:")
    print(f"  Mean: {valid_ind_counts.mean():.1f}")
    print(f"  Min:  {valid_ind_counts.min():.0f}")
    print(f"  Max:  {valid_ind_counts.max():.0f}")
    
    # Average stocks per industry
    print(f"\nAverage stocks per industry:")
    for ind_name in industry_counts.columns:
        avg = industry_counts[ind_name].mean()
        print(f"  {ind_name}: {avg:.1f}")
    
    return industry_returns, industry_weights, industry_counts


def load_fred_data(fred_path, start_date_str='1960-01-01'):
    """
    Load and merge FRED economic data (BAA, DGS10, CPIAUCSL, M2SL, UNRATE).
    Resamples daily data to monthly (end of month).
    """
    start_date = pd.Timestamp(start_date_str)
    
    # Files to load
    files = {
        'BAA': 'BAA.csv',
        'DGS10': 'DGS10.csv',
        'CPI': 'CPIAUCSL.csv',
        'M2SL': 'M2SL.csv',
        'UNRATE': 'UNRATE.csv',
        'TB3MS': 'TB3MS.csv'
    }
    
    merged_df = pd.DataFrame()
    
    for name, filename in files.items():
        path = os.path.join(fred_path, filename)
        if os.path.exists(path):
            # Read CSV
            # FRED CSVs usually have 'observation_date' and the value column
            try:
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                
                # Rename value column to 'name'
                if len(df.columns) > 0:
                    df.columns = [name]
                
                # Resample to Monthly End if needed (taking last value)
                # This handles both daily (DGS10) and monthly (BAA, CPI, etc.)
                # 'ME' is month end frequency (pandas 2.2+)
                # FIX: Shift by 1 month to account for publication lag (Look-Ahead Bias)
                df_monthly = df.resample('ME').last().shift(1)
                
                if merged_df.empty:
                    merged_df = df_monthly
                else:
                    merged_df = merged_df.join(df_monthly, how='outer')
                
                print(f"✓ Loaded {name} from {filename}")
            except Exception as e:
                print(f"✗ Error loading {filename}: {e}")
        else:
            print(f"✗ Missing {filename} in {fred_path}")
            
    # Filter by start date
    if not merged_df.empty:
        merged_df = merged_df[merged_df.index >= start_date]
        # Forward fill missing values (e.g. if some series end early or have gaps)
        merged_df = merged_df.ffill()
        
    return merged_df

def load_economic_data(prep_path, series_list, start_date_str='1960-01-01'):
    start_date = pd.Timestamp(start_date_str)
    econ_dict = {}
    for series in series_list:
        file = os.path.join(prep_path, f'{series}_quarterly.csv')
        if os.path.exists(file):
            # FIX: Use 'observation_date' as index column (FRED format)
            df_econ = pd.read_csv(file, index_col='observation_date', parse_dates=True)
            # Filter economic data to start from start_date
            df_econ = df_econ[df_econ.index >= start_date]
            econ_dict[series] = df_econ[series]
            print(f"✓ Loaded {series}: {len(df_econ)} observations (from {df_econ.index.min().date()})")
        else:
            print(f"✗ Missing: {series}")
            
    print(f"\nLoaded {len(econ_dict)}/{len(series_list)} economic series (all from {start_date.year}+)")
    return econ_dict

def load_risk_free_rate(prep_path, target_dates, start_date_str='1960-01-01'):
    start_date = pd.Timestamp(start_date_str)
    ff_file = os.path.join(prep_path, 'F-F_Research_Data_Factors.csv')
    
    if os.path.exists(ff_file):
        # Read only rows 4-1194 (monthly data, excluding annual data)
        # This hardcoding might be fragile if file changes, but matches original notebook logic
        ff_data = pd.read_csv(ff_file, skiprows=3, nrows=1190)
        ff_data.columns = ff_data.columns.str.strip()  # Remove whitespace
        
        # Parse date: format is YYYYMM (e.g., 192607 = 1926-07)
        ff_data['date'] = pd.to_datetime(ff_data.iloc[:, 0].astype(str), format='%Y%m')
        ff_data.set_index('date', inplace=True)
        
        # RF is in PERCENTAGE form and is MONTHLY (not annualized) - convert to decimal
        rf_monthly = ff_data['RF'].astype(float) / 100.0  # Convert from % to decimal
        rf_monthly = rf_monthly[rf_monthly.index >= start_date]  # Filter to start_date+
        
        # Reindex to match returns_all dates (forward-fill for missing dates)
        rf_monthly_aligned = rf_monthly.reindex(target_dates, method='ffill')
        
        print(f"✓ Loaded Fama-French RF: {len(rf_monthly)} monthly observations")
        print(f"  Aligned to {len(rf_monthly_aligned)} trading periods")
    else:
        print(f"✗ WARNING: Fama-French data not found. Using RF=0.")
        rf_monthly_aligned = pd.Series(0.0, index=target_dates)
        
    return rf_monthly_aligned
