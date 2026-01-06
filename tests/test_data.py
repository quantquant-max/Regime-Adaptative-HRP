import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add utils to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

# We can't easily import hrp_data without the file structure, 
# but we can test the logic if we extract the filtering logic to a pure function.
# For now, let's test a mock universe filter logic similar to what's in hrp_data.

def mock_universe_filter(df):
    """
    Replicates the logic in hrp_data.load_market_data
    """
    # Calculate median liquidity per date
    median_liquidity = df.groupby('DATE')['LIQUIDITY'].transform('median')
    
    # Calculate 20th percentile of Market Cap per date
    mkt_cap_20pct = df.groupby('DATE')['MKT_CAP'].transform(lambda x: x.quantile(0.2))
    
    # Define Universe Conditions
    cond_liquidity = df['LIQUIDITY'] >= median_liquidity
    cond_price = df['ABS_PRC'] > 3
    cond_mkt_cap = df['MKT_CAP'] >= mkt_cap_20pct
    
    df['in_universe_hrp'] = (cond_liquidity & cond_price).astype(int)
    df['in_universe_ML'] = cond_mkt_cap.astype(int)
    
    return df

def test_universe_filter():
    # Create mock data
    dates = pd.to_datetime(['2020-01-01'] * 4)
    data = {
        'DATE': dates,
        'PERMNO': [1, 2, 3, 4],
        'LIQUIDITY': [100, 200, 50, 300], # Median is (100+200)/2 = 150
        'MKT_CAP': [10, 20, 5, 100],      # 20th pct is small
        'ABS_PRC': [2, 10, 10, 10]        # Stock 1 fails price check
    }
    df = pd.DataFrame(data)
    
    df_filtered = mock_universe_filter(df)
    
    # Stock 1: Price 2 (<3) -> Fail HRP
    assert df_filtered.loc[0, 'in_universe_hrp'] == 0
    
    # Stock 3: Liquidity 50 (<150) -> Fail HRP
    assert df_filtered.loc[2, 'in_universe_hrp'] == 0
    
    # Stock 2: Liq 200 (>150), Price 10 (>3) -> Pass HRP
    assert df_filtered.loc[1, 'in_universe_hrp'] == 1
    
    # Stock 4: Liq 300 (>150), Price 10 (>3) -> Pass HRP
    assert df_filtered.loc[3, 'in_universe_hrp'] == 1
