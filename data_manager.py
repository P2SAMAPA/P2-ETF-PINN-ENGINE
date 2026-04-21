"""
Data loading and preprocessing for PINN engine.
"""

import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import config

def load_master_data() -> pd.DataFrame:
    print(f"Downloading {config.HF_DATA_FILE} from {config.HF_DATA_REPO}...")
    file_path = hf_hub_download(
        repo_id=config.HF_DATA_REPO,
        filename=config.HF_DATA_FILE,
        repo_type="dataset",
        token=config.HF_TOKEN,
        cache_dir="./hf_cache"
    )
    df = pd.read_parquet(file_path)
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={'index': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def prepare_returns_series(df_wide: pd.DataFrame, ticker: str) -> pd.Series:
    """Extract log returns for a single ticker."""
    if ticker not in df_wide.columns:
        return pd.Series(dtype=float)
    prices = df_wide.set_index('Date')[ticker].dropna()
    returns = np.log(prices / prices.shift(1)).dropna()
    return returns
