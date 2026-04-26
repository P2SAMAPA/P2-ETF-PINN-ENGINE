import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import config

def load_master_data():
    path = hf_hub_download(repo_id=config.HF_DATA_REPO, filename=config.HF_DATA_FILE,
                           repo_type="dataset", token=config.HF_TOKEN)
    df = pd.read_parquet(path)
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={'index': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def prepare_returns_matrix(df_wide, tickers):
    available = [t for t in tickers if t in df_wide.columns]
    df_long = df_wide.melt(id_vars=['Date'], value_vars=available, var_name='ticker', value_name='price')
    df_long = df_long.sort_values(['ticker', 'Date'])
    df_long['log_return'] = df_long.groupby('ticker')['price'].transform(lambda x: np.log(x / x.shift(1)))
    df_long = df_long.dropna(subset=['log_return'])
    return df_long.pivot(index='Date', columns='ticker', values='log_return')[available].dropna()

def prepare_macro(df_wide):
    macro_cols = [c for c in config.MACRO_COLS if c in df_wide.columns]
    macro_df = df_wide[['Date'] + macro_cols].set_index('Date').ffill().dropna()
    return macro_df

def build_sequences(returns, macro):
    common = returns.index.intersection(macro.index)
    returns, macro = returns.loc[common], macro.loc[common]
    tickers = returns.columns.tolist()
    X, y = [], []
    for i in range(21, len(returns)-1):
        feat = []
        for t in tickers:
            feat.extend(returns[t].iloc[i-21:i].values)
        feat.extend(macro.iloc[i].values)
        X.append(feat)
        y.append(returns.iloc[i+1].values)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), tickers
