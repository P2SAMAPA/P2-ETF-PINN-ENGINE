"""
Configuration for P2-ETF-PINN-ENGINE (repurposed).
"""
import os
from datetime import datetime

HF_DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_DATA_FILE = "master_data.parquet"
HF_OUTPUT_REPO = "P2SAMAPA/p2-etf-pinn-results"

FI_COMMODITIES_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
EQUITY_SECTORS_TICKERS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV",
    "XLI", "XLY", "XLP", "XLU", "GDX", "XME",
    "IWF", "XSD", "XBI", "IWM"
]
ALL_TICKERS = list(set(FI_COMMODITIES_TICKERS + EQUITY_SECTORS_TICKERS))

UNIVERSES = {
    "FI_COMMODITIES": FI_COMMODITIES_TICKERS,
    "EQUITY_SECTORS": EQUITY_SECTORS_TICKERS,
    "COMBINED": ALL_TICKERS
}

MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M"]

# Neural network
HIDDEN_DIMS = [128, 64]
EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 0.001
FACTOR_LAMBDA = 0.2                # strength of economic constraint
RANDOM_SEED = 42

# Data
DAILY_LOOKBACK = 504
GLOBAL_TRAIN_START = "2008-01-01"
MIN_OBSERVATIONS = 252

# Shrinking
SHRINKING_WINDOW_START_YEARS = list(range(2010, 2025))

TODAY = datetime.utcnow().strftime("%Y-%m-%d")
HF_TOKEN = os.environ.get("HF_TOKEN", None)
