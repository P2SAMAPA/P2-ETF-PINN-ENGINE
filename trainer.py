"""
Main training script for PINN engine.
"""

import json
import pandas as pd
import numpy as np

import config
import data_manager
from pinn_model import PINNTrainer
import push_results

def run_pinn():
    print(f"=== P2-ETF-PINN Run: {config.TODAY} ===")
    df_master = data_manager.load_master_data()

    trainer = PINNTrainer(
        hidden_layers=config.HIDDEN_LAYERS,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        lr=config.LEARNING_RATE,
        physics_weight=config.PHYSICS_WEIGHT,
        seed=config.RANDOM_SEED
    )

    all_results = {}
    top_picks = {}

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n--- Processing Universe: {universe_name} ---")
        universe_results = {}

        for ticker in tickers:
            print(f"  Training PINN for {ticker}...")
            returns = data_manager.prepare_returns_series(df_master, ticker)
            if len(returns) < config.MIN_OBSERVATIONS:
                continue
            prices = (1 + returns.iloc[-config.LOOKBACK_WINDOW:]).cumprod() * 100
            success = trainer.fit(prices.values)
            if not success:
                continue
            forecast = trainer.forecast(prices.values)
            universe_results[ticker] = {
                'ticker': ticker,
                'forecast': forecast
            }

        all_results[universe_name] = universe_results
        sorted_tickers = sorted(universe_results.items(),
                                key=lambda x: x[1]['forecast'], reverse=True)
        top_picks[universe_name] = [{'ticker': t, 'forecast': d['forecast']}
                                    for t, d in sorted_tickers[:3]]

    # Shrinking windows (simplified)
    shrinking_results = {}
    for start_year in config.SHRINKING_WINDOW_START_YEARS:
        start_date = pd.Timestamp(f"{start_year}-01-01")
        window_label = f"{start_year}-{config.TODAY[:4]}"
        mask = df_master['Date'] >= start_date
        df_window = df_master[mask].copy()
        if len(df_window) < config.MIN_OBSERVATIONS:
            continue
        window_top = {}
        for universe_name, tickers in config.UNIVERSES.items():
            best_ticker = None
            best_forecast = -np.inf
            for ticker in tickers:
                returns = data_manager.prepare_returns_series(df_window, ticker)
                if len(returns) < config.MIN_OBSERVATIONS:
                    continue
                prices = (1 + returns.iloc[-config.LOOKBACK_WINDOW:]).cumprod() * 100
                trainer.fit(prices.values)
                fc = trainer.forecast(prices.values)
                if fc > best_forecast:
                    best_forecast = fc
                    best_ticker = ticker
            if best_ticker:
                window_top[universe_name] = {'ticker': best_ticker, 'forecast': best_forecast}
        shrinking_results[window_label] = {
            'start_year': start_year,
            'top_picks': window_top
        }

    output_payload = {
        "run_date": config.TODAY,
        "config": {
            "hidden_layers": config.HIDDEN_LAYERS,
            "epochs": config.EPOCHS,
            "physics_weight": config.PHYSICS_WEIGHT
        },
        "daily_trading": {
            "universes": all_results,
            "top_picks": top_picks
        },
        "shrinking_windows": shrinking_results
    }

    push_results.push_daily_result(output_payload)
    print("\n=== Run Complete ===")

if __name__ == "__main__":
    run_pinn()
