# P2-ETF-PINN

**Physics‑Informed Neural Network with Black‑Scholes‑Merton Constraints for ETF Forecasting**

[![Daily Run](https://github.com/P2SAMAPA/P2-ETF-PINN/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-PINN/actions/workflows/daily_run.yml)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-p2--etf--pinn--results-blue)](https://huggingface.co/datasets/P2SAMAPA/p2-etf-pinn-results)

## Overview

`P2-ETF-PINN` uses a Physics‑Informed Neural Network that enforces the Black‑Scholes‑Merton PDE as a soft constraint in its loss function. This ensures that predictions remain consistent with no‑arbitrage principles, even during high‑volatility regimes. The engine forecasts next‑day returns and ranks ETFs accordingly.

## Methodology

- **Neural Network**: Multilayer perceptron with Tanh activations.
- **Physics Constraint**: Penalizes deviations from the risk‑neutral drift (r * S * dt).
- **Training**: 200 epochs on 252‑day rolling windows.
- **Forecast**: Predicted next‑day return.

## Usage

```bash
pip install -r requirements.txt
python trainer.py
streamlit run streamlit_app.py
