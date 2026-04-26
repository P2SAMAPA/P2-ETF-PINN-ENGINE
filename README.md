# P2-ETF-PINN-ENGINE

**Macro‑Informed Neural Network with Economic Factor Constraint for ETF Return Prediction**

[![Daily Run](https://github.com/P2SAMAPA/P2-ETF-PINN-ENGINE/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-PINN-ENGINE/actions/workflows/daily_run.yml)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-p2--etf--pinn--results-blue)](https://huggingface.co/datasets/P2SAMAPA/p2-etf-pinn-results)

## Overview

`P2-ETF-PINN-ENGINE` uses a multi‑output neural network trained on **lagged returns (21‑day windows)** and **current macro features (VIX, DXY, T10Y2Y, TBILL_3M)** . Instead of the BSM PDE, the model is regularized by an **economic factor constraint** that anchors cross‑sectional predictions to a benchmark ETF (SPY for equity, TLT for FI). This prevents drift while allowing individual ETF differentiation. The engine produces next‑day return predictions and ranks ETFs across three training modes.

## Methodology

1. **Feature Engineering** – 21‑day lagged returns per ETF + current macro variables.
2. **Multi‑Output Neural Network** – Shared hidden layers (128→64) predict all ETF returns simultaneously.
3. **Economic Factor Constraint** – The cross‑sectional average of predictions is lightly anchored to the predicted return of SPY (or TLT), discouraging economically implausible forecasts.
4. **Training on Full History** – 200 epochs on 2008‑2026 YTD (or a 504‑day daily window).
5. **Ranking** – Top 3 ETFs per universe by predicted next‑day return.

## Training Modes

- **Daily (504d)** – Most recent 2 years for current regime awareness.
- **Global (2008‑YTD)** – Full history for long‑term learning.
- **Shrinking Windows Consensus** – 15 rolling windows (2010‑2024), consensus ETF across windows.

## Universe Coverage

| Universe | Tickers |
|----------|---------|
| **FI / Commodities** | TLT, VCIT, LQD, HYG, VNQ, GLD, SLV |
| **Equity Sectors** | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM |
| **Combined** | All tickers above |

## Usage

```bash
pip install -r requirements.txt
python trainer.py
streamlit run streamlit_app.py
Dashboard
Three sub‑tabs per universe: Daily, Global, Shrinking Consensus.

Hero card with predicted return.

Full ETF ranking tables.

License
MIT License
