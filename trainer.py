import json, os, numpy as np, pandas as pd
import torch
import config, data_manager
from pinn_model import EconomicNet, train_model, predict_latest
import push_results

def run_mode(returns, macro, tickers, mode, device, anchor):
    X, y, _ = data_manager.build_sequences(returns, macro)
    if len(X) < config.MIN_OBSERVATIONS: return None
    model = train_model(X, y, tickers, config.EPOCHS, config.LEARNING_RATE,
                        device, config.FACTOR_LAMBDA, anchor)
    latest = X[-1:]
    preds = predict_latest(model, latest, device)
    scores = {t: float(p) for t, p in zip(tickers, preds)}
    top = sorted(scores, key=scores.get, reverse=True)[:3]
    return {"top_picks": [{"ticker": t, "score": scores[t]} for t in top],
            "all_scores": scores}

def shrinking(data, returns, macro, tickers, device, anchor):
    windows = []
    for yr in config.SHRINKING_WINDOW_START_YEARS:
        sd = pd.Timestamp(f"{yr}-01-01")
        ed = pd.Timestamp("2024-12-31")
        mask = (data['Date'] >= sd) & (data['Date'] <= ed)
        r = returns.loc[mask]
        m = macro.loc[r.index]
        if len(r) < config.MIN_OBSERVATIONS: continue
        X, y, _ = data_manager.build_sequences(r, m)
        if len(X) < config.MIN_OBSERVATIONS: continue
        model = train_model(X, y, tickers, config.EPOCHS, config.LEARNING_RATE,
                            device, config.FACTOR_LAMBDA, anchor)
        p = predict_latest(model, X[-1:], device)
        best = tickers[np.argmax(p)]
        windows.append({"window_start": yr, "window_end": 2024, "ticker": best,
                        "score": float(p.max())})
    if not windows: return None
    vote = {}
    for w in windows: vote[w['ticker']] = vote.get(w['ticker'],0)+1
    pick = max(vote, key=vote.get)
    return {"ticker": pick, "conviction": vote[pick]/len(windows)*100,
            "windows": windows}

def main():
    tok = os.getenv("HF_TOKEN")
    if not tok: return
    dev = torch.device('cpu')
    df = data_manager.load_master_data()
    macro = data_manager.prepare_macro(df)
    results = {}
    for uni, tks in config.UNIVERSES.items():
        ret = data_manager.prepare_returns_matrix(df, tks)
        m = macro.loc[ret.index].dropna()
        ret = ret.loc[m.index]
        anchor = "SPY" if "SPY" in tks else "TLT"
        u = {}
        # Daily
        d = run_mode(ret.iloc[-config.DAILY_LOOKBACK:], m.iloc[-config.DAILY_LOOKBACK:],
                     tks, 'daily', dev, anchor)
        if d: u['daily'] = d
        # Global
        g = run_mode(ret, m, tks, 'global', dev, anchor)
        if g: u['global'] = g
        # Shrinking
        s = shrinking(df, ret, m, tks, dev, anchor)
        if s: u['shrinking'] = s
        results[uni] = u
    push_results.push_daily_result({"run_date": config.TODAY, "universes": results})

if __name__ == "__main__":
    main()
