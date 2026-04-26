"""
Streamlit Dashboard for PINN Engine (Macro‑Informed Neural Network).
"""

import streamlit as st
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
import json
import config
from us_calendar import USMarketCalendar

st.set_page_config(page_title="P2Quant PINN Engine", page_icon="🔬", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 600; color: #1f77b4; }
    .hero-card { background: linear-gradient(135deg, #1f77b4 0%, #2C5282 100%); border-radius: 16px; padding: 2rem; color: white; text-align: center; }
    .hero-ticker { font-size: 4rem; font-weight: 800; }
    .score-positive { color: #28a745; font-weight: 600; }
    .score-negative { color: #dc3545; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_latest_results():
    try:
        api = HfApi(token=config.HF_TOKEN)
        files = api.list_repo_files(repo_id=config.HF_OUTPUT_REPO, repo_type="dataset")
        json_files = sorted([f for f in files if f.endswith('.json')], reverse=True)
        if not json_files:
            return None
        local_path = hf_hub_download(
            repo_id=config.HF_OUTPUT_REPO, filename=json_files[0],
            repo_type="dataset", token=config.HF_TOKEN, cache_dir="./hf_cache"
        )
        with open(local_path) as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

def score_badge(s):
    try:
        s = float(s)
    except:
        return f'{s}'
    if s >= 0:
        return f'<span class="score-positive">+{s*100:.2f}%</span>'
    return f'<span class="score-negative">{s*100:.2f}%</span>'

def render_mode_tab(mode_data, mode_name):
    if not mode_data or 'top_picks' not in mode_data:
        st.warning(f"No {mode_name} data available.")
        return
    top = mode_data['top_picks']
    if not top:
        st.info(f"No predictions for {mode_name}.")
        return
    pick = top[0]
    ticker = pick['ticker']
    score = pick['score']

    st.markdown(f"""
    <div class="hero-card">
        <div style="font-size: 1.2rem; opacity: 0.8;">🔬 {mode_name} TOP PICK</div>
        <div class="hero-ticker">{ticker}</div>
        <div style="font-size: 1.5rem;">Predicted Return: {score_badge(score)}</div>
    </div>
    """, unsafe_allow_html=True)

    # Top 3 table
    st.markdown("### Top 3 Picks")
    rows = [{"Ticker": p['ticker'], "Predicted Return": score_badge(p['score'])} for p in top]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # All ETFs
    all_scores = mode_data.get('all_scores', {})
    if all_scores:
        st.markdown("### All ETFs")
        all_rows = [{"Ticker": t, "Predicted Return": score_badge(s)} for t, s in all_scores.items()]
        df_all = pd.DataFrame(all_rows).sort_values("Predicted Return", ascending=False)
        st.dataframe(df_all, use_container_width=True, hide_index=True)

def render_shrinking_tab(shrinking_data):
    if not shrinking_data:
        st.warning("No shrinking data.")
        return
    st.markdown(f"""
    <div class="hero-card">
        <div style="font-size: 1.2rem; opacity: 0.8;">🔄 SHRINKING CONSENSUS</div>
        <div class="hero-ticker">{shrinking_data['ticker']}</div>
        <div>{shrinking_data['conviction']:.0f}% conviction across {len(shrinking_data.get('windows',[]))} windows</div>
    </div>
    """, unsafe_allow_html=True)
    with st.expander("📋 All Windows"):
        rows = []
        for w in shrinking_data.get('windows', []):
            rows.append({
                "Window": f"{w['window_start']}-{w['window_end']}",
                "ETF": w['ticker'],
                "Score": score_badge(w.get('score', 0))
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# --- Sidebar ---
st.sidebar.markdown("## ⚙️ Configuration")
calendar = USMarketCalendar()
st.sidebar.markdown(f"**📅 Next Trading Day:** {calendar.next_trading_day().strftime('%Y-%m-%d')}")
data = load_latest_results()
if data:
    st.sidebar.markdown(f"**Run Date:** {data.get('run_date', 'Unknown')}")

st.markdown('<div class="main-header">🔬 P2Quant PINN Engine</div>', unsafe_allow_html=True)
st.markdown('<div>Macro‑Informed Neural Network · Economic Factor Constraint</div>', unsafe_allow_html=True)

if data is None:
    st.warning("No data available.")
    st.stop()

universes_data = data.get('universes', {})
tabs = st.tabs(["📊 Combined", "📈 Equity Sectors", "💰 FI/Commodities"])
keys = ["COMBINED", "EQUITY_SECTORS", "FI_COMMODITIES"]

for tab, key in zip(tabs, keys):
    uni = universes_data.get(key, {})
    if not uni:
        with tab:
            st.info(f"No data for {key}.")
        continue
    with tab:
        d, g, s = st.tabs(["📅 Daily (504d)", "🌍 Global (2008‑YTD)", "🔄 Shrinking Consensus"])
        with d:
            render_mode_tab(uni.get('daily'), "Daily")
        with g:
            render_mode_tab(uni.get('global'), "Global")
        with s:
            render_shrinking_tab(uni.get('shrinking'))
