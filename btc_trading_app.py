"""
BTC Trading Strategy Dashboard - Streamlit Web App
Fetches pre-computed backtest results directly from GitHub.
This ensures fresh data on every refresh without needing to redeploy.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import requests
from datetime import datetime
from io import StringIO

# ============================================================
# CONFIGURATION - UPDATE THESE WITH YOUR GITHUB DETAILS
# ============================================================
GITHUB_USERNAME = "ha0zh"
GITHUB_REPO = "btc-db-v3"
GITHUB_BRANCH = "main"
# ============================================================

# Construct raw GitHub URLs
GITHUB_RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/{GITHUB_BRANCH}"

# Page configuration
st.set_page_config(
    page_title="BTC Trading Strategy Dashboard",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stDataFrame { font-size: 12px; }
</style>
""", unsafe_allow_html=True)

# Title
st.title("📈 BTC Trading Strategy Dashboard")
st.markdown("**Asian Hours Strategy Backtest & Live Signals**")

# ===== REFRESH BUTTON =====
col1, col2 = st.columns([1, 5])
with col1:
    refresh_clicked = st.button("🔄 Refresh Data", type="primary")
with col2:
    st.caption(f"Fetches latest data from GitHub | Last loaded: {datetime.now().strftime('%H:%M:%S')}")

if refresh_clicked:
    st.cache_data.clear()
    st.rerun()

# ===== FETCH DATA FROM GITHUB =====
@st.cache_data(ttl=60)  # Cache for 60 seconds only
def fetch_from_github(url):
    """Fetch a file from GitHub raw URL"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except Exception as e:
        return None

@st.cache_data(ttl=60)
def load_results():
    """Load pre-computed backtest results from GitHub"""
    results = {}
    
    # Load metrics JSON
    metrics_url = f"{GITHUB_RAW_BASE}/backtest_results/metrics.json"
    metrics_text = fetch_from_github(metrics_url)
    
    if metrics_text:
        try:
            data = json.loads(metrics_text)
            results['metrics'] = data.get('metrics', {})
            results['live_position'] = data.get('live_position')
            results['equity_curve'] = data.get('equity_curve', [])
            results['last_updated'] = data.get('last_updated', 'Unknown')
            results['data_timestamp'] = data.get('data_latest_timestamp', 'Unknown')
        except:
            results['metrics'] = {}
            results['live_position'] = None
            results['equity_curve'] = []
            results['last_updated'] = 'Error parsing data'
            results['data_timestamp'] = 'Error'
    else:
        results['metrics'] = {}
        results['live_position'] = None
        results['equity_curve'] = []
        results['last_updated'] = 'Could not fetch from GitHub'
        results['data_timestamp'] = 'N/A'
    
    # Load trade log
    trade_url = f"{GITHUB_RAW_BASE}/backtest_results/trade_log.csv"
    trade_text = fetch_from_github(trade_url)
    if trade_text:
        try:
            results['trade_log'] = pd.read_csv(StringIO(trade_text))
        except:
            results['trade_log'] = pd.DataFrame()
    else:
        results['trade_log'] = pd.DataFrame()
    
    # Load conditions
    conditions_url = f"{GITHUB_RAW_BASE}/backtest_results/conditions.csv"
    conditions_text = fetch_from_github(conditions_url)
    if conditions_text:
        try:
            results['conditions'] = pd.read_csv(StringIO(conditions_text), index_col=0)
        except:
            results['conditions'] = pd.DataFrame()
    else:
        results['conditions'] = pd.DataFrame()
    
    # Load indicators
    indicators_url = f"{GITHUB_RAW_BASE}/backtest_results/indicators.csv"
    indicators_text = fetch_from_github(indicators_url)
    if indicators_text:
        try:
            results['indicators'] = pd.read_csv(StringIO(indicators_text), index_col=0)
        except:
            results['indicators'] = pd.DataFrame()
    else:
        results['indicators'] = pd.DataFrame()
    
    return results

# Load results
results = load_results()
metrics = results['metrics']
live_position = results['live_position']
trade_log = results['trade_log']
conditions = results['conditions']
indicators = results['indicators']
equity_curve = results['equity_curve']

# ===== HEADER INFO =====
st.markdown(f"""
**🕐 Backtest Last Run:** {results['last_updated']}  
**📊 Data Until:** {results['data_timestamp']}
""")
st.markdown("---")

# ===== CHECK IF DATA EXISTS =====
if not metrics:
    st.error("⚠️ Could not load backtest results from GitHub.")
    st.warning(f"""
    **Please check:**
    1. Update the GitHub username and repo at the top of `btc_trading_app.py`
    2. Make sure the `backtest_results/` folder exists in your repo
    3. Run the GitHub Actions workflow at least once
    
    **Current configuration:**
    - GitHub URL: `{GITHUB_RAW_BASE}`
    """)
    st.info("""
    **To run the backtest:**
    1. Go to your GitHub repository
    2. Click on the **Actions** tab
    3. Select **Update BTC Data Hourly**
    4. Click **Run workflow**
    """)
    st.stop()

# ===== KEY METRICS =====
st.subheader("📊 Strategy Metrics Summary")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Trades", f"{metrics.get('Trades', 0):,}")
with col2:
    st.metric("Win Rate", f"{metrics.get('Win_rate_pct', 0):.0f}%")
with col3:
    st.metric("Cumulative Return", f"{metrics.get('Cum_return_pct', 0):,.0f}%")
with col4:
    st.metric("Max Drawdown", f"{metrics.get('Max_DD_pct', 0):.0f}%")

# Full metrics table
metrics_display = {
    "Variant": metrics.get('Variant', ''),
    "Capital Risked": metrics.get('Capital_Risked', ''),
    "Trades": metrics.get('Trades', 0),
    "Win-rate %": metrics.get('Win_rate_pct', 0),
    "Win/Loss ratio": metrics.get('Win_Loss_ratio', 0),
    "Cum return %": metrics.get('Cum_return_pct', 0),
    "Max DD %": metrics.get('Max_DD_pct', 0),
    "Max consec losses": metrics.get('Max_consec_losses', 0),
    "Max consec wins": metrics.get('Max_consec_wins', 0),
    "Win-rate 7d %": metrics.get('Win_rate_7d_pct', 0),
    "Trades 7d": metrics.get('Trades_7d', 0),
    "PnL 7d": f"${metrics.get('PnL_7d', 0):,.0f}",
    "Win-rate 30d %": metrics.get('Win_rate_30d_pct', 0),
    "Trades 30d": metrics.get('Trades_30d', 0),
    "PnL 30d": f"${metrics.get('PnL_30d', 0):,.0f}",
}
metrics_df = pd.DataFrame([metrics_display])
st.dataframe(metrics_df, use_container_width=True, hide_index=True)

# ===== LIVE POSITIONS =====
st.markdown("---")
st.subheader("🔴 LIVE POSITIONS")

if live_position:
    live_df = pd.DataFrame([live_position])
    st.dataframe(live_df, use_container_width=True, hide_index=True)
    
    if live_position["position"] == "long":
        st.success(f"📈 **LONG** position open at ${live_position['entry_price']:,.0f}")
    else:
        st.error(f"📉 **SHORT** position open at ${live_position['entry_price']:,.0f}")
    
    st.info(f"Stop: ${live_position['stop_price']:,.0f} | Target: ${live_position['tp_price']:,.0f}")
else:
    st.info("No live position open.")

# ===== SIGNAL CONDITIONS =====
st.markdown("---")
st.subheader("🎯 Signal Conditions (Last 12 Hours)")

if not conditions.empty:
    st.dataframe(conditions, use_container_width=True)
else:
    st.warning("No conditions data available.")

# ===== TECHNICAL INDICATORS =====
st.markdown("---")
st.subheader("📈 Technical Indicators (Last 12 Hours)")

if not indicators.empty:
    st.dataframe(indicators, use_container_width=True)
else:
    st.warning("No indicators data available.")

# ===== TRADE LOG =====
st.markdown("---")
st.subheader("📝 Trade Log (Last 59 Trades)")

if not trade_log.empty:
    # Format numeric columns
    display_df = trade_log.copy()
    for col in ['entry_price', 'stop', 'target', 'exit_price', 'pnl']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f'{x:,.0f}' if pd.notna(x) else '')
    
    st.dataframe(display_df, use_container_width=True, hide_index=True, height=600)
    
    # Summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Displayed Trades", len(trade_log))
    with col2:
        total_pnl = trade_log['pnl'].sum() if 'pnl' in trade_log.columns else 0
        st.metric("Total PnL (displayed)", f"${total_pnl:,.0f}")
    with col3:
        avg_pnl = trade_log['pnl'].mean() if 'pnl' in trade_log.columns else 0
        st.metric("Avg PnL/Trade", f"${avg_pnl:,.0f}")
else:
    st.warning("No trade log data available.")

# ===== EQUITY CURVE =====
st.markdown("---")
st.subheader("📈 Equity Curve")

if equity_curve:
    eq_df = pd.DataFrame({
        'Trade #': range(1, len(equity_curve) + 1),
        'Equity': equity_curve
    })
    st.line_chart(eq_df.set_index('Trade #'))
else:
    st.info("No equity curve data available.")

# ===== STRATEGY RULES =====
st.markdown("---")
st.subheader("📖 Asian Hours Strategy Summary")

st.markdown("""
| Element | Rule |
|---------|------|
| **Session** | Enter only 00:00–11:00 UTC (Asia hours) |
| **Volatility Filter** | ATR-20 > median ATR-20 |
| **Mean-Reversion Long** | Price < lower Bollinger band AND RSI-14 < 30 |
| **Mean-Reversion Short** | Price > upper Bollinger band AND RSI-14 > 70 |
| **Breakout Long** | Price > 3-hour high AND RSI-14 > 60 |
| **Breakout Short** | Price < 3-hour low AND RSI-14 < 40 |
| **Stop-Loss** | 0.50% from entry |
| **Profit-Take** | Entry ± 3.0 × ATR-20 |
| **Exit** | First of stop-loss / profit-take / US session open (15:00–20:00 UTC) |
""")

# Footer
st.markdown("---")
st.caption("BTC Trading Strategy Dashboard | Data fetched directly from GitHub | Click 'Refresh Data' for latest results")
