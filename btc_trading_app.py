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
import time
from datetime import datetime, timedelta
from io import StringIO
import pytz
import altair as alt

# ============================================================
# CONFIGURATION - UPDATE THESE WITH YOUR GITHUB DETAILS
# ============================================================
GITHUB_USERNAME = "ha0zh"
GITHUB_REPO = "btc-db-v3-cron"
GITHUB_BRANCH = "main"
# ============================================================

# Construct raw GitHub URLs
GITHUB_RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/{GITHUB_BRANCH}"

# Timezone for display
GMT8 = pytz.timezone('Asia/Singapore')

# ===== LIVE BTC PRICE API FUNCTIONS =====
# Using Bybit and OKX APIs (no geo-restrictions)

def fetch_btc_price_bybit():
    """Fetch current BTC-USDT price from Bybit"""
    try:
        response = requests.get(
            "https://api.bybit.com/v5/market/tickers",
            params={"category": "spot", "symbol": "BTCUSDT"},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            if data.get('retCode') == 0:
                result = data.get('result', {}).get('list', [])
                if result:
                    return float(result[0]['lastPrice']), None
        return None, f"Bybit error: {response.status_code}"
    except Exception as e:
        return None, str(e)

def fetch_btc_price_okx():
    """Fetch current BTC-USDT price from OKX"""
    try:
        response = requests.get(
            "https://www.okx.com/api/v5/market/ticker",
            params={"instId": "BTC-USDT"},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == '0':
                result = data.get('data', [])
                if result:
                    return float(result[0]['last']), None
        return None, f"OKX error: {response.status_code}"
    except Exception as e:
        return None, str(e)

def fetch_btc_price():
    """Fetch current BTC price with fallback"""
    price, error = fetch_btc_price_bybit()
    if price:
        return price, None
    price, error = fetch_btc_price_okx()
    if price:
        return price, None
    return None, error

def fetch_btc_klines_bybit(interval='15', limit=96):
    """Fetch BTC-USDT klines from Bybit

    Intervals: 1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, M, W
    """
    try:
        response = requests.get(
            "https://api.bybit.com/v5/market/kline",
            params={
                "category": "spot",
                "symbol": "BTCUSDT",
                "interval": interval,
                "limit": limit
            },
            timeout=15
        )
        if response.status_code == 200:
            data = response.json()
            if data.get('retCode') == 0:
                klines = data.get('result', {}).get('list', [])
                if klines:
                    # Bybit returns: [startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover]
                    df = pd.DataFrame(klines, columns=[
                        'open_time', 'open', 'high', 'low', 'close', 'volume_btc', 'volume'
                    ])
                    df['open_time'] = pd.to_datetime(df['open_time'].astype(int), unit='ms')
                    # Convert to GMT+8
                    df['time_gmt8'] = df['open_time'] + pd.Timedelta(hours=8)
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = df[col].astype(float)
                    # Sort ascending (Bybit returns newest first)
                    df = df.sort_values('time_gmt8')
                    return df, None
        return None, f"Bybit klines error: {response.status_code}"
    except Exception as e:
        return None, str(e)

def fetch_btc_klines_okx(interval='1H', limit=96):
    """Fetch BTC-USDT klines from OKX

    Intervals: 1m, 3m, 5m, 15m, 30m, 1H, 2H, 4H, 6H, 12H, 1D
    """
    try:
        response = requests.get(
            "https://www.okx.com/api/v5/market/candles",
            params={
                "instId": "BTC-USDT",
                "bar": interval,
                "limit": str(limit)
            },
            timeout=15
        )
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == '0':
                klines = data.get('data', [])
                if klines:
                    # OKX returns: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
                    df = pd.DataFrame(klines, columns=[
                        'open_time', 'open', 'high', 'low', 'close', 'volume_btc',
                        'volume', 'volCcyQuote', 'confirm'
                    ])
                    df['open_time'] = pd.to_datetime(df['open_time'].astype(int), unit='ms')
                    # Convert to GMT+8
                    df['time_gmt8'] = df['open_time'] + pd.Timedelta(hours=8)
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = df[col].astype(float)
                    # Sort ascending (OKX returns newest first)
                    df = df.sort_values('time_gmt8')
                    return df, None
        return None, f"OKX klines error: {response.status_code}"
    except Exception as e:
        return None, str(e)

def fetch_btc_klines(interval_key):
    """Fetch BTC klines with fallback

    interval_key: '1m', '3m', '15m', '1h', '4h'
    """
    # Map to API-specific intervals and limits for 24 hours
    bybit_map = {
        '1m': ('1', 1440),
        '3m': ('3', 480),
        '15m': ('15', 96),
        '1h': ('60', 24),
        '4h': ('240', 6)
    }
    okx_map = {
        '1m': ('1m', 1440),
        '3m': ('3m', 480),
        '15m': ('15m', 96),
        '1h': ('1H', 24),
        '4h': ('4H', 6)
    }

    bybit_interval, limit = bybit_map.get(interval_key, ('15', 96))
    df, error = fetch_btc_klines_bybit(interval=bybit_interval, limit=limit)
    if df is not None:
        return df, None

    okx_interval, limit = okx_map.get(interval_key, ('15m', 96))
    df, error = fetch_btc_klines_okx(interval=okx_interval, limit=limit)
    if df is not None:
        return df, None

    return None, error

def convert_utc_to_gmt8(utc_str):
    """Convert UTC timestamp string to GMT+8"""
    if not utc_str or utc_str == 'Unknown':
        return utc_str
    try:
        # Parse UTC timestamp (format: "2026-01-11 13:02:26 UTC")
        utc_str_clean = utc_str.replace(' UTC', '')
        utc_dt = datetime.strptime(utc_str_clean, '%Y-%m-%d %H:%M:%S')
        utc_dt = pytz.UTC.localize(utc_dt)
        gmt8_dt = utc_dt.astimezone(GMT8)
        return gmt8_dt.strftime('%Y-%m-%d %H:%M:%S GMT+8')
    except Exception:
        return utc_str

# Page configuration
st.set_page_config(
    page_title="BTC Trading Strategy Dashboard (Cronjob)",
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
st.title("📈 BTC Trading Strategy Dashboard (Cronjob)")
st.markdown("**Asian Hours Strategy Backtest & Live Signals**")

# ===== REFRESH BUTTON =====
col1, col2, col3 = st.columns([1, 1, 3])
with col1:
    refresh_clicked = st.button("🔄 Refresh Data", type="primary")
with col2:
    show_debug = st.checkbox("Show debug", value=False)

# Current timestamp for display
load_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
st.caption(f"Page loaded: {load_time}")

if refresh_clicked:
    st.rerun()

# ===== FETCH FUNCTIONS (NO CACHING) =====
def fetch_from_github(filepath):
    """
    Fetch a file from GitHub raw URL with cache-busting.
    Returns (content, status_message)
    """
    # Cache-busting: add timestamp to URL
    cache_buster = int(time.time() * 1000)
    url = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/{GITHUB_BRANCH}/{filepath}?cb={cache_buster}"
    
    headers = {
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            return response.text, f"✓ Fetched {filepath} ({len(response.text)} bytes)"
        else:
            return None, f"✗ HTTP {response.status_code} for {filepath}"
    except Exception as e:
        return None, f"✗ Error fetching {filepath}: {str(e)}"

def load_results():
    """Load pre-computed backtest results from GitHub - NO CACHING"""
    results = {
        'metrics': {},
        'live_position': None,
        'equity_curve': [],
        'equity_curve_ts': pd.DataFrame(),
        'last_updated': 'Unknown',
        'data_timestamp': 'Unknown',
        'trade_log': pd.DataFrame(),
        'conditions': pd.DataFrame(),
        'indicators': pd.DataFrame(),
        'debug_messages': []
    }
    
    # Load metrics JSON
    metrics_text, msg = fetch_from_github("backtest_results/metrics.json")
    results['debug_messages'].append(msg)
    
    if metrics_text:
        try:
            data = json.loads(metrics_text)
            results['metrics'] = data.get('metrics', {})
            results['live_position'] = data.get('live_position')
            results['equity_curve'] = data.get('equity_curve', [])
            results['last_updated'] = data.get('last_updated', 'Unknown')
            results['data_timestamp'] = data.get('data_latest_timestamp', 'Unknown')
            results['debug_messages'].append(f"  → Backtest timestamp: {results['last_updated']}")
        except Exception as e:
            results['debug_messages'].append(f"  → JSON parse error: {e}")
    
    # Load trade log
    trade_text, msg = fetch_from_github("backtest_results/trade_log.csv")
    results['debug_messages'].append(msg)
    if trade_text:
        try:
            results['trade_log'] = pd.read_csv(StringIO(trade_text))
            results['debug_messages'].append(f"  → {len(results['trade_log'])} trades loaded")
        except Exception as e:
            results['debug_messages'].append(f"  → CSV parse error: {e}")
    
    # Load conditions
    conditions_text, msg = fetch_from_github("backtest_results/conditions.csv")
    results['debug_messages'].append(msg)
    if conditions_text:
        try:
            results['conditions'] = pd.read_csv(StringIO(conditions_text), index_col=0)
        except:
            pass
    
    # Load indicators
    indicators_text, msg = fetch_from_github("backtest_results/indicators.csv")
    results['debug_messages'].append(msg)
    if indicators_text:
        try:
            results['indicators'] = pd.read_csv(StringIO(indicators_text), index_col=0)
        except:
            pass

    # Load time-based equity curve
    equity_ts_text, msg = fetch_from_github("backtest_results/equity_curve_ts.csv")
    results['debug_messages'].append(msg)
    if equity_ts_text:
        try:
            results['equity_curve_ts'] = pd.read_csv(StringIO(equity_ts_text))
            results['debug_messages'].append(f"  → {len(results['equity_curve_ts'])} equity points loaded")
        except Exception as e:
            results['debug_messages'].append(f"  → Equity curve CSV parse error: {e}")

    return results

# ===== LOAD DATA =====
results = load_results()
metrics = results['metrics']
live_position = results['live_position']
trade_log = results['trade_log']
conditions = results['conditions']
indicators = results['indicators']
equity_curve = results['equity_curve']
equity_curve_ts = results['equity_curve_ts']

# ===== DEBUG INFO =====
if show_debug:
    st.markdown("### 🔧 Debug Info")
    st.markdown(f"**GitHub URL base:** `https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/{GITHUB_BRANCH}/`")
    st.markdown("**Fetch log:**")
    for msg in results['debug_messages']:
        st.text(msg)
    st.markdown("---")

# ===== HEADER INFO =====
last_run_gmt8 = convert_utc_to_gmt8(results['last_updated'])
st.markdown(f"""
**🕐 Backtest Last Run:** {last_run_gmt8}
**📊 Data Until:** {results['data_timestamp']}
""")
st.markdown("---")

# ===== CHECK IF DATA EXISTS =====
if not metrics:
    st.error("⚠️ Could not load backtest results from GitHub.")
    
    # Show debug by default when there's an error
    st.markdown("### 🔧 Debug Info")
    st.markdown(f"**GitHub URL base:** `https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/{GITHUB_BRANCH}/`")
    st.markdown("**Fetch log:**")
    for msg in results['debug_messages']:
        st.text(msg)
    
    st.warning(f"""
    **Please check:**
    1. Update `GITHUB_USERNAME` and `GITHUB_REPO` at the top of `btc_trading_app.py`
    2. Make sure the `backtest_results/` folder exists in your repo
    3. Run the GitHub Actions workflow at least once
    
    **Current configuration:**
    - Username: `{GITHUB_USERNAME}`
    - Repo: `{GITHUB_REPO}`
    """)
    st.stop()

# ===== LIVE BTC PRICE =====
st.subheader("💰 BTC-USDT Perpetual (Live)")

# Fetch current price from Bybit/OKX API
btc_price, price_error = fetch_btc_price()

if btc_price:
    # Display current price prominently
    price_col1, price_col2 = st.columns([1, 2])
    with price_col1:
        st.metric("Current Price", f"${btc_price:,.2f}")
    with price_col2:
        current_time_gmt8 = datetime.now(pytz.UTC).astimezone(GMT8).strftime('%Y-%m-%d %H:%M:%S')
        st.caption(f"Last updated: {current_time_gmt8} GMT+8")
        st.caption("Click 'Refresh' to get latest price")
else:
    st.warning(f"Could not fetch BTC price: {price_error}")

# Price chart section
st.markdown("##### 24-Hour Price Chart")

# Interval selector
interval_options = {
    '1 min': '1m',
    '3 min': '3m',
    '15 min': '15m',
    '1 hour': '1h',
    '4 hour': '4h'
}
selected_interval = st.selectbox(
    'Interval',
    list(interval_options.keys()),
    index=2,  # Default to 15 min
    key='btc_interval'
)

interval_code = interval_options[selected_interval]
klines_df, klines_error = fetch_btc_klines(interval_code)

if klines_df is not None and not klines_df.empty:
    # Prepare candlestick data
    candle_df = klines_df[['time_gmt8', 'open', 'high', 'low', 'close', 'volume']].copy()
    candle_df.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']

    # Ensure numeric types
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        candle_df[col] = pd.to_numeric(candle_df[col], errors='coerce')

    # Convert volume to millions (USDT)
    candle_df['VolumeMil'] = candle_df['Volume'] / 1_000_000

    # Determine candle color (green for up, red for down)
    candle_df['Color'] = candle_df.apply(lambda x: 'green' if x['Close'] >= x['Open'] else 'red', axis=1)

    # Calculate y-axis range with padding
    price_min = candle_df['Low'].min()
    price_max = candle_df['High'].max()
    price_range = price_max - price_min
    padding = price_range * 0.1
    y_min = price_min - padding
    y_max = price_max + padding

    # === CANDLESTICK CHART ===
    # Base chart for shared x-axis encoding
    base = alt.Chart(candle_df).encode(
        x=alt.X('Time:T', title=None, axis=alt.Axis(labels=False, ticks=False))
    )

    # Candlestick wicks (high-low line)
    wicks = base.mark_rule().encode(
        y=alt.Y('Low:Q', title='Price (USDT)', scale=alt.Scale(domain=[y_min, y_max])),
        y2=alt.Y2('High:Q'),
        color=alt.Color('Color:N', scale=alt.Scale(domain=['green', 'red'], range=['#00d4aa', '#ff4d4d']), legend=None),
        tooltip=[
            alt.Tooltip('Time:T', title='Time', format='%Y-%m-%d %H:%M'),
            alt.Tooltip('Open:Q', title='Open', format='$,.2f'),
            alt.Tooltip('High:Q', title='High', format='$,.2f'),
            alt.Tooltip('Low:Q', title='Low', format='$,.2f'),
            alt.Tooltip('Close:Q', title='Close', format='$,.2f'),
            alt.Tooltip('VolumeMil:Q', title='Vol (M USDT)', format=',.2f')
        ]
    )

    # Candlestick bodies (open-close bar)
    bodies = base.mark_bar(size=6).encode(
        y=alt.Y('Open:Q', scale=alt.Scale(domain=[y_min, y_max])),
        y2=alt.Y2('Close:Q'),
        color=alt.Color('Color:N', scale=alt.Scale(domain=['green', 'red'], range=['#00d4aa', '#ff4d4d']), legend=None),
        tooltip=[
            alt.Tooltip('Time:T', title='Time', format='%Y-%m-%d %H:%M'),
            alt.Tooltip('Open:Q', title='Open', format='$,.2f'),
            alt.Tooltip('High:Q', title='High', format='$,.2f'),
            alt.Tooltip('Low:Q', title='Low', format='$,.2f'),
            alt.Tooltip('Close:Q', title='Close', format='$,.2f'),
            alt.Tooltip('VolumeMil:Q', title='Vol (M USDT)', format=',.2f')
        ]
    )

    # Combine wicks and bodies for candlestick chart
    candlestick_chart = (wicks + bodies).properties(height=300)

    # === VOLUME CHART ===
    volume_chart = alt.Chart(candle_df).mark_bar().encode(
        x=alt.X('Time:T', title='Time (GMT+8)', axis=alt.Axis(format='%H:%M', labelAngle=-45)),
        y=alt.Y('VolumeMil:Q', title='Volume (M USDT)'),
        color=alt.Color('Color:N', scale=alt.Scale(domain=['green', 'red'], range=['#00d4aa', '#ff4d4d']), legend=None),
        tooltip=[
            alt.Tooltip('Time:T', title='Time', format='%Y-%m-%d %H:%M'),
            alt.Tooltip('VolumeMil:Q', title='Vol (M USDT)', format=',.2f')
        ]
    ).properties(height=100)

    # Stack candlestick and volume charts vertically
    combined_chart = alt.vconcat(
        candlestick_chart,
        volume_chart,
        spacing=0
    ).resolve_scale(
        x='shared'
    ).configure_view(
        strokeWidth=0
    ).configure_axis(
        grid=True,
        gridColor='#333333',
        gridOpacity=0.3
    )

    st.altair_chart(combined_chart, use_container_width=True)

    # Show price range info
    high_24h = candle_df['High'].max()
    low_24h = candle_df['Low'].min()
    total_volume_mil = candle_df['VolumeMil'].sum()
    st.caption(f"24h High: ${high_24h:,.2f} | 24h Low: ${low_24h:,.2f} | Range: ${high_24h - low_24h:,.2f} | Total Volume: {total_volume_mil:,.2f}M USDT")
else:
    st.warning(f"Could not fetch price chart: {klines_error}")

st.markdown("---")

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

# ===== POSITION SIZE CALCULATOR =====
st.markdown("---")
st.subheader("📐 Position Size Calculator")

# Capital risked input (numbers only)
capital_risked = st.number_input(
    "Capital Risked (USDT)",
    min_value=1.0,
    max_value=1000000.0,
    value=500.0,
    step=100.0,
    format="%.2f",
    help="Enter the amount you're willing to risk on this trade"
)

if live_position:
    entry_price = live_position['entry_price']
    stop_price = live_position['stop_price']
    position_type = live_position['position']

    # Calculate risk per unit (absolute difference between entry and stop)
    risk_per_unit = abs(entry_price - stop_price)

    if risk_per_unit > 0:
        # Position size = Capital Risked / Risk per Unit
        position_size_btc = capital_risked / risk_per_unit
        position_size_usdt = position_size_btc * entry_price

        # Display results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Position Size (BTC)", f"{position_size_btc:.6f}")
        with col2:
            st.metric("Position Size (USDT)", f"${position_size_usdt:,.2f}")
        with col3:
            st.metric("Risk per Unit", f"${risk_per_unit:,.2f}")

        # Show calculation breakdown
        st.caption(f"Formula: Position Size = Capital Risked / |Entry - Stop| = ${capital_risked:,.2f} / ${risk_per_unit:,.2f} = {position_size_btc:.6f} BTC")
    else:
        st.warning("Cannot calculate position size: Entry and Stop prices are the same.")
else:
    st.info("No live position available. Position size will be calculated when a signal is generated.")

    # Show example calculation
    st.caption("Example: With $500 capital risked and a $100 risk per unit (entry-stop), position size = 0.005 BTC")

# ===== SIGNAL CONDITIONS =====
st.markdown("---")
st.subheader("🎯 Signal Conditions (Last 12 Hours)")

if not conditions.empty:
    # Sort by index (timestamp) in descending order - latest first
    conditions_sorted = conditions.sort_index(ascending=False)
    st.dataframe(conditions_sorted, use_container_width=True)
else:
    st.warning("No conditions data available.")

# ===== TECHNICAL INDICATORS =====
st.markdown("---")
st.subheader("📈 Technical Indicators (Last 12 Hours)")

if not indicators.empty:
    # Sort by index (timestamp) in descending order - latest first
    indicators_sorted = indicators.sort_index(ascending=False)
    st.dataframe(indicators_sorted, use_container_width=True)
else:
    st.warning("No indicators data available.")

# ===== TRADE LOG =====
st.markdown("---")
st.subheader("📝 Trade Log (Last 59 Trades)")

if not trade_log.empty:
    # Sort by exit_time in descending order - latest first
    display_df = trade_log.copy()
    if 'exit_time' in display_df.columns:
        display_df = display_df.sort_values('exit_time', ascending=False)

    # Format numeric columns
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

# ===== EQUITY CURVE (Trade Number) =====
st.markdown("---")
st.subheader("📈 Equity Curve (by Trade #)")

if equity_curve:
    eq_df = pd.DataFrame({
        'Trade #': range(1, len(equity_curve) + 1),
        'Equity': equity_curve
    })
    st.line_chart(eq_df.set_index('Trade #'))
else:
    st.info("No equity curve data available.")

# ===== EQUITY CURVE (Time-Based) =====
st.markdown("---")
st.subheader("📈 Equity Curve (Time-Based)")

if not equity_curve_ts.empty:
    # Convert exit_time to datetime and to GMT+8
    eq_ts_df = equity_curve_ts.copy()
    eq_ts_df['exit_time'] = pd.to_datetime(eq_ts_df['exit_time'])

    # The data is in UTC, convert to GMT+8
    eq_ts_df['exit_time_gmt8'] = eq_ts_df['exit_time'] + pd.Timedelta(hours=8)

    # Filter controls
    col1, col2 = st.columns(2)

    with col1:
        duration_options = {
            'Last Month': 30,
            'Last 3 Months': 90,
            'Last 6 Months (Default)': 180,
            'Last 1 Year': 365,
            'All Time': 0
        }
        selected_duration = st.selectbox(
            'Duration',
            list(duration_options.keys()),
            index=2  # Default to 6 months
        )
        days_filter = duration_options[selected_duration]

    with col2:
        interval_options = ['Hourly', 'Daily']
        selected_interval = st.selectbox(
            'Interval',
            interval_options,
            index=0  # Default to Hourly
        )

    # Apply duration filter
    if days_filter > 0:
        latest_date = eq_ts_df['exit_time_gmt8'].max()
        cutoff_date = latest_date - pd.Timedelta(days=days_filter)
        eq_ts_filtered = eq_ts_df[eq_ts_df['exit_time_gmt8'] >= cutoff_date].copy()
    else:
        eq_ts_filtered = eq_ts_df.copy()

    if not eq_ts_filtered.empty:
        # Resample based on interval selection
        eq_ts_filtered = eq_ts_filtered.set_index('exit_time_gmt8')

        if selected_interval == 'Daily':
            # Take the last equity value for each day
            eq_resampled = eq_ts_filtered['equity'].resample('D').last().dropna()
        else:
            # Hourly: take the last equity value for each hour
            eq_resampled = eq_ts_filtered['equity'].resample('h').last().dropna()

        if not eq_resampled.empty:
            # Format the index for display
            chart_df = pd.DataFrame({
                'Equity': eq_resampled.values
            }, index=eq_resampled.index)

            st.line_chart(chart_df)

            # Show date range info
            start_date = eq_resampled.index.min().strftime('%Y-%m-%d %H:%M')
            end_date = eq_resampled.index.max().strftime('%Y-%m-%d %H:%M')
            st.caption(f"Showing {len(eq_resampled)} data points from {start_date} to {end_date} (GMT+8)")
        else:
            st.info("No data available for the selected filters.")
    else:
        st.info("No data available for the selected duration.")
else:
    st.info("No time-based equity curve data available.")

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

# ===== ESTIMATED ENTRY POINTS =====
st.markdown("---")
st.subheader("🎯 Estimated Entry Points")

if not indicators.empty and live_position is None:
    # Get the latest indicator values
    latest_idx = indicators.index[-1]
    latest = indicators.loc[latest_idx]

    # Extract key values
    current_close = latest.get('close', 0)
    upper_band = latest.get('upper_band', 0)
    lower_band = latest.get('lower_band', 0)
    rsi = latest.get('rsi14', 50)
    high_3h = latest.get('high_3h', 0)
    low_3h = latest.get('low_3h', 0)
    atr20 = latest.get('atr20', 0)
    atr20_median = latest.get('atr20_median_all', 0)

    # Check volatility filter
    vol_filter_met = atr20 > atr20_median if atr20_median > 0 else False

    # Calculate entry points for each strategy
    entry_points = []

    # Mean-Reversion Long: Price < lower band AND RSI < 30
    mr_long_price = lower_band
    mr_long_rsi_met = rsi < 30
    entry_points.append({
        'Strategy': 'Mean-Reversion Long',
        'Direction': 'LONG',
        'Entry Price': f"< ${mr_long_price:,.2f}",
        'RSI Condition': f"RSI < 30 (current: {rsi:.1f})",
        'RSI Met': mr_long_rsi_met,
        'Vol Filter': vol_filter_met
    })

    # Mean-Reversion Short: Price > upper band AND RSI > 70
    mr_short_price = upper_band
    mr_short_rsi_met = rsi > 70
    entry_points.append({
        'Strategy': 'Mean-Reversion Short',
        'Direction': 'SHORT',
        'Entry Price': f"> ${mr_short_price:,.2f}",
        'RSI Condition': f"RSI > 70 (current: {rsi:.1f})",
        'RSI Met': mr_short_rsi_met,
        'Vol Filter': vol_filter_met
    })

    # Breakout Long: Price > 3h high AND RSI > 60
    bo_long_price = high_3h
    bo_long_rsi_met = rsi > 60
    entry_points.append({
        'Strategy': 'Breakout Long',
        'Direction': 'LONG',
        'Entry Price': f"> ${bo_long_price:,.2f}",
        'RSI Condition': f"RSI > 60 (current: {rsi:.1f})",
        'RSI Met': bo_long_rsi_met,
        'Vol Filter': vol_filter_met
    })

    # Breakout Short: Price < 3h low AND RSI < 40
    bo_short_price = low_3h
    bo_short_rsi_met = rsi < 40
    entry_points.append({
        'Strategy': 'Breakout Short',
        'Direction': 'SHORT',
        'Entry Price': f"< ${bo_short_price:,.2f}",
        'RSI Condition': f"RSI < 40 (current: {rsi:.1f})",
        'RSI Met': bo_short_rsi_met,
        'Vol Filter': vol_filter_met
    })

    # Create DataFrame for display
    entry_df = pd.DataFrame(entry_points)

    # Add status column
    def get_status(row):
        if row['RSI Met'] and row['Vol Filter']:
            return "Ready"
        elif row['RSI Met']:
            return "RSI Met, Vol Filter Not Met"
        elif row['Vol Filter']:
            return "Vol Filter Met, RSI Not Met"
        else:
            return "Not Met"

    entry_df['Status'] = entry_df.apply(get_status, axis=1)

    # Display table
    display_df = entry_df[['Strategy', 'Direction', 'Entry Price', 'RSI Condition', 'Status']]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Show volatility filter status
    if vol_filter_met:
        st.success(f"Volatility Filter: MET (ATR-20: {atr20:.2f} > Median: {atr20_median:.2f})")
    else:
        st.warning(f"Volatility Filter: NOT MET (ATR-20: {atr20:.2f} < Median: {atr20_median:.2f})")

    # Show current price context
    st.caption(f"Based on hourly data at {latest_idx} | Current close: ${current_close:,.2f}")

    # Add note about live price
    if btc_price:
        st.caption(f"Live BTC price: ${btc_price:,.2f} - Use 'Refresh Data' to update hourly indicators")

elif live_position:
    st.info(f"Live position open: {live_position['position'].upper()} @ ${live_position['entry_price']:,.0f}")
    st.caption("Entry points not shown when a position is already open.")
else:
    st.warning("No indicator data available to calculate entry points.")

# Footer
st.markdown("---")
st.caption(f"BTC Trading Strategy Dashboard | Data fetched from GitHub at {load_time}")
