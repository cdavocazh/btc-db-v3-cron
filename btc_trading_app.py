"""
BTC Trading Strategy Dashboard - Streamlit Web App
Converted from Jupyter Notebook: data_refresh_1h_OHLC_updated_Aug25.ipynb
"""

import streamlit as st
import pandas as pd
import numpy as np
import itertools
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="BTC Trading Strategy Dashboard",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stDataFrame {
        font-size: 12px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("📈 BTC Trading Strategy Dashboard")
st.markdown("**Asian Hours Strategy Backtest & Live Signals**")

# ===== DATA LOADING =====
@st.cache_data
def load_data(file_path='BTC_OHLC_1h_gmt8_updated.csv'):
    """Load the BTC OHLC data"""
    try:
        df = pd.read_csv(file_path, parse_dates=['timestamp'])
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        return df
    except FileNotFoundError:
        st.error(f"Data file not found: {file_path}")
        return None

# Load data
df_raw = load_data()

if df_raw is not None:
    st.success(f"✓ Data loaded: {len(df_raw):,} rows | {df_raw.index.min()} to {df_raw.index.max()}")
    
    # ===== SIDEBAR PARAMETERS =====
    st.sidebar.header("⚙️ Strategy Parameters")
    
    INITIAL_EQ = st.sidebar.number_input("Initial Equity ($)", value=100000, step=10000)
    RISK_PCT_INIT = st.sidebar.slider("Risk % per Trade", 0.01, 0.20, 0.05, 0.01)
    STOP_PCT = st.sidebar.selectbox("Stop Loss %", [0.005, 0.0075, 0.01], index=0)
    ATR_MULT = st.sidebar.selectbox("ATR Multiplier for TP", [2.5, 3.0, 3.5], index=1)
    
    # Constants
    ASIA_HRS = set(range(0, 12))
    US_HRS = set(range(15, 21))
    ATR_PERIOD = 14
    BB_PERIOD = 20
    RSI_PERIOD = 14
    MA200_PERIOD = 200

    # ===== INDICATOR CALCULATIONS =====
    @st.cache_data
    def calculate_indicators(df):
        """Calculate all technical indicators"""
        df = df.copy()
        
        # Bollinger Bands
        df['sma20'] = df['close'].rolling(BB_PERIOD).mean()
        df['std20'] = df['close'].rolling(BB_PERIOD).std()
        df['upper_band'] = df['sma20'] + 2 * df['std20']
        df['lower_band'] = df['sma20'] - 2 * df['std20']
        
        # ATR
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift()).abs(),
            (df['low'] - df['close'].shift()).abs()
        ], axis=1).max(axis=1)
        df['atr20'] = tr.rolling(ATR_PERIOD).mean()
        df['atr20_median_all'] = df['atr20'].expanding().median()
        df['atr20_roll_med180'] = df['atr20'].rolling(window=180).median()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.clip(lower=0).rolling(RSI_PERIOD).mean()
        loss = (-delta.clip(upper=0)).rolling(RSI_PERIOD).mean()
        df['rsi14'] = 100 - 100 / (1 + gain / loss)
        
        # SMA 200
        df['sma200'] = df['close'].rolling(MA200_PERIOD).mean()
        
        # Breakout levels
        df['high_3h'] = df['high'].shift(1).rolling(3).max()
        df['low_3h'] = df['low'].shift(1).rolling(3).min()
        
        return df

    df = calculate_indicators(df_raw)
    atr_med = df['atr20'].median()

    # ===== BACKTEST FUNCTION =====
    @st.cache_data
    def run_backtest(df, initial_eq, risk_pct, stop_pct, atr_mult, atr_med):
        """Run the Asian Hours backtest"""
        
        # Convert to UTC for backtesting
        df_bt = df.copy()
        df_bt.index = df_bt.index.tz_localize('Asia/Singapore').tz_convert('UTC').tz_localize(None)
        df_bt = df_bt.sort_index()
        
        variant = f"{stop_pct*100:.2f}% stop, ATR×{atr_mult}"
        risk_amount = initial_eq * risk_pct
        equity = initial_eq
        open_trade = None
        pnl_history = []
        exit_times = []
        eq_hist = []
        trade_log = []
        
        for r in df_bt.itertuples():
            hr = r.Index.hour
            
            # Skip if indicators not ready
            if pd.isna(r.sma20) or pd.isna(r.atr20):
                continue
            
            # ENTRY
            if open_trade is None and hr in ASIA_HRS and r.atr20 > atr_med:
                long_mr = (r.close < r.lower_band) and (r.rsi14 < 30)
                short_mr = (r.close > r.upper_band) and (r.rsi14 > 70)
                long_bo = (r.close > r.high_3h) and (r.rsi14 > 60)
                short_bo = (r.close < r.low_3h) and (r.rsi14 < 40)
                
                if long_mr or long_bo or short_mr or short_bo:
                    side = "long" if (long_mr or long_bo) else "short"
                    entry_price = r.close
                    stop_price = entry_price * (1 - stop_pct) if side == "long" else entry_price * (1 + stop_pct)
                    target_price = entry_price + atr_mult * r.atr20 if side == "long" else entry_price - atr_mult * r.atr20
                    unit_risk = abs(entry_price - stop_price)
                    size = risk_amount / unit_risk if unit_risk > 0 else 0
                    
                    open_trade = {
                        "variant": variant,
                        "side": side,
                        "entry_time": r.Index,
                        "entry_price": entry_price,
                        "stop": stop_price,
                        "target": target_price,
                        "size": size
                    }
            
            # EXIT
            elif open_trade:
                exit_price = None
                
                # Exit at US session open
                if hr in US_HRS and hr not in ASIA_HRS:
                    exit_price = r.close
                else:
                    if open_trade["side"] == "long":
                        if r.low <= open_trade["stop"]:
                            exit_price = open_trade["stop"]
                        elif r.high >= open_trade["target"]:
                            exit_price = open_trade["target"]
                    else:
                        if r.high >= open_trade["stop"]:
                            exit_price = open_trade["stop"]
                        elif r.low <= open_trade["target"]:
                            exit_price = open_trade["target"]
                
                if exit_price is not None:
                    pnl = ((exit_price - open_trade["entry_price"]) if open_trade["side"] == "long"
                           else (open_trade["entry_price"] - exit_price)) * open_trade["size"]
                    
                    trade_log.append({
                        "variant": open_trade["variant"],
                        "side": open_trade["side"],
                        "entry_time": open_trade["entry_time"],
                        "entry_price": open_trade["entry_price"],
                        "stop": open_trade["stop"],
                        "target": open_trade["target"],
                        "size": int(open_trade["size"]),
                        "exit_time": r.Index,
                        "exit_price": exit_price,
                        "pnl": pnl
                    })
                    
                    pnl_history.append(pnl)
                    exit_times.append(r.Index)
                    equity += pnl
                    eq_hist.append(equity)
                    open_trade = None
        
        # Calculate metrics
        pnl_arr = np.array(pnl_history)
        wins = pnl_arr[pnl_arr > 0]
        losses = pnl_arr[pnl_arr < 0]
        win_rate = len(wins) / len(pnl_arr) * 100 if len(pnl_arr) > 0 else 0
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.nan
        cum_return = (equity - initial_eq) / initial_eq * 100
        
        # Max drawdown
        eq_array = initial_eq + np.cumsum(pnl_history)
        drawdowns = np.maximum.accumulate(eq_array) - eq_array
        max_dd = drawdowns.max() / np.maximum.accumulate(eq_array).max() * 100 if len(eq_array) > 0 else 0
        
        # Consecutive losses/wins
        consec_losses = [sum(1 for _ in grp) for k, grp in itertools.groupby(pnl_arr < 0) if k]
        consec_wins = [sum(1 for _ in grp) for k, grp in itertools.groupby(pnl_arr > 0) if k]
        max_consec_losses = max(consec_losses) if consec_losses else 0
        max_consec_wins = max(consec_wins) if consec_wins else 0
        
        # Recent performance
        trade_df = pd.DataFrame(trade_log)
        if len(trade_df) > 0:
            now = df_bt.index.max()
            
            # 7 day
            trades_7d = trade_df[trade_df['exit_time'] >= now - pd.Timedelta(days=7)]
            pnl_7d = trades_7d['pnl'].sum() if len(trades_7d) > 0 else 0
            wins_7d = len(trades_7d[trades_7d['pnl'] > 0])
            win_rate_7d = wins_7d / len(trades_7d) * 100 if len(trades_7d) > 0 else 0
            
            # 30 day
            trades_30d = trade_df[trade_df['exit_time'] >= now - pd.Timedelta(days=30)]
            pnl_30d = trades_30d['pnl'].sum() if len(trades_30d) > 0 else 0
            wins_30d = len(trades_30d[trades_30d['pnl'] > 0])
            win_rate_30d = wins_30d / len(trades_30d) * 100 if len(trades_30d) > 0 else 0
            
            # 3 month
            trades_3m = trade_df[trade_df['exit_time'] >= now - pd.Timedelta(days=90)]
            pnl_3m = trades_3m['pnl'].sum() if len(trades_3m) > 0 else 0
            wins_3m = len(trades_3m[trades_3m['pnl'] > 0])
            win_rate_3m = wins_3m / len(trades_3m) * 100 if len(trades_3m) > 0 else 0
        else:
            trades_7d = pd.DataFrame()
            trades_30d = pd.DataFrame()
            trades_3m = pd.DataFrame()
            pnl_7d = pnl_30d = pnl_3m = 0
            win_rate_7d = win_rate_30d = win_rate_3m = 0
        
        metrics = {
            "Variant": variant,
            "Capital Risked": f"{risk_pct*100:.1f}%",
            "Trades": len(pnl_arr),
            "Win-rate %": round(win_rate, 0),
            "Win/Loss ratio": round(win_loss_ratio, 0) if not np.isnan(win_loss_ratio) else 0,
            "Cum return %": round(cum_return, 0),
            "Max DD %": round(max_dd, 0),
            "Max consec losses": max_consec_losses,
            "Max consec wins": max_consec_wins,
            "Win-rate 30d %": round(win_rate_30d, 0),
            "Trades 30d": len(trades_30d),
            "PnL 30d": round(pnl_30d, 0),
            "Win-rate 7d %": round(win_rate_7d, 0),
            "Trades 7d": len(trades_7d),
            "PnL 7d": round(pnl_7d, 0),
            "Win-rate 3m %": round(win_rate_3m, 0),
            "Trades 3m": len(trades_3m),
            "PnL 3m": round(pnl_3m, 0),
        }
        
        # Live position
        live_position = None
        if open_trade:
            live_position = {
                "variant": variant,
                "entry_time": open_trade["entry_time"],
                "position": open_trade["side"],
                "entry_price": round(open_trade["entry_price"], 0),
                "stop_price": round(open_trade["stop"], 0),
                "tp_price": round(open_trade["target"], 0)
            }
        
        return trade_df, metrics, live_position, eq_hist

    # Run backtest
    with st.spinner("Running backtest..."):
        trade_log_df, metrics, live_position, equity_curve = run_backtest(
            df, INITIAL_EQ, RISK_PCT_INIT, STOP_PCT, ATR_MULT, atr_med
        )

    # ===== DISPLAY SECTIONS =====
    
    # Section 1: Metrics Summary (Cell 22)
    st.markdown("---")
    st.subheader("📊 Strategy Metrics Summary")
    
    metrics_df = pd.DataFrame([metrics])
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Trades", metrics["Trades"])
    with col2:
        st.metric("Win Rate", f"{metrics['Win-rate %']}%")
    with col3:
        st.metric("Cumulative Return", f"{metrics['Cum return %']:,.0f}%")
    with col4:
        st.metric("Max Drawdown", f"{metrics['Max DD %']}%")

    # Section 2: Live Positions (DISPLAY LIVE POSITIONS)
    st.markdown("---")
    st.subheader("🔴 LIVE POSITIONS")
    
    if live_position:
        live_df = pd.DataFrame([live_position])
        st.dataframe(live_df, use_container_width=True, hide_index=True)
        
        # Highlight the position
        if live_position["position"] == "long":
            st.success(f"📈 **LONG** position open at ${live_position['entry_price']:,.0f}")
        else:
            st.error(f"📉 **SHORT** position open at ${live_position['entry_price']:,.0f}")
        
        st.info(f"Stop: ${live_position['stop_price']:,.0f} | Target: ${live_position['tp_price']:,.0f}")
    else:
        st.info("No live position open.")

    # Section 3: Signal Conditions (Cell 23 + Cell 27 - conditions.tail(12))
    st.markdown("---")
    st.subheader("🎯 Signal Conditions (Last 12 Hours)")
    
    # Calculate conditions
    df_cond = df.copy()
    df_cond['cond_mr_long'] = (df_cond['close'] < df_cond['lower_band']) & (df_cond['rsi14'] < 30)
    df_cond['cond_mr_short'] = (df_cond['close'] > df_cond['upper_band']) & (df_cond['rsi14'] > 70)
    df_cond['cond_bo_long'] = (df_cond['close'] > df_cond['high_3h']) & (df_cond['rsi14'] > 60)
    df_cond['cond_bo_short'] = (df_cond['close'] < df_cond['low_3h']) & (df_cond['rsi14'] < 40)
    df_cond['cond_vol'] = df_cond['atr20'] > df_cond['atr20_median_all']
    
    df_cond['potential_side'] = 0
    df_cond.loc[df_cond['cond_vol'] & (df_cond['cond_mr_long'] | df_cond['cond_bo_long']), 'potential_side'] = 1
    df_cond.loc[df_cond['cond_vol'] & (df_cond['cond_mr_short'] | df_cond['cond_bo_short']), 'potential_side'] = -1
    df_cond['potential_stop'] = df_cond['close'] * (1 - STOP_PCT * df_cond['potential_side'])
    
    conditions = pd.DataFrame({
        'close': df_cond['close'],
        'potential_side': df_cond['potential_side'],
        'below_lower_MR_long': df_cond['cond_mr_long'],
        'above_upper_MR_short': df_cond['cond_mr_short'],
        'rsi_lt_30_MR_long': df_cond['cond_mr_long'],
        'rsi_gt_70_MR_short': df_cond['cond_mr_short'],
        'price_above_high3': df_cond['close'] > df_cond['high_3h'],
        'price_below_low3': df_cond['close'] < df_cond['low_3h'],
        'rsi_gt_60_BO_long': df_cond['rsi14'] > 60,
        'rsi_lt_40_BO_short': df_cond['rsi14'] < 40,
        'potential_stop': df_cond['potential_stop'],
        'atr_gt_median_vol': df_cond['cond_vol']
    })
    
    # conditions.tail(12) output
    st.dataframe(conditions.tail(12).style.format({
        'close': '{:,.0f}',
        'potential_stop': '{:,.0f}'
    }), use_container_width=True)

    # Section 4: Indicators Table (Cell 25)
    st.markdown("---")
    st.subheader("📈 Technical Indicators (Last 12 Hours)")
    
    indicators_cols = ['open', 'high', 'low', 'close', 'volume', 'sma20', 'std20', 
                       'upper_band', 'lower_band', 'rsi14', 'high_3h', 'low_3h', 
                       'atr20', 'atr20_median_all', 'atr20_roll_med180']
    
    available_cols = [col for col in indicators_cols if col in df.columns]
    indicators_df = df[available_cols].tail(12)
    
    st.dataframe(indicators_df.style.format({
        col: '{:,.0f}' for col in available_cols if col != 'rsi14'
    }), use_container_width=True)

    # Section 5: Trade Log (trade_log_df.tail(59))
    st.markdown("---")
    st.subheader("📝 Trade Log (Last 59 Trades)")
    
    if len(trade_log_df) > 0:
        display_cols = ['variant', 'side', 'entry_time', 'entry_price', 'stop', 
                        'target', 'size', 'exit_time', 'exit_price', 'pnl']
        
        trade_display = trade_log_df[display_cols].tail(59).copy()
        trade_display.columns = ['variant', 'side', 'entry_time', 'entry_price', 
                                 'stop', 'target', 'size', 'exit_time', 'exit_price', 'pnl']
        
        # Color code PnL
        def color_pnl(val):
            if val > 0:
                return 'background-color: #90EE90'
            elif val < 0:
                return 'background-color: #FFB6C1'
            return ''
        
        st.dataframe(
            trade_display.style.applymap(color_pnl, subset=['pnl']).format({
                'entry_price': '{:,.0f}',
                'stop': '{:,.0f}',
                'target': '{:,.0f}',
                'exit_price': '{:,.0f}',
                'pnl': '{:,.0f}'
            }),
            use_container_width=True,
            height=800
        )
        
        # Summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Displayed Trades", len(trade_display))
        with col2:
            st.metric("Total PnL (displayed)", f"${trade_display['pnl'].sum():,.0f}")
        with col3:
            st.metric("Avg PnL/Trade", f"${trade_display['pnl'].mean():,.0f}")
    else:
        st.warning("No trades in the backtest period.")

    # Section 6: Equity Curve Chart
    st.markdown("---")
    st.subheader("📈 Equity Curve")
    
    if len(equity_curve) > 0:
        eq_df = pd.DataFrame({
            'Trade #': range(1, len(equity_curve) + 1),
            'Equity': equity_curve
        })
        st.line_chart(eq_df.set_index('Trade #'))
    else:
        st.info("No equity curve data available.")

    # Strategy Description (Cell 23 markdown)
    st.markdown("---")
    st.subheader("📖 Asian Hours Strategy Summary")
    
    st.markdown(f"""
    **Strategy Configuration: {STOP_PCT*100:.2f}% stop, ATR×{ATR_MULT}**
    
    | Element | Rule |
    |---------|------|
    | **Session** | Enter only 00:00–11:00 UTC (Asia hours) |
    | **Volatility Filter** | ATR-20 > median ATR-20 |
    | **Mean-Reversion Long** | Price < lower Bollinger band AND RSI-14 < 30 |
    | **Mean-Reversion Short** | Price > upper Bollinger band AND RSI-14 > 70 |
    | **Breakout Long** | Price > 3-hour high AND RSI-14 > 60 |
    | **Breakout Short** | Price < 3-hour low AND RSI-14 < 40 |
    | **Stop-Loss** | {STOP_PCT*100:.2f}% from entry |
    | **Profit-Take** | Entry ± {ATR_MULT} × ATR-20 |
    | **Exit** | First of stop-loss / profit-take / US session open (15:00–20:00 UTC) |
    """)

else:
    st.error("Please ensure the data file 'BTC_OHLC_1h_gmt8_updated.csv' is in the same directory as this app.")
    st.info("Upload the CSV file to the app directory and refresh the page.")

# Footer
st.markdown("---")
st.caption("BTC Trading Strategy Dashboard | Converted from Jupyter Notebook")
