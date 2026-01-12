#!/usr/bin/env python3
"""
BTC Trading Strategy Backtest Runner
Runs the full backtest and saves results to files for Streamlit to display.
Designed to run via GitHub Actions every hour after data update.
"""

import pandas as pd
import numpy as np
import itertools
import json
from datetime import datetime, timezone
import os

# Configuration
CSV_FILE = 'BTC_OHLC_1h_gmt8_updated.csv'
RESULTS_DIR = 'backtest_results'

# Strategy Parameters
INITIAL_EQ = 100000
RISK_PCT_INIT = 0.05
STOP_PCT = 0.005
ATR_MULT = 3.0
ASIA_HRS = set(range(0, 12))
US_HRS = set(range(15, 21))
ATR_PERIOD = 14
BB_PERIOD = 20
RSI_PERIOD = 14
MA200_PERIOD = 200

def log(message):
    """Print timestamped log message"""
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    print(f"[{timestamp}] {message}")

def load_data():
    """Load BTC OHLC data"""
    df = pd.read_csv(CSV_FILE, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    return df

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

def run_backtest(df, atr_med):
    """Run the Asian Hours backtest"""
    
    # Convert to UTC for backtesting
    df_bt = df.copy()
    df_bt.index = df_bt.index.tz_localize('Asia/Singapore').tz_convert('UTC').tz_localize(None)
    df_bt = df_bt.sort_index()
    
    variant = f"{STOP_PCT*100:.2f}% stop, ATR×{ATR_MULT}"
    risk_amount = INITIAL_EQ * RISK_PCT_INIT
    equity = INITIAL_EQ
    open_trade = None
    pnl_history = []
    trade_log = []
    
    for r in df_bt.itertuples():
        hr = r.Index.hour
        
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
                stop_price = entry_price * (1 - STOP_PCT) if side == "long" else entry_price * (1 + STOP_PCT)
                target_price = entry_price + ATR_MULT * r.atr20 if side == "long" else entry_price - ATR_MULT * r.atr20
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
                    "entry_time": str(open_trade["entry_time"]),
                    "entry_price": open_trade["entry_price"],
                    "stop": open_trade["stop"],
                    "target": open_trade["target"],
                    "size": int(open_trade["size"]),
                    "exit_time": str(r.Index),
                    "exit_price": exit_price,
                    "pnl": pnl
                })
                
                pnl_history.append(pnl)
                equity += pnl
                open_trade = None
    
    # Calculate metrics
    pnl_arr = np.array(pnl_history)
    wins = pnl_arr[pnl_arr > 0]
    losses = pnl_arr[pnl_arr < 0]
    win_rate = len(wins) / len(pnl_arr) * 100 if len(pnl_arr) > 0 else 0
    avg_win = float(wins.mean()) if len(wins) > 0 else 0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    cum_return = (equity - INITIAL_EQ) / INITIAL_EQ * 100
    
    eq_array = INITIAL_EQ + np.cumsum(pnl_history)
    drawdowns = np.maximum.accumulate(eq_array) - eq_array
    max_dd = float(drawdowns.max() / np.maximum.accumulate(eq_array).max() * 100) if len(eq_array) > 0 else 0
    
    consec_losses = [sum(1 for _ in grp) for k, grp in itertools.groupby(pnl_arr < 0) if k]
    consec_wins = [sum(1 for _ in grp) for k, grp in itertools.groupby(pnl_arr > 0) if k]
    max_consec_losses = max(consec_losses) if consec_losses else 0
    max_consec_wins = max(consec_wins) if consec_wins else 0
    
    trade_df = pd.DataFrame(trade_log)
    
    # Recent performance
    if len(trade_df) > 0:
        trade_df['exit_time_dt'] = pd.to_datetime(trade_df['exit_time'])
        now = trade_df['exit_time_dt'].max()
        
        trades_7d = trade_df[trade_df['exit_time_dt'] >= now - pd.Timedelta(days=7)]
        pnl_7d = float(trades_7d['pnl'].sum()) if len(trades_7d) > 0 else 0
        wins_7d = len(trades_7d[trades_7d['pnl'] > 0])
        win_rate_7d = wins_7d / len(trades_7d) * 100 if len(trades_7d) > 0 else 0
        
        trades_30d = trade_df[trade_df['exit_time_dt'] >= now - pd.Timedelta(days=30)]
        pnl_30d = float(trades_30d['pnl'].sum()) if len(trades_30d) > 0 else 0
        wins_30d = len(trades_30d[trades_30d['pnl'] > 0])
        win_rate_30d = wins_30d / len(trades_30d) * 100 if len(trades_30d) > 0 else 0
        
        trades_3m = trade_df[trade_df['exit_time_dt'] >= now - pd.Timedelta(days=90)]
        pnl_3m = float(trades_3m['pnl'].sum()) if len(trades_3m) > 0 else 0
        wins_3m = len(trades_3m[trades_3m['pnl'] > 0])
        win_rate_3m = wins_3m / len(trades_3m) * 100 if len(trades_3m) > 0 else 0
        
        trade_df = trade_df.drop(columns=['exit_time_dt'])
    else:
        pnl_7d = pnl_30d = pnl_3m = 0
        win_rate_7d = win_rate_30d = win_rate_3m = 0
    
    metrics = {
        "Variant": variant,
        "Capital_Risked": f"{RISK_PCT_INIT*100:.1f}%",
        "Trades": int(len(pnl_arr)),
        "Win_rate_pct": round(win_rate, 0),
        "Win_Loss_ratio": round(win_loss_ratio, 0),
        "Cum_return_pct": round(cum_return, 0),
        "Max_DD_pct": round(max_dd, 0),
        "Max_consec_losses": int(max_consec_losses),
        "Max_consec_wins": int(max_consec_wins),
        "Win_rate_30d_pct": round(win_rate_30d, 0),
        "Trades_30d": int(len(trades_30d)) if len(trade_df) > 0 else 0,
        "PnL_30d": round(pnl_30d, 0),
        "Win_rate_7d_pct": round(win_rate_7d, 0),
        "Trades_7d": int(len(trades_7d)) if len(trade_df) > 0 else 0,
        "PnL_7d": round(pnl_7d, 0),
        "Win_rate_3m_pct": round(win_rate_3m, 0),
        "Trades_3m": int(len(trades_3m)) if len(trade_df) > 0 else 0,
        "PnL_3m": round(pnl_3m, 0),
    }
    
    live_position = None
    if open_trade:
        live_position = {
            "variant": variant,
            "entry_time": str(open_trade["entry_time"]),
            "position": open_trade["side"],
            "entry_price": round(open_trade["entry_price"], 0),
            "stop_price": round(open_trade["stop"], 0),
            "tp_price": round(open_trade["target"], 0)
        }
    
    # Equity curve (last 500 points for chart - trade number based)
    equity_curve = [float(x) for x in list(eq_array)[-500:]]

    # Time-based equity curve (all trades with timestamps)
    equity_curve_ts = []
    cumulative_equity = INITIAL_EQ
    for trade in trade_log:
        cumulative_equity = cumulative_equity + trade['pnl'] if len(equity_curve_ts) == 0 else equity_curve_ts[-1]['equity'] + trade['pnl']
        # Store in GMT+8 format (the original data is in UTC, convert to GMT+8 for display)
        exit_time = trade['exit_time']
        equity_curve_ts.append({
            'exit_time': exit_time,
            'equity': cumulative_equity
        })

    return trade_df, metrics, live_position, equity_curve, equity_curve_ts

def calculate_conditions(df):
    """Calculate signal conditions"""
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
        'price_above_high3': df_cond['close'] > df_cond['high_3h'],
        'price_below_low3': df_cond['close'] < df_cond['low_3h'],
        'rsi_gt_60_BO_long': df_cond['rsi14'] > 60,
        'rsi_lt_40_BO_short': df_cond['rsi14'] < 40,
        'potential_stop': df_cond['potential_stop'],
        'atr_gt_median_vol': df_cond['cond_vol']
    })
    
    return conditions

def main():
    log("=" * 50)
    log("Running BTC Trading Strategy Backtest")
    log("=" * 50)
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load data
    log("Loading data...")
    df_raw = load_data()
    log(f"✓ Loaded {len(df_raw):,} rows")
    log(f"  Latest timestamp: {df_raw.index.max()}")
    
    # Calculate indicators
    log("Calculating indicators...")
    df = calculate_indicators(df_raw)
    atr_med = df['atr20'].median()
    
    # Run backtest
    log("Running backtest...")
    trade_log_df, metrics, live_position, equity_curve, equity_curve_ts = run_backtest(df, atr_med)
    log(f"✓ Backtest complete: {metrics['Trades']} trades")
    
    # Calculate conditions
    log("Calculating signal conditions...")
    conditions = calculate_conditions(df)
    
    # Prepare indicators for display (last 12 rows)
    indicators_cols = ['open', 'high', 'low', 'close', 'volume', 'sma20', 'std20', 
                      'upper_band', 'lower_band', 'rsi14', 'high_3h', 'low_3h', 
                      'atr20', 'atr20_median_all', 'atr20_roll_med180']
    available_cols = [col for col in indicators_cols if col in df.columns]
    indicators = df[available_cols].tail(12)
    
    # Save results
    log("Saving results...")
    
    # 1. Metrics JSON
    metrics_data = {
        "metrics": metrics,
        "live_position": live_position,
        "equity_curve": equity_curve,
        "last_updated": datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
        "data_latest_timestamp": str(df_raw.index.max())
    }
    with open(f"{RESULTS_DIR}/metrics.json", 'w') as f:
        json.dump(metrics_data, f, indent=2)
    log(f"✓ Saved {RESULTS_DIR}/metrics.json")
    
    # 2. Trade log CSV (last 59 trades)
    if len(trade_log_df) > 0:
        trade_log_df.tail(59).to_csv(f"{RESULTS_DIR}/trade_log.csv", index=False)
        log(f"✓ Saved {RESULTS_DIR}/trade_log.csv")
    
    # 3. Conditions CSV (last 12 rows)
    conditions.tail(12).to_csv(f"{RESULTS_DIR}/conditions.csv")
    log(f"✓ Saved {RESULTS_DIR}/conditions.csv")
    
    # 4. Indicators CSV (last 12 rows)
    indicators.to_csv(f"{RESULTS_DIR}/indicators.csv")
    log(f"✓ Saved {RESULTS_DIR}/indicators.csv")

    # 5. Equity curve with timestamps CSV (all trades)
    if equity_curve_ts:
        equity_df = pd.DataFrame(equity_curve_ts)
        equity_df.to_csv(f"{RESULTS_DIR}/equity_curve_ts.csv", index=False)
        log(f"✓ Saved {RESULTS_DIR}/equity_curve_ts.csv ({len(equity_curve_ts)} points)")

    # Print summary
    log("")
    log("=" * 50)
    log("BACKTEST SUMMARY")
    log("=" * 50)
    log(f"Total Trades: {metrics['Trades']}")
    log(f"Win Rate: {metrics['Win_rate_pct']:.0f}%")
    log(f"Cumulative Return: {metrics['Cum_return_pct']:,.0f}%")
    log(f"Max Drawdown: {metrics['Max_DD_pct']:.0f}%")
    log(f"7-Day PnL: ${metrics['PnL_7d']:,.0f}")
    log(f"30-Day PnL: ${metrics['PnL_30d']:,.0f}")
    if live_position:
        log(f"LIVE POSITION: {live_position['position'].upper()} @ ${live_position['entry_price']:,.0f}")
        log(f"  Stop: ${live_position['stop_price']:,.0f} | Target: ${live_position['tp_price']:,.0f}")
    else:
        log("No live position")
    log("=" * 50)
    log("✓ Backtest completed successfully")

if __name__ == "__main__":
    main()
