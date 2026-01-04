#!/usr/bin/env python3
"""
BTC Trading Strategy Report Generator
Generates a static HTML report with strategy metrics, signals, and trade log.
Designed to run via GitHub Actions after data update.
"""

import pandas as pd
import numpy as np
import itertools
from datetime import datetime, timezone
import os

# Configuration
CSV_FILE = 'BTC_OHLC_1h_gmt8_updated.csv'
OUTPUT_FILE = 'report.html'

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
                equity += pnl
                open_trade = None
    
    # Calculate metrics
    pnl_arr = np.array(pnl_history)
    wins = pnl_arr[pnl_arr > 0]
    losses = pnl_arr[pnl_arr < 0]
    win_rate = len(wins) / len(pnl_arr) * 100 if len(pnl_arr) > 0 else 0
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.nan
    cum_return = (equity - INITIAL_EQ) / INITIAL_EQ * 100
    
    eq_array = INITIAL_EQ + np.cumsum(pnl_history)
    drawdowns = np.maximum.accumulate(eq_array) - eq_array
    max_dd = drawdowns.max() / np.maximum.accumulate(eq_array).max() * 100 if len(eq_array) > 0 else 0
    
    consec_losses = [sum(1 for _ in grp) for k, grp in itertools.groupby(pnl_arr < 0) if k]
    consec_wins = [sum(1 for _ in grp) for k, grp in itertools.groupby(pnl_arr > 0) if k]
    max_consec_losses = max(consec_losses) if consec_losses else 0
    max_consec_wins = max(consec_wins) if consec_wins else 0
    
    trade_df = pd.DataFrame(trade_log)
    
    # Recent performance
    if len(trade_df) > 0:
        now = df_bt.index.max()
        
        trades_7d = trade_df[trade_df['exit_time'] >= now - pd.Timedelta(days=7)]
        pnl_7d = trades_7d['pnl'].sum() if len(trades_7d) > 0 else 0
        wins_7d = len(trades_7d[trades_7d['pnl'] > 0])
        win_rate_7d = wins_7d / len(trades_7d) * 100 if len(trades_7d) > 0 else 0
        
        trades_30d = trade_df[trade_df['exit_time'] >= now - pd.Timedelta(days=30)]
        pnl_30d = trades_30d['pnl'].sum() if len(trades_30d) > 0 else 0
        wins_30d = len(trades_30d[trades_30d['pnl'] > 0])
        win_rate_30d = wins_30d / len(trades_30d) * 100 if len(trades_30d) > 0 else 0
        
        trades_3m = trade_df[trade_df['exit_time'] >= now - pd.Timedelta(days=90)]
        pnl_3m = trades_3m['pnl'].sum() if len(trades_3m) > 0 else 0
        wins_3m = len(trades_3m[trades_3m['pnl'] > 0])
        win_rate_3m = wins_3m / len(trades_3m) * 100 if len(trades_3m) > 0 else 0
    else:
        pnl_7d = pnl_30d = pnl_3m = 0
        win_rate_7d = win_rate_30d = win_rate_3m = 0
        trades_7d = trades_30d = trades_3m = pd.DataFrame()
    
    metrics = {
        "Variant": variant,
        "Capital Risked": f"{RISK_PCT_INIT*100:.1f}%",
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
    
    return trade_df, metrics, live_position

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

def generate_html_report(metrics, live_position, conditions, indicators, trade_log, last_update):
    """Generate HTML report"""
    
    # Format metrics table
    metrics_html = pd.DataFrame([metrics]).to_html(index=False, classes='table table-striped')
    
    # Format live position
    if live_position:
        pos_color = 'success' if live_position['position'] == 'long' else 'danger'
        pos_icon = '📈' if live_position['position'] == 'long' else '📉'
        live_pos_html = f"""
        <div class="alert alert-{pos_color}">
            <h4>{pos_icon} {live_position['position'].upper()} Position Open</h4>
            <p><strong>Entry:</strong> ${live_position['entry_price']:,.0f} | 
               <strong>Stop:</strong> ${live_position['stop_price']:,.0f} | 
               <strong>Target:</strong> ${live_position['tp_price']:,.0f}</p>
            <p><strong>Entry Time:</strong> {live_position['entry_time']}</p>
        </div>
        """
    else:
        live_pos_html = '<div class="alert alert-info">No live position open.</div>'
    
    # Format conditions table
    conditions_tail = conditions.tail(12).copy()
    conditions_tail['close'] = conditions_tail['close'].apply(lambda x: f'{x:,.0f}')
    conditions_tail['potential_stop'] = conditions_tail['potential_stop'].apply(lambda x: f'{x:,.0f}')
    conditions_html = conditions_tail.to_html(classes='table table-striped table-sm')
    
    # Format indicators table
    indicators_tail = indicators.tail(12).copy()
    for col in indicators_tail.columns:
        if col != 'rsi14':
            indicators_tail[col] = indicators_tail[col].apply(lambda x: f'{x:,.0f}' if pd.notna(x) else '')
        else:
            indicators_tail[col] = indicators_tail[col].apply(lambda x: f'{x:.0f}' if pd.notna(x) else '')
    indicators_html = indicators_tail.to_html(classes='table table-striped table-sm')
    
    # Format trade log
    if len(trade_log) > 0:
        trade_tail = trade_log.tail(59).copy()
        for col in ['entry_price', 'stop', 'target', 'exit_price', 'pnl']:
            trade_tail[col] = trade_tail[col].apply(lambda x: f'{x:,.0f}')
        trade_html = trade_tail.to_html(index=False, classes='table table-striped table-sm')
    else:
        trade_html = '<p>No trades available.</p>'
    
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BTC Trading Strategy Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{ padding: 20px; background-color: #f8f9fa; }}
        .card {{ margin-bottom: 20px; }}
        .table {{ font-size: 0.85rem; }}
        h1 {{ color: #2c3e50; }}
        .last-update {{ color: #6c757d; font-size: 0.9rem; }}
        .metric-value {{ font-size: 1.5rem; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container-fluid">
        <h1>📈 BTC Trading Strategy Dashboard</h1>
        <p class="last-update">Last updated: {last_update}</p>
        
        <div class="row">
            <div class="col-md-3">
                <div class="card text-white bg-primary">
                    <div class="card-body text-center">
                        <h5>Total Trades</h5>
                        <p class="metric-value">{metrics['Trades']}</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card text-white bg-success">
                    <div class="card-body text-center">
                        <h5>Win Rate</h5>
                        <p class="metric-value">{metrics['Win-rate %']:.0f}%</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card text-white bg-info">
                    <div class="card-body text-center">
                        <h5>Cumulative Return</h5>
                        <p class="metric-value">{metrics['Cum return %']:,.0f}%</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card text-white bg-warning">
                    <div class="card-body text-center">
                        <h5>Max Drawdown</h5>
                        <p class="metric-value">{metrics['Max DD %']:.0f}%</p>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <div class="card-header"><h4>📊 Strategy Metrics Summary</h4></div>
            <div class="card-body table-responsive">{metrics_html}</div>
        </div>
        
        <div class="card">
            <div class="card-header"><h4>🔴 LIVE POSITIONS</h4></div>
            <div class="card-body">{live_pos_html}</div>
        </div>
        
        <div class="card">
            <div class="card-header"><h4>🎯 Signal Conditions (Last 12 Hours)</h4></div>
            <div class="card-body table-responsive">{conditions_html}</div>
        </div>
        
        <div class="card">
            <div class="card-header"><h4>📈 Technical Indicators (Last 12 Hours)</h4></div>
            <div class="card-body table-responsive">{indicators_html}</div>
        </div>
        
        <div class="card">
            <div class="card-header"><h4>📝 Trade Log (Last 59 Trades)</h4></div>
            <div class="card-body table-responsive" style="max-height: 600px; overflow-y: auto;">{trade_html}</div>
        </div>
        
        <div class="card">
            <div class="card-header"><h4>📖 Strategy Rules</h4></div>
            <div class="card-body">
                <table class="table table-bordered">
                    <tr><td><strong>Session</strong></td><td>00:00–11:00 UTC (Asia hours)</td></tr>
                    <tr><td><strong>Volatility Filter</strong></td><td>ATR-20 > median ATR-20</td></tr>
                    <tr><td><strong>Mean-Reversion Long</strong></td><td>Price < lower BB AND RSI-14 < 30</td></tr>
                    <tr><td><strong>Mean-Reversion Short</strong></td><td>Price > upper BB AND RSI-14 > 70</td></tr>
                    <tr><td><strong>Breakout Long</strong></td><td>Price > 3h high AND RSI-14 > 60</td></tr>
                    <tr><td><strong>Breakout Short</strong></td><td>Price < 3h low AND RSI-14 < 40</td></tr>
                    <tr><td><strong>Stop-Loss</strong></td><td>{STOP_PCT*100:.2f}% from entry</td></tr>
                    <tr><td><strong>Profit-Take</strong></td><td>Entry ± {ATR_MULT} × ATR-20</td></tr>
                </table>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""
    return html

def main():
    log("=" * 50)
    log("Generating BTC Trading Strategy Report")
    log("=" * 50)
    
    try:
        # Load and process data
        log("Loading data...")
        df_raw = load_data()
        log(f"✓ Loaded {len(df_raw):,} rows")
        
        log("Calculating indicators...")
        df = calculate_indicators(df_raw)
        atr_med = df['atr20'].median()
        
        log("Running backtest...")
        trade_log_df, metrics, live_position = run_backtest(df, atr_med)
        log(f"✓ Backtest complete: {metrics['Trades']} trades")
        
        log("Calculating conditions...")
        conditions = calculate_conditions(df)
        
        # Prepare indicators for display
        indicators_cols = ['open', 'high', 'low', 'close', 'volume', 'sma20', 'std20', 
                          'upper_band', 'lower_band', 'rsi14', 'high_3h', 'low_3h', 
                          'atr20', 'atr20_median_all', 'atr20_roll_med180']
        available_cols = [col for col in indicators_cols if col in df.columns]
        indicators = df[available_cols]
        
        # Generate report
        log("Generating HTML report...")
        last_update = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        html = generate_html_report(metrics, live_position, conditions, indicators, trade_log_df, last_update)
        
        # Save report
        with open(OUTPUT_FILE, 'w') as f:
            f.write(html)
        
        log(f"✓ Report saved to {OUTPUT_FILE}")
        
        # Print summary
        log("")
        log("=" * 50)
        log("SUMMARY")
        log("=" * 50)
        log(f"Total Trades: {metrics['Trades']}")
        log(f"Win Rate: {metrics['Win-rate %']:.0f}%")
        log(f"Cumulative Return: {metrics['Cum return %']:,.0f}%")
        log(f"Max Drawdown: {metrics['Max DD %']:.0f}%")
        if live_position:
            log(f"LIVE POSITION: {live_position['position'].upper()} @ ${live_position['entry_price']:,.0f}")
        else:
            log("No live position")
        log("=" * 50)
        
    except Exception as e:
        log(f"❌ Error generating report: {e}")
        raise

if __name__ == "__main__":
    main()
