# BTC Trading Strategy Dashboard

A Streamlit web application for visualizing and analyzing the Asian Hours BTC trading strategy, with automated hourly data updates via GitHub Actions.

## Features

This dashboard displays:
1. **Strategy Metrics Summary** - Overall backtest performance including win rate, cumulative return, max drawdown
2. **Live Positions** - Current open positions with entry, stop, and target prices
3. **Signal Conditions (Last 12 Hours)** - Real-time signal conditions for entry detection
4. **Technical Indicators (Last 12 Hours)** - Bollinger Bands, RSI, ATR, and breakout levels
5. **Trade Log (Last 59 Trades)** - Historical trade entries and exits with P&L
6. **Equity Curve** - Visual representation of account growth over time

## Quick Start

### Local Installation

1. Make sure you have Python 3.8+ installed

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Place the data file `BTC_OHLC_1h_gmt8_updated.csv` in the same directory as the app

4. Run the Streamlit app:
```bash
streamlit run btc_trading_app.py
```

The app will open in your default web browser at `http://localhost:8501`

---

## GitHub Actions Setup (Automated Hourly Updates)

### Step 1: Create GitHub Repository

1. Create a new repository on GitHub
2. Push all files to the repository:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### Step 2: Repository Structure

Ensure your repository has this structure:
```
your-repo/
├── .github/
│   └── workflows/
│       └── update_data.yml      # GitHub Actions workflow
├── btc_trading_app.py           # Streamlit app
├── update_data.py               # Data update script
├── generate_report.py           # Report generation script
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── BTC_OHLC_1h_gmt8_updated.csv # BTC data file
```

### Step 3: Enable GitHub Actions

1. Go to your repository on GitHub
2. Click on "Actions" tab
3. GitHub Actions should be automatically enabled

### Step 4: Verify Workflow

The workflow will:
- Run automatically every hour at minute 5 (e.g., 1:05, 2:05, etc.)
- Fetch latest BTC data from Binance API
- Update the CSV file
- Generate an HTML report (`report.html`)
- Commit and push changes back to the repository

You can also trigger the workflow manually:
1. Go to "Actions" tab
2. Select "Update BTC Data Hourly"
3. Click "Run workflow"

### Step 5: View the Report

After the workflow runs, you can:
- View `report.html` directly on GitHub (raw) or download it
- Enable GitHub Pages to host the report:
  1. Go to Settings → Pages
  2. Select "main" branch and "/" (root) folder
  3. Save
  4. Access at `https://YOUR_USERNAME.github.io/YOUR_REPO/report.html`

---

## Deploy Streamlit App (Optional)

### Streamlit Cloud (Free)

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub account
3. Select your repository
4. Set main file path: `btc_trading_app.py`
5. Deploy!

The app will auto-update when GitHub Actions pushes new data.

---

## Strategy Parameters

Adjustable parameters in the Streamlit sidebar:
- **Initial Equity**: Starting capital for backtest
- **Risk % per Trade**: Percentage of equity risked per trade
- **Stop Loss %**: Distance to stop loss from entry
- **ATR Multiplier for TP**: Multiplier for ATR to set take profit

## Strategy Rules

**Asian Hours Strategy (0.50% stop, ATR×3.0)**

| Element | Rule |
|---------|------|
| Session | Enter only 00:00–11:00 UTC (Asia hours) |
| Volatility Filter | ATR-20 > median ATR-20 |
| Mean-Reversion Long | Price < lower Bollinger band AND RSI-14 < 30 |
| Mean-Reversion Short | Price > upper Bollinger band AND RSI-14 > 70 |
| Breakout Long | Price > 3-hour high AND RSI-14 > 60 |
| Breakout Short | Price < 3-hour low AND RSI-14 < 40 |
| Stop-Loss | 0.50% from entry |
| Profit-Take | Entry ± 3.0 × ATR-20 |
| Exit | First of stop-loss / profit-take / US session open (15:00–20:00 UTC) |

## Files

| File | Description |
|------|-------------|
| `btc_trading_app.py` | Main Streamlit application |
| `update_data.py` | Script to fetch latest data from Binance API |
| `generate_report.py` | Script to generate static HTML report |
| `.github/workflows/update_data.yml` | GitHub Actions workflow for hourly updates |
| `requirements.txt` | Python dependencies |
| `BTC_OHLC_1h_gmt8_updated.csv` | BTC OHLC data |
| `report.html` | Auto-generated HTML report (created by GitHub Actions) |

## Troubleshooting

### GitHub Actions not running?
- Check Actions tab for errors
- Ensure workflow file is in `.github/workflows/` directory
- Verify repository has Actions enabled

### Data not updating?
- Check Binance API status
- Review workflow logs in Actions tab
- Ensure CSV file exists in repository

### Streamlit app errors?
- Verify all dependencies installed: `pip install -r requirements.txt`
- Check CSV file is in same directory as app
- Review console output for error messages

## Converted From

Original Jupyter Notebook: `data_refresh_1h_OHLC_updated_Aug25.ipynb`
