#!/usr/bin/env python3
"""
BTC OHLC Data Updater
Fetches latest hourly candles from Binance Futures API and updates the CSV file.
Designed to run via GitHub Actions every hour.
"""

import requests
import pandas as pd
from datetime import datetime, timezone
import os
import sys

# Configuration
CSV_FILE = 'BTC_OHLC_1h_gmt8_updated.csv'
BINANCE_API_URL = "https://fapi.binance.com/fapi/v1/klines"
SYMBOL = 'BTCUSDT'
INTERVAL = '1h'
LIMIT = 100  # Fetch last 100 candles to ensure overlap

def log(message):
    """Print timestamped log message"""
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    print(f"[{timestamp}] {message}")

def get_latest_timestamp_from_csv(csv_file):
    """Get the latest timestamp from the existing CSV file"""
    try:
        df = pd.read_csv(csv_file, parse_dates=['timestamp'])
        latest_timestamp = df['timestamp'].max()
        log(f"✓ Current CSV latest timestamp: {latest_timestamp}")
        return latest_timestamp, df
    except FileNotFoundError:
        log(f"⚠ CSV file not found: {csv_file}")
        return None, None
    except Exception as e:
        log(f"❌ Error reading CSV: {e}")
        return None, None

def fetch_recent_klines(limit=100):
    """
    Fetch the most recent klines from Binance Futures API
    No API key required - uses public endpoints
    """
    params = {
        'symbol': SYMBOL,
        'interval': INTERVAL,
        'limit': limit
    }
    
    try:
        log(f"Fetching {limit} most recent klines from Binance API...")
        response = requests.get(BINANCE_API_URL, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if not data:
            log("⚠ No data returned from API")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert timestamp to datetime (UTC)
        df['timestamp'] = pd.to_datetime(df['open_time'].astype(int), unit='ms', utc=True)
        
        # Convert to GMT+8 (Asia/Singapore timezone)
        df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Singapore').dt.tz_localize(None)
        
        # Select and rename columns to match existing format
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        
        # Convert price columns to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        log(f"✓ Fetched {len(df)} rows from API")
        log(f"  Date range: {df.index.min()} to {df.index.max()}")
        
        return df
        
    except requests.exceptions.RequestException as e:
        log(f"❌ API request failed: {e}")
        return None
    except Exception as e:
        log(f"❌ Error processing API response: {e}")
        return None

def update_csv():
    """Main function to update CSV with latest data"""
    log("=" * 50)
    log("Starting BTC OHLC Data Update")
    log("=" * 50)
    
    # Get current CSV data
    latest_csv_time, df_existing = get_latest_timestamp_from_csv(CSV_FILE)
    
    if df_existing is None:
        log("❌ Cannot proceed without existing CSV file")
        return False
    
    # Ensure existing data has proper index
    if 'timestamp' in df_existing.columns:
        df_existing.set_index('timestamp', inplace=True)
    
    # Fetch recent data
    df_new = fetch_recent_klines(limit=LIMIT)
    
    if df_new is None or len(df_new) == 0:
        log("❌ Failed to fetch new data from API")
        return False
    
    # Combine with existing data
    df_combined = pd.concat([df_existing, df_new])
    
    # Remove duplicates (keep last - API data is more recent)
    df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
    
    # Sort by timestamp
    df_combined.sort_index(inplace=True)
    
    # Check if we have new data
    new_rows = len(df_combined) - len(df_existing)
    
    if new_rows > 0:
        # Save updated data
        df_combined.to_csv(CSV_FILE)
        
        log(f"✓ CSV file updated successfully:")
        log(f"  • Total rows: {len(df_combined)}")
        log(f"  • Earliest timestamp: {df_combined.index.min()}")
        log(f"  • Latest timestamp: {df_combined.index.max()}")
        log(f"  • New rows added: {new_rows}")
        
        return True
    else:
        # Still save to update any corrected values
        df_combined.to_csv(CSV_FILE)
        log("✓ CSV updated (no new rows, but data refreshed)")
        log(f"  • Latest timestamp: {df_combined.index.max()}")
        return True

def main():
    try:
        success = update_csv()
        if success:
            log("✓ Data update completed successfully")
            sys.exit(0)
        else:
            log("❌ Data update failed")
            sys.exit(1)
    except Exception as e:
        log(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
