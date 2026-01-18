#!/usr/bin/env python3
"""
Telegram Bot for BTC Trading Signal Notifications

This bot sends notifications when a new trading signal is generated from the backtest.
It can be integrated with run_backtest.py to send alerts for live positions.

Setup Instructions:
===================
1. Create a Telegram bot:
   - Open Telegram and search for @BotFather
   - Send /newbot and follow the prompts
   - Save the bot token (looks like: 123456789:ABCdefGHIjklMNOpqrsTUVwxyz)

2. Get your Chat ID:
   - Start a conversation with your new bot
   - Send any message to the bot
   - Visit: https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates
   - Look for "chat":{"id": YOUR_CHAT_ID} in the response

3. Set environment variables:
   - TELEGRAM_BOT_TOKEN: Your bot token from BotFather
   - TELEGRAM_CHAT_ID: Your chat ID (or group chat ID)

   For GitHub Actions, add these as repository secrets:
   - Go to Settings > Secrets and variables > Actions
   - Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID

4. Integration with run_backtest.py:
   - Import this module in run_backtest.py
   - Call send_signal_notification() when live_position is detected

Usage:
======
    from telegram_bot import send_signal_notification

    # After detecting a live position in run_backtest.py:
    if live_position:
        send_signal_notification(live_position)
"""

import os
import requests
from datetime import datetime, timezone

# Telegram configuration from environment variables
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"


def send_telegram_message(message: str, parse_mode: str = 'HTML') -> bool:
    """
    Send a message via Telegram bot.

    Args:
        message: The message text to send
        parse_mode: 'HTML' or 'Markdown' for formatting

    Returns:
        True if message was sent successfully, False otherwise
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[Telegram] Bot token or chat ID not configured. Skipping notification.")
        return False

    try:
        url = f"{TELEGRAM_API_URL}/sendMessage"
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': parse_mode
        }

        response = requests.post(url, json=payload, timeout=10)

        if response.status_code == 200:
            print("[Telegram] ✓ Message sent successfully")
            return True
        else:
            print(f"[Telegram] ✗ Failed to send message: {response.status_code} - {response.text}")
            return False

    except Exception as e:
        print(f"[Telegram] ✗ Error sending message: {e}")
        return False


def send_signal_notification(live_position: dict, capital_risked: float = 500.0) -> bool:
    """
    Send a trading signal notification via Telegram.

    Args:
        live_position: Dictionary containing position details:
            - variant: Strategy variant name
            - entry_time: Entry timestamp
            - position: 'long' or 'short'
            - entry_price: Entry price
            - stop_price: Stop loss price
            - tp_price: Take profit price
        capital_risked: Amount risked for position size calculation (default: 500 USDT)

    Returns:
        True if notification was sent successfully, False otherwise
    """
    if not live_position:
        return False

    position_type = live_position.get('position', 'unknown').upper()
    entry_price = live_position.get('entry_price', 0)
    stop_price = live_position.get('stop_price', 0)
    tp_price = live_position.get('tp_price', 0)
    entry_time = live_position.get('entry_time', 'Unknown')
    variant = live_position.get('variant', 'Unknown')

    # Calculate position size
    risk_per_unit = abs(entry_price - stop_price)
    position_size_btc = capital_risked / risk_per_unit if risk_per_unit > 0 else 0
    position_size_usdt = position_size_btc * entry_price

    # Calculate potential profit/loss
    potential_profit = abs(tp_price - entry_price) * position_size_btc
    potential_loss = capital_risked
    risk_reward = potential_profit / potential_loss if potential_loss > 0 else 0

    # Emoji based on position type
    emoji = "🟢📈" if position_type == "LONG" else "🔴📉"

    # Current time in GMT+8
    current_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

    message = f"""
{emoji} <b>NEW TRADING SIGNAL</b> {emoji}

<b>Position:</b> {position_type}
<b>Strategy:</b> {variant}

📊 <b>Price Levels:</b>
• Entry: ${entry_price:,.2f}
• Stop Loss: ${stop_price:,.2f}
• Take Profit: ${tp_price:,.2f}

💰 <b>Position Sizing (${capital_risked:,.0f} risk):</b>
• Size: {position_size_btc:.6f} BTC
• Value: ${position_size_usdt:,.2f}
• Risk/Unit: ${risk_per_unit:,.2f}

📈 <b>Risk/Reward:</b>
• Potential Profit: ${potential_profit:,.2f}
• Potential Loss: ${potential_loss:,.2f}
• R:R Ratio: {risk_reward:.2f}

⏰ Signal Time: {entry_time}
🔄 Alert Sent: {current_time}

<i>Asian Hours BTC Strategy</i>
"""

    return send_telegram_message(message.strip())


def send_position_closed_notification(trade_result: dict) -> bool:
    """
    Send a notification when a position is closed.

    Args:
        trade_result: Dictionary containing trade result:
            - side: 'long' or 'short'
            - entry_price: Entry price
            - exit_price: Exit price
            - pnl: Profit/Loss amount
            - entry_time: Entry timestamp
            - exit_time: Exit timestamp

    Returns:
        True if notification was sent successfully, False otherwise
    """
    if not trade_result:
        return False

    side = trade_result.get('side', 'unknown').upper()
    entry_price = trade_result.get('entry_price', 0)
    exit_price = trade_result.get('exit_price', 0)
    pnl = trade_result.get('pnl', 0)
    entry_time = trade_result.get('entry_time', 'Unknown')
    exit_time = trade_result.get('exit_time', 'Unknown')

    # Emoji based on P&L
    if pnl > 0:
        emoji = "✅💰"
        result = "WIN"
    else:
        emoji = "❌💸"
        result = "LOSS"

    message = f"""
{emoji} <b>POSITION CLOSED - {result}</b> {emoji}

<b>Position:</b> {side}
<b>P&L:</b> ${pnl:,.2f}

📊 <b>Trade Details:</b>
• Entry: ${entry_price:,.2f}
• Exit: ${exit_price:,.2f}
• Return: {((exit_price - entry_price) / entry_price * 100):+.2f}%

⏰ Entry: {entry_time}
⏰ Exit: {exit_time}

<i>Asian Hours BTC Strategy</i>
"""

    return send_telegram_message(message.strip())


def send_daily_summary(metrics: dict) -> bool:
    """
    Send a daily performance summary.

    Args:
        metrics: Dictionary containing performance metrics

    Returns:
        True if notification was sent successfully, False otherwise
    """
    if not metrics:
        return False

    total_trades = metrics.get('Trades', 0)
    win_rate = metrics.get('Win_rate_pct', 0)
    cum_return = metrics.get('Cum_return_pct', 0)
    max_dd = metrics.get('Max_DD_pct', 0)
    pnl_7d = metrics.get('PnL_7d', 0)
    pnl_30d = metrics.get('PnL_30d', 0)
    trades_7d = metrics.get('Trades_7d', 0)
    trades_30d = metrics.get('Trades_30d', 0)

    current_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

    message = f"""
📊 <b>DAILY PERFORMANCE SUMMARY</b> 📊

<b>All-Time Stats:</b>
• Total Trades: {total_trades:,}
• Win Rate: {win_rate:.1f}%
• Cumulative Return: {cum_return:,.1f}%
• Max Drawdown: {max_dd:.1f}%

<b>Recent Performance:</b>
• 7-Day P&L: ${pnl_7d:,.2f} ({trades_7d} trades)
• 30-Day P&L: ${pnl_30d:,.2f} ({trades_30d} trades)

⏰ Report Time: {current_time}

<i>Asian Hours BTC Strategy</i>
"""

    return send_telegram_message(message.strip())


def test_connection() -> bool:
    """
    Test the Telegram bot connection by sending a test message.

    Returns:
        True if test message was sent successfully, False otherwise
    """
    message = """
🔔 <b>Telegram Bot Test</b>

✓ Connection successful!
✓ Bot is configured correctly.

<i>BTC Trading Signal Bot is ready to send notifications.</i>
"""
    return send_telegram_message(message.strip())


if __name__ == "__main__":
    # Test the bot when run directly
    print("Testing Telegram Bot Connection...")
    print(f"Bot Token configured: {'Yes' if TELEGRAM_BOT_TOKEN else 'No'}")
    print(f"Chat ID configured: {'Yes' if TELEGRAM_CHAT_ID else 'No'}")

    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        if test_connection():
            print("✓ Test message sent successfully!")
        else:
            print("✗ Failed to send test message")
    else:
        print("\nTo configure the bot, set these environment variables:")
        print("  export TELEGRAM_BOT_TOKEN='your_bot_token'")
        print("  export TELEGRAM_CHAT_ID='your_chat_id'")
        print("\nThen run this script again to test.")
