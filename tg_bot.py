#!/usr/bin/env python
import os
import sys
import threading
import urllib.parse
import urllib.request
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# Global event flag to indicate whether trading is active.
trading_event = threading.Event()
trading_event.set()  # Trading is active by default.

# Global variable to hold the latest trade status.
last_trade_status = "No trades executed yet."

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = (
        "Hello! I am your trading bot.\n"
        "Use /status to get the latest trade update,\n"
        "/restart to restart the AI process,\n"
        "/stop_trading to pause trading, and\n"
        "/start_trading to resume trading."
    )
    await update.message.reply_text(message)

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"AI is alive: {last_trade_status}")

async def restart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Restarting the AI process...")
    python = sys.executable
    os.execv(python, [python] + sys.argv)

async def stop_trading(update: Update, context: ContextTypes.DEFAULT_TYPE):
    trading_event.clear()  # Pauses the trading loop.
    await update.message.reply_text("Trading paused.")

async def start_trading(update: Update, context: ContextTypes.DEFAULT_TYPE):
    trading_event.set()  # Resumes the trading loop.
    await update.message.reply_text("Trading resumed.")

def send_telegram_message(text: str) -> None:
    """Send a simple text message via the Telegram bot."""
    try:
        from local_config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
    except ImportError:
        TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
        TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials not provided; message not sent.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": TELEGRAM_CHAT_ID, "text": text}).encode()
    try:
        with urllib.request.urlopen(url, data=data, timeout=10) as resp:
            if resp.status != 200:
                print(f"Failed to send Telegram message: {resp.status}")
    except Exception as exc:
        print(f"Error sending Telegram message: {exc}")

def run_telegram_bot():
    try:
        from local_config import TELEGRAM_BOT_TOKEN
    except ImportError:
        TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("Telegram bot token not provided")

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("restart", restart))
    app.add_handler(CommandHandler("stop_trading", stop_trading))
    app.add_handler(CommandHandler("start_trading", start_trading))

    print("Starting Telegram bot...")
    app.run_polling()

if __name__ == "__main__":
    run_telegram_bot()
