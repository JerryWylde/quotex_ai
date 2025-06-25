#auto_live_runner.py

import pandas as pd
from binance.client import Client
from datetime import datetime
import schedule
import time
import subprocess

API_KEY = ''  # Optional if using public data
API_SECRET = ''  # Optional if using public data
client = Client(api_key=API_KEY, api_secret=API_SECRET)

def fetch_and_save_binance_data():
    print("🔄 Fetching latest BTC/USDT 1m candles...")
    klines = client.get_klines(symbol='BTCUSDT', interval=Client.KLINE_INTERVAL_1MINUTE, limit=1000)
    
    df = pd.DataFrame(klines, columns=[
        'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close time', 'Quote asset volume', 'Number of trades',
        'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
    ])
    
    df = df[['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df.columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    
    df.to_csv('data/btc_1min_raw.csv', index=False)
    print("✅ Saved to data/btc_1min_raw.csv")

    print("🚀 Running live_predictor.py...")
    subprocess.run(["python", "-m", "modules.live_predictor"])

# Schedule it to run every 60 seconds
schedule.every(60).seconds.do(fetch_and_save_binance_data)

if __name__ == "__main__":
    print("🧠 Auto Live Runner started (fetch + predict every minute)")
    fetch_and_save_binance_data()  # run immediately on start
    while True:
        schedule.run_pending()
        time.sleep(1)
