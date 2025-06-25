# modules/data_fetch.py

import pandas as pd
import requests
import time
import os
from datetime import datetime, timedelta

TRADERMADE_API_KEY = "sCsja0b7xS95OXzz7pMs"  # Replace with your actual TraderMade key

def fetch_tradermade_data(
    currency_pair="EURUSD",
    interval="1min",
    lookback_days=10,
    save_path="data/eurusd_1min_raw.csv"
):
    """
    Fetch historical FX data from TraderMade API and save to CSV with deduplication.
    """
    print(f"📡 Fetching {currency_pair} {interval} data for {lookback_days} days...")

    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=lookback_days)

    url = "https://marketdata.tradermade.com/api/v1/history"
    params = {
        "currency": currency_pair,
        "api_key": TRADERMADE_API_KEY,
        "format": "records",
        "start_date": start_time.strftime("%Y-%m-%d"),
        "end_date": end_time.strftime("%Y-%m-%d"),
        "interval": interval
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise RuntimeError(f"❌ API request failed: {response.text}")

    data = response.json()
    if not data.get("quotes"):
        raise ValueError("❌ No quote data received.")

    df = pd.DataFrame(data["quotes"])
    df.rename(columns={
        "date": "Timestamp",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close"
    }, inplace=True)

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["Volume"] = 0  # TraderMade has no volume

    df = df[["Timestamp", "Open", "High", "Low", "Close", "Volume"]]
    df = df.astype({"Open": float, "High": float, "Low": float, "Close": float})

    # === Deduplication if file exists
    if os.path.exists(save_path):
        existing_df = pd.read_csv(save_path, parse_dates=["Timestamp"])
        df = pd.concat([existing_df, df]).drop_duplicates("Timestamp").sort_values("Timestamp")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"✅ {len(df)} candles saved to {save_path}")
    return df

def fetch_latest_candle(currency_pair="EURUSD") -> pd.DataFrame:
    """
    Fetch synthetic latest 1-minute candle from live tick quote.
    """
    url = "https://marketdata.tradermade.com/api/v1/live"
    params = {
        "currency": currency_pair,
        "api_key": TRADERMADE_API_KEY,
        "format": "records"
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise RuntimeError(f"❌ Live API request failed: {response.text}")

    data = response.json()
    quote = data.get("quotes", [{}])[0]
    if not quote or "mid" not in quote:
        raise ValueError("❌ No valid live quote received.")

    # Build a synthetic 1-min candle from mid price
    mid_price = quote["mid"]
    timestamp = datetime.utcnow().replace(second=0, microsecond=0)

    df = pd.DataFrame([{
        "Timestamp": timestamp,
        "Open": mid_price,
        "High": mid_price,
        "Low": mid_price,
        "Close": mid_price,
        "Volume": 0
    }])
    return df

if __name__ == "__main__":
    fetch_tradermade_data()
    print(fetch_latest_candle())
