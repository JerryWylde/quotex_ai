import os 
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

DATA_DIR = "data"
RAW_CSV = os.path.join(DATA_DIR, "eurusd_raw.csv")  # ✅ Updated

def generate_synthetic_1min_data(days=10):
    """
    Generate synthetic 1-minute OHLCV EURUSD-like data for 'days' days.
    Only used if raw file missing.
    """
    np.random.seed(42)
    end_time = datetime.utcnow().replace(second=0, microsecond=0)
    start_time = end_time - timedelta(days=days)

    date_range = pd.date_range(start=start_time, end=end_time, freq="1T")
    length = len(date_range)

    # Generate synthetic prices around 1.08 for EURUSD
    price = 1.08 + np.cumsum(np.random.normal(0, 0.001, length))

    df = pd.DataFrame({"Timestamp": date_range})
    df["Open"] = price
    df["Close"] = price + np.random.normal(0, 0.0005, length)
    df["High"] = df[["Open", "Close"]].max(axis=1) + np.abs(np.random.normal(0, 0.0003, length))
    df["Low"] = df[["Open", "Close"]].min(axis=1) - np.abs(np.random.normal(0, 0.0003, length))
    df["Volume"] = 0  # No real volume in fallback

    # Ensure no negative lows
    df["Low"] = df["Low"].clip(lower=0)

    return df

def resample_to_3min(df_1min):
    """
    Resample 1-min OHLCV data to 3-min candles.
    """
    df_1min = df_1min.set_index("Timestamp")
    ohlcv_3min = df_1min.resample("3T").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna().reset_index()

    return ohlcv_3min

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    one_min_csv = os.path.join(DATA_DIR, "eurusd_1min_raw.csv")  # ✅ Changed
    if os.path.exists(one_min_csv):
        print(f"Found 1-min raw data at {one_min_csv}, loading for resampling...")
        df_1min = pd.read_csv(one_min_csv, parse_dates=["Timestamp"])
    else:
        print("No 1-min raw data found, generating synthetic data...")
        df_1min = generate_synthetic_1min_data(days=30)
        df_1min.to_csv(one_min_csv, index=False)
        print(f"Synthetic 1-min data saved to {one_min_csv}")

    print("Resampling to 3-minute candles...")
    df_3min = resample_to_3min(df_1min)

    print(f"Saving 3-minute OHLCV data to {RAW_CSV} ...")
    df_3min.to_csv(RAW_CSV, index=False)

    print("CSV generation completed.")

if __name__ == "__main__":
    main()
