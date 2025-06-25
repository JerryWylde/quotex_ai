# daily_auto_pipeline.py

import os
import time
import logging
from datetime import datetime
import pandas as pd
from modules.data_fetch import fetch_binance_data
os.makedirs("logs", exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("logs/daily_pipeline.log"),
        logging.StreamHandler()
    ]
)

def run_step("Run model training pipeline", "python train_model.py"):
    logging.info(f" Starting: {description}")
    code = os.system(command)
    if code != 0:
        logging.error(f" Failed: {description} (exit code: {code})")
        raise RuntimeError(f"Step failed: {description}")
    logging.info(f" Completed: {description}\n")


def main():
    start_time = time.time()
    logging.info(" Starting Daily Auto-Pipeline (Quotex AI)")

    try:
        logging.info(" Fetching fresh BTC 1-min data...")
        df = fetch_binance_data(symbol="BTCUSDT", interval="1m", lookback_days=7)
        df.to_csv("data/btc_1min_raw.csv", index=False)
        run_step("Run model training pipeline", "python run_training_pipeline.py")
        run_step("Tune signal thresholds", "python -m modules.threshold_tuner")
        run_step("Run live predictions", "python -m modules.live_predictor")

        # Optional: Log confident trades summary
        confident_path = 'data/predictions/confident_trades.csv'
        if os.path.exists(confident_path):
            trades = pd.read_csv(confident_path)
            logging.info(f" Confident trades today: {len(trades)}")
            logging.info(trades.tail())


    except Exception as e:
        logging.exception(f"Pipeline failed with error: {e}")

    finally:
        duration = round(time.time() - start_time, 2)
        logging.info(f" Daily Auto-Pipeline finished in {duration} seconds")


if __name__ == "__main__":
    main()
