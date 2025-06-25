# live_predictor.py

import os
import time
import json
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import traceback

from modules.data_fetch import fetch_latest_candle
from modules.feature_engineering import engineer_topnotch_features
from modules.signal_generator import create_signal
from telegram_module.trade_executor import TradeExecutor
from telegram_module.telebot_alerts import send_telegram_alert
from modules.wallet_manager import initialize_wallet, update_wallet_after_trade, can_trade

# === Logging ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

# === Paths ===
BASE_DIR = os.getcwd()
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'eurusd_1min_raw.csv')
THRESHOLD = 0.51
TRADE_LOG_PATH = os.path.join(DATA_DIR, 'real_trades.csv')

RETRAIN_INTERVAL = 68

def load_models():
    logging.info("🔄 Loading models...")
    xgb_model = joblib.load(os.path.join(MODELS_DIR, "xgb_best_model.joblib"))
    lgb_model = joblib.load(os.path.join(MODELS_DIR, "lgb_best_model.joblib"))
    cat_model = joblib.load(os.path.join(MODELS_DIR, "cat_best_model.joblib"))
    meta_model = joblib.load(os.path.join(MODELS_DIR, "meta_best_model.joblib"))
    return xgb_model, lgb_model, cat_model, meta_model

def ensemble_predict(xgb, lgb, cat, meta, X: pd.DataFrame):
    xgb_p = float(xgb.predict_proba(X)[0][1])
    lgb_p = float(lgb.predict_proba(X)[0][1])
    cat_p = float(cat.predict_proba(X)[0][1])
    
    print(f"[DEBUG] Model outputs → XGB: {xgb_p:.4f}, LGB: {lgb_p:.4f}, CAT: {cat_p:.4f}")
    
    meta_input = np.array([[xgb_p, lgb_p, cat_p]])
    prob_up = float(meta.predict_proba(meta_input)[0][1])
    prob_down = 1 - prob_up
    
    return prob_up, prob_down, xgb_p, lgb_p, cat_p

def log_trade(ts, signal, prob_up, prob_down, result, wallet_amount, features, xgb_p, lgb_p, cat_p):
    row = {
        "timestamp": str(ts),
        "signal": signal,
        "prob_up": prob_up,
        "prob_down": prob_down,
        "xgb": xgb_p,
        "lgb": lgb_p,
        "cat": cat_p,
        "result": result,
        "wallet_amount": wallet_amount
    }
    row.update(features)
    df = pd.DataFrame([row])
    if not os.path.exists(TRADE_LOG_PATH):
        df.to_csv(TRADE_LOG_PATH, index=False)
    else:
        df.to_csv(TRADE_LOG_PATH, mode='a', header=False, index=False)

def evaluate_trade(prev_close, current_close, signal):
    if signal == 1 and current_close > prev_close:
        return "WON"
    elif signal == 0 and current_close < prev_close:
        return "WON"
    else:
        return "LOST"

def retrain_meta_model():
    logging.info("🔁 Retraining meta-model using real_trades.csv ...")
    try:
        df = pd.read_csv(TRADE_LOG_PATH)
        df.dropna(inplace=True)
        X = df[["xgb", "lgb", "cat"]]
        y = (df["result"] == "WON").astype(int)

        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        model.fit(X, y)
        joblib.dump(model, os.path.join(MODELS_DIR, "meta_best_model.joblib"))
        logging.info("✅ Meta-model retrained and saved.")
    except Exception as e:
        logging.error(f"❌ Retrain failed: {e}")

def main_loop():
    logging.info("🚀 Starting live EUR/USD sniper loop...")

    try:
        xgb, lgb, cat, meta = load_models()
    except Exception as e:
        logging.critical(f"❌ Failed to load models: {e}")
        return

    wallet = initialize_wallet()
    executor = TradeExecutor(real_trading=False, headless_browser=False)
    trade_count = 0

    try:
        while True:
            try:
                if not can_trade(wallet):
                    logging.warning("🛑 Drawdown limit exceeded. Waiting...")
                    time.sleep(60)
                    continue

                logging.info("📡 Fetching latest EURUSD candle...")
                new_candle = fetch_latest_candle("EURUSD")
                print(f"[DEBUG] Appending candle with Timestamp: {new_candle['Timestamp'].values[0]}")
                new_candle.to_csv(RAW_DATA_PATH, mode='a', header=not os.path.exists(RAW_DATA_PATH), index=False)

                df_raw = pd.read_csv(RAW_DATA_PATH, parse_dates=["Timestamp"])
                features = engineer_topnotch_features(df_raw.copy())
                X = features.select_dtypes(include=[np.number]).drop(columns=["Signal"], errors="ignore").iloc[-1:]
                feature_snapshot = X.to_dict(orient="records")[0]

                print("[DEBUG] Latest engineered features snapshot:")
                print(X.T)

                if X.isnull().values.any():
                    logging.error("❌ NaNs detected. Skipping cycle.")
                    time.sleep(10)
                    continue

                prob_up, prob_down, xgb_p, lgb_p, cat_p = ensemble_predict(xgb, lgb, cat, meta, X)
                signal, confidence = create_signal(prob_up, threshold=THRESHOLD)

                logging.info(
                    f"📊 Signal: {'CALL' if signal == 1 else 'PUT' if signal == 0 else 'NO TRADE'} | ↑: {prob_up:.3f} ↓: {prob_down:.3f} | Confidence: {confidence:.3f} | Threshold: {THRESHOLD}"
                )

                if signal in [0, 1]:
                    send_telegram_alert(signal, prob_up, prob_down, THRESHOLD)
                    direction = "CALL" if signal == 1 else "PUT"

                    logging.info("🕒 Waiting for next candle to evaluate result...")
                    time.sleep(190)

                    df_updated = pd.read_csv(RAW_DATA_PATH, parse_dates=["Timestamp"])
                    prev_close = df_updated["Close"].iloc[-2]
                    current_close = df_updated["Close"].iloc[-1]
                    result = evaluate_trade(prev_close, current_close, signal)

                    if result == "WON":
                        profit = wallet['current_invest_amount'] * 0.68
                        wallet = update_wallet_after_trade(wallet, profit, won=True)
                        logging.info(f"✅ Trade WON. New Wallet: ₹{wallet['current_invest_amount']}")
                    else:
                        wallet = update_wallet_after_trade(wallet, wallet['current_invest_amount'], won=False)
                        logging.error(f"❌ Trade LOST. New Wallet: ₹{wallet['current_invest_amount']}")

                    log_trade(
                        datetime.now(timezone.utc),
                        direction,
                        prob_up,
                        prob_down,
                        result,
                        wallet['current_invest_amount'],
                        feature_snapshot,
                        xgb_p,
                        lgb_p,
                        cat_p
                    )

                    trade_count += 1

                    if trade_count % RETRAIN_INTERVAL == 0:
                        retrain_meta_model()

                else:
                    logging.info("⚠️ No confident signal. Skipping.")

                now = datetime.utcnow()
                sleep_time = 180 - (now.minute % 3) * 60 - now.second
                logging.info(f"⏳ Sleeping {sleep_time}s until next cycle...")
                time.sleep(sleep_time)

            except KeyboardInterrupt:
                logging.info("🛑 Interrupted manually. Exiting loop.")
                break
            except Exception as e:
                logging.error(f"❌ Error in loop:\n{traceback.format_exc()}")
                time.sleep(10)

    finally:
        executor.close()
        logging.info("🔚 Sniper loop terminated.")

if __name__ == "__main__":
    main_loop()
