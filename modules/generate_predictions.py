# modules/generate_predictions.py

import pandas as pd
import joblib
import json
import os
from modules.feature_engineering import engineer_topnotch_features

# === Paths ===
DATA_PATH = "data/eurusd_1min_raw.csv"
FEATURE_COLS_PATH = "models/feature_cols.json"
MODELS_DIR = "models"
THRESHOLD_PATH = "config/best_thresholds.json"
OUTPUT_PATH = "data/predictions/live_predictions.csv"
CONFIDENT_PATH = "data/predictions/confident_trades.csv"

# === Load everything ===
df_raw = pd.read_csv(DATA_PATH)
df = engineer_topnotch_features(df_raw, add_signal=True)

with open(FEATURE_COLS_PATH) as f:
    feature_cols = json.load(f)

with open(THRESHOLD_PATH) as f:
    best_threshold = json.load(f)["threshold"]

X = df[feature_cols]
y = df['Signal']

# === Models ===
xgb = joblib.load(f"{MODELS_DIR}/xgb_best_model.joblib")
lgb = joblib.load(f"{MODELS_DIR}/lgb_best_model.joblib")
cat = joblib.load(f"{MODELS_DIR}/cat_best_model.joblib")
meta = joblib.load(f"{MODELS_DIR}/meta_best_model.joblib")

# === Predictions ===
xgb_preds = xgb.predict_proba(X)[:, 1]
lgb_preds = lgb.predict_proba(X)[:, 1]
cat_preds = cat.predict_proba(X)[:, 1]

meta_input = pd.DataFrame({
    'xgb': xgb_preds,
    'lgb': lgb_preds,
    'cat': cat_preds
})
meta_preds = meta.predict_proba(meta_input)[:, 1]

df_out = pd.DataFrame({
    'probability': meta_preds,
    'actual': y
})

# === Filter confident trades ===
confident_df = df_out[df_out['probability'] > best_threshold]
os.makedirs("data/predictions", exist_ok=True)
df_out.to_csv(OUTPUT_PATH, index=False)
confident_df.to_csv(CONFIDENT_PATH, index=False)

print(f"✅ All predictions saved to {OUTPUT_PATH}")
print(f"🎯 Confident sniper trades saved to {CONFIDENT_PATH} with threshold {best_threshold}")
print(f"📈 Total sniper trades: {len(confident_df)}")
