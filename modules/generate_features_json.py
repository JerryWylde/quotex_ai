# modules/generate_features_json.py

import pandas as pd
import json
import os
from modules.feature_engineering import engineer_topnotch_features

RAW_DATA_PATH = "data/eurusd_1min_raw.csv"
OUTPUT_PATH = "models/feature_cols.json"

def main():
    print("📦 Loading raw EUR/USD data...")
    df_raw = pd.read_csv(RAW_DATA_PATH)

    print("🧠 Engineering features + generating signal...")
    df = engineer_topnotch_features(df_raw, add_signal=True)

    print("🧹 Extracting clean feature columns (excluding Signal)...")
    feature_cols = [col for col in df.columns if col != 'Signal']

    os.makedirs("models", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(feature_cols, f, indent=4)

    print(f"✅ Saved {len(feature_cols)} safe feature columns to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
