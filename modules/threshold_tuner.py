# modules/threshold_tuner.py

import numpy as np
import pandas as pd
import joblib
import os
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_thresholds(y_true, probs, steps=50):
    results = []
    thresholds = np.linspace(0.5, 0.99, steps)

    for threshold in thresholds:
        preds = (probs[:, 1] > threshold).astype(int)

        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)

        results.append({
            "threshold": threshold,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1
        })

    df = pd.DataFrame(results)

    if df.empty or df['f1'].max() == 0:
        print("⚠️ No valid threshold found — check model output.")
        return pd.DataFrame()

    df.sort_values(by="precision", ascending=False, inplace=True)
    return df


def main():
    print("📊 Tuning decision threshold for EUR/USD with confidence + accuracy filtering...")

    # === Load models ===
    meta = joblib.load("models/meta_best_model.joblib")
    xgb = joblib.load("models/xgb_best_model.joblib")
    lgbm = joblib.load("models/lgb_best_model.joblib")
    cat = joblib.load("models/cat_best_model.joblib")

    df = pd.read_csv("data/eurusd_1min_labelled.csv")
    with open("models/feature_cols.json") as f:
        feature_cols = json.load(f)

    df.dropna(subset=feature_cols + ["Signal"], inplace=True)
    df = df[(df["volatility_rank"] > 0.7) & (df["body_ratio"] > 0.1)]
    df = df[df["hour"].between(9, 12) | df["hour"].between(14, 17)]

    X = df[feature_cols]
    y = df["Signal"].values

    # === Generate meta-features (3D input to meta-model) ===
    xgb_prob = xgb.predict_proba(X)[:, 1]
    lgb_prob = lgbm.predict_proba(X)[:, 1]
    cat_prob = cat.predict_proba(X)[:, 1]

    meta_input = np.column_stack([xgb_prob, lgb_prob, cat_prob])
    meta_probs = meta.predict_proba(meta_input)

    print(f"🔍 meta_input shape: {meta_input.shape}")
    print(f"🔍 meta_probs shape: {meta_probs.shape}")
    print(f"🔍 y shape: {y.shape}")
    print(f"🔍 Sample meta_probs: {meta_probs[:5]}")

    # === Evaluate Thresholds ===
    results = evaluate_thresholds(y, meta_probs)

    if results.empty:
        print("🚫 No threshold results generated.")
        return

    best_row = results.iloc[0]
    best_threshold = best_row["threshold"]
    print(f"\n✅ Best Threshold: {best_threshold:.4f} (Precision: {best_row['precision']:.2%}, F1: {best_row['f1']:.2%})")

    # === Save ===
    os.makedirs("config", exist_ok=True)
    with open("config/best_thresholds.json", "w") as f:
        json.dump({"meta_model": best_threshold}, f)

    print("💾 Saved to config/best_thresholds.json")
    print("\n🔝 Top 10 Thresholds by Precision:")
    print(results.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
