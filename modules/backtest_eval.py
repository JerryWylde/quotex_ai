import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# === CONFIG ===
DATA_PATH = "data/eurusd_1min_labelled.csv"  # Pre-engineered file with 'Signal'
MODELS_DIR = "models"

# === Load Data ===
print("[INFO] Loading labelled data...")
df = pd.read_csv(DATA_PATH)
df.dropna(subset=['Signal'], inplace=True)

# === Load feature columns used during training ===
with open(os.path.join(MODELS_DIR, "feature_cols.json")) as f:
    feature_cols = json.load(f)
label_col = 'Signal'

# === Use last X% of data as unseen test (e.g., last 20%) ===
split_index = int(len(df) * 0.8)
test_df = df.iloc[split_index:]

X_test = test_df[feature_cols]
y_test = test_df[label_col]

# === Load Models ===
print("[INFO] Loading trained models...")
xgb_model = joblib.load(os.path.join(MODELS_DIR, "xgb_best_model.joblib"))
lgb_model = joblib.load(os.path.join(MODELS_DIR, "lgb_best_model.joblib"))
cat_model = joblib.load(os.path.join(MODELS_DIR, "cat_best_model.joblib"))
meta_model = joblib.load(os.path.join(MODELS_DIR, "meta_best_model.joblib"))

# === Base Model Predictions ===
print("[INFO] Making predictions with base models...")
xgb_pred = xgb_model.predict_proba(X_test)[:, 1]
lgb_pred = lgb_model.predict_proba(X_test)[:, 1]
cat_pred = cat_model.predict_proba(X_test)[:, 1]

# === Meta Ensemble ===
meta_input = np.column_stack([xgb_pred, lgb_pred, cat_pred])
meta_pred = meta_model.predict(meta_input)

# === Evaluation ===
acc = accuracy_score(y_test, meta_pred)
cm = confusion_matrix(y_test, meta_pred)
report = classification_report(y_test, meta_pred)

print("\n📊 [RESULTS - ENSEMBLE META MODEL]")
print(f"Accuracy: {acc:.4f}")
print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(report)
