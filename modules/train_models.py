import os
import joblib
import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import json

# === MODULES ===
from modules.signal_generator import create_signal
from modules.feature_engineering import engineer_topnotch_features

# === LOGGING CONFIG ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# === PATHS ===
DATA_CSV_PATH = "data/eurusd_1min_labelled.csv"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# === MODEL TRAINING ===
def train_base_models(X_train, y_train, X_val, y_val):
    logging.info("Training XGBoost...")
    with open("best_params/best_xgboost_params.json") as f:
        xgb_params = json.load(f)
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[xgb.callback.EarlyStopping(rounds=50, save_best=True)],
        verbose=False
    )

    logging.info("Training LightGBM...")
    with open("best_params/best_lightgbm_params.json") as f:
        lgb_params = json.load(f)
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )

    logging.info("Training CatBoost...")
    with open("best_params/best_catboost_params.json") as f:
        cat_params = json.load(f)
    cat_model = CatBoostClassifier(**cat_params)
    cat_model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=False
    )

    return xgb_model, lgb_model, cat_model

# === MAIN ===
def main():
    logging.info(f"Loading dataset from {DATA_CSV_PATH} ...")
    df = pd.read_csv(DATA_CSV_PATH)
    df.dropna(inplace=True)

    # === FILTER STRONG SIGNALS ===
    df = df[(df['volatility_rank'] > 0.7) & (df['body_ratio'] > 0.1)]

    # === TRADING SESSION FILTER ===
    if 'hour' in df.columns:
        df = df[df['hour'].between(9, 12) | df['hour'].between(14, 17)]
    else:
        logging.warning("⚠️ 'hour' column not found — skipping time filtering.")

    feature_cols = [col for col in df.columns if col != 'Signal']
    label_col = 'Signal'
    df.dropna(subset=feature_cols + [label_col], inplace=True)

    tscv = TimeSeriesSplit(n_splits=5)
    meta_features, meta_labels = [], []

    base_models = {"xgb": None, "lgb": None, "cat": None}

    for fold, (train_idx, val_idx) in enumerate(tscv.split(df), 1):
        logging.info(f"Starting fold {fold}...")

        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
        X_train, y_train = train_df[feature_cols], train_df[label_col]
        X_val, y_val = val_df[feature_cols], val_df[label_col]

        xgb_model, lgb_model, cat_model = train_base_models(X_train, y_train, X_val, y_val)

        # Store final models trained on full data
        X_all, y_all = df[feature_cols], df[label_col]
        final_xgb, final_lgb, final_cat = train_base_models(X_all, y_all, X_all, y_all)
        base_models["xgb"] = final_xgb
        base_models["lgb"] = final_lgb
        base_models["cat"] = final_cat

        xgb_val_pred = xgb_model.predict_proba(X_val)[:, 1]
        lgb_val_pred = lgb_model.predict_proba(X_val)[:, 1]
        cat_val_pred = cat_model.predict_proba(X_val)[:, 1]

        meta_features.append(np.column_stack([xgb_val_pred, lgb_val_pred, cat_val_pred]))
        meta_labels.append(y_val.values)

        logging.info(f"Fold {fold} Accuracies:")
        logging.info(f" - XGB: {accuracy_score(y_val, (xgb_val_pred > 0.5).astype(int)):.4f}")
        logging.info(f" - LGB: {accuracy_score(y_val, (lgb_val_pred > 0.5).astype(int)):.4f}")
        logging.info(f" - CAT: {accuracy_score(y_val, (cat_val_pred > 0.5).astype(int)):.4f}")

    meta_features = np.vstack(meta_features)
    meta_labels = np.concatenate(meta_labels)

    pos = np.sum(meta_labels)
    neg = len(meta_labels) - pos
    logging.info(f"🔍 Meta training class distribution: {pos} positive | {neg} negative ({(pos/len(meta_labels))*100:.2f}% positive)")

    logging.info("Training hybrid meta-model (VotingClassifier)...")
    hybrid_meta = VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=1000, class_weight='balanced')),
            ("rf", RandomForestClassifier(n_estimators=50, max_depth=5, class_weight='balanced', random_state=42))
        ],
        voting='soft'
    )

    calibrated_meta = CalibratedClassifierCV(hybrid_meta, cv=3)
    calibrated_meta.fit(meta_features, meta_labels)

    # Save meta probs for threshold tuning
    meta_probs = calibrated_meta.predict_proba(meta_features)
    np.save("data/meta_probs.npy", meta_probs)
    np.save("data/meta_labels.npy", meta_labels)

    meta_acc = accuracy_score(meta_labels, calibrated_meta.predict(meta_features))
    logging.info(f"🔥 Final Meta-Model Accuracy: {meta_acc:.4f}")

    # Save all models
    joblib.dump(base_models["xgb"], os.path.join(MODELS_DIR, "xgb_best_model.joblib"))
    joblib.dump(base_models["lgb"], os.path.join(MODELS_DIR, "lgb_best_model.joblib"))
    joblib.dump(base_models["cat"], os.path.join(MODELS_DIR, "cat_best_model.joblib"))
    joblib.dump(calibrated_meta, os.path.join(MODELS_DIR, "meta_best_model.joblib"))

    # Save feature columns
    with open(os.path.join(MODELS_DIR, "feature_cols.json"), "w") as f:
        json.dump(feature_cols, f)

    logging.info("✅ Training process completed successfully.")

if __name__ == "__main__":
    main()
