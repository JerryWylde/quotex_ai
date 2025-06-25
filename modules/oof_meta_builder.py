# oof_meta_builder.py

import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from modules.feature_engineering import engineer_topnotch_features
from modules.signal_generator import create_signal_from_price

logging.basicConfig(level=logging.INFO)

# === Load and preprocess data ===
df = pd.read_csv("data/eurusd_1min_raw.csv")
df = create_signal_from_price(df)
df, feature_cols, target = engineer_topnotch_features(df)
X, y = df[feature_cols], target.reset_index(drop=True)

# === Base model definitions ===
base_models = {
    "xgb": XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6, n_jobs=2, random_state=42),
    "lgb": LGBMClassifier(n_estimators=300, learning_rate=0.1, max_depth=6, n_jobs=2, random_state=42),
    "cat": CatBoostClassifier(iterations=300, learning_rate=0.1, depth=6, verbose=0, random_seed=42)
}

# === Out-of-Fold Meta Feature Builder ===
n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)
meta_features = {name: np.zeros(len(X)) for name in base_models}

for fold, (train_idx, valid_idx) in enumerate(tscv.split(X)):
    logging.info(f"Fold {fold+1}/{n_splits}")
    X_train, X_val = X.iloc[train_idx], X.iloc[valid_idx]
    y_train = y.iloc[train_idx]

    for name, model in base_models.items():
        model.fit(X_train, y_train)
        meta_features[name][valid_idx] = model.predict_proba(X_val)[:, 1]  # probability for class 1

# === Assemble meta dataset ===
X_meta = pd.DataFrame(meta_features)

# === Meta Model: Hybrid Stack ===
hybrid_meta = StackingClassifier(
    estimators=[
        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ("rf", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)),
        ("gb", GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)),
        ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=500, random_state=42))
    ],
    final_estimator=CatBoostClassifier(iterations=100, learning_rate=0.1, depth=4, verbose=0, random_seed=42),
    passthrough=True,
    n_jobs=-1
)

hybrid_meta.fit(X_meta, y)
preds = hybrid_meta.predict(X_meta)
acc = accuracy_score(y, preds)
logging.info(f"🔥 FINAL META MODEL OOF ACCURACY: {acc:.4f}")

# Save models
joblib.dump(hybrid_meta, "models/meta_best_model.joblib")
logging.info("💾 Saved meta_best_model.joblib")
