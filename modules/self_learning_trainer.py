# self_learning_trainer.py

import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import lightgbm as lgb

DATA_PATH = 'data/real_trades.csv'
MODEL_PATH = 'models/self_learning_model.joblib'

def train_from_real_trades():
    if not os.path.exists(DATA_PATH):
        print("❌ No real_trades.csv found. Run live predictor first.")
        return

    df = pd.read_csv(DATA_PATH)

    # Clean and prepare
    if 'result' not in df.columns or 'signal' not in df.columns:
        print("❌ Missing required columns.")
        return

    df = df.dropna()
    df = df[df['result'].isin(['WON', 'LOST'])]

    df['target'] = df['result'].map({'WON': 1, 'LOST': 0})

    # Drop irrelevant columns
    drop_cols = ['timestamp', 'result', 'wallet_amount', 'signal']
    X = df.drop(columns=drop_cols + ['target'], errors='ignore')
    y = df['target']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

    # Train LightGBM model
    clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy on validation set: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # Save
    joblib.dump(clf, MODEL_PATH)
    print(f"✅ Self-learning model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_from_real_trades()
