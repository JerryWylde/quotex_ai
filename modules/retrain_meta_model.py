import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

# === Load confident trade data ===
print("[INFO] Loading confident trade data...")
df = pd.read_csv("data/meta_training_data.csv")

X = df[["xgb_pred", "lgb_pred", "cat_pred"]]
y = df["Signal"]

# === Split for evaluation ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# === Train meta-model ===
print("[INFO] Training sniper-tuned meta-model...")
meta_model = LogisticRegression()
meta_model.fit(X_train, y_train)

# === Evaluate ===
y_pred = meta_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"[RESULT] Meta-model accuracy on test set: {accuracy * 100:.2f}%")

# === Save model ===
joblib.dump(meta_model, os.path.join("models", "meta_best_model.joblib"))
print("[SAVED] Sniper meta-model → models/meta_best_model.joblib")
