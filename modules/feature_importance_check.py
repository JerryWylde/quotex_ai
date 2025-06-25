# feature_importance_check.py

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from modules.feature_engineering import engineer_topnotch_features
from modules.signal_generator import create_signal_from_price

# ====== LOAD DATA ======
df = pd.read_csv("data/eurusd_1min_raw.csv")
df = create_signal_from_price(df)
df, feature_cols, target = engineer_topnotch_features(df)
X = df[feature_cols]

# ====== LOAD MODELS ======
xgb_model = joblib.load("models/xgb_best_model.joblib")
lgb_model = joblib.load("models/lgb_best_model.joblib")
cat_model = joblib.load("models/cat_best_model.joblib")

# ====== FEATURE IMPORTANCE PLOTS ======
def plot_importance(model, model_name, fallback_features):
    # Try getting feature names from model
    try:
        if hasattr(model, "get_booster"):
            features = model.get_booster().feature_names
        elif hasattr(model, "feature_name_"):
            features = model.feature_name_
        else:
            features = fallback_features  # fallback
    except:
        features = fallback_features

    # Now get importances
    importances = model.feature_importances_
    if len(importances) != len(features):
        print(f"⚠️ WARNING: Mismatch in features vs importances for {model_name}")
        min_len = min(len(importances), len(features))
        features = features[:min_len]
        importances = importances[:min_len]

    imp_df = pd.DataFrame({"Feature": features, "Importance": importances})
    imp_df = imp_df.sort_values(by="Importance", ascending=False).head(25)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=imp_df, palette="viridis")
    plt.title(f"{model_name} Feature Importances")
    plt.tight_layout()
    plt.savefig(f"feature_importance_{model_name.lower()}.png")
    plt.close()


# ====== RUN ======
plot_importance(xgb_model, "XGBoost", feature_cols)
plot_importance(lgb_model, "LightGBM", feature_cols)
plot_importance(cat_model, "CatBoost", feature_cols)

print("✅ Feature importance plots saved as PNG.")