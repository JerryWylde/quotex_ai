# modules/analyze_predictions.py

import pandas as pd

df = pd.read_csv("data/predictions/confident_trades.csv")

accuracy = (df['actual'] == (df['probability'] > 0.5).astype(int)).mean()
num_trades = len(df)

print(f"🎯 Sniper Trade Accuracy: {accuracy:.4f}")
print(f"📌 Number of Sniper Trades: {num_trades}")

# Optional: Distribution inspection
print(df['actual'].value_counts(normalize=True))
