#profit_simulation.py
import pandas as pd
import numpy as np
import joblib
import json
import os

# === CONFIG ===
CSV_PATH = "data/eurusd_1min_labelled.csv"
MODELS_DIR = "models"
INITIAL_BALANCE = 1000
PAYOUT = 0.68
MAX_DAYS = 10
TRADE_INTERVAL_MINUTES = 3
CONFIDENCE_THRESHOLD = 0.76  # 💥 Brutally high-confidence
HOLD_MARGIN = 0.10           # 🔒 Min directional clarity
MAX_TRADES_PER_DAY = 12
MIN_TRADES_PER_DAY = 10
MAX_LOSSES_PER_DAY = 2
TRADING_WINDOWS = [(9, 12), (14, 17)]  # Only trade during these hour windows

# === Load Data ===
print("[INFO] Loading labelled data...")
df = pd.read_csv(CSV_PATH)
df.dropna(subset=["Signal"], inplace=True)

# === Filter trading windows ===
df = df[df['hour'].apply(lambda h: any(start <= h < end for start, end in TRADING_WINDOWS))]
df = df.iloc[::TRADE_INTERVAL_MINUTES].reset_index(drop=True)

# === Features ===
with open(os.path.join(MODELS_DIR, "feature_cols.json")) as f:
    feature_cols = json.load(f)

X = df[feature_cols]
y_true = df["Signal"]

# === Load Models ===
print("[INFO] Loading trained models...")
xgb = joblib.load(os.path.join(MODELS_DIR, "xgb_best_model.joblib"))
lgb = joblib.load(os.path.join(MODELS_DIR, "lgb_best_model.joblib"))
cat = joblib.load(os.path.join(MODELS_DIR, "cat_best_model.joblib"))
meta = joblib.load(os.path.join(MODELS_DIR, "meta_best_model.joblib"))

# === Predictions ===
print("[INFO] Running predictions...")
xgb_pred = xgb.predict_proba(X)[:, 1]
lgb_pred = lgb.predict_proba(X)[:, 1]
cat_pred = cat.predict_proba(X)[:, 1]
meta_input = np.column_stack([xgb_pred, lgb_pred, cat_pred])
meta_probs = meta.predict_proba(meta_input)

# === Sniper Simulation ===
print("[INFO] Starting SNIPER simulation...")
balance = INITIAL_BALANCE
trade_log = []
daily_summary = []

skipped_confidence = 0
skipped_hold = 0
wins = 0
losses = 0

entries_per_day = len(df) // MAX_DAYS
total_entries = len(meta_probs)
day_start = 0

for day in range(MAX_DAYS):
    trades_today = 0
    losses_today = 0
    winrate_check = []
    daily_trades = []

    print(f"\n[DAY {day+1}] Starting with balance ₹{balance:.2f}")
    day_end = min(day_start + entries_per_day, total_entries)
    day_indices = list(range(day_start, day_end))

    # First pass: strict filter
    for i in day_indices:
        prob_down, prob_up = meta_probs[i]
        confidence = max(prob_up, prob_down)
        margin = abs(prob_up - prob_down)

        if confidence < CONFIDENCE_THRESHOLD:
            skipped_confidence += 1
            continue

        if margin < HOLD_MARGIN:
            skipped_hold += 1
            continue

        pred = 1 if prob_up > prob_down else 0
        true = y_true.iloc[i]
        outcome = "WON" if pred == true else "LOST"
        if balance <= 1000:
            trade_amount = 100
        elif balance <= 1999:
            trade_amount = 200
        elif balance <= 4999:
            trade_amount = 400
        elif balance <= 9999:
            trade_amount = 800
        elif balance <= 19999:
            trade_amount = 2000
        elif balance <= 39999:
            trade_amount = 5000
        else:
            trade_amount = 10000
        profit = trade_amount * PAYOUT if outcome == "WON" else -trade_amount
        balance += profit

        winrate_check.append(outcome == "WON")
        if outcome == "WON":
            wins += 1
        else:
            losses += 1
            losses_today += 1

        trades_today += 1

        print(
            f"Trade {trades_today:2} | Time: {int(df.iloc[i]['hour']):02d}:{int(df.iloc[i]['minute']):02d} "
            f"| Conf: {confidence:.3f} | Pred: {pred} | Actual: {true} "
            f"| Result: {outcome} | Bal: ₹{balance:.2f}"
        )

        trade_log.append({
            "Day": day + 1,
            "Index": i,
            "Time": f"{int(df.iloc[i]['hour']):02d}:{int(df.iloc[i]['minute']):02d}",
            "Confidence": confidence,
            "Margin": margin,
            "Prediction": pred,
            "Actual": int(true),
            "Outcome": outcome,
            "Trade Amount": trade_amount,
            "Profit": profit,
            "Balance": balance,
        })

        if trades_today >= MAX_TRADES_PER_DAY or losses_today >= MAX_LOSSES_PER_DAY:
            break

    # Fallback: loosen conditions if < MIN_TRADES_PER_DAY
    if trades_today < MIN_TRADES_PER_DAY and losses_today < MAX_LOSSES_PER_DAY:
        for i in day_indices:
            if trades_today >= MIN_TRADES_PER_DAY:
                break
            if any(t["Index"] == i for t in trade_log):
                continue

            prob_down, prob_up = meta_probs[i]
            confidence = max(prob_up, prob_down)
            margin = abs(prob_up - prob_down)

            if confidence < CONFIDENCE_THRESHOLD - 0.03:
                continue
            if margin < HOLD_MARGIN - 0.05:
                continue

            pred = 1 if prob_up > prob_down else 0
            true = y_true.iloc[i]
            outcome = "WON" if pred == true else "LOST"
            trade_amount = round(balance * 0.05)
            profit = trade_amount * PAYOUT if outcome == "WON" else -trade_amount
            balance += profit

            winrate_check.append(outcome == "WON")
            if outcome == "WON":
                wins += 1
            else:
                losses += 1
                losses_today += 1

            trades_today += 1

            print(
                f"(Fallback) Trade {trades_today:2} | Time: {int(df.iloc[i]['hour']):02d}:{int(df.iloc[i]['minute']):02d} "
                f"| Conf: {confidence:.3f} | Pred: {pred} | Actual: {true} "
                f"| Result: {outcome} | Bal: ₹{balance:.2f}"
            )

            trade_log.append({
                "Day": day + 1,
                "Index": i,
                "Time": f"{int(df.iloc[i]['hour']):02d}:{int(df.iloc[i]['minute']):02d}",
                "Confidence": confidence,
                "Margin": margin,
                "Prediction": pred,
                "Actual": int(true),
                "Outcome": outcome,
                "Trade Amount": trade_amount,
                "Profit": profit,
                "Balance": balance,
            })

            if losses_today >= MAX_LOSSES_PER_DAY:
                break

    winrate = winrate_check.count(True) / len(winrate_check) if winrate_check else 0
    print(f"[DAY {day+1} END] Trades: {trades_today}, Losses: {losses_today}, Winrate: {winrate:.2%}")

    daily_summary.append({
        "Day": day + 1,
        "Ending Balance": balance,
        "Trades": trades_today,
        "Losses": losses_today
    })

    day_start = day_end
    if day_start >= total_entries:
        break

# === Summary ===
results = pd.DataFrame(trade_log)
final_balance = results["Balance"].iloc[-1] if not results.empty else INITIAL_BALANCE

print("\n📈 SNIPER SIMULATION RESULTS")
print(f"Total Executed Trades: {len(results)}")
print(f"Skipped (Low Confidence): {skipped_confidence}")
print(f"Skipped (HOLD Zone): {skipped_hold}")
print(f"Wins: {wins} | Losses: {losses}")
print(f"Final Balance: ₹{final_balance:.2f}")
print(f"Net Profit: ₹{final_balance - INITIAL_BALANCE:.2f}")
print(f"Accuracy: {wins / len(results) * 100:.2f}%" if len(results) > 0 else "No trades executed.")

print("\n📅 DAILY BALANCES")
for entry in daily_summary:
    print(f"🔹 Day {entry['Day']}: ₹{entry['Ending Balance']:.2f} | Trades: {entry['Trades']} | Losses: {entry['Losses']}")

# === Save Logs ===
results.to_csv("data/pnl_simulation.csv", index=False)
pd.DataFrame(daily_summary).to_csv("data/daily_balances.csv", index=False)
print("[SAVED] Trade log → data/pnl_simulation.csv")
print("[SAVED] Daily balances → data/daily_balances.csv")