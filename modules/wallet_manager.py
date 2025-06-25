import json
import os
import threading
import logging

# Path to wallet balance storage
WALLET_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'wallet_balance.json')
_lock = threading.Lock()


def initialize_wallet():
    """
    Creates a new wallet file with default values if not already present.
    Returns the wallet dictionary.
    """
    if not os.path.exists(WALLET_PATH):
        wallet = {
            "total_profits_earned": 0.0,
            "peak_profit": 0.0,
            "current_loss": 0.0,
            "current_invest_amount": 70.0,
            "consecutive_losses": 0,
            "trades_executed": 0,
            "was_stopped": False
        }
        save_wallet(wallet)
        return wallet
    return load_wallet()


def load_wallet():
    with _lock:
        with open(WALLET_PATH, 'r') as f:
            return json.load(f)


def save_wallet(wallet):
    with _lock:
        with open(WALLET_PATH, 'w') as f:
            json.dump(wallet, f, indent=4)


def update_wallet_after_trade(wallet, profit_loss_amount: float, won: bool):
    """
    Updates wallet after a trade is executed.
    - Adds to total profits if won.
    - Adds to current_loss if lost.
    - Tracks trade count and loss streak.
    - Dynamically adjusts investment size based on growth tiers.
    """
    if won:
        wallet['total_profits_earned'] += profit_loss_amount
        wallet['peak_profit'] = max(wallet['peak_profit'], wallet['total_profits_earned'])
        wallet['current_loss'] = 0.0
        wallet['consecutive_losses'] = 0
    else:
        wallet['current_loss'] += abs(profit_loss_amount)
        wallet['consecutive_losses'] += 1

    # Dynamically adjust investment amount based on total profits
    profit = wallet['total_profits_earned']
    wallet['current_invest_amount'] = (
        1500.0 if profit >= 50000 else
        500.0 if profit >= 10000 else
        200.0 if profit >= 1000 else
        70.0
    )

    wallet['trades_executed'] += 1
    save_wallet(wallet)
    log_wallet_snapshot(wallet)
    return wallet


def can_trade(wallet):
    """
    Risk management rule:
    - Block trading if drawdown from peak exceeds 20%.
    - If no profit yet, allow trade.
    """
    peak = wallet.get('peak_profit', wallet['total_profits_earned'])
    current = wallet['total_profits_earned']
    drawdown = peak - current

    if peak > 0 and drawdown >= 0.20 * peak:
        wallet["was_stopped"] = True
        save_wallet(wallet)
        return False
    return True


def reset_losses(wallet):
    """
    Resets loss counters after recovery or manual override.
    """
    wallet['current_loss'] = 0.0
    wallet['consecutive_losses'] = 0
    wallet['was_stopped'] = False
    save_wallet(wallet)
    return wallet


def reset_wallet():
    """
    Fully resets the wallet state to initial conditions.
    Use this only manually.
    """
    wallet = {
        "total_profits_earned": 0.0,
        "peak_profit": 0.0,
        "current_loss": 0.0,
        "current_invest_amount": 70.0,
        "consecutive_losses": 0,
        "trades_executed": 0,
        "was_stopped": False
    }
    save_wallet(wallet)
    return wallet


def log_wallet_snapshot(wallet):
    """
    Logs current wallet state to console after each trade.
    """
    logging.info(
        f"💼 Wallet Snapshot → ₹{wallet['current_invest_amount']} | "
        f"Earned: ₹{wallet['total_profits_earned']} | Peak: ₹{wallet['peak_profit']} | "
        f"Drawdown: ₹{wallet['current_loss']} | Losses: {wallet['consecutive_losses']} | "
        f"Trades: {wallet['trades_executed']}"
    )
