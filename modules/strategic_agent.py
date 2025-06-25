import datetime

def get_live_threshold(default=0.60):
    # Can be dynamic based on market hours or volatility
    return default

def detect_market_pattern(df):
    # Basic: check recent volatility to label market as 'trending' or 'sideways'
    recent = df.tail(20)
    vol = (recent['High'] - recent['Low']).mean()
    if vol > 0.5:  # Adjust threshold per asset/timeframe
        return "trending"
    else:
        return "sideways"

def is_blacklisted_time():
    # Blacklist illiquid times, e.g. weekends, market close times, etc.
    now = datetime.datetime.utcnow()
    # Binance operates 24/7, but liquidity can be low on weekends
    if now.weekday() in [6]:  # Saturday, Sunday
        return True
    # Add your own time blackout periods here
    return False
