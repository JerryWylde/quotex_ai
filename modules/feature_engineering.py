#feature_engineering.py

import pandas as pd 
import numpy as np
from ta.trend import MACD, CCIIndicator, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from modules.signal_generator import create_signal

def engineer_topnotch_features(
    df: pd.DataFrame,
    add_signal: bool = True,
    threshold: float = 0.00015,
    high_confidence_labels_only: bool = False
) -> pd.DataFrame:
    df = df.copy()

    # Handle timestamp
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True, errors='coerce')
        df.set_index('Timestamp', inplace=True)

    # Temporal features
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['session'] = pd.cut(df['hour'], bins=[0, 6, 12, 18, 24], labels=[1, 2, 3, 4]).astype(float).fillna(0).astype(int)

    # Breakout logic
    df['is_breakout_high'] = (df['Close'] > df['High'].rolling(10).max().shift(1)).astype(int)
    df['is_breakout_low'] = (df['Close'] < df['Low'].rolling(10).min().shift(1)).astype(int)

    # Heikin-Ashi
    df['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    df['HA_Open'] = (df['Open'].shift(1) + df['Close'].shift(1)) / 2
    df['HA_High'] = df[['High', 'HA_Open', 'HA_Close']].max(axis=1)
    df['HA_Low'] = df[['Low', 'HA_Open', 'HA_Close']].min(axis=1)
    df['HA_Body'] = df['HA_Close'] - df['HA_Open']

    # Price movement
    df['price_change'] = df['Close'] - df['Close'].shift(1)
    df['candle_direction'] = np.sign(df['price_change'])
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['rolling_return_3'] = df['Close'].pct_change(periods=3)

    # MAs and trend
    df['ema_12'] = df['Close'].ewm(span=12).mean()
    df['ema_26'] = df['Close'].ewm(span=26).mean()
    df['ema_ratio'] = df['ema_12'] / (df['ema_26'] + 1e-9)

    # Indicators
    df['rsi'] = RSIIndicator(df['Close'], window=14).rsi()
    df['macd'] = MACD(df['Close']).macd_diff()
    df['cci'] = CCIIndicator(df['High'], df['Low'], df['Close'], window=20).cci()
    df['adx'] = ADXIndicator(df['High'], df['Low'], df['Close'], window=14).adx()

    # Oscillators
    stoch = StochasticOscillator(df['High'], df['Low'], df['Close'], window=14)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()

    # BB
    bb = BollingerBands(df['Close'], window=20, window_dev=2)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_width'] = df['bb_high'] - df['bb_low']
    df['bb_position'] = (df['Close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'] + 1e-9)

    # Volatility
    atr = AverageTrueRange(df['High'], df['Low'], df['Close'], window=14)
    df['atr'] = atr.average_true_range()
    df['volatility_rank'] = df['atr'] / (df['atr'].rolling(50).mean() + 1e-9)
    df['price_range_ratio'] = (df['High'] - df['Low']) / (df['Close'] + 1e-9)
    df['vol_scaled_change'] = df['price_change'] / (df['atr'] + 1e-9)
    df['body_ratio'] = (df['Close'] - df['Open']) / (df['High'] - df['Low'] + 1e-9)

    # === SNIPER ENHANCERS ===
    df['ema_slope'] = df['ema_12'] - df['ema_12'].shift(1)
    df['macd_slope'] = df['macd'] - df['macd'].shift(1)
    df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
    df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
    df['ha_strong_bull'] = ((df['HA_Body'] > 0) & (df['HA_Body'] > df['atr'] * 0.5)).astype(int)
    df['ha_strong_bear'] = ((df['HA_Body'] < 0) & (abs(df['HA_Body']) > df['atr'] * 0.5)).astype(int)

    # Drop volume, future leaks if present
    df.drop(columns=[col for col in ['future_close', 'delta', 'Volume'] if col in df.columns], errors='ignore', inplace=True)

    # Remove highly null rows before signal creation
    df = df[df.isnull().mean(axis=1) < 0.3]

    # === Signal Creation ===
    if create_signal and ('Signal' not in df.columns or df['Signal'].isnull().all()):
        df['future_close'] = df['Close'].shift(-3)
        df['delta'] = df['future_close'] - df['Close']
        df['Signal'] = np.where(df['delta'] > threshold, 1,
                         np.where(df['delta'] < -threshold, 0, np.nan))

        print("[DEBUG] Δ distribution:\n", df['delta'].describe())
        print("[DEBUG] Signal class distribution:\n", df['Signal'].value_counts(dropna=False))

        # ✅ High-Confidence Label Filtering
        if high_confidence_labels_only:
            print("[INFO] Applying high-confidence signal filtering...")
            df['abs_delta'] = df['delta'].abs()
            df = df[
                (df['abs_delta'] > 0.0002) &
                (df['atr'] > df['atr'].rolling(50).mean()) &
                (df['volatility_rank'] > 0.8) &
                (df['body_ratio'] > 0.1)
            ]

            # Time window filter for consistency
            df = df[((df['hour'] >= 9) & (df['hour'] <= 12)) | ((df['hour'] >= 14) & (df['hour'] <= 17))]

    # Final filter: keep only binary labeled rows
    df = df[df['Signal'].isin([0, 1])]

    # Drop remaining NaNs
    df.dropna(inplace=True)

    # Drop low-variance or high-null columns
    df.drop(columns=[col for col in df.columns if df[col].std() < 1e-6 or df[col].isnull().mean() > 0.3], inplace=True)

    # Clean up any leftover leak columns
    df.drop(columns=['future_close', 'delta', 'abs_delta'], errors='ignore', inplace=True)

    # Sanity check
    if 'Signal' not in df.columns or df['Signal'].nunique() < 2:
        raise ValueError("🚨 Signal column missing or only one class after generation.")

    return df.reset_index(drop=True)
