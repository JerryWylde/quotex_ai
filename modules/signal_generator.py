import numpy as np
import pandas as pd

def create_signal(prob_up, threshold=0.51, volatility_rank=None):
    """
    Generate a sniper-grade signal using confidence threshold only.

    Args:
        prob_up (float): Probability of upward movement.
        threshold (float): Confidence threshold (e.g., 0.51)
        volatility_rank (float): Optional (unused here now)

    Returns:
        signal (int or None): 1 for BUY, 0 for SELL, None for HOLD
        confidence (float): Max(prob_up, prob_down)
    """
    prob_down = 1 - prob_up
    confidence = max(prob_up, prob_down)

    # ✅ Only apply confidence filter
    if confidence < threshold:
        return None, confidence

    return (1 if prob_up > prob_down else 0), confidence


def apply_signals_to_dataframe(df, prob_col='meta_prob', threshold=0.51):
    """
    Apply sniper-grade signal logic across an entire DataFrame.

    Args:
        df (pd.DataFrame): Must contain a prediction column (e.g., meta_prob)
        prob_col (str): Column containing prob_up values.
        threshold (float): Confidence threshold (default = 0.51)

    Returns:
        df (pd.DataFrame): With 'Signal' and 'Confidence' columns added.
    """
    df = df.copy()
    signals = []
    confidences = []

    for _, row in df.iterrows():
        prob_up = row[prob_col]
        signal, confidence = create_signal(prob_up, threshold)
        signals.append(signal)
        confidences.append(confidence)

    df['Signal'] = signals
    df['Confidence'] = confidences
    return df
