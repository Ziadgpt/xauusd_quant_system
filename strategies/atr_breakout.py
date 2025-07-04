# strategies/atr_breakout.py

import pandas as pd
import ta


def apply_atr_breakout(df: pd.DataFrame, atr_period=14, lookback=20) -> pd.DataFrame:
    df = df.copy()

    df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=atr_period)
    df["high_break"] = df["high"].rolling(window=lookback).max()
    df["low_break"] = df["low"].rolling(window=lookback).min()

    df["signal_atr_breakout"] = 0
    latest_close = df["close"].iloc[-1]
    latest_high = df["high_break"].iloc[-2]
    latest_low = df["low_break"].iloc[-2]

    if latest_close > latest_high:
        df.at[df.index[-1], "signal_atr_breakout"] = 1
    elif latest_close < latest_low:
        df.at[df.index[-1], "signal_atr_breakout"] = -1

    return df

