import pandas as pd

def detect_hh_ll_breakout(df: pd.DataFrame, lookback=20) -> pd.DataFrame:
    df = df.copy()

    df["signal_structure"] = 0

    recent_high = df["high"].rolling(window=lookback).max()
    recent_low = df["low"].rolling(window=lookback).min()

    breakout_up = df["close"] > recent_high.shift(1)
    breakout_down = df["close"] < recent_low.shift(1)

    df.loc[breakout_up, "signal_structure"] = 1  # BUY
    df.loc[breakout_down, "signal_structure"] = -1  # SELL

    return df
