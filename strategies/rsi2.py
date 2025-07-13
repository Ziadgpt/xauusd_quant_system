import pandas as pd
from ta.momentum import RSIIndicator

def apply_rsi2(df: pd.DataFrame, rsi_col="rsi2") -> pd.DataFrame:
    """
    Applies 2-period RSI strategy to generate buy/sell signals.
    - BUY when RSI2 < 10
    - SELL when RSI2 > 90
    """
    rsi = RSIIndicator(df["close"], window=2).rsi()
    df[rsi_col] = rsi

    df["signal"] = 0
    df.loc[rsi < 10, "signal"] = 1   # Buy
    df.loc[rsi > 90, "signal"] = -1  # Sell

    return df
