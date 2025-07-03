from ta.momentum import RSIIndicator

def apply_rsi2(df):
    rsi = RSIIndicator(close=df["close"], window=2)
    df["rsi2"] = rsi.rsi()
    df["signal"] = 0
    df.loc[df["rsi2"] < 10, "signal"] = 1
    df.loc[df["rsi2"] > 90, "signal"] = -1
    return df
