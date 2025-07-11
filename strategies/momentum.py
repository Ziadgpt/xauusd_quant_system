from indicators.rsi import calculate_rsi

def apply_rsi2(df):
    df["rsi2"] = calculate_rsi(df["close"], 2)
    df["rsi14"] = calculate_rsi(df["close"], 14)  # ✅ Add this line
    df["signal"] = 0
    df.loc[df["rsi2"] < 10, "signal"] = 1
    df.loc[df["rsi2"] > 90, "signal"] = -1
    return df
