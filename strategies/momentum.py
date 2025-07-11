import pandas as pd
from indicators.rsi import calculate_rsi


def apply_rsi2(df: pd.DataFrame) -> pd.DataFrame:
    df["rsi2"] = calculate_rsi(df["close"], period=2)

    def get_signal(rsi):
        if rsi < 10:
            return 1  # Buy
        elif rsi > 90:
            return -1  # Sell
        else:
            return 0

    df["signal"] = df["rsi2"].apply(get_signal)
    return df
