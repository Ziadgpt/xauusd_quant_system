import pandas as pd
import ta

def apply_macd_bollinger(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # MACD
    macd = ta.trend.MACD(df["close"])
    df["macd_line"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df["close"], window=21, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()

    # Signal logic
    df["signal_macd_bb"] = 0

    # Entry Criteria
    df.loc[
        (df["close"] < df["bb_lower"]) & (df["macd_hist"] > 0),
        "signal_macd_bb"
    ] = 1  # BUY

    df.loc[
        (df["close"] > df["bb_upper"]) & (df["macd_hist"] < 0),
        "signal_macd_bb"
    ] = -1  # SELL

    return df
