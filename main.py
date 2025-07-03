from data.fetch_data import get_ohlcv
from strategies.momentum import apply_rsi2

df = get_ohlcv()
df = apply_rsi2(df)
print(df[["time", "close", "rsi2", "signal"]].tail())

from execution.trade_manager import open_trade

if df.iloc[-1]["signal"] == 1:
    print("BUY Signal")
    open_trade("XAUUSD", 0.1, 1)
elif df.iloc[-1]["signal"] == -1:
    print("SELL Signal")
    open_trade("XAUUSD", 0.1, -1)

import time

while True:
    df = get_ohlcv()
    df = apply_rsi2(df)

    signal = df.iloc[-1]["signal"]
    if signal != 0:
        open_trade("XAUUSD", 0.1, signal)
        log_trade(signal, df.iloc[-1]["close"], df.iloc[-1]["rsi2"], sl=150, tp=300)

    time.sleep(60 * 15)  # Wait 15 minutes
