import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime
import time

# === Config ===
SYMBOL = "XAUUSDc"
TIMEFRAME = mt5.TIMEFRAME_M15
TP = 300
SL = 150
FORWARD_BARS = 40

# === Init MT5 ===
if not mt5.initialize():
    print("❌ MT5 init failed")
    quit()

# === Load trade log ===
df = pd.read_csv("../logs/trade_log.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["label"] = -1  # default

# === Loop through each trade ===
for idx, row in df.iterrows():
    time_from = row["timestamp"]
    entry = row["entry_price"]
    direction = 1 if row["signal"] == "BUY" else -1

    rates = mt5.copy_rates_from(SYMBOL, TIMEFRAME, time_from, FORWARD_BARS)
    if rates is None or len(rates) < 5:
        print(f"⛔ No data for trade {idx} at {time_from}")
        continue

    hit = -1
    for bar in rates:
        high, low = bar["high"], bar["low"]
        if direction == 1:
            if high >= entry + TP:
                hit = 1
                break
            elif low <= entry - SL:
                hit = 0
                break
        else:
            if low <= entry - TP:
                hit = 1
                break
            elif high >= entry + SL:
                hit = 0
                break

    df.at[idx, "label"] = hit
    print(f"✅ Trade {idx} labeled as {hit}")

# === Save ===
df.to_csv("ml/labeled_trades.csv", index=False)
print("🎯 Labeled dataset saved as ml/labeled_trades.csv")

# Shutdown MT5
mt5.shutdown()
