import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import time
import os

# === Setup ===
SYMBOL = "XAUUSDc"
TIMEFRAME = mt5.TIMEFRAME_M15
TP = 300
SL = 150
FORWARD_BARS = 40

# === Connect to MT5 ===
if not mt5.initialize():
    print("❌ MT5 init failed.")
    quit()

# === Load trade log ===
df = pd.read_csv("logs/trade_log.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["label"] = None

# === Label trades ===
for idx, row in df.iterrows():
    entry_time = row["timestamp"]
    price = row["entry_price"]
    direction = 1 if str(row["signal"]).upper() == "BUY" else -1

    rates = mt5.copy_rates_from(SYMBOL, TIMEFRAME, entry_time, FORWARD_BARS)
    if rates is None or len(rates) < 5:
        print(f"⛔ No data for trade {idx} at {entry_time}")
        continue

    hit = -1
    for bar in rates:
        high = bar["high"]
        low = bar["low"]

        if direction == 1:
            if high >= price + TP * 0.01:
                hit = 1
                break
            elif low <= price - SL * 0.01:
                hit = 0
                break
        else:
            if low <= price - TP * 0.01:
                hit = 1
                break
            elif high >= price + SL * 0.01:
                hit = 0
                break

    df.at[idx, "label"] = hit
    print(f"✅ Trade {idx} → Label = {hit}")

# === Save labeled trades ===
os.makedirs("data", exist_ok=True)
df.to_csv("data/labeled_trades.csv", index=False)
print("🎯 Saved labeled data to data/labeled_trades.csv")

mt5.shutdown()
