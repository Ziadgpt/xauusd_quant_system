import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import time
import os
import numpy as np

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
df = pd.read_csv("logs/cleaned_trade_log.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["label"] = -1  # Default to -1 (unlabeled)

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
                hit = 1  # TP hit
                break
            elif low <= price - SL * 0.01:
                hit = 0  # SL hit
                break
        else:
            if low <= price - TP * 0.01:
                hit = 1  # TP hit
                break
            elif high >= price + SL * 0.01:
                hit = 0  # SL hit
                break

    df.at[idx, "label"] = hit
    print(f"✅ Trade {idx} → Label = {hit}")

# Ensure column is numeric
df["label"] = pd.to_numeric(df["label"], errors="coerce")

# === Save labeled trades ===
os.makedirs("data", exist_ok=True)
df.to_csv("data/labeled_trades.csv", index=False)
print("🎯 Saved labeled data to data/labeled_trades.csv")

mt5.shutdown()
