import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime
import time
import os

from indicators.rsi import calculate_rsi
from indicators.macd import calculate_macd
from indicators.atr import calculate_atr
from indicators.obv import calculate_obv
from indicators.bollinger import calculate_bollinger_bands

SYMBOL = "XAUUSDc"
TIMEFRAME = mt5.TIMEFRAME_M15
BARS = 100

# Init MT5
if not mt5.initialize():
    print("❌ MT5 init failed")
    exit()

df = pd.read_csv("labeled_trades.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

features = []

for idx, row in df.iterrows():
    timestamp = row["timestamp"]
    direction = 1 if str(row["signal"]).upper() == "BUY" else -1
    label = row["label"]

    rates = mt5.copy_rates_from(SYMBOL, TIMEFRAME, timestamp, BARS)
    if rates is None or len(rates) < 50:
        print(f"⚠️ Not enough data for trade {idx}")
        continue

    hist_df = pd.DataFrame(rates)
    hist_df["time"] = pd.to_datetime(hist_df["time"], unit="s")

    # Compute indicators
    hist_df["rsi"] = calculate_rsi(hist_df["close"], period=14)
    macd_line, macd_signal, _ = calculate_macd(hist_df["close"])
    hist_df["macd"] = macd_line
    hist_df["macd_signal"] = macd_signal
    hist_df["atr"] = calculate_atr(hist_df)
    hist_df["obv"] = calculate_obv(hist_df)
    hist_df = calculate_bollinger_bands(hist_df)

    # Select most recent row (latest indicators)
    last = hist_df.iloc[-1]

    features.append({
        "timestamp": timestamp,
        "direction": direction,
        "rsi": last["rsi"],
        "macd": last["macd"],
        "macd_signal": last["macd_signal"],
        "atr": last["atr"],
        "obv": last["obv"],
        "bb_upper": last["bb_upper"],
        "bb_middle": last["bb_middle"],
        "bb_lower": last["bb_lower"],
        "label": label
    })

# Final Dataset
features_df = pd.DataFrame(features)
os.makedirs("features", exist_ok=True)
features_df.to_csv("features/dataset.csv", index=False)
print("✅ Dataset saved to features/dataset.csv")

mt5.shutdown()
