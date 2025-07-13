import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import numpy as np
import os

from indicators.rsi import calculate_rsi
from indicators.macd import calculate_macd
from indicators.atr import calculate_atr
from models.garch_model import forecast_garch_volatility
from models.hmm_model import detect_market_regime

# === CONFIG ===
SYMBOL = "XAUUSDc"
TIMEFRAME = mt5.TIMEFRAME_M15
LOOKBACK = 100  # bars before and after entry to analyze

# === Init MT5 ===
if not mt5.initialize():
    raise RuntimeError("MT5 initialization failed")

# === Load Cleaned Trade Log ===
df = pd.read_csv("logs/cleaned_trade_log.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

exit_data = []

for i, row in df.iterrows():
    entry_time = row["timestamp"]
    signal = str(row["signal"]).upper()
    direction = 1 if signal == "BUY" else -1
    entry_price = float(row["entry_price"])
    sl = float(row["sl"])
    tp = float(row["tp"])

    # Fetch candles from before to after the entry
    candles = mt5.copy_rates_from(SYMBOL, TIMEFRAME, entry_time - timedelta(minutes=LOOKBACK * 15), LOOKBACK * 2)
    if candles is None or len(candles) < LOOKBACK:
        print(f"⛔ Skipping trade {i} — not enough data")
        continue

    df_candles = pd.DataFrame(candles)
    df_candles["timestamp"] = pd.to_datetime(df_candles["time"], unit="s")

    # Find entry index
    entry_idx = df_candles[df_candles["timestamp"] >= entry_time].index.min()
    if entry_idx is None or entry_idx + 1 >= len(df_candles):
        continue

    hit = None
    exit_idx = None

    for j in range(entry_idx + 1, len(df_candles)):
        high = df_candles.at[j, "high"]
        low = df_candles.at[j, "low"]

        if direction == 1:
            if high >= tp:
                hit = 1
                exit_idx = j
                break
            elif low <= sl:
                hit = 0
                exit_idx = j
                break
        else:
            if low <= tp:
                hit = 1
                exit_idx = j
                break
            elif high >= sl:
                hit = 0
                exit_idx = j
                break

    # If neither TP nor SL hit → skip or mark as undecided
    if hit is None:
        continue  # Or set hit = -1 for decay

    exit_candle = df_candles.iloc[exit_idx]
    since_entry = (exit_idx - entry_idx)  # in candles

    df_candles["rsi"] = calculate_rsi(df_candles["close"])
    macd_line, macd_signal, _ = calculate_macd(df_candles["close"])
    df_candles["macd_line"] = macd_line
    df_candles["macd_signal"] = macd_signal
    df_candles["atr"] = calculate_atr(df_candles)

    snapshot = df_candles.iloc[entry_idx]

    # Volatility & regime
    numeric = df_candles[["open", "high", "low", "close", "tick_volume"]].select_dtypes(include="number")
    try:
        vol = forecast_garch_volatility(numeric)
        regime, _ = detect_market_regime(numeric)
    except:
        vol, regime = 0, 1

    exit_data.append({
        "direction": direction,
        "entry_price": entry_price,
        "time_elapsed": since_entry,
        "rsi": snapshot["rsi"],
        "macd_line": snapshot["macd_line"],
        "macd_signal": snapshot["macd_signal"],
        "atr": snapshot["atr"],
        "volatility": vol,
        "regime": regime,
        "label": hit  # 1 = TP, 0 = SL
    })

    print(f"✅ Trade {i} processed — {'TP' if hit==1 else 'SL'} after {since_entry} candles.")

# === Save Dataset ===
df_exit = pd.DataFrame(exit_data)
os.makedirs("data", exist_ok=True)
df_exit.to_csv("data/exit_dataset.csv", index=False)
print("🎯 Exit dataset saved to data/exit_dataset.csv")

mt5.shutdown()
