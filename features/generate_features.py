import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime
import os

from indicators.rsi import calculate_rsi
from indicators.macd import calculate_macd
from indicators.atr import calculate_atr
from indicators.obv import calculate_obv
from indicators.bollinger import calculate_bollinger_bands
from models.garch_model import forecast_garch_volatility
from models.hmm_model import detect_market_regime

# === Config ===
SYMBOL = "XAUUSDc"
TIMEFRAME = mt5.TIMEFRAME_M15
BARS = 100

# === Init MT5 ===
if not mt5.initialize():
    print("❌ MT5 init failed")
    exit()

# === Load labeled trades ===
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

    # === Compute indicators ===
    hist_df["rsi_2"] = calculate_rsi(hist_df["close"], period=2)
    hist_df["rsi_14"] = calculate_rsi(hist_df["close"], period=14)

    macd_line, macd_signal, macd_hist = calculate_macd(hist_df["close"])
    hist_df["macd_line"] = macd_line
    hist_df["macd_signal"] = macd_signal
    hist_df["macd_hist"] = macd_hist

    hist_df["atr"] = calculate_atr(hist_df)
    hist_df["obv"] = calculate_obv(hist_df)
    hist_df = calculate_bollinger_bands(hist_df)

    # === Compute custom features ===
    last = hist_df.iloc[-1]
    prev_close = hist_df["close"].iloc[-2]
    body = abs(last["close"] - last["open"])
    range_ = last["high"] - last["low"] + 1e-6

    # === Rolling volatility ===
    hist_df["rolling_std_5"] = hist_df["close"].rolling(5).std()
    bb_width = last["bb_upper"] - last["bb_lower"]

    # === Optional: Volatility + Regime Detection ===
    numeric_df = hist_df.select_dtypes(include="number")
    try:
        garch_vol = forecast_garch_volatility(numeric_df)
        regime, _ = detect_market_regime(numeric_df)
    except:
        garch_vol, regime = 0.0, 1

    # === Add to features ===
    features.append({
        "timestamp": timestamp,
        "direction": direction,
        "label": label,

        # Indicators
        "rsi_2": last["rsi_2"],
        "rsi_14": last["rsi_14"],
        "macd_line": last["macd_line"],
        "macd_signal": last["macd_signal"],
        "macd_hist": last["macd_hist"],
        "atr": last["atr"],
        "obv": last["obv"],
        "bb_upper": last["bb_upper"],
        "bb_middle": last["bb_middle"],
        "bb_lower": last["bb_lower"],
        "bb_width": bb_width,
        "bb_distance": (last["close"] - last["bb_lower"]) / (bb_width + 1e-6),
        "rolling_std_5": last["rolling_std_5"],

        # Price action
        "body_ratio": body / range_,
        "return_pct": (last["close"] - prev_close) / prev_close * 100,
        "trend_slope": np.polyfit(np.arange(10), hist_df["close"].tail(10).values, 1)[0],

        # Market context
        "garch_vol": garch_vol,
        "regime": regime,

        # Time
        "hour": timestamp.hour,
        "weekday": timestamp.weekday()
    })

    print(f"✅ Trade {idx} processed.")

# === Save final dataset ===
features_df = pd.DataFrame(features)
os.makedirs("features", exist_ok=True)
features_df.to_csv("features/dataset.csv", index=False)
print("🎯 Saved to features/dataset.csv")

mt5.shutdown()
