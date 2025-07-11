import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime
import numpy as np
import os

from indicators.rsi import calculate_rsi
from indicators.macd import calculate_macd
from indicators.bollinger import calculate_bollinger_bands
from models.garch_model import forecast_garch_volatility
from models.hmm_model import detect_market_regime

# === Config ===
SYMBOL = "XAUUSDc"
TIMEFRAME = mt5.TIMEFRAME_M15
BARS_LOOKBACK = 100

# === Initialize MT5 ===
if not mt5.initialize():
    print("❌ MT5 init failed")
    quit()

# === Load Labeled Trades ===
df = pd.read_csv("data/labeled_trades.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["label"] = pd.to_numeric(df["label"], errors="coerce")

features = []

# === Feature Extraction Loop ===
for i, row in df.iterrows():
    entry_time = row["timestamp"]
    direction = 1 if str(row["signal"]).upper() == "BUY" else -1
    label = row["label"]

    # Fetch historical data at time of entry
    rates = mt5.copy_rates_from(SYMBOL, TIMEFRAME, entry_time, BARS_LOOKBACK)
    if rates is None or len(rates) < 30:
        print(f"⚠️ Skipping trade {i} — not enough data")
        continue

    df_candles = pd.DataFrame(rates)
    df_candles["time"] = pd.to_datetime(df_candles["time"], unit="s")

    # Calculate Indicators
    df_candles["rsi2"] = calculate_rsi(df_candles["close"], period=2)
    df_candles["rsi14"] = calculate_rsi(df_candles["close"], period=14)
    macd_line, macd_sig, _ = calculate_macd(df_candles["close"])
    df_candles["macd"] = macd_line
    df_candles["macd_signal"] = macd_sig

    bb_upper, bb_lower = calculate_bollinger_bands(df_candles["close"], period=21)
    df_candles["bb_upper"] = bb_upper
    df_candles["bb_lower"] = bb_lower

    # Drop non-numeric data before passing to model
    df_model_input = df_candles.select_dtypes(include=["number"]).copy()

    vol_forecast = forecast_garch_volatility(df_model_input)
    regime, _ = detect_market_regime(df_model_input)

    # Price action features
    close = df_candles["close"].iloc[-1]
    high = df_candles["high"].iloc[-1]
    low = df_candles["low"].iloc[-1]
    open_ = df_candles["open"].iloc[-1]
    body_ratio = abs(close - open_) / (high - low + 1e-6)
    prev_return = (close - df_candles["close"].iloc[-2]) / df_candles["close"].iloc[-2] * 100

    # Trend (slope of linear regression)
    y = df_candles["close"].tail(10).values
    x = np.arange(len(y))
    slope = np.polyfit(x, y, 1)[0]

    # Timestamp features (for ML)
    hour = entry_time.hour
    weekday = entry_time.weekday()

    features.append({
        "direction": direction,
        "hour": hour,
        "weekday": weekday,
        "rsi2": df_candles["rsi2"].iloc[-1],
        "rsi14": df_candles["rsi14"].iloc[-1],
        "macd": df_candles["macd"].iloc[-1],
        "macd_signal": df_candles["macd_signal"].iloc[-1],
        "boll_upper": df_candles["bb_upper"].iloc[-1],
        "boll_lower": df_candles["bb_lower"].iloc[-1],
        "volatility": vol_forecast,
        "regime": regime,
        "body_ratio": body_ratio,
        "prev_return": prev_return,
        "trend_strength": slope,
        "label": label
    })

# === Save Final Feature Set ===
features_df = pd.DataFrame(features)
os.makedirs("data", exist_ok=True)
features_df.to_csv("data/trade_features.csv", index=False)
print("✅ Features saved to data/trade_features.csv")

mt5.shutdown()
