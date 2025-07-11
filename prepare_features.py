import pandas as pd
import MetaTrader5 as mt5
import time
from datetime import datetime, timedelta

from indicators.rsi import calculate_rsi
from indicators.macd import calculate_macd
from indicators.bollinger import calculate_bollinger_bands
from indicators.obv import calculate_obv
from indicators.atr import calculate_atr
from models.garch_model import forecast_garch_volatility
from models.hmm_model import detect_market_regime

# === Config ===
SYMBOL = "XAUUSDc"
TIMEFRAME = mt5.TIMEFRAME_M15
LOOKBACK = 100

# === Init MT5 ===
if not mt5.initialize():
    print("❌ MT5 initialization failed")
    exit()

# === Load Labeled Trades ===
df_trades = pd.read_csv("data/labeled_trades.csv")
df_trades["timestamp"] = pd.to_datetime(df_trades["timestamp"])
rows = []

# === Loop through each trade ===
for i, row in df_trades.iterrows():
    timestamp = row["timestamp"]
    label = row["label"]

    # Get past candles before the trade
    rates = mt5.copy_rates_from(SYMBOL, TIMEFRAME, timestamp - timedelta(minutes=15 * LOOKBACK), LOOKBACK)
    if rates is None or len(rates) < 30:
        print(f"⛔ Not enough data for trade {i}")
        continue

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")

    # === Feature Engineering ===
    try:
        df["rsi2"] = calculate_rsi(df["close"], 2)
        df["rsi14"] = calculate_rsi(df["close"], 14)

        macd_line, macd_signal, macd_hist = calculate_macd(df["close"])
        df["macd_line"] = macd_line
        df["macd_signal"] = macd_signal
        df["macd_hist"] = macd_hist

        df = calculate_bollinger_bands(df, period=21)
        df["obv"] = calculate_obv(df)
        df["atr"] = calculate_atr(df, period=14)

        # ✅ Use numeric-only data for GARCH and HMM models
        numeric_df = df.select_dtypes(include=["number"]).copy()
        vol = forecast_garch_volatility(numeric_df)
        regime, _ = detect_market_regime(numeric_df)

        latest = df.iloc[-1]

        rows.append({
            "timestamp": timestamp,
            "rsi2": latest["rsi2"],
            "rsi14": latest["rsi14"],
            "macd_line": latest["macd_line"],
            "macd_signal": latest["macd_signal"],
            "macd_hist": latest["macd_hist"],
            "bb_upper": latest["bb_upper"],
            "bb_lower": latest["bb_lower"],
            "bb_width": latest["bb_upper"] - latest["bb_lower"],
            "obv": latest["obv"],
            "atr": latest["atr"],
            "volatility": vol,
            "regime": regime,
            "label": label
        })

        print(f"✅ Processed trade {i}")

    except Exception as e:
        print(f"❌ Error at trade {i}: {e}")
        continue

# === Save Features ===
df_final = pd.DataFrame(rows)
df_final.to_csv("ml_dataset.csv", index=False)
print("🎯 Features saved to ml_dataset.csv")

mt5.shutdown()
