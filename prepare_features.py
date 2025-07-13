import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta
from models.garch_model import forecast_garch_volatility
from models.hmm_model import detect_market_regime
from features.extract_features import extract_features  # your new file

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

# === Loop through each labeled trade ===
for i, row in df_trades.iterrows():
    timestamp = row["timestamp"]
    label = row["label"]

    # Fetch past data
    start_time = timestamp - timedelta(minutes=15 * LOOKBACK)
    rates = mt5.copy_rates_from(SYMBOL, TIMEFRAME, start_time, LOOKBACK)
    if rates is None or len(rates) < 30:
        print(f"⛔ Not enough data for trade {i}")
        continue

    df = pd.DataFrame(rates)
    df["timestamp"] = pd.to_datetime(df["time"], unit="s")

    try:
        # === Extract Alpha Features ===
        df = extract_features(df)

        # === Add GARCH + HMM ===
        numeric_df = df.select_dtypes(include=["number"]).copy()
        vol = forecast_garch_volatility(numeric_df)
        regime, _ = detect_market_regime(numeric_df)
        df["garch_vol"] = vol
        df["regime"] = regime

        latest = df.iloc[-1].copy()
        latest["timestamp"] = timestamp
        latest["label"] = label

        rows.append(latest)

        print(f"✅ Processed trade {i}")

    except Exception as e:
        print(f"❌ Error at trade {i}: {e}")
        continue

# === Save Dataset ===
df_final = pd.DataFrame(rows)
df_final.to_csv("ml_dataset.csv", index=False)
print("🎯 Features saved to ml_dataset.csv")

mt5.shutdown()
