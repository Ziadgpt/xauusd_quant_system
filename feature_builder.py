import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import numpy as np

from features.extract_features import extract_features  # central feature file
from models.garch_model import forecast_garch_volatility
from models.hmm_model import detect_market_regime

# === Config ===
SYMBOL = "XAUUSDc"
TIMEFRAME = mt5.TIMEFRAME_M15
LOOKBACK = 100

if not mt5.initialize():
    raise RuntimeError("❌ MT5 Initialization Failed")

df_trades = pd.read_csv("data/labeled_trades.csv")
df_trades["timestamp"] = pd.to_datetime(df_trades["timestamp"])

results = []

for i, row in df_trades.iterrows():
    entry_time = row["timestamp"]
    label = row["label"]
    direction = 1 if str(row["signal"]).upper() == "BUY" else -1

    # Fetch historical candles before trade
    rates = mt5.copy_rates_from(SYMBOL, TIMEFRAME, entry_time - timedelta(minutes=15 * LOOKBACK), LOOKBACK)
    if rates is None or len(rates) < 30:
        print(f"⛔ Trade {i} skipped — not enough data.")
        continue

    df = pd.DataFrame(rates)
    df["timestamp"] = pd.to_datetime(df["time"], unit="s")

    try:
        # Feature engineering
        df_feat = extract_features(df)
        numeric_df = df_feat.select_dtypes(include='number')

        # Statistical filters
        garch_vol = forecast_garch_volatility(numeric_df)
        hmm_regime, _ = detect_market_regime(numeric_df)

        # Final features (last row of extracted df)
        latest = df_feat.iloc[-1].copy()
        latest["garch_vol"] = garch_vol
        latest["regime"] = hmm_regime
        latest["label"] = label
        latest["direction"] = direction

        results.append(latest)
        print(f"✅ Trade {i} processed")

    except Exception as e:
        print(f"❌ Error at trade {i}: {e}")
        continue

df_out = pd.DataFrame(results)
df_out.to_csv("ml_dataset.csv", index=False)
print("🎯 Features saved to ml_dataset.csv")

mt5.shutdown()
