import pandas as pd
import MetaTrader5 as mt5
from datetime import timedelta
from features.extract_features import extract_features
from models.garch_model import forecast_garch_volatility
from models.hmm_model import detect_market_regime

# === Config ===
SYMBOL = "XAUUSDc"
TIMEFRAME = mt5.TIMEFRAME_M15
LOOKBACK = 100

# === Init MT5 ===
if not mt5.initialize():
    raise RuntimeError("MT5 initialization failed.")

# === Load Your Labeled Trades ===
df_trades = pd.read_csv("data/labeled_trades.csv")  # Must contain: timestamp, label
df_trades["timestamp"] = pd.to_datetime(df_trades["timestamp"])

all_rows = []

for i, row in df_trades.iterrows():
    trade_time = row["timestamp"]
    label = row["label"]

    # Fetch 100 candles before trade
    rates = mt5.copy_rates_from(SYMBOL, TIMEFRAME, trade_time - timedelta(minutes=15 * LOOKBACK), LOOKBACK)
    if rates is None or len(rates) < 30:
        print(f"⛔ Skipping trade {i} — Not enough candles.")
        continue

    df = pd.DataFrame(rates)
    df["timestamp"] = pd.to_datetime(df["time"], unit="s")

    try:
        # Apply feature engineering
        df_feat = extract_features(df)

        # Add GARCH/HMM
        numeric = df_feat.select_dtypes(include='number')
        df_feat["garch_vol"] = forecast_garch_volatility(numeric)
        df_feat["regime"], _ = detect_market_regime(numeric)

        last = df_feat.iloc[-1].copy()
        last["label"] = label
        all_rows.append(last)

        print(f"✅ Trade {i} processed.")

    except Exception as e:
        print(f"❌ Error at trade {i}: {e}")
        continue

# === Save Final Dataset ===
df_final = pd.DataFrame(all_rows)
df_final.to_csv("ml_dataset.csv", index=False)
print("🎯 Saved labeled dataset to ml_dataset.csv")

mt5.shutdown()
