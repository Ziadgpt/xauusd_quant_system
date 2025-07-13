import pandas as pd
from data.fetch_data import get_ohlcv
from features.extract_features import extract_features
from models.garch_model import forecast_garch_volatility
from models.hmm_model import detect_market_regime

# === Load Labeled Trades ===
trades_df = pd.read_csv("labeled_trades.csv")
features_list = []

for _, row in trades_df.iterrows():
    try:
        ts = pd.to_datetime(row["time"])
        symbol = row["symbol"]
        label = int(row["label"])

        # Fetch historical context before trade
        df = get_ohlcv(symbol=symbol, timeframe="M15", count=1500)
        df["timestamp"] = df["time"]
        df = df[df["timestamp"] < ts]

        if len(df) < 50:
            continue

        # Add GARCH + HMM
        df["garch_vol"] = forecast_garch_volatility(df)
        regime, _ = detect_market_regime(df)
        df["regime"] = regime

        # Feature Engineering
        df = extract_features(df)
        last_row = df.iloc[-1].to_dict()
        last_row["label"] = label
        features_list.append(last_row)

    except Exception as e:
        print(f"⚠️ Skipping trade at {row['time']}: {e}")

# === Save ML Dataset ===
ml_df = pd.DataFrame(features_list)
ml_df.to_csv("ml_dataset.csv", index=False)
print("✅ Saved features to ml_dataset.csv")
