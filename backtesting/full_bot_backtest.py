import pandas as pd
import joblib
import os
from data.fetch_data import get_ohlcv
from models.garch_model import forecast_garch_volatility
from models.hmm_model import detect_market_regime
from features.extract_features import extract_features

# Load Models
rf_model = joblib.load("ml/models/rf_model.pkl")
xgb_model = joblib.load("ml/models/xgb_model.pkl")

# Get OHLCV
df = get_ohlcv(symbol="XAUUSDc", timeframe="M15", count=500)
df["timestamp"] = df["time"]  # for consistency with feature engineering

# Add Volatility + Regime
df["garch_vol"] = forecast_garch_volatility(df)
regime, dominant = detect_market_regime(df)
df["regime"] = regime

# Feature Engineering
features_df = extract_features(df)
features_df.reset_index(drop=True, inplace=True)

# Drop datetime column before ML prediction
X = features_df.drop(columns=["timestamp", "time"], errors="ignore")

# Ensure model input alignment
required_features = list(rf_model.feature_names_in_)
X = X[required_features]

# Predict
rf_preds = rf_model.predict(X)
xgb_preds = xgb_model.predict(X)

features_df["rf_signal"] = rf_preds
features_df["xgb_signal"] = xgb_preds
features_df["final_signal"] = (rf_preds == xgb_preds) & (rf_preds != 0)
features_df["final_signal"] = features_df["final_signal"].astype(int) * rf_preds

# Save to CSV
features_df.to_csv("backtest_trades.csv", index=False)
print("✅ Backtest complete. Results saved to backtest_trades.csv")
