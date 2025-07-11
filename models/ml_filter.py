# models/ml_filter.py

import joblib
import pandas as pd
import os

# === Load the trained Random Forest model ===
MODEL_PATH = "models/random_forest_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Random Forest model not found at 'models/random_forest_model.pkl'.")

model = joblib.load(MODEL_PATH)

# === Define the expected input features (must match training set order) ===
FEATURES = [
    "rsi2",
    "rsi14",
    "macd",
    "macd_signal",
    "obv",
    "atr",
    "bb_upper",
    "bb_lower",
    "volatility",  # GARCH output
    "regime"       # Trending = 1, Non-trending = 0
]

def predict_trade_signal(latest_features: dict) -> int:
    """
    Predicts whether the trade is likely to succeed using the trained model.
    Returns 1 for likely success, 0 for likely failure.
    """
    try:
        df = pd.DataFrame([latest_features])
        df = df[FEATURES]  # Ensure order consistency
        prediction = model.predict(df)[0]
        return int(prediction)
    except Exception as e:
        print(f"⚠️ ML Prediction Error: {e}")
        return 0  # Conservative fallback: reject trade
