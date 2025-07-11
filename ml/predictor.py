import pandas as pd
import joblib
import os

# Load model once
model_path = "ml/trade_model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")

model = joblib.load(model_path)

def predict_trade(features: dict):
    """
    Accepts a dict of features and returns prediction: 1 (profit) or 0 (loss).
    """
    df = pd.DataFrame([features])
    prediction = model.predict(df)[0]
    return prediction
