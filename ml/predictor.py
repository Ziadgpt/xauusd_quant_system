# ml/predictor.py
import pandas as pd
import joblib
import os

# Load model once
model_path = "ml/trade_model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")

model = joblib.load(model_path)

def predict_trade(features: dict) -> int:
    """
    Predicts trade outcome using trained model.
    Returns: 1 (profitable trade) or 0 (loss)
    """
    df = pd.DataFrame([features])
    prediction = model.predict(df)[0]
    return int(prediction)
