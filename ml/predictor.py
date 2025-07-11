# ml/predictor.py
import pandas as pd
import joblib
import os

model_path = "ml/trade_model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")

model, expected_features = joblib.load(model_path)

def predict_trade(features: dict) -> int:
    """
    Predicts trade outcome using trained model.
    Returns: 1 (profit) or 0 (loss)
    """
    df = pd.DataFrame([features])
    try:
        df = df[expected_features]
    except KeyError as e:
        missing = set(expected_features) - set(df.columns)
        raise ValueError(f"Missing required features for prediction: {missing}")
    prediction = model.predict(df)[0]
    return int(prediction)
