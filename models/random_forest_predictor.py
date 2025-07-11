# models/random_forest_predictor.py
import joblib
import pandas as pd

model = joblib.load("models/random_forest_model.pkl")

def predict_with_rf(row):
    features = ["rsi2", "macd", "obv", "atr", "bollinger_width"]
    X = pd.DataFrame([row[features].values], columns=features)
    return model.predict(X)[0]
