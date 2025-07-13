import pandas as pd
import joblib

model, features = joblib.load("ml/exit_model.pkl")

def predict_exit_probability(snapshot: dict) -> float:
    df = pd.DataFrame([snapshot])
    df = df[features]
    return model.predict_proba(df)[0][1]
