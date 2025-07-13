import pandas as pd
import joblib
import os

# === Load Exit Model ===
model_path = "ml/exit_model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Exit model not found at: {model_path}")

model, expected_features = joblib.load(model_path)


def predict_exit_probability(features: dict) -> float:
    """
    Predict the probability that a trade will hit TP before SL.
    Returns:
        float: Probability (between 0 and 1)
    """
    df = pd.DataFrame([features])
    try:
        df = df[expected_features]
    except KeyError as e:
        missing = set(expected_features) - set(df.columns)
        raise ValueError(f"Missing features for exit prediction: {missing}")

    probability = model.predict_proba(df)[0][1]  # Probability of class "1" (TP)
    return float(probability)
