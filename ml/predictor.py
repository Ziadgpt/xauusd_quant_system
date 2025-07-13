import pandas as pd
import joblib
import os

# === Load Models ===
model_paths = {
    "RandomForest": "ml/randomforest_model.pkl",
    "XGBoost": "ml/xgboost_model.pkl"
}
models = {name: joblib.load(path) for name, path in model_paths.items() if os.path.exists(path)}

if not models:
    raise RuntimeError("❌ No ML models loaded.")

# === Load Features ===
expected_features = joblib.load("ml/feature_names.pkl")

# === Predict ===
def predict_trade(features: dict) -> int:
    df = pd.DataFrame([features])
    try:
        df = df[expected_features]
    except KeyError:
        missing = set(expected_features) - set(df.columns)
        raise ValueError(f"Missing required features: {missing}")

    votes = []
    for name, model in models.items():
        try:
            pred = model.predict(df)[0]
            votes.append(int(pred))
        except Exception as e:
            print(f"❌ Error in {name} prediction: {e}")

    return 1 if sum(votes) >= len(votes) / 2 else 0
