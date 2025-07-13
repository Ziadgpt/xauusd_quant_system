# ml/predictor.py
import pandas as pd
import joblib
import os

# === Load All Models ===
model_paths = {
    "RandomForest": "ml/randomforest_model.pkl",
    "XGBoost": "ml/xgboost_model.pkl",
    "KNN": "ml/knn_model.pkl"
}

models = {}
for name, path in model_paths.items():
    if os.path.exists(path):
        models[name] = joblib.load(path)
    else:
        print(f"⚠️ {name} model not found at {path} — skipping.")

# Load expected feature names
feature_names_path = "ml/feature_names.pkl"
if not os.path.exists(feature_names_path):
    raise FileNotFoundError("Missing feature list (ml/feature_names.pkl)")

expected_features = joblib.load(feature_names_path)

def predict_trade(features: dict) -> int:
    """
    Uses ensemble of models to predict trade outcome.
    Returns:
        1 (take trade) or 0 (reject)
    """
    df = pd.DataFrame([features])
    try:
        df = df[expected_features]
    except KeyError as e:
        missing = set(expected_features) - set(df.columns)
        raise ValueError(f"Missing required features: {missing}")

    votes = []
    for name, model in models.items():
        try:
            pred = model.predict(df)[0]
            votes.append(int(pred))
        except Exception as e:
            print(f"❌ Error in {name} prediction: {e}")

    if not votes:
        print("⚠️ No valid model predictions.")
        return 0

    # Majority voting
    final = 1 if sum(votes) >= len(votes) / 2 else 0
    return final
