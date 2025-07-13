import joblib
import numpy as np
import os

rf_model = joblib.load("ml/models/rf_model.pkl")
xgb_model = joblib.load("ml/models/xgb_model.pkl")

def predict_trade(features: dict) -> int:
    try:
        input_df = pd.DataFrame([features])
        input_df = input_df[rf_model.feature_names_in_]

        rf_pred = rf_model.predict(input_df)[0]
        xgb_pred = xgb_model.predict(input_df)[0]

        if rf_pred == xgb_pred:
            return rf_pred
        else:
            return 0  # Disagreement = no trade
    except Exception as e:
        print(f"⚠️ ML error: {e}")
        return 0
