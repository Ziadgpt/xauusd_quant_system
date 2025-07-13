import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib

# Load Data
df = pd.read_csv("ml_dataset.csv")
df.dropna(inplace=True)

# Drop unused or datetime columns
df = df.drop(columns=["timestamp", "time"], errors="ignore")

# Features and Label
X = df.drop(columns=["label"])
y = df["label"]

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
rf_model.fit(X, y)

# Train XGBoost
xgb_model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, use_label_encoder=False, eval_metric="logloss")
xgb_model.fit(X, y)

# Save models
joblib.dump(rf_model, "ml/models/rf_model.pkl")
joblib.dump(xgb_model, "ml/models/xgb_model.pkl")

print("✅ Models trained and saved.")
