import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# === Load ===
df = pd.read_csv("ml_dataset.csv")
df = df.dropna()

# Remove non-numeric features
if "time" in df.columns:
    df.drop(columns=["time"], inplace=True)
if "timestamp" in df.columns:
    df.drop(columns=["timestamp"], inplace=True)

X = df.drop(columns=["label"])
y = df["label"]

os.makedirs("ml", exist_ok=True)
joblib.dump(list(X.columns), "ml/feature_names.pkl")

# === Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Models ===
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, max_depth=6, use_label_encoder=False, eval_metric="logloss")
}

for name, model in models.items():
    print(f"\n🚀 Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"📊 {name} Report:")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, f"ml/{name.lower()}_model.pkl")
    print(f"✅ {name} saved to ml/{name.lower()}_model.pkl")
