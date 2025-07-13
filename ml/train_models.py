# ml/train_models.py

import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# === Load Dataset ===
df = pd.read_csv("features/dataset.csv")
df.dropna(inplace=True)

X = df.drop(columns=["label", "timestamp"], errors="ignore")
y = df["label"]

# Save features used
os.makedirs("ml", exist_ok=True)
joblib.dump(list(X.columns), "ml/feature_names.pkl")

# === Split Dataset ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Define Models ===
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, max_depth=6, use_label_encoder=False, eval_metric="logloss"),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

# === Train and Evaluate ===
for name, model in models.items():
    print(f"\n🚀 Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"📊 {name} Report:")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(model, f"ml/{name.lower()}_model.pkl")
    print(f"✅ {name} saved to ml/{name.lower()}_model.pkl")
