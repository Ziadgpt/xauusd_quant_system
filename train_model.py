# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# === Load Dataset ===
df = pd.read_csv("data/trade_features.csv")  # ✅ Use the engineered features
df.dropna(inplace=True)

# === Define Features and Target ===
X = df.drop(columns=["label", "timestamp"], errors="ignore")  # Ignore if "timestamp" doesn't exist
y = df["label"]

# === Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train Model ===
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# === Evaluate ===
y_pred = model.predict(X_test)
print("📊 Classification Report:\n", classification_report(y_test, y_pred))
print("🧱 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# === Save Model ===
os.makedirs("ml", exist_ok=True)
joblib.dump(model, "ml/trade_model.pkl")
print("✅ Model saved to ml/trade_model.pkl")
