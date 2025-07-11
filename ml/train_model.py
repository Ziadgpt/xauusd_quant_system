import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# Load dataset
df = pd.read_csv("data/labeled_trades.csv")  # or the correct path if different

df.dropna(inplace=True)

# Features and target
X = df.drop(columns=["timestamp", "label"])
y = df["label"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("📊 Classification Report:\n", classification_report(y_test, y_pred))
print("🧱 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model
os.makedirs("ml", exist_ok=True)
joblib.dump(model, "ml/trade_model.pkl")
print("✅ Model saved to ml/trade_model.pkl")
