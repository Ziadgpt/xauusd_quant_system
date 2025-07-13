import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

# === Load Dataset ===
df = pd.read_csv("data/exit_dataset.csv")
df.dropna(inplace=True)

# === Features and Target ===
X = df.drop(columns=["label"])
y = df["label"]

# === Save Feature Names for Inference
feature_names = list(X.columns)

# === Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train Logistic Regression ===
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# === Evaluate ===
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # probability of hitting TP

print("📊 Classification Report:\n", classification_report(y_test, y_pred))
print(f"🎯 ROC-AUC Score: {roc_auc_score(y_test, y_prob):.3f}")

# === Save Model and Feature List ===
os.makedirs("ml", exist_ok=True)
joblib.dump((model, feature_names), "ml/exit_model.pkl")
print("✅ Exit model saved to ml/exit_model.pkl")
