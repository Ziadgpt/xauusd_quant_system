# train_model.py

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score
)

# === Load Dataset ===
df = pd.read_csv("data/trade_features.csv")
df.dropna(inplace=True)

# === Define Features and Target ===
X = df.drop(columns=["label", "timestamp"], errors="ignore")
y = df["label"]

# === Save column names for later use
feature_names = list(X.columns)

# === Split Dataset ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === Train Random Forest Model ===
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# === Evaluate ===
y_pred = model.predict(X_test)
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("🧱 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# === Confidence Scores ===
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy:.2f}")

if len(y.unique()) == 2:
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    print(f"✅ AUC Score: {auc:.2f}")

# === Cross Validation (optional) ===
cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')
print(f"🔁 Cross-validated F1 scores: {cv_scores}")
print(f"📈 Mean F1: {cv_scores.mean():.3f}")

# === Feature Importances ===
importances = model.feature_importances_
feat_df = pd.DataFrame({"feature": feature_names, "importance": importances})
feat_df = feat_df.sort_values(by="importance", ascending=False)

print("\n🔍 Top 10 Features:")
print(feat_df.head(10))

# === Save to file ===
os.makedirs("ml", exist_ok=True)
joblib.dump((model, feature_names), "ml/trade_model.pkl")
feat_df.to_csv("ml/feature_importance.csv", index=False)
print("✅ Model and feature importance saved to ml/")

# === Plot (optional) ===
feat_df.head(10).plot(kind="barh", x="feature", y="importance", title="Top 10 Features")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
