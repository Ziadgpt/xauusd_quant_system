# ml/train_exit_model.py
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load your labeled snapshot dataset
df = pd.read_csv("data/exit_snapshots.csv")
df.dropna(inplace=True)

X = df.drop(columns=["outcome", "timestamp"], errors="ignore")
y = df["outcome"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
print(classification_report(y_test, model.predict(X_test)))

# Save
joblib.dump((model, list(X.columns)), "ml/exit_model.pkl")
print("✅ Exit model saved to ml/exit_model.pkl")
