# models/random_forest_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def train_rf_model(csv_path="data/labeled_trades.csv"):
    df = pd.read_csv(csv_path)
    features = ["rsi2", "macd", "obv", "atr", "bollinger_width"]
    X = df[features]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"🎯 RF Model Accuracy: {acc:.2f}")

    joblib.dump(clf, "models/random_forest_model.pkl")
