import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import os

def train_rf_model(csv_path="data/trade_features.csv"):
    df = pd.read_csv(csv_path)

    # Features to use
    features = [
        "rsi2", "rsi14", "macd", "macd_signal",
        "boll_upper", "boll_lower", "boll_middle",
        "volatility", "regime", "body_ratio", "prev_return",
        "trend_strength", "direction", "hour", "weekday"
    ]

    df.dropna(inplace=True)
    X = df[features]
    y = df["label"]

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluation
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    print(f"🎯 Random Forest Accuracy: {acc:.2%}")
    print("📊 Confusion Matrix:")
    print(cm)

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/random_forest_model.pkl")
    print("✅ Model saved to models/random_forest_model.pkl")

if __name__ == "__main__":
    train_rf_model()
