import pandas as pd

df = pd.read_csv("logs/trade_log.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Separate filtered trades
executed = df[df["ml_decision"] == 1]
skipped = df[df["ml_decision"] == 0]

print(f"✅ Executed trades: {len(executed)}")
print(f"❌ Skipped trades (ML said no): {len(skipped)}")

# Optional: You can evaluate % win/loss once you add the `label` column again
