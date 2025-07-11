import csv

with open("logs/trade_log.csv", "r") as infile:
    rows = list(csv.reader(infile))

# Keep only rows that have 8 columns
cleaned_rows = [r for r in rows if len(r) == 8]

# Write clean file
with open("logs/cleaned_trade_log.csv", "w", newline="") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["timestamp", "symbol", "strategy", "signal", "entry_price", "indicator", "sl", "tp"])
    writer.writerows(cleaned_rows)

print("✅ Cleaned log saved as logs/cleaned_trade_log.csv")
