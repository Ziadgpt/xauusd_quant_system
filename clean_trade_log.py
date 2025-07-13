import csv

input_file = "logs/trade_log.csv"
output_file = "logs/cleaned_trade_log.csv"

with open(input_file, "r") as infile:
    rows = list(csv.reader(infile))

# Clean rows and report corrupted ones
cleaned_rows = []
corrupted_rows = []

for i, row in enumerate(rows):
    if len(row) == 8:
        cleaned_rows.append(row)
    else:
        corrupted_rows.append((i, row))

# Save cleaned log
with open(output_file, "w", newline="") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["timestamp", "symbol", "strategy", "signal", "entry_price", "indicator", "sl", "tp"])
    writer.writerows(cleaned_rows)

print(f"✅ Cleaned log saved as {output_file}")
print(f"🧹 Removed {len(corrupted_rows)} corrupted rows.")

if corrupted_rows:
    print("⚠️ Corrupted rows (index, data):")
    for idx, row in corrupted_rows[:5]:
        print(f"Row {idx}: {row}")
