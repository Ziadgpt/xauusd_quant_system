import csv
import os
from datetime import datetime

LOG_FILE = "logs/trade_log.csv"

def log_trade(signal, price, indicator_value, sl=150, tp=300):
    """Logs trade to CSV."""
    direction = "BUY" if signal == 1 else "SELL"
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    header = ["timestamp", "direction", "price", "indicator", "sl", "tp"]

    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([now, direction, price, indicator_value, sl, tp])

def log_exit(ticket, symbol, direction, entry, exit_price, result, pnl):
    with open("logs/trade_log.csv", "a") as f:
        f.write(f"{ticket},{symbol},{'BUY' if direction==1 else 'SELL'},{entry},{exit_price},{result},{pnl:.2f}\n")
