import csv
from datetime import datetime
import os


def log_trade(signal, entry_price, indicator_value, sl, tp, strategy="RSI2", symbol="XAUUSD"):
    filename = "logs/trade_log.csv"
    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "symbol", "strategy", "signal", "entry_price", "indicator", "sl", "tp"])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            symbol,
            strategy,
            "BUY" if signal == 1 else "SELL",
            round(entry_price, 2),
            round(indicator_value, 2),
            round(sl, 2),
            round(tp, 2)
        ])

def log_exit(ticket, symbol, direction, entry_price, exit_price, reason, pnl):
    filename = "logs/exit_log.csv"
    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "ticket", "symbol", "direction",
                "entry_price", "exit_price", "reason", "PnL"
            ])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ticket,
            symbol,
            "BUY" if direction == 1 else "SELL",
            round(entry_price, 2),
            round(exit_price, 2),
            reason,
            round(pnl, 2)
        ])
