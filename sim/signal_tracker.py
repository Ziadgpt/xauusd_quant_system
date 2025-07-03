# sim/signal_tracker.py

tracked_signals = {}

def record_signal(timestamp, symbol, direction, entry_price, sl, tp):
    tracked_signals[symbol] = {
        "timestamp": timestamp,
        "direction": direction,
        "entry": entry_price,
        "sl": sl,
        "tp": tp,
        "active": True
    }

def check_exits(current_price, symbol):
    trade = tracked_signals.get(symbol)
    if not trade or not trade["active"]:
        return None

    direction = trade["direction"]
    sl, tp = trade["sl"], trade["tp"]
    entry = trade["entry"]
    hit = None

    if direction == 1:  # BUY
        if current_price <= sl:
            hit = "SL"
        elif current_price >= tp:
            hit = "TP"
    elif direction == -1:  # SELL
        if current_price >= sl:
            hit = "SL"
        elif current_price <= tp:
            hit = "TP"

    if hit:
        trade["active"] = False
        pnl = (current_price - entry) * 100 if direction == 1 else (entry - current_price) * 100
        return {
            "exit_type": hit,
            "entry": entry,
            "exit": current_price,
            "pnl": pnl,
            "direction": "BUY" if direction == 1 else "SELL"
        }

    return None
