import MetaTrader5 as mt5
import pandas as pd
from utils.notifier import send_alert
from logs.logger import log_exit
from ml.exit_predictor import predict_exit_probability

open_positions = {}

def track_trade(ticket, symbol, direction, entry_price, sl, tp, lot=0.1):
    open_positions[ticket] = {
        "symbol": symbol,
        "direction": direction,
        "entry": entry_price,
        "sl": sl,
        "tp": tp,
        "lot": lot,
        "open_time": pd.Timestamp.now()
    }

def get_rsi14(symbol):
    bars = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 100)
    df = pd.DataFrame(bars)
    from indicators.rsi import calculate_rsi
    df["rsi"] = calculate_rsi(df["close"], 14)
    return df["rsi"].iloc[-1]

def get_macd_hist(symbol):
    bars = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 100)
    df = pd.DataFrame(bars)
    from indicators.macd import calculate_macd
    _, _, hist = calculate_macd(df["close"])
    return hist.iloc[-1]

def manage_exits():
    positions = mt5.positions_get()
    if positions is None:
        return

    for pos in positions:
        ticket = pos.ticket
        if ticket not in open_positions:
            continue

        data = open_positions[ticket]
        price = mt5.symbol_info_tick(data["symbol"]).bid if data["direction"] == 1 else mt5.symbol_info_tick(data["symbol"]).ask
        hit = None

        # Check SL/TP hits
        if data["direction"] == 1:
            if price <= data["sl"]:
                hit = "SL"
            elif price >= data["tp"]:
                hit = "TP"
        else:
            if price >= data["sl"]:
                hit = "SL"
            elif price <= data["tp"]:
                hit = "TP"

        if hit:
            pnl = (price - data["entry"]) * data["lot"] * 100 if data["direction"] == 1 else (data["entry"] - price) * data["lot"] * 100
            msg = (
                f"💰 Exit {hit} | {data['symbol']} {('BUY' if data['direction'] == 1 else 'SELL')}\n"
                f"Entry: {data['entry']:.2f} | Exit: {price:.2f} | PnL: ${pnl:.2f}"
            )
            send_alert(msg)
            log_exit(ticket, data['symbol'], data['direction'], data['entry'], price, hit, pnl)

            mt5.order_send(mt5.OrderSendRequest(
                action=mt5.TRADE_ACTION_DEAL,
                symbol=data["symbol"],
                volume=data["lot"],
                type=mt5.ORDER_TYPE_SELL if data["direction"] == 1 else mt5.ORDER_TYPE_BUY,
                position=ticket,
                magic=123456,
                deviation=10
            ))
            del open_positions[ticket]
            continue

        # === ML-Based Dynamic Exit Logic ===
        elapsed_minutes = (pd.Timestamp.now() - data["open_time"]).total_seconds() / 60
        pnl_pct = (price - data["entry"]) / data["entry"] * 100 * data["direction"]

        snapshot = {
            "elapsed_time": elapsed_minutes,
            "unrealized_pnl": pnl_pct,
            "direction": data["direction"],
            "rsi14": get_rsi14(data["symbol"]),
            "macd_hist": get_macd_hist(data["symbol"])
        }

        p_win = predict_exit_probability(snapshot)

        if p_win < 0.35:
            pnl = pnl_pct * data["lot"]
            msg = f"⚠️ ML Exit: P(win)={p_win:.2f} | {data['symbol']} {'BUY' if data['direction']==1 else 'SELL'}"
            send_alert(msg)
            log_exit(ticket, data['symbol'], data['direction'], data['entry'], price, "ML_EXIT", pnl)

            mt5.order_send(mt5.OrderSendRequest(
                action=mt5.TRADE_ACTION_DEAL,
                symbol=data["symbol"],
                volume=data["lot"],
                type=mt5.ORDER_TYPE_SELL if data["direction"] == 1 else mt5.ORDER_TYPE_BUY,
                position=ticket,
                magic=123456,
                deviation=10
            ))
            del open_positions[ticket]
