# execution/exit_manager.py

import MetaTrader5 as mt5
from utils.notifier import send_alert
from logs.logger import log_exit

open_positions = {}

def track_trade(ticket, symbol, direction, entry_price, sl, tp, lot=0.1):
    open_positions[ticket] = {
        "symbol": symbol,
        "direction": direction,
        "entry": entry_price,
        "sl": sl,
        "tp": tp,
        "lot": lot
    }

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

        if data["direction"] == 1:  # BUY
            if price <= data["sl"]:
                hit = "SL"
            elif price >= data["tp"]:
                hit = "TP"
        else:  # SELL
            if price >= data["sl"]:
                hit = "SL"
            elif price <= data["tp"]:
                hit = "TP"

        if hit:
            pnl = (price - data["entry"]) * data["lot"] * 100 if data["direction"] == 1 else (data["entry"] - price) * data["lot"] * 100
            msg = (
                f"💰 Exit {hit} | {data['symbol']} {('BUY' if data['direction'] == 1 else 'SELL')}\n"
                f"Entry: {data['entry']:.2f}\n"
                f"Exit: {price:.2f}\n"
                f"PnL: ${pnl:.2f}"
            )
            send_alert(msg)
            log_exit(ticket, data['symbol'], data['direction'], data['entry'], price, hit, pnl)
            mt5.order_send(mt5.OrderSendRequest(  # optional close order
                action=mt5.TRADE_ACTION_DEAL,
                symbol=data["symbol"],
                volume=data["lot"],
                type=mt5.ORDER_TYPE_SELL if data["direction"] == 1 else mt5.ORDER_TYPE_BUY,
                position=ticket,
                magic=123456,
                deviation=10
            ))
            del open_positions[ticket]
