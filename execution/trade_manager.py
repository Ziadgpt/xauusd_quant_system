import MetaTrader5 as mt5
import pandas as pd

from utils.notifier import send_alert
from indicators.trailing_stop import calculate_trailing_stop
from indicators.rsi import calculate_rsi
from indicators.macd import calculate_macd
from utils.risk import calculate_lot_size

def open_trade(symbol="XAUUSDc", direction=1, sl=150, tp=300, strategy="Unknown", magic=234000, risk_percent=1.0):
    account = mt5.account_info()
    if account is None:
        print("❌ Account info not available.")
        return

    balance = account.balance
    lot = calculate_lot_size(balance, sl, risk_percent)
    if lot <= 0:
        print("⚠️ Invalid lot size.")
        return

    # Get latest price
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        print("❌ Failed to get tick data.")
        return

    price = tick.ask if direction == 1 else tick.bid
    sl_price = price - sl * 0.01 if direction == 1 else price + sl * 0.01
    tp_price = price + tp * 0.01 if direction == 1 else price - tp * 0.01

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": round(lot, 2),
        "type": mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": round(sl_price, 2),
        "tp": round(tp_price, 2),
        "deviation": 10,
        "magic": magic,
        "comment": f"Entry via {strategy}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC
    }

    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        msg = f"✅ Trade Opened: {symbol} {'BUY' if direction==1 else 'SELL'} @ {price:.2f} | SL: {sl_price:.2f} | TP: {tp_price:.2f}"
        print(msg)
        send_alert(msg)
    else:
        msg = f"❌ Trade Failed: {result.retcode} → {result.comment}"
        print(msg)
        send_alert(msg)


def manage_open_positions(symbol="XAUUSDc"):
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        print("📭 No open positions.")
        return

    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        print("❌ Failed to get current price tick.")
        return

    last_price = tick.bid

    # Get recent candles for indicator analysis
    bars = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 100)
    if bars is None or len(bars) == 0:
        print("❌ Failed to fetch OHLCV.")
        return

    df = pd.DataFrame(bars)
    df["rsi"] = calculate_rsi(df["close"], period=14)
    macd_line, macd_signal, _ = calculate_macd(df["close"])
    df["macd"] = macd_line
    df["macd_signal"] = macd_signal

    rsi_now = df["rsi"].iloc[-1]
    macd_now = df["macd"].iloc[-1]
    signal_now = df["macd_signal"].iloc[-1]

    for pos in positions:
        ticket = pos.ticket
        direction = 1 if pos.type == mt5.ORDER_TYPE_BUY else -1
        entry = pos.price_open
        tp = pos.tp
        current_sl = pos.sl or 0
        current_magic = pos.magic

        # === Trailing Stop ===
        new_sl = calculate_trailing_stop(entry, last_price, direction, distance=100)

        if new_sl:
            should_update_sl = (
                (direction == 1 and (current_sl == 0 or new_sl > current_sl)) or
                (direction == -1 and (current_sl == 0 or new_sl < current_sl))
            )

            if should_update_sl:
                sl_request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": ticket,
                    "sl": round(new_sl, 2),
                    "tp": tp,
                }

                sl_result = mt5.order_send(sl_request)
                if sl_result.retcode == mt5.TRADE_RETCODE_DONE:
                    msg = f"🔒 SL Updated: ticket {ticket} → {new_sl:.2f}"
                    print(msg)
                    send_alert(msg)
                else:
                    print(f"❌ Failed SL update: {sl_result.retcode}")

        # === Exit Conditions (MACD crossover + RSI overbought/sold) ===
        should_exit = (
            (direction == 1 and macd_now < signal_now and rsi_now > 70) or
            (direction == -1 and macd_now > signal_now and rsi_now < 30)
        )

        if should_exit:
            close_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": round(pos.volume, 2),
                "type": mt5.ORDER_TYPE_SELL if direction == 1 else mt5.ORDER_TYPE_BUY,
                "position": ticket,
                "price": last_price,
                "deviation": 10,
                "magic": current_magic,
                "comment": "Exit via MACD/RSI",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC
            }

            close_result = mt5.order_send(close_request)
            if close_result.retcode == mt5.TRADE_RETCODE_DONE:
                msg = f"✅ Trade Closed: {symbol} {'BUY' if direction==1 else 'SELL'} @ {last_price:.2f} via exit logic"
                print(msg)
                send_alert(msg)
            else:
                print(f"❌ Close failed: {close_result.retcode} → {close_result.comment}")
