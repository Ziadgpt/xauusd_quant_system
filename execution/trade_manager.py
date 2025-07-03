import MetaTrader5 as mt5
import pandas as pd

from utils.notifier import send_alert
from indicators.trailing_stop import calculate_trailing_stop
from indicators.rsi import calculate_rsi
from indicators.macd import calculate_macd


def open_trade(symbol="XAUUSD", lot=0.1, direction=1, sl=150, tp=300, strategy="RSI2", magic=234000):
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        msg = "❌ No tick data available for trade entry."
        print(msg)
        send_alert(msg)
        return

    price = tick.ask if direction == 1 else tick.bid
    order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
    sl_price = price - sl * 0.01 if direction == 1 else price + sl * 0.01
    tp_price = price + tp * 0.01 if direction == 1 else price - tp * 0.01

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "sl": round(sl_price, 2),
        "tp": round(tp_price, 2),
        "deviation": 10,
        "magic": magic,
        "comment": f"{strategy} Entry",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        msg = f"✅ Trade Executed: {symbol} {'BUY' if direction == 1 else 'SELL'} @ {price:.2f}"
        print(msg)
        send_alert(msg)
    else:
        error_msg = f"❌ Trade Failed: retcode {result.retcode} | {mt5.last_error()}"
        print(error_msg)
        send_alert(error_msg)


def manage_open_positions(symbol="XAUUSD"):
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        print("📭 No open positions.")
        return

    # Price and indicator data
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        print("❌ Failed to get current price tick.")
        return

    last_price = tick.bid  # for exits and SL

    # Get latest 100 candles for indicator analysis
    bars = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 100)
    if bars is None or len(bars) == 0:
        print("❌ Failed to fetch OHLCV for indicator analysis.")
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
        current_magic = pos.magic

        # --- Trailing Stop Logic ---
        new_sl = calculate_trailing_stop(entry, last_price, direction, distance=100)
        if new_sl:
            sl_request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "sl": round(new_sl, 2),
                "tp": tp,
            }

            sl_result = mt5.order_send(sl_request)
            if sl_result.retcode == mt5.TRADE_RETCODE_DONE:
                msg = f"🔒 Updated SL for ticket {ticket} → {new_sl:.2f}"
                print(msg)
                send_alert(msg)

        # --- Exit Logic ---
        should_exit = (
            (direction == 1 and macd_now < signal_now and rsi_now > 70) or
            (direction == -1 and macd_now > signal_now and rsi_now < 30)
        )

        if should_exit:
            close_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": pos.volume,
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
                msg = f"✅ Trade Closed: {symbol} {'BUY' if direction==1 else 'SELL'} @ {last_price:.2f} by exit logic"
                print(msg)
                send_alert(msg)
