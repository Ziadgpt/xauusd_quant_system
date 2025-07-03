import MetaTrader5 as mt5
from utils.notifier import send_alert
import pandas as pd
import time
from indicators.trailing_stop import calculate_trailing_stop
from indicators.rsi import calculate_rsi
from indicators.macd import calculate_macd

def manage_exits():
    positions = mt5.positions_get(symbol="XAUUSD")
    if not positions:
        return

    # Get latest OHLCV to compute indicators
    from data.fetch_data import get_ohlcv
    df = get_ohlcv()
    if df is None or len(df) < 30:
        return

    df["rsi"] = calculate_rsi(df["close"], period=14)
    df["macd"], df["signal"], _ = calculate_macd(df["close"])
    last_price = df.iloc[-1]["close"]
    last_rsi = df.iloc[-1]["rsi"]
    macd = df.iloc[-1]["macd"]
    signal = df.iloc[-1]["signal"]

    for pos in positions:
        ticket = pos.ticket
        open_price = pos.price_open
        volume = pos.volume
        direction = 1 if pos.type == 0 else -1  # 0 = BUY, 1 = SELL
        sl = pos.sl
        tp = pos.tp
        time_open = pos.time

        # === Trailing Stop Logic ===
        new_sl = calculate_trailing_stop(open_price, last_price, direction, distance=100)
        if new_sl and new_sl != sl:
            mt5.order_modify(ticket, sl=new_sl, tp=tp)
            print(f"🔄 Trailing SL updated for {ticket}: {new_sl:.2f}")

        # === RSI Exit ===
        if (direction == 1 and last_rsi > 85) or (direction == -1 and last_rsi < 15):
            close_position(ticket, volume)
            send_alert(f"📉 RSI Exit Triggered on {ticket}")
            continue

        # === MACD Exit ===
        if (direction == 1 and macd < signal) or (direction == -1 and macd > signal):
            close_position(ticket, volume)
            send_alert(f"📉 MACD Exit Triggered on {ticket}")
            continue

        # === Time-Based Exit (e.g., 2h = 8 candles) ===
        age = (time.time() - time_open) / 60  # minutes
        if age > 120:
            close_position(ticket, volume)
            send_alert(f"⏱️ Timed Exit Triggered after {int(age)} mins.")
            continue

def close_position(ticket, volume):
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "position": ticket,
        "volume": volume,
        "type": mt5.ORDER_TYPE_SELL if mt5.positions_get(ticket=ticket)[0].type == 0 else mt5.ORDER_TYPE_BUY,
        "price": mt5.symbol_info_tick("XAUUSD").bid if mt5.positions_get(ticket=ticket)[0].type == 0 else mt5.symbol_info_tick("XAUUSD").ask,
        "deviation": 10,
        "magic": 234000,
        "comment": "Smart Exit Trigger",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC
    }

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"❌ Exit Failed: {result.retcode}")
    else:
        print(f"✅ Position Closed: {ticket}")
