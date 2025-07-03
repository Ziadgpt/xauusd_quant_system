import MetaTrader5 as mt5
from datetime import datetime, timedelta

# Parameters (customize as needed)
TRAILING_DISTANCE_POINTS = 100       # points for trailing stop (e.g., 10.0 USD for XAUUSD)
MAX_TRADE_DURATION_MINUTES = 90      # max time to hold a trade
VOLATILITY_THRESHOLD = 2.0            # example GARCH volatility limit

def manage_exits():
    positions = mt5.positions_get(symbol="XAUUSD")
    if positions is None:
        print("No open positions found.")
        return

    now = datetime.utcnow()

    for pos in positions:
        open_time = datetime.fromtimestamp(pos.time)
        duration = (now - open_time).total_seconds() / 60  # minutes

        # Current price
        tick = mt5.symbol_info_tick("XAUUSD")
        if tick is None:
            print("Failed to get tick data.")
            continue
        price = tick.ask if pos.type == mt5.POSITION_TYPE_SELL else tick.bid

        # Trailing stop logic
        if pos.type == mt5.POSITION_TYPE_BUY:
            new_sl = price - TRAILING_DISTANCE_POINTS * mt5.symbol_info("XAUUSD").point
            if pos.sl < new_sl:
                modify_sl(pos, new_sl)
        elif pos.type == mt5.POSITION_TYPE_SELL:
            new_sl = price + TRAILING_DISTANCE_POINTS * mt5.symbol_info("XAUUSD").point
            if pos.sl > new_sl or pos.sl == 0.0:
                modify_sl(pos, new_sl)

        # Time-based exit
        if duration > MAX_TRADE_DURATION_MINUTES:
            close_position(pos)
            print(f"Closed position due to max duration: {pos.ticket}")
            continue

        # Volatility/regime exit placeholder
        current_vol = get_current_volatility()  # you define this based on your GARCH model
        current_regime = get_current_regime()   # you define this with your HMM model
        if current_vol > VOLATILITY_THRESHOLD or current_regime == "non-trend":
            close_position(pos)
            print(f"Closed position due to volatility/regime change: {pos.ticket}")

def modify_sl(position, new_sl):
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": position.symbol,
        "position": position.ticket,
        "sl": new_sl,
        "tp": position.tp,
        "deviation": 10,
        "type": position.type,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Failed to modify SL: {result.comment}")
    else:
        print(f"SL modified for position {position.ticket} to {new_sl}")

def close_position(position):
    price = mt5.symbol_info_tick(position.symbol).bid if position.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(position.symbol).ask
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "position": position.ticket,
        "symbol": position.symbol,
        "volume": position.volume,
        "type": mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
        "price": price,
        "deviation": 10,
        "magic": 234000,
        "comment": "Exit by bot",
        "type_filling": mt5.ORDER_FILLING_RETURN,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Failed to close position {position.ticket}: {result.comment}")
    else:
        print(f"Position {position.ticket} closed successfully")
