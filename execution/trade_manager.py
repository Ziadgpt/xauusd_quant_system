import MetaTrader5 as mt5

def open_trade(symbol, lot, signal, sl_points=150, tp_points=300):
    price = mt5.symbol_info_tick(symbol).ask if signal == 1 else mt5.symbol_info_tick(symbol).bid
    order_type = mt5.ORDER_TYPE_BUY if signal == 1 else mt5.ORDER_TYPE_SELL
    deviation = 20
    sl = price - sl_points * mt5.symbol_info(symbol).point if signal == 1 else price + sl_points * mt5.symbol_info(symbol).point
    tp = price + tp_points * mt5.symbol_info(symbol).point if signal == 1 else price - tp_points * mt5.symbol_info(symbol).point

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": deviation,
        "magic": 10032025,
        "comment": "RSI entry",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    return result
