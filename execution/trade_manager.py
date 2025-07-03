import MetaTrader5 as mt5
from datetime import datetime, timedelta

# === Parameters ===
TRAILING_DISTANCE_POINTS = 100           # Trailing SL in points (e.g., 10.0 USD = 1000 pts for XAUUSD)
MAX_TRADE_DURATION_MINUTES = 90          # Max holding time
VOLATILITY_THRESHOLD = 2.0               # Vol threshold to force exit

# === Trade Entry ===
def open_trade(symbol: str, volume: float, signal: int, sl=150, tp=300):
    tick = mt5.symbol_info_tick(symbol)
    point = mt5.symbol_info(symbol).point
    price = tick.ask if signal == 1 else tick.bid
    order_type = mt5.ORDER_TYPE_BUY if signal == 1 else mt5.ORDER_TYPE_SELL

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": price - sl * point if signal == 1 else price + sl * point,
        "tp": price + tp * point if signal == 1 else price - tp * point,
        "deviation": 20,
        "magic": 234000,
        "comment": "XAUUSD Quant Entry",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"❌ Trade failed: {result.comment}")
    else:
        print(f"✅ Trade executed: {symbol} {volume} lots {'BUY' if signal == 1 else 'SELL'} at {price:.2f}")

# === Trade Exit Manager ===
def manage_exits():
    positions = mt5.positions_get(symbol="XAUUSD")
    if positions is None or len(positions) == 0:
        print("📭 No open positions found.")
        return

    now = datetime.utcnow()
    point = mt5.symbol_info("XAUUSD").point
    tick = mt5.symbol_info_tick("XAUUSD")

    for pos in positions:
        open_time = datetime.fromtimestamp(pos.time)
        duration = (now - open_time).total_seconds() / 60  # minutes

        price = tick.ask if pos.type == mt5.POSITION_TYPE_SELL else tick.bid

        # === Trailing SL ===
        if pos.type == mt5.POSITION_TYPE_BUY:
            new_sl = price - TRAILING_DISTANCE_POINTS * point
            if pos.sl is None or pos.sl < new_sl:
                modify_sl(pos, new_sl)
        elif pos.type == mt5.POSITION_TYPE_SELL:
            new_sl = price + TRAILING_DISTANCE_POINTS * point
            if pos.sl is None or pos.sl > new_sl:
                modify_sl(pos, new_sl)

        # === Max Time Exit ===
        if duration > MAX_TRADE_DURATION_MINUTES:
            close_position(pos)
            print(f"⏰ Exited due to time limit: {pos.ticket}")
            continue

        # === Volatility or Regime Exit (placeholders) ===
        try:
            from models.garch_model import forecast_garch_volatility
            from models.hmm_model import detect_market_regime
            from data.fetch_data import get_ohlcv

            df = get_ohlcv()
            current_vol = forecast_garch_volatility(df)
            regime, dominant = detect_market_regime(df)

            if current_vol > VOLATILITY_THRESHOLD or regime != dominant:
                close_position(pos)
                print(f"⚠️ Exit due to volatility or regime shift: {pos.ticket}")
        except Exception as e:
            print(f"⚠️ Exit model skipped: {e}")

# === SL Modifier ===
def modify_sl(position, new_sl):
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": position.symbol,
        "position": position.ticket,
        "sl": new_sl,
        "tp": position.tp,
        "deviation": 10,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"❌ SL update failed for {position.ticket}: {result.comment}")
    else:
        print(f"🔒 SL updated for {position.ticket} → {new_sl:.2f}")

# === Close Order Logic ===
def close_position(position):
    tick = mt5.symbol_info_tick(position.symbol)
    price = tick.bid if position.type == mt5.POSITION_TYPE_BUY else tick.ask
    close_type = mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "position": position.ticket,
        "symbol": position.symbol,
        "volume": position.volume,
        "type": close_type,
        "price": price,
        "deviation": 10,
        "magic": 234000,
        "comment": "Auto Exit",
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"❌ Failed to close position {position.ticket}: {result.comment}")
    else:
        print(f"✅ Position {position.ticket} closed at {price:.2f}")
