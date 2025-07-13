import MetaTrader5 as mt5
import pandas as pd

def get_ohlcv(symbol="XAUUSDc", timeframe="M15", count=1000):
    tf_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "H1": mt5.TIMEFRAME_H1
    }

    if not mt5.initialize():
        raise RuntimeError("❌ MT5 Initialization failed")

    tf = tf_map.get(timeframe.upper(), mt5.TIMEFRAME_M15)
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
    mt5.shutdown()

    if rates is None:
        raise ValueError("❌ No OHLCV data returned from MT5")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df
