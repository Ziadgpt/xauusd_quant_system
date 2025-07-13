import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

# === Connect to MT5 ===
if not mt5.initialize():
    raise RuntimeError("❌ MT5 initialization failed")

# === Fetch OHLCV Data ===
def get_ohlcv(symbol, timeframe="M15", days=None, count=None):
    if count is not None:
        bars = count
    elif days is not None:
        bars = days * 96  # ~96 candles/day for M15
    else:
        bars = 500  # default fallback

    # Use MetaTrader5 constants for timeframes
    tf = getattr(mt5, timeframe, mt5.TIMEFRAME_M15)

    rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
    if rates is None or len(rates) == 0:
        raise ValueError("❌ No data fetched from MT5")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df
