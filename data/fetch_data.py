import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta

def get_ohlcv(symbol="XAUUSD", timeframe=mt5.TIMEFRAME_M15, bars=100):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df
