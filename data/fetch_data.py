def get_ohlcv(symbol="XAUUSDc", timeframe="M15", days=60):
    import MetaTrader5 as mt5
    import pandas as pd
    from datetime import datetime, timedelta

    if not mt5.initialize():
        raise RuntimeError("Failed to initialize MT5")

    tf_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1
    }

    tf = tf_map.get(timeframe.upper(), mt5.TIMEFRAME_M15)
    utc_from = datetime.now() - timedelta(days=days)
    rates = mt5.copy_rates_from(symbol, tf, utc_from, days * 96)  # ~96 bars/day for M15

    mt5.shutdown()

    if rates is None:
        print("❌ Could not fetch OHLCV data.")
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df
