import pandas as pd
import numpy as np
import ta
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetch_data import get_ohlcv
from strategies.momentum import apply_rsi2
from strategies.macd_bollinger import apply_macd_bollinger
from strategies.structure_breakout import detect_hh_ll_breakout
from models.garch_model import forecast_garch_volatility
from models.hmm_model import detect_market_regime
from ml.predictor import predict_trade
from indicators.trailing_stop import calculate_trailing_stop

# === Config ===
SYMBOL = "XAUUSDc"
TF = "M15"
SL_PIPS = 150
TP_PIPS = 300
BACKTEST_BARS = 4000

# === Load Historical Data ===
df = get_ohlcv(SYMBOL, timeframe=TF, count=BACKTEST_BARS)
if df is None or len(df) < 300:
    raise ValueError("❌ Not enough data to backtest.")

# === Add Core Strategy Signals ===
df = apply_rsi2(df)
df = apply_macd_bollinger(df)
df = detect_hh_ll_breakout(df)

# === Feature Engineering (match feature_builder.py) ===
df["return_pct"] = df["close"].pct_change() * 100
df["body"] = df["close"] - df["open"]
df["range"] = df["high"] - df["low"]
df["body_ratio"] = df["body"] / df["range"].replace(0, np.nan)
df["wick_ratio"] = (df["high"] - df["close"]) / df["range"].replace(0, np.nan)
df["engulfing"] = ((df["body"].shift(1) < 0) & (df["body"] > abs(df["body"].shift(1)))).astype(int)
df["trend_slope"] = df["close"].rolling(5).apply(lambda x: np.polyfit(range(5), x, 1)[0], raw=True)
df["roc"] = ta.momentum.ROCIndicator(df["close"]).roc()
df["cci"] = ta.trend.CCIIndicator(df["high"], df["low"], df["close"]).cci()
df["stoch"] = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"]).stoch()
df["williams_r"] = ta.momentum.WilliamsRIndicator(df["high"], df["low"], df["close"]).williams_r()
df["bb_distance"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
df["bb_bandwidth"] = df["bb_upper"] - df["bb_lower"]
df["rolling_std_5"] = df["close"].rolling(5).std()
df["sma_20"] = ta.trend.SMAIndicator(df["close"], 20).sma_indicator()
df["ema_20"] = ta.trend.EMAIndicator(df["close"], 20).ema_indicator()

# === Missing ML Features ===
# OBV
df["obv"] = ta.volume.OnBalanceVolumeIndicator(close=df["close"], volume=df["tick_volume"]).on_balance_volume()

# RSI 14
df["rsi_14"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

# RSI 2
df["rsi_2"] = ta.momentum.RSIIndicator(df["close"], window=2).rsi()

# VWAP workaround (no volume support in `ta`)
df["vwap"] = (df["close"] * df["tick_volume"]).cumsum() / df["tick_volume"].cumsum()

# Accumulation/Distribution (manual calculation)
df["accum_dist"] = ((2 * df["close"] - df["high"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)) * df["tick_volume"]
df["accum_dist"] = df["accum_dist"].cumsum()

df["volume_delta"] = df["tick_volume"].diff()
df["garch_vol"] = df["return_pct"].rolling(20).std()
df["hour"] = pd.to_datetime(df["time"]).dt.hour
df["weekday"] = pd.to_datetime(df["time"]).dt.weekday
df["direction"] = np.where(df["close"].shift(-1) > df["close"], 1, 0)

# === Backtest Simulation ===
open_trade = None
trade_log = []

for i in range(100, len(df)):
    row = df.iloc[i]
    timestamp = row["time"]

    if open_trade:
        price = row["close"]
        dir = open_trade["direction"]
        sl_updated = calculate_trailing_stop(open_trade["entry"], price, dir, distance=100)
        if sl_updated:
            open_trade["sl"] = max(open_trade["sl"], sl_updated) if dir == 1 else min(open_trade["sl"], sl_updated)

        if (dir == 1 and price <= open_trade["sl"]) or (dir == -1 and price >= open_trade["sl"]):
            result = "SL"
        elif (dir == 1 and price >= open_trade["tp"]) or (dir == -1 and price <= open_trade["tp"]):
            result = "TP"
        else:
            continue

        pnl = (price - open_trade["entry"]) * dir * 10
        trade_log.append({
            "timestamp": open_trade["timestamp"],
            "exit_time": timestamp,
            "direction": "BUY" if dir == 1 else "SELL",
            "entry": open_trade["entry"],
            "exit": price,
            "pnl": pnl,
            "result": result,
            "strategy": open_trade["strategy"]
        })
        open_trade = None
        continue

    # === Filters ===
    window = df.iloc[i-100:i]
    vol = forecast_garch_volatility(window.select_dtypes(include='number'))
    regime, dominant = detect_market_regime(window.select_dtypes(include='number'))
    if vol > 2.0 or regime != dominant:
        continue

    # === Signal Generation ===
    rsi_signal = row.get("signal", 0)
    macd_signal = row.get("signal_macd_bb", 0)
    structure_signal = row.get("signal_structure", 0)
    combined = rsi_signal + macd_signal + structure_signal

    if combined >= 2:
        signal = 1
        strategy = "Ensemble Long"
    elif combined <= -2:
        signal = -1
        strategy = "Ensemble Short"
    elif structure_signal != 0:
        signal = structure_signal
        strategy = "Structure Only"
    elif macd_signal != 0:
        signal = macd_signal
        strategy = "MACD_BB Only"
    elif rsi_signal != 0:
        signal = rsi_signal
        strategy = "RSI2 Only"
    else:
        continue

    # === ML Filter ===
    try:
        features = row.to_dict()
        features["volatility"] = vol
        features["regime"] = regime
        ml_result = predict_trade(features)
        if ml_result == 0:
            continue
    except Exception as e:
        print(f"⚠️ ML error: {e}")
        continue

    # === Execute Trade ===
    entry = row["close"]
    sl = entry - SL_PIPS * 0.01 if signal == 1 else entry + SL_PIPS * 0.01
    tp = entry + TP_PIPS * 0.01 if signal == 1 else entry - TP_PIPS * 0.01

    open_trade = {
        "timestamp": timestamp,
        "entry": entry,
        "direction": signal,
        "sl": sl,
        "tp": tp,
        "strategy": strategy
    }

# === Save ===
df_trades = pd.DataFrame(trade_log)
df_trades.to_csv("backtest_trades.csv", index=False)
print(f"✅ Backtest complete. {len(df_trades)} trades saved to backtest_trades.csv")
