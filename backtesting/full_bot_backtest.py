import pandas as pd
import numpy as np
import ta
import sys
import os
from datetime import datetime, timedelta

# === Add root path ===
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# === Imports ===
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
INITIAL_BALANCE = 10000
RISK_PER_TRADE = 0.01
BACKTEST_BARS = 4000

# === Load Historical Data ===
df = get_ohlcv(SYMBOL, timeframe=TF, count=BACKTEST_BARS)
if df is None or len(df) < 300:
    raise ValueError("❌ Not enough data to backtest.")

# === Apply Core Indicators ===
df = apply_rsi2(df)
df = apply_macd_bollinger(df)
df = detect_hh_ll_breakout(df)

# === Add Matching ML Features ===
df["return_pct"] = df["close"].pct_change() * 100
df["body"] = df["close"] - df["open"]
df["range"] = df["high"] - df["low"]
df["body_ratio"] = df["body"] / df["range"].replace(0, np.nan)
df["wick_ratio"] = (df["high"] - df["close"]) / df["range"].replace(0, np.nan)

# Engulfing pattern
df["engulfing"] = ((df["body"].shift(1) < 0) & (df["body"] > abs(df["body"].shift(1)))).astype(int)

# Trend slope
df["trend_slope"] = df["close"].rolling(5).apply(lambda x: np.polyfit(range(5), x, 1)[0], raw=True)

# Additional Indicators
df["roc"] = ta.momentum.ROCIndicator(df["close"]).roc()
df["cci"] = ta.trend.CCIIndicator(df["high"], df["low"], df["close"]).cci()
df["stoch"] = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"]).stoch()
df["williams_r"] = ta.momentum.WilliamsRIndicator(df["high"], df["low"], df["close"]).williams_r()
df["bb_distance"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
df["bb_bandwidth"] = df["bb_upper"] - df["bb_lower"]
df["rolling_std_5"] = df["close"].rolling(5).std()
df["sma_20"] = ta.trend.SMAIndicator(df["close"], 20).sma_indicator()
df["ema_20"] = ta.trend.EMAIndicator(df["close"], 20).ema_indicator()
df["vwap"] = ta.volume.VolumeWeightedAveragePrice(
    high=df["high"],
    low=df["low"],
    close=df["close"],
    volume=df["tick_volume"]
).vwap
df["accum_dist"] = ta.volume.AccumulationDistributionIndicator(df["high"], df["low"], df["close"], df["tick_volume"]).acc_dist_index()
df["volume_delta"] = df["tick_volume"].diff()
df["garch_vol"] = df["return_pct"].rolling(20).std()
df["hour"] = pd.to_datetime(df["time"]).dt.hour
df["weekday"] = pd.to_datetime(df["time"]).dt.weekday
df["direction"] = np.where(df["close"].shift(-1) > df["close"], 1, 0)

# === Simulation ===
open_trade = None
trade_log = []

for i in range(100, len(df)):
    row = df.iloc[i]
    timestamp = row["time"]

    if open_trade:
        # Manage exit
        current_price = row["close"]
        direction = open_trade["direction"]
        entry_price = open_trade["entry"]

        new_sl = calculate_trailing_stop(entry_price, current_price, direction, distance=100)
        if new_sl:
            open_trade["sl"] = max(open_trade["sl"], new_sl) if direction == 1 else min(open_trade["sl"], new_sl)

        if (direction == 1 and current_price <= open_trade["sl"]) or \
           (direction == -1 and current_price >= open_trade["sl"]):
            result = "SL"
        elif (direction == 1 and current_price >= open_trade["tp"]) or \
             (direction == -1 and current_price <= open_trade["tp"]):
            result = "TP"
        else:
            continue

        pnl = (current_price - entry_price) * direction * 10
        trade_log.append({
            "timestamp": open_trade["timestamp"],
            "exit_time": timestamp,
            "direction": "BUY" if direction == 1 else "SELL",
            "entry": entry_price,
            "exit": current_price,
            "pnl": pnl,
            "result": result,
            "strategy": open_trade["strategy"]
        })
        open_trade = None
        continue

    # Filters
    window = df.iloc[i-100:i]
    volatility = forecast_garch_volatility(window.select_dtypes(include='number'))
    if volatility > 2.0:
        continue

    regime, dominant = detect_market_regime(window.select_dtypes(include='number'))
    if regime != dominant:
        continue

    # Signals
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
    features = row.to_dict()
    features["volatility"] = volatility
    features["regime"] = regime

    try:
        ml_decision = predict_trade(features)
        if ml_decision == 0:
            continue
    except Exception as e:
        print(f"⚠️ ML error: {e}")
        continue

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
