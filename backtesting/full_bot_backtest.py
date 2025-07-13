import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta

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
TF_MINUTES = 15
SL_PIPS = 150
TP_PIPS = 300
INITIAL_BALANCE = 10000
RISK_PER_TRADE = 0.01
BACKTEST_DAYS = 60

# === Load Historical Data ===
df = get_ohlcv(SYMBOL, timeframe="M15", days=BACKTEST_DAYS)
if df is None or len(df) < 300:
    raise ValueError("❌ Not enough data to backtest.")

# === Add Indicators & Signals ===
df = apply_rsi2(df)
df = apply_macd_bollinger(df)
df = detect_hh_ll_breakout(df)

# === Simulation State ===
open_trade = None
trade_log = []

for i in range(100, len(df)):
    row = df.iloc[i]
    timestamp = row["time"]

    # Skip if already in trade
    if open_trade:
        # Manage exit
        current_price = row["close"]
        direction = open_trade["direction"]
        entry_price = open_trade["entry"]

        # Trailing Stop
        new_sl = calculate_trailing_stop(entry_price, current_price, direction, distance=100)
        if new_sl:
            open_trade["sl"] = max(open_trade["sl"], new_sl) if direction == 1 else min(open_trade["sl"], new_sl)

        # Check SL/TP
        if (direction == 1 and current_price <= open_trade["sl"]) or \
           (direction == -1 and current_price >= open_trade["sl"]):
            result = "SL"
        elif (direction == 1 and current_price >= open_trade["tp"]) or \
             (direction == -1 and current_price <= open_trade["tp"]):
            result = "TP"
        else:
            continue  # still holding

        # Close trade
        pnl = (current_price - entry_price) * direction * 10  # simple multiplier
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

    # === GARCH Filter ===
    window = df.iloc[i-100:i]
    volatility = forecast_garch_volatility(window.select_dtypes(include='number'))
    if volatility > 2.0:
        continue

    # === HMM Regime Detection ===
    regime, dominant = detect_market_regime(window.select_dtypes(include='number'))
    if regime != dominant:
        continue

    # === Strategy Ensemble ===
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
        signal = 0
        strategy = None

    # === ML Filter ===
    if signal != 0:
        features = {
            "rsi2": row["rsi2"],
            "rsi14": row["rsi14"],
            "macd_line": row["macd"],  # ✅ Fix: use the correct column name
            "macd_signal": row["macd_signal"],
            "macd_hist": row["macd_hist"],
            "bb_upper": row["bb_upper"],
            "bb_lower": row["bb_lower"],
            "bb_width": row["bb_upper"] - row["bb_lower"],
            "obv": row["obv"],
            "atr": row["atr"],
            "volatility": volatility,
            "regime": regime
        }
        ml_decision = predict_trade(features)
        if ml_decision == 0:
            continue  # ML rejected

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

# === Save Results ===
df_trades = pd.DataFrame(trade_log)
df_trades.to_csv("backtest_trades.csv", index=False)
print(f"✅ Backtest complete. {len(df_trades)} trades saved to backtest_trades.csv")
