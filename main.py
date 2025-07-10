import time
import sys
import os

from dotenv import load_dotenv
from strategies.momentum import apply_rsi2
from strategies.macd_bollinger import apply_macd_bollinger
from logs.logger import log_trade
from utils.notifier import send_alert
from models.garch_model import forecast_garch_volatility
from models.hmm_model import detect_market_regime
from utils.mt5_connector import connect_mt5, shutdown_mt5
from data.fetch_data import get_ohlcv
from execution.trade_manager import open_trade
from execution.trade_manager import manage_open_positions
from sim.signal_tracker import record_signal
from strategies.structure_breakout import detect_hh_ll_breakout
# from strategies.atr_breakout import apply_atr_breakout  # Optional if used

# === Load Environment Variables ===
load_dotenv()

# === Symbol Setup for Cent Account ===
symbol = "XAUUSDc"  # Update this if your broker uses "XAUUSD.cent" instead

# --- MT5 Setup ---
mt5_enabled = True

try:
    if not connect_mt5():
        print("❌ MT5 initialization failed.")
        mt5_enabled = False
    else:
        print("✅ MT5 connected.")
except Exception as e:
    print(f"⚠️ MT5 import or init failed: {e}")
    mt5_enabled = False

try:
    while True:
        print("\n🔁 Starting new cycle...")

        # === Fetch OHLCV ===
        df = get_ohlcv(symbol=symbol)
        if df is None or len(df) < 30:
            print("⚠️ Data fetch failed or insufficient bars.")
            time.sleep(60)
            continue

        # === Volatility Filter ===
        vol = forecast_garch_volatility(df)
        print(f"📉 Forecasted Volatility: {vol:.2f}%")
        if vol > 2.0:
            send_alert("⚠️ High volatility — skipping trade.")
            time.sleep(60 * 15)
            continue

        # === Market Regime Filter ===
        regime, dominant = detect_market_regime(df)
        print(f"📊 Market Regime: {regime}, Dominant: {dominant}")
        if regime != dominant:
            send_alert("📉 Non-trending regime — no trades.")
            time.sleep(60 * 15)
            continue

        # === Strategy 1: RSI(2) ===
        df = apply_rsi2(df)
        rsi2_signal = df.iloc[-1]["signal"]
        rsi_val = df.iloc[-1]["rsi2"]

        # === Strategy 2: MACD + Bollinger Bands ===
        df = apply_macd_bollinger(df)
        macd_signal = df.iloc[-1]["signal_macd_bb"]

        # === Strategy 3: HH/LL Structure Breakout ===
        df = detect_hh_ll_breakout(df)
        structure_signal = df.iloc[-1]["signal_structure"]

        # === (Optional) Strategy 4: ATR Breakout ===
        # df = apply_atr_breakout(df)
        # atr_signal = df.iloc[-1]["signal_atr"]

        # === Ensemble Logic ===
        combined = rsi2_signal + macd_signal + structure_signal

        if combined >= 2:
            signal = 1
            strategy_used = "Ensemble (RSI2 + MACD + Structure)"
        elif combined <= -2:
            signal = -1
            strategy_used = "Ensemble (RSI2 + MACD + Structure)"
        elif structure_signal != 0:
            signal = structure_signal
            strategy_used = "Structure Only"
        elif macd_signal != 0:
            signal = macd_signal
            strategy_used = "MACD_BB Only"
        elif rsi2_signal != 0:
            signal = rsi2_signal
            strategy_used = "RSI2 Only"
        else:
            signal = 0
            strategy_used = None

        # === Trade Execution ===
        if signal != 0:
            direction = "BUY" if signal == 1 else "SELL"
            price = df.iloc[-1]["close"]
            print(f"🚨 Signal Detected: {direction} from {strategy_used} @ {price:.2f}")
            send_alert(f"🚨 {strategy_used} → {direction} on {symbol} @ {price:.2f}")

            if mt5_enabled:
                open_trade(
                    symbol=symbol,
                    direction=direction,
                    sl=150,
                    tp=300,
                    strategy=strategy_used,
                    risk_percent=1.0
                )

            log_trade(signal, price, rsi_val, sl=150, tp=300, strategy=strategy_used)

            # Signal Tracker for Exits
            record_signal(
                timestamp=df.iloc[-1]["time"],
                symbol=symbol,
                direction=signal,
                entry_price=price,
                sl=price - 1.5 if signal == 1 else price + 1.5,
                tp=price + 3.0 if signal == 1 else price - 3.0
            )
        else:
            print("ℹ️ No signal this cycle.")

        # === Exit Logic ===
        manage_open_positions()

        # === Wait until next 15M candle ===
        time.sleep(60 * 15)

except KeyboardInterrupt:
    print("🛑 Manually stopped.")
except Exception as e:
    print(f"❌ Fatal Error: {e}")
    send_alert(f"❌ Bot Error: {e}")
finally:
    shutdown_mt5()
