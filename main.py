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
from strategies.atr_breakout import apply_atr_breakout


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


# Load env vars
load_dotenv()

# === Initialize MT5 ===
if not connect_mt5():
    print("❌ MT5 initialization failed. Exiting.")
    sys.exit()
print("✅ MT5 connected.")

try:
    while True:
        print("\n🔁 Starting new cycle...")

        # === Fetch OHLCV ===
        df = get_ohlcv()
        if df is None or len(df) < 30:
            print("⚠️ Data fetch failed or insufficient bars.")
            time.sleep(60)
            continue

        # === Volatility Check ===
        vol = forecast_garch_volatility(df)
        print(f"📉 Forecasted Volatility: {vol:.2f}%")
        if vol > 2.0:
            send_alert("⚠️ High volatility — skipping trade.")
            time.sleep(60 * 15)
            continue

        # === Regime Check ===
        regime, dominant = detect_market_regime(df)
        print(f"📊 Market Regime: {regime}, Dominant: {dominant}")
        if regime != dominant:
            send_alert("📉 Non-trending regime — no trades.")
            time.sleep(60 * 15)
            continue

        # === Strategy 1: RSI2 ===
        df = apply_rsi2(df)
        rsi2_signal = df.iloc[-1]["signal"]
        rsi_val = df.iloc[-1]["rsi2"]


        # === Strategy 2: MACD + BB ===
        df = apply_macd_bollinger(df)
        macd_signal = df.iloc[-1]["signal_macd_bb"]

        if macd_signal != 0:
            direction = "BUY" if macd_signal == 1 else "SELL"
            price = df.iloc[-1]["close"]

            print(f"🚨 MACD+BB Signal: {direction} at {price:.2f}")
            send_alert(f"⚡ MACD+BB Entry: {direction} @ {price:.2f}")

            if mt5_enabled:
                open_trade("XAUUSD", direction=signal, sl=150, tp=300, strategy=strategy_used, risk_percent=1.0)

            log_trade(macd_signal, price, df.iloc[-1]["macd"], sl=150, tp=300)

        # === Strategy 3: HH/LL Breakout ===
        df = detect_hh_ll_breakout(df)
        structure_signal = df.iloc[-1]["signal_structure"]

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

        # === Execute Trade ===
        if signal != 0:
            direction = "BUY" if signal == 1 else "SELL"
            price = df.iloc[-1]["close"]
            print(f"🚨 Signal Detected: {direction} from {strategy_used} @ {price:.2f}")
            send_alert(f"🚨 {strategy_used} → {direction} on XAUUSD @ {price:.2f}")

            if mt5_enabled:
                open_trade("XAUUSD", 0.1, signal)

            log_trade(signal, price, rsi_val, sl=150, tp=300, strategy=strategy_used)

            # Track for exit logic
            record_signal(
                timestamp=df.iloc[-1]["time"],
                symbol="XAUUSD",
                direction=signal,
                entry_price=price,
                sl=price - 1.5 if signal == 1 else price + 1.5,
                tp=price + 3.0 if signal == 1 else price - 3.0
            )

        # === Exit Logic ===
        manage_open_positions()

        # === Wait for next 15M candle ===
        time.sleep(60 * 15)

except KeyboardInterrupt:
    print("🛑 Manually stopped.")
except Exception as e:
    print(f"❌ Fatal Error: {e}")
    send_alert(f"❌ Bot Error: {e}")
finally:
    shutdown_mt5()
