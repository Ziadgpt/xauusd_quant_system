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
from execution.mt5_connector import initialize, shutdown
from data.fetch_data import get_ohlcv
from execution.trade_manager import open_trade
from execution.trade_manager import manage_exits

# Load env vars
load_dotenv()

# === Initialize MT5 ===
if not initialize():
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

        # === Ensemble Logic ===
        combined = rsi2_signal + macd_signal
        if combined == 2:
            signal = 1
            strategy_used = "RSI2 + MACD_BB"
        elif combined == -2:
            signal = -1
            strategy_used = "RSI2 + MACD_BB"
        elif rsi2_signal != 0:
            signal = rsi2_signal
            strategy_used = "RSI2 Only"
        else:
            signal = 0
            strategy_used = None

        print(f"📊 Signals — RSI2: {rsi2_signal}, MACD_BB: {macd_signal}, Final: {signal}")

        # === Execute Trade ===
        if signal != 0:
            direction = "BUY" if signal == 1 else "SELL"
            price = df.iloc[-1]["close"]
            print(f"🚨 Signal Detected: {direction} from {strategy_used} @ {price:.2f}")
            send_alert(f"🚨 {strategy_used} → {direction} on XAUUSD @ {price:.2f}")

            open_trade("XAUUSD", 0.1, signal)
            log_trade(signal, price, rsi_val, sl=150, tp=300)
        else:
            print("📈 No valid signal this cycle.")

        # === Exit Logic ===
        manage_exits()

        # === Wait for next 15M candle ===
        time.sleep(60 * 15)

except KeyboardInterrupt:
    print("🛑 Manually stopped.")
except Exception as e:
    print(f"❌ Fatal Error: {e}")
    send_alert(f"❌ Bot Error: {e}")
finally:
    shutdown()
