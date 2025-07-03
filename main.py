import time
import sys

from strategies.momentum import apply_rsi2
from logs.logger import log_trade
from utils.notifier import send_alert
from models.garch_model import forecast_garch_volatility
from models.hmm_model import detect_market_regime
from execution.trade_manager import manage_exits  # ✅ only this

# Start in simulation mode by default
simulation_mode = True

try:
    from data.fetch_data import get_ohlcv
    from execution.mt5_connector import initialize, shutdown
    from execution.trade_manager import open_trade  # ✅ no exit_manager here

    if initialize():
        print("✅ MT5 connected.")
        simulation_mode = False
    else:
        print("❌ MT5 initialization failed. Running in simulation mode.")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("⚠️ Required MT5 modules missing. Running in simulation mode.")
    simulation_mode = True

print("🧠 Simulation Mode:", simulation_mode)

try:
    while True:
        if simulation_mode:
            print("⏳ Running strategy logic (simulation mode)...")
            time.sleep(60 * 15)
            continue

        # Fetch latest OHLCV data
        df = get_ohlcv()
        if df is None or len(df) < 30:
            print("⚠️ Data fetch failed or insufficient candles.")
            time.sleep(60)
            continue

        # Volatility filter (GARCH)
        vol = forecast_garch_volatility(df)
        print(f"📉 Forecasted Volatility: {vol:.2f}%")
        if vol > 2.0:
            send_alert("⚠️ High volatility — skipping trade.")
            time.sleep(60 * 15)
            continue

        # Market regime filter (HMM)
        regime, dominant = detect_market_regime(df)
        print(f"📊 Market Regime: {regime}, Dominant: {dominant}")
        if regime != dominant:
            send_alert("📉 Non-trending regime — no trend trades.")
            time.sleep(60 * 15)
            continue

        # Signal generation (RSI2)
        df = apply_rsi2(df)
        signal = df.iloc[-1]["signal"]
        price = df.iloc[-1]["close"]
        rsi_val = df.iloc[-1]["rsi2"]

        if signal != 0:
            direction = "BUY" if signal == 1 else "SELL"
            print(f"🚨 Signal Detected: {direction} at {price:.2f}")
            send_alert(f"🚨 Signal: {direction} on XAUUSD @ {price:.2f}")

            if not simulation_mode:
                open_trade("XAUUSD", 0.1, signal)

            log_trade(signal, price, rsi_val, sl=150, tp=300)
        else:
            print("📈 No trade signal this cycle.")

        # Exit logic
        if not simulation_mode:
            manage_exits()

        time.sleep(60 * 15)

except KeyboardInterrupt:
    print("🛑 Manual exit received. Shutting down...")

except Exception as e:
    print(f"❌ Fatal Error: {e}")
    send_alert(f"❌ Bot Fatal Error: {e}")

finally:
    if not simulation_mode:
        shutdown()
