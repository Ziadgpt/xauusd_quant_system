import time
import sys

# === Always-available modules ===
from strategies.momentum import apply_rsi2
from logs.logger  import log_trade
from utils.notifier import send_alert
from models.garch_model import forecast_garch_volatility
from models.hmm_model import detect_market_regime

# === Try to import MT5-related modules (only work on Windows VPS) ===
mt5_enabled = True
try:
    from data.fetch_data import get_ohlcv
    from execution.mt5_connector import initialize, shutdown
    from execution.trade_manager import open_trade
except ImportError:
    mt5_enabled = False
    print("⚠️ MT5 environment not detected. Running in simulation mode.")


# === Initialize MT5 if available ===
if mt5_enabled:
    if not initialize():
        print("❌ MT5 initialization failed.")
        sys.exit()
    print("✅ MT5 connected.")


# === Main Loop ===
while True:
    try:
        if not mt5_enabled:
            print("⏳ Running strategy logic (simulation mode)...")
            time.sleep(60 * 15)
            continue

        df = get_ohlcv()
        if df is None or len(df) < 30:
            print("⚠️ Data fetch failed or not enough bars.")
            time.sleep(60)
            continue

        # Step 1: Volatility Filter
        vol = forecast_garch_volatility(df)
        print(f"📉 Forecasted Volatility: {vol:.2f}%")
        if vol > 2.0:
            send_alert("⚠️ High volatility — skipping trade.")
            time.sleep(60 * 15)
            continue

        # Step 2: Regime Detection
        regime, dominant = detect_market_regime(df)
        print(f"📊 Market Regime: {regime}, Dominant: {dominant}")
        if regime != dominant:
            send_alert("📉 Non-trending regime — no trend trades.")
            time.sleep(60 * 15)
            continue

        # Step 3: Apply Strategy
        df = apply_rsi2(df)
        signal = df.iloc[-1]["signal"]
        price = df.iloc[-1]["close"]
        rsi_val = df.iloc[-1]["rsi2"]

        if signal != 0:
            direction = "BUY" if signal == 1 else "SELL"
            print(f"🚨 Signal Detected: {direction} at {price:.2f}")
            send_alert(f"🚨 Signal: {direction} on XAUUSD @ {price:.2f}")

            # Step 4: Execute & Log
            if mt5_enabled:
                open_trade("XAUUSD", 0.1, signal)
            log_trade(signal, price, rsi_val, sl=150, tp=300)

        else:
            print("📈 No trade signal this cycle.")

    except Exception as e:
        print(f"❌ Error: {e}")
        send_alert(f"❌ Bot Error: {e}")

    time.sleep(60 * 15)  # Wait 15 minutes before next cycle

# === Shutdown MT5 if enabled ===
if mt5_enabled:
    shutdown()
