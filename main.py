import time
import sys
from dotenv import load_dotenv

from strategies.momentum import apply_rsi2
from strategies.macd_bollinger import apply_macd_bollinger
from strategies.structure_breakout import detect_hh_ll_breakout
# from strategies.atr_breakout import apply_atr_breakout  # Optional

from models.garch_model import forecast_garch_volatility
from models.hmm_model import detect_market_regime

from utils.mt5_connector import connect_mt5, shutdown_mt5
from data.fetch_data import get_ohlcv
from execution.trade_manager import open_trade, manage_open_positions
from logs.logger import log_trade
from utils.notifier import send_alert
from sim.signal_tracker import record_signal

# === Load Env ===
load_dotenv()

# === Symbol for Cent Account ===
symbol = "XAUUSDc"

# === MT5 Connect ===
if not connect_mt5():
    print("❌ MT5 initialization failed.")
    sys.exit()
else:
    print("✅ MT5 connected.")

# === Trading Loop ===
try:
    while True:
        print("\n🔁 Starting new 15M cycle...")

        # === Get Market Data ===
        df = get_ohlcv(symbol=symbol)
        if df is None or len(df) < 30:
            print("⚠️ Insufficient OHLCV data.")
            time.sleep(60)
            continue

        # === Volatility Filter (GARCH) ===
        vol = forecast_garch_volatility(df)
        print(f"📉 Forecasted Volatility: {vol:.2f}%")
        if vol > 2.0:
            send_alert("⚠️ High volatility — skipping.")
            time.sleep(60 * 15)
            continue

        # === Regime Filter (HMM) ===
        regime, dominant = detect_market_regime(df)
        print(f"📊 Market Regime: {regime} | Dominant: {dominant}")
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

        # === Strategy 3: Structure Breakout ===
        df = detect_hh_ll_breakout(df)
        structure_signal = df.iloc[-1]["signal_structure"]

        # === (Optional) Strategy 4: ATR breakout ===
        # df = apply_atr_breakout(df)
        # atr_signal = df.iloc[-1]["signal_atr"]

        # === Ensemble Signal Logic ===
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

            print(f"🚨 Signal: {direction} from {strategy_used} @ {price:.2f}")
            send_alert(f"🚨 {strategy_used} → {direction} on {symbol} @ {price:.2f}")

            open_trade(
                symbol=symbol,
                direction=signal,
                sl=150,
                tp=300,
                strategy=strategy_used,
                risk_percent=1.0
            )

            log_trade(
                signal=direction,
                price=price,
                rsi_value=rsi_val,
                sl=150,
                tp=300
            )

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

        # === Manage Open Positions (Trailing Stop + Exit Logic) ===
        manage_open_positions(symbol=symbol)

        # === Wait Until Next Candle ===
        time.sleep(60 * 15)

except KeyboardInterrupt:
    print("🛑 Bot manually stopped.")
except Exception as e:
    print(f"❌ Fatal Error: {e}")
    send_alert(f"❌ Bot Error: {e}")
finally:
    shutdown_mt5()
