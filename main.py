import time
import os
import sys
from dotenv import load_dotenv

from strategies.momentum import apply_rsi2
from strategies.macd_bollinger import apply_macd_bollinger
from strategies.structure_breakout import detect_hh_ll_breakout
# from strategies.atr_breakout import apply_atr_breakout

from models.garch_model import forecast_garch_volatility
from models.hmm_model import detect_market_regime
from ml.predictor import predict_trade

from data.fetch_data import get_ohlcv
from execution.trade_manager import open_trade, manage_open_positions
from logs.logger import log_trade
from utils.mt5_connector import connect_mt5, shutdown_mt5
from utils.notifier import send_alert
from sim.signal_tracker import record_signal

# === Load Environment Variables ===
load_dotenv()

# === Symbol Setup ===
symbol = "XAUUSDc"

# === Connect to MT5 ===
mt5_enabled = True
if not connect_mt5():
    print("❌ MT5 initialization failed.")
    mt5_enabled = False
else:
    print("✅ Connected to MT5.")

try:
    while True:
        print("\n🔁 Starting new cycle...")

        # === Fetch OHLCV Data ===
        df = get_ohlcv(symbol=symbol)
        if df is None or len(df) < 30:
            print("⚠️ Insufficient data.")
            time.sleep(60)
            continue

        # === GARCH Volatility Filter ===
        vol = forecast_garch_volatility(df)
        print(f"📉 Forecasted Volatility: {vol:.2f}%")
        if vol > 2.0:
            send_alert("⚠️ High volatility — skipping trade.")
            time.sleep(900)
            continue

        # === HMM Market Regime Detection ===
        regime, dominant = detect_market_regime(df)
        print(f"📊 Market Regime: {regime}, Dominant: {dominant}")
        if regime != dominant:
            send_alert("📉 Non-trending regime — no trades.")
            time.sleep(900)
            continue

        # === Apply Strategies ===
        df = apply_rsi2(df)
        rsi2_signal = df.iloc[-1]["signal"]
        rsi_val = df.iloc[-1]["rsi2"]

        df = apply_macd_bollinger(df)
        macd_signal = df.iloc[-1]["signal_macd_bb"]

        df = detect_hh_ll_breakout(df)
        structure_signal = df.iloc[-1]["signal_structure"]

        # === (Optional) ATR breakout ===
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

        # === ML Prediction Filter ===
        if signal != 0:
            features = {
                "rsi2": df.iloc[-1]["rsi2"],
                "rsi14": df.iloc[-1].get("rsi14", 50),
                "macd": df.iloc[-1].get("macd", 0),
                "macd_signal": df.iloc[-1].get("macd_signal", 0),
                "obv": df.iloc[-1].get("obv", 0),
                "atr": df.iloc[-1].get("atr", 0),
                "bb_upper": df.iloc[-1].get("bb_upper", 0),
                "bb_lower": df.iloc[-1].get("bb_lower", 0),
                "volatility": vol,
                "regime": 1 if regime == "trending" else 0
            }

            ml_decision = predict_trade(features)
            if ml_decision == 0:
                print("🤖 ML rejected trade.")
                signal = 0

        # === Execute Trade ===
        if signal != 0:
            direction = "BUY" if signal == 1 else "SELL"
            price = df.iloc[-1]["close"]

            print(f"🚨 Signal: {direction} from {strategy_used} @ {price:.2f}")
            send_alert(f"🚨 {strategy_used} → {direction} on {symbol} @ {price:.2f}")

            if mt5_enabled:
                open_trade(
                    symbol=symbol,
                    direction=signal,
                    sl=150,
                    tp=300,
                    strategy=strategy_used,
                    risk_percent=1.0
                )

            log_trade(signal, price, rsi_val, sl=150, tp=300, symbol=symbol, strategy=strategy_used)


            record_signal(
                timestamp=df.iloc[-1]["time"],
                symbol=symbol,
                direction=signal,
                entry_price=price,
                sl=price - 1.5 if signal == 1 else price + 1.5,
                tp=price + 3.0 if signal == 1 else price - 3.0
            )
        else:
            print("ℹ️ No valid signal this cycle.")

        # === Manage Open Positions ===
        manage_open_positions(symbol)

        # === Wait Until Next Candle ===
        time.sleep(900)  # 15M

except KeyboardInterrupt:
    print("🛑 Stopped manually.")

except Exception as e:
    print(f"❌ Fatal error: {e}")
    send_alert(f"❌ Bot error: {e}")

finally:
    shutdown_mt5()
