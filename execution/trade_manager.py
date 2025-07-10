def manage_open_positions(symbol="XAUUSDc"):
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        print("📭 No open positions.")
        return

    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        print("❌ Failed to get current price tick.")
        return

    last_price = tick.bid

    # Get latest candles for indicators
    bars = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 100)
    if bars is None or len(bars) == 0:
        print("❌ Failed to fetch OHLCV for indicator analysis.")
        return

    df = pd.DataFrame(bars)
    df["rsi"] = calculate_rsi(df["close"], period=14)
    macd_line, macd_signal, _ = calculate_macd(df["close"])
    df["macd"] = macd_line
    df["macd_signal"] = macd_signal

    rsi_now = df["rsi"].iloc[-1]
    macd_now = df["macd"].iloc[-1]
    signal_now = df["macd_signal"].iloc[-1]

    for pos in positions:
        ticket = pos.ticket
        direction = 1 if pos.type == mt5.ORDER_TYPE_BUY else -1
        entry = pos.price_open
        tp = pos.tp
        current_sl = pos.sl or 0
        current_magic = pos.magic

        # === Trailing Stop Logic ===
        new_sl = calculate_trailing_stop(entry, last_price, direction, distance=100)

        if new_sl:
            should_update_sl = (
                (direction == 1 and (current_sl == 0 or new_sl > current_sl)) or
                (direction == -1 and (current_sl == 0 or new_sl < current_sl))
            )

            if should_update_sl:
                sl_request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": ticket,
                    "sl": round(new_sl, 2),
                    "tp": tp,
                }

                sl_result = mt5.order_send(sl_request)
                if sl_result.retcode == mt5.TRADE_RETCODE_DONE:
                    msg = f"🔒 Trailing SL updated for ticket {ticket} → {new_sl:.2f}"
                    print(msg)
                    send_alert(msg)

        # === Exit Logic (MACD Divergence + RSI Overbought/Oversold) ===
        should_exit = (
            (direction == 1 and macd_now < signal_now and rsi_now > 70) or
            (direction == -1 and macd_now > signal_now and rsi_now < 30)
        )

        if should_exit:
            close_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": round(pos.volume, 2),
                "type": mt5.ORDER_TYPE_SELL if direction == 1 else mt5.ORDER_TYPE_BUY,
                "position": ticket,
                "price": last
