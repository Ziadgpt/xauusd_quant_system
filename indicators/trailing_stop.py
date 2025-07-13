def calculate_trailing_stop(entry_price, current_price, direction, distance=100):
    if direction == 1:  # Buy
        new_sl = current_price - distance * 0.01
        return round(new_sl, 2) if new_sl > entry_price else None
    else:  # Sell
        new_sl = current_price + distance * 0.01
        return round(new_sl, 2) if new_sl < entry_price else None
