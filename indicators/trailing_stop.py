def calculate_trailing_stop(entry_price, current_price, direction, distance=100):
    """
    Calculates new SL based on current price and trailing distance in points (e.g., 100 = $1.00).
    """
    if direction == 1:  # Buy
        new_sl = current_price - distance * 0.01
        return round(new_sl, 2) if new_sl > entry_price else None
    else:  # Sell
        new_sl = current_price + distance * 0.01
        return round(new_sl, 2) if new_sl < entry_price else None
