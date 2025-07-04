# utils/risk.py

def calculate_lot_size(balance, sl_pips, risk_percent=1.0, pip_value=1.0):
    """
    Calculates lot size based on risk % and SL.

    Args:
        balance: Account balance in USD
        sl_pips: Stop loss in pips (e.g. 150 for 1.50 USD on gold)
        risk_percent: % of account to risk (default: 1.0)
        pip_value: Value per pip for 0.1 lot (XAUUSD ≈ $1.0)

    Returns:
        Lot size (float), rounded to 2 decimals
    """
    risk_amount = balance * (risk_percent / 100)
    lot = risk_amount / (sl_pips * pip_value)
    return round(lot, 2)
