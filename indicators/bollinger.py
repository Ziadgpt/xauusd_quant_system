import pandas as pd

def calculate_bollinger_bands(series, period=21, std_dev=2):
    """
    Calculate Bollinger Bands for a price series.

    Parameters:
        series (pd.Series): Series of prices (e.g., close).
        period (int): Moving average period.
        std_dev (float): Number of standard deviations.

    Returns:
        tuple: upper_band, middle_band (SMA), lower_band
    """
    middle_band = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()

    upper_band = middle_band + std_dev * std
    lower_band = middle_band - std_dev * std

    return upper_band, middle_band, lower_band
