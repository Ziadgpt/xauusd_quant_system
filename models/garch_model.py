from arch import arch_model
import pandas as pd
import numpy as np


def forecast_garch_volatility(df, price_col="close", horizon=1):
    try:
        # Ensure clean numeric returns
        series = df[price_col].astype(float).pct_change().dropna() * 100
        if len(series) < 30:
            raise ValueError("Not enough data for GARCH")

        model = arch_model(series, vol='Garch', p=1, q=1, rescale=True)
        res = model.fit(disp="off")

        forecast = res.forecast(horizon=horizon)
        vol_forecast = forecast.variance.values[-1][0] ** 0.5  # Std Dev
        return vol_forecast  # In %

    except Exception as e:
        print(f"⚠️ GARCH model failed: {e}")
        return np.nan  # or return a fixed fallback like 1.5
