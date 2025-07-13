from arch import arch_model
import pandas as pd
import numpy as np

def forecast_garch_volatility(df, price_col="close", horizon=1):
    try:
        series = df[price_col].astype(float).pct_change().dropna() * 100
        if len(series) < 30:
            return np.nan

        model = arch_model(series, vol='Garch', p=1, q=1, rescale=True)
        res = model.fit(disp="off")

        forecast = res.forecast(horizon=horizon)
        vol_forecast = forecast.variance.values[-1][0] ** 0.5
        return vol_forecast  # %
    except Exception as e:
        print(f"⚠️ GARCH failed: {e}")
        return np.nan
