from arch import arch_model
import pandas as pd

def forecast_garch_volatility(df, price_col="close", horizon=1):
    # Ensure we're working with a clean numeric Series
    series = df[price_col].astype(float).pct_change().dropna() * 100

    model = arch_model(series, vol='Garch', p=1, q=1, rescale=True)
    res = model.fit(disp="off")

    forecast = res.forecast(horizon=horizon)
    vol_forecast = forecast.variance.values[-1][0] ** 0.5  # Std Dev
    return vol_forecast  # In %
