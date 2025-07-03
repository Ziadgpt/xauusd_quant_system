from arch import arch_model
import pandas as pd

def forecast_garch_volatility(df, price_col="close", horizon=1):
    returns = 100 * df[price_col].pct_change().dropna()
    model = arch_model(returns, vol='Garch', p=1, q=1, rescale=True)
    res = model.fit(disp="off")

    forecast = res.forecast(horizon=horizon)
    vol_forecast = forecast.variance.values[-1][0] ** 0.5  # Std Dev
    return vol_forecast  # In %
