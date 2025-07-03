import numpy as np
from hmmlearn.hmm import GaussianHMM

def detect_market_regime(df, price_col="close", n_states=2):
    log_returns = np.log(df[price_col] / df[price_col].shift(1)).dropna().values.reshape(-1, 1)
    model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=1000).fit(log_returns)
    hidden_states = model.predict(log_returns)

    df = df.iloc[-len(hidden_states):].copy()
    df["regime"] = hidden_states

    counts = df["regime"].value_counts()
    trend_regime = counts.idxmax()  # Most frequent regime

    current_regime = df["regime"].iloc[-1]
    return current_regime, trend_regime  # e.g., (0, 1)
