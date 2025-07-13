import numpy as np
from hmmlearn.hmm import GaussianHMM

def detect_market_regime(df, price_col="close", n_states=2):
    df = df[[price_col]].copy()
    df["log_return"] = np.log(df[price_col] / df[price_col].shift(1))
    df.dropna(inplace=True)

    if len(df) < n_states * 10:
        return 0, 0

    X = df["log_return"].values.reshape(-1, 1)
    model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=1000)
    model.fit(X)
    hidden_states = model.predict(X)

    df["regime"] = hidden_states
    trend_regime = df["regime"].value_counts().idxmax()
    current_regime = df["regime"].iloc[-1]

    return current_regime, trend_regime
