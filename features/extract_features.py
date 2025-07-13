import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, ROCIndicator, StochasticOscillator, WilliamsRIndicator
from ta.trend import MACD, CCIIndicator, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator, AccDistIndexIndicator, VolumeWeightedAveragePrice

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ===== Price Action =====
    df['return_pct'] = df['close'].pct_change() * 100
    df['body'] = df['close'] - df['open']
    df['range'] = df['high'] - df['low'] + 1e-6
    df['body_ratio'] = abs(df['body']) / df['range']
    df['wick_ratio'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['range']
    df['engulfing'] = ((df['body'].shift(1) < 0) & (df['body'] > abs(df['body'].shift(1)))).astype(int)
    df['trend_slope'] = df['close'].diff().rolling(window=5).mean()

    # ===== Momentum =====
    df['rsi_2'] = RSIIndicator(close=df['close'], window=2).rsi()
    df['rsi_14'] = RSIIndicator(close=df['close'], window=14).rsi()
    df['roc'] = ROCIndicator(close=df['close'], window=5).roc()
    df['cci'] = CCIIndicator(high=df['high'], low=df['low'], close=df['close'], window=14).cci()
    df['stoch'] = StochasticOscillator(high=df['high'], low=df['low'], close=df['close']).stoch()
    df['williams_r'] = WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close']).williams_r()

    # ===== MACD =====
    macd = MACD(close=df['close'])
    df['macd_line'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()

    # ===== Bollinger Bands =====
    bb = BollingerBands(close=df['close'], window=21, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_distance'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-6)
    df['bb_bandwidth'] = df['bb_upper'] - df['bb_lower']
    df['rolling_std_5'] = df['close'].rolling(5).std()

    # ===== Trend Indicators =====
    df['sma_20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
    df['ema_20'] = EMAIndicator(close=df['close'], window=20).ema_indicator()

    # ===== VWAP =====
    if 'volume' in df.columns:
        vwap = VolumeWeightedAveragePrice(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            volume=df['volume']
        )
        df['vwap'] = vwap.volume_weighted_average_price()
    else:
        df['vwap'] = 0

    # ===== Volume =====
    if 'volume' in df.columns:
        df['obv'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
        df['accum_dist'] = AccDistIndexIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume']).acc_dist_index()
        df['volume_delta'] = df['volume'].diff()
    else:
        df['obv'] = 0
        df['accum_dist'] = 0
        df['volume_delta'] = 0

    # ===== Market Regime Features =====
    df['regime'] = df.get('regime', 1)
    df['garch_vol'] = df.get('garch_vol', 0.0)

    # ===== Time Features =====
    df['hour'] = df['timestamp'].dt.hour
    df['weekday'] = df['timestamp'].dt.weekday

    # ===== Final Clean =====
    df.dropna(inplace=True)

    return df
