import pandas as pd
import numpy as np

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.clip(lower=0)).ewm(alpha=1/length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/length, adjust=False).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.fillna(50)

def add_stoch_rsi(df: pd.DataFrame, rsi_len=14, stoch_len=14, k=3, d=3) -> pd.DataFrame:
    r = rsi(df["close"], rsi_len)
    # RSI를 0~100으로 보고 다시 스토캐스틱 처리
    lowest = r.rolling(stoch_len).min()
    highest = r.rolling(stoch_len).max()
    stoch = (r - lowest) / (highest - lowest + 1e-9)  # 0~1
    k_line = stoch.rolling(k).mean() * 100.0
    d_line = k_line.rolling(d).mean()
    df["StochRSI_K"] = k_line
    df["StochRSI_D"] = d_line
    return df
