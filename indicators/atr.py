import pandas as pd
import numpy as np

def add_atr(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder's smoothing (EMA with alpha=1/length)
    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    df[f"ATR_{length}"] = atr
    return df

def calculate_atr(df: pd.DataFrame, length: int = 14) -> float:
    """ATR 계산 함수"""
    if df.empty or len(df) < length:
        return 0.0
    
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder's smoothing (EMA with alpha=1/length)
    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    
    # 최신 ATR 값 반환
    return atr.iloc[-1] if not atr.empty else 0.0
