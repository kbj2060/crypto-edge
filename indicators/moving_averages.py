import pandas as pd

def ma(series: pd.Series, length: int, mode: str = "ema") -> pd.Series:
    mode = mode.lower()
    if mode == "sma":
        return series.rolling(length).mean()
    return series.ewm(span=length, adjust=False).mean()

def calculate_ema(series: pd.Series, length: int) -> pd.Series:
    """EMA 계산 함수"""
    return series.ewm(span=length, adjust=False).mean()

def add_mas(df: pd.DataFrame, cfg_ma=("ema",50), cfg_ma2=("ema",200)) -> pd.DataFrame:
    mode1, L1 = cfg_ma
    mode2, L2 = cfg_ma2
    df[f"{mode1.upper()}_{L1}"] = ma(df["close"], L1, mode1)
    df[f"{mode2.upper()}_{L2}"] = ma(df["close"], L2, mode2)
    # 호환성을 위해 공통 이름도 채워줌(전략 코드에서 참조)
    df["EMA_20"]  = ma(df["close"], 20, "ema")  # 스캘핑 전략용 EMA_20 추가
    df["EMA_50"]  = df.get("EMA_50",  df[f"{mode1.upper()}_{L1}"] if L1==50 else ma(df["close"],50,"ema"))
    df["EMA_200"] = df.get("EMA_200", df[f"{mode2.upper()}_{L2}"] if L2==200 else ma(df["close"],200,"ema"))
    df["SMA_200"] = ma(df["close"], 200, "sma")
    return df
