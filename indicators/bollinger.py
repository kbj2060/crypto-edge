import pandas as pd

def add_bollinger(df: pd.DataFrame, length: int = 20, mult: float = 2.0) -> pd.DataFrame:
    basis = df["close"].rolling(length).mean()
    dev = df["close"].rolling(length).std(ddof=0)
    df["BB_basis"] = basis
    df["BB_upper"] = basis + mult * dev
    df["BB_lower"] = basis - mult * dev
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["BB_basis"]
    return df
