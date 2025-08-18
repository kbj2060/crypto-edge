import requests
import pandas as pd
from typing import List, Any, Literal

SPOT_BASE = "https://api.binance.com"
FUTURES_BASE = "https://fapi.binance.com"

def fetch_klines(
    symbol: str,
    interval: str = "5m",
    limit: int = 500,
    market: Literal["spot","futures"] = "futures",
    price_source: Literal["last","mark"] = "last",  # futures일 때만 의미 있음
) -> pd.DataFrame:
    if market == "spot":
        base = SPOT_BASE
        path = "/api/v3/klines"
    else:
        base = FUTURES_BASE
        if price_source == "mark":
            path = "/fapi/v1/markPriceKlines"
        else:
            path = "/fapi/v1/klines"  # last trade price

    url = f"{base}{path}"
    params = dict(symbol=symbol, interval=interval, limit=min(1500, max(2, limit)))
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    raw: List[List[Any]] = r.json()

    cols = ["open_time","open","high","low","close","volume",
            "close_time","quote_asset_volume","number_of_trades",
            "taker_buy_base","taker_buy_quote","ignore"]
    df = pd.DataFrame(raw, columns=cols)
    num_cols = ["open","high","low","close","volume","quote_asset_volume","taker_buy_base","taker_buy_quote"]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    df["open_time"]  = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df.set_index("close_time", inplace=True)
    return df
