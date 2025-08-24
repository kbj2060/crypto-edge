import requests
import pandas as pd
from typing import List, Any, Literal
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

SPOT_BASE = "https://api.binance.com"
FUTURES_BASE = "https://fapi.binance.com"

def fetch_klines(
    symbol: str,
    interval: str = "5m",
    limit: int = 1500,
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
    
    # 재시도 로직이 포함된 세션 생성
    session = requests.Session()
    retry_strategy = Retry(
        total=3,  # 총 3번 재시도
        backoff_factor=1,  # 재시도 간격 증가
        status_forcelist=[429, 500, 502, 503, 504],  # 재시도할 HTTP 상태 코드
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # 타임아웃 증가 및 재시도
    max_retries = 3
    for attempt in range(max_retries):
        try:
            r = session.get(url, params=params, timeout=30)  # 타임아웃 30초로 증가
            r.raise_for_status()
            raw: List[List[Any]] = r.json()
            break
        except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
            if attempt == max_retries - 1:  # 마지막 시도였다면
                print(f"❌ Binance API 연결 실패 (최대 재시도 초과): {e}")
                raise
            else:
                wait_time = (attempt + 1) * 2  # 2초, 4초, 6초 대기
                print(f"⚠️ Binance API 연결 실패, {wait_time}초 후 재시도 ({attempt + 1}/{max_retries})")
                time.sleep(wait_time)

    cols = ["open_time","open","high","low","close","volume",
            "close_time","quote_asset_volume","number_of_trades",
            "taker_buy_base","taker_buy_quote","ignore"]
    df = pd.DataFrame(raw, columns=cols)
    num_cols = ["open","high","low","close","volume","quote_asset_volume","taker_buy_base","taker_buy_quote"]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    df["open_time"]  = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df.set_index("close_time", inplace=True)
    df.index.name = 'timestamp'  
    return df
