# htf_rsi_divergence.py
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

from data.binance_dataloader import BinanceDataLoader
from utils.time_manager import get_time_manager

def _clamp(x, a=0.0, b=1.0):
    try:
        return max(a, min(b, float(x)))
    except Exception:
        return a

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    down = -delta.clip(upper=0).ewm(alpha=1/period, adjust=False).mean()
    rs = up / (down + 1e-9)
    return 100 - (100 / (1 + rs))

def _stoch_rsi(rsi: pd.Series, period: int = 14) -> pd.Series:
    min_rsi = rsi.rolling(window=period).min()
    max_rsi = rsi.rolling(window=period).max()
    return (rsi - min_rsi) / (max_rsi - min_rsi + 1e-9)

@dataclass
class HTFRSIDivCfg:
    symbol: str = "ETHUSDT"
    interval: str = "4h"       # "4h" 또는 "1d"
    rsi_period: int = 14
    stoch_period: int = 14
    lookback_bars: int = 200
    divergence_lookback: int = 8   # 최근 N봉에서 다이버전스 확인
    min_score: float = 0.5

class RSIDivergence:
    """
    Higher Timeframe RSI / StochRSI Divergence Detector.
    - Bullish divergence: 가격 저점 하락 but RSI 저점 상승 → BUY
    - Bearish divergence: 가격 고점 상승 but RSI 고점 하락 → SELL
    """
    def __init__(self, cfg: HTFRSIDivCfg = HTFRSIDivCfg()):
        self.cfg = cfg
        self.loader = BinanceDataLoader()
        self.tm = get_time_manager()

    def on_kline_close_htf(self) -> Optional[Dict[str, Any]]:
        end_time = self.tm.get_current_time()
        start_time = end_time - timedelta(days=90)  # 충분한 기간 확보
        df = self.loader.fetch_data(
            interval=self.cfg.interval,
            symbol=self.cfg.symbol,
            start_time=start_time,
            end_time=end_time
        )
        
        if df is None or len(df) < self.cfg.rsi_period + self.cfg.divergence_lookback + 2:
            return None

        close = pd.to_numeric(df["close"].astype(float))
        rsi_series = _rsi(close, self.cfg.rsi_period)
        stoch_series = _stoch_rsi(rsi_series, self.cfg.stoch_period)

        # 최근 N봉에서 고점/저점 찾기
        recent_close = close.iloc[-self.cfg.divergence_lookback:]
        recent_rsi = rsi_series.iloc[-self.cfg.divergence_lookback:]

        action, score, conf = None, 0.0, "LOW"

        # Bullish divergence: 가격 저점 하락, RSI 저점 상승
        if recent_close.iloc[-1] < recent_close.min() and recent_rsi.iloc[-1] > recent_rsi.min():
            action = "BUY"
            score = 1.0
        # Bearish divergence: 가격 고점 상승, RSI 고점 하락
        elif recent_close.iloc[-1] > recent_close.max() and recent_rsi.iloc[-1] < recent_rsi.max():
            action = "SELL"
            score = 1.0

        if score >= 0.8:
            conf = "HIGH"
        elif score >= 0.5:
            conf = "MEDIUM"
        else:
            conf = "LOW"

        if action is None:
            return None

        return {
            "name": f"RSI_DIV",
            "action": action,
            "score": float(score),
            "confidence": conf,
            "timestamp": self.tm.get_current_time(),
            "context": {
                "last_close": float(recent_close.iloc[-1]),
                "last_rsi": float(recent_rsi.iloc[-1]),
                "last_stoch_rsi": float(stoch_series.iloc[-1]),
            },
        }
