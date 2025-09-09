# htf_trend_bot.py
from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime
import pandas as pd
import numpy as np

@dataclass
class HTFConfig:
    ema_period: int = 50
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    lookback_15m: int = 20
    atr_period: int = 14
    min_data_length: int = 300

class HTFTrend:
    """Higher Timeframe Trend Bot - 클래스 기반"""
    
    def __init__(self, config: HTFConfig = HTFConfig()):
        self.config = config
    
    @staticmethod
    def ensure_index(df: pd.DataFrame) -> pd.DataFrame:
        """DataFrame 인덱스를 DatetimeIndex로 변환"""
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        return df.sort_index()
    
    @staticmethod
    def ema(series: pd.Series, span: int) -> pd.Series:
        """지수이동평균 계산"""
        return series.ewm(span=span, adjust=False).mean()
    
    @staticmethod
    def macd(series: pd.Series, fast=12, slow=26, signal=9):
        """MACD 지표 계산"""
        f = HTFTrend.ema(series, fast)
        s = HTFTrend.ema(series, slow)
        macd_line = f - s
        sig = HTFTrend.ema(macd_line, signal)
        hist = macd_line - sig
        return macd_line, sig, hist
    
    @staticmethod
    def atr(df: pd.DataFrame, period=14) -> pd.Series:
        """Average True Range 계산"""
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        close = df['close'].astype(float)
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period, min_periods=1).mean()

    def on_kline_close_15m(
        self,
        df_15m: pd.DataFrame,
        df_1h: Optional[pd.DataFrame] = None,
        df_4h: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        트렌드 신호 생성
        
        Returns:
            {'name','action'('LONG'/'SHORT'/'HOLD'),'score'(0..1),'entry','stop','context'}
        """
        df_15m = self.ensure_index(df_15m)
        if len(df_15m) < max(30, self.config.lookback_15m + 2):
            return {
                'name': 'HTF_TREND_15M', 
                'action': 'HOLD', 
                'score': 0.0,
                'context': {'reason': 'insufficient_15m', 'n': len(df_15m)}
            }

        # prefer 1H then 4H as HTF; if absent approximate using aggregated 15m
        htf_df = df_1h if df_1h is not None else df_4h
        if htf_df is not None and len(htf_df) >= self.config.ema_period + 5:
            htf = self.ensure_index(htf_df)
            close_htf = htf['close'].astype(float)
            ema_htf = self.ema(close_htf, self.config.ema_period)
            slope = ema_htf.iloc[-1] - ema_htf.iloc[-3] if len(ema_htf) > 3 else 0.0
            ema_trend = 'UP' if slope > 0 else 'DOWN' if slope < 0 else 'FLAT'
            _, _, macd_hist_series = self.macd(close_htf, self.config.macd_fast, self.config.macd_slow, self.config.macd_signal)
            macd_hist = float(macd_hist_series.iloc[-1])
        else:
            close_15 = df_15m['close'].astype(float)
            ema_approx = self.ema(close_15, self.config.ema_period)
            slope = ema_approx.iloc[-1] - ema_approx.iloc[-3] if len(ema_approx) > 3 else 0.0
            ema_trend = 'UP' if slope > 0 else 'DOWN' if slope < 0 else 'FLAT'
            _, _, macd_hist_series = self.macd(close_15, self.config.macd_fast, self.config.macd_slow, self.config.macd_signal)
            macd_hist = float(macd_hist_series.iloc[-1])

        # Execution / entry logic on 15m
        recent = df_15m.iloc[-(self.config.lookback_15m + 1):].copy()
        last_close = float(recent['close'].iloc[-1])
        ema15 = self.ema(recent['close'].astype(float), 20)
        last_ema15 = float(ema15.iloc[-1])
        atr_val = float(self.atr(df_15m, self.config.atr_period).iloc[-1])

        action = 'HOLD'; score = 0.0; conf = 0.0; entry = None; stop = None
        if ema_trend == 'UP' and last_close >= last_ema15:
            action = 'BUY'
            score = min(1.0, max(0.0, (macd_hist / (abs(macd_hist) + 1e-9)) * 0.8 + 0.2))
            conf = 0.8 if macd_hist > 0 else 0.6
            entry = last_close
            stop = entry - 1.5 * atr_val
        elif ema_trend == 'DOWN' and last_close <= last_ema15:
            action = 'SELL'
            score = min(1.0, max(0.0, (-macd_hist / (abs(macd_hist) + 1e-9)) * 0.8 + 0.2))
            conf = 0.8 if macd_hist < 0 else 0.6
            entry = last_close
            stop = entry + 1.5 * atr_val

        return {
            'name': 'HTF_TREND_15M',
            'action': action,
            'score': float(score),
            'entry': float(entry) if entry is not None else None,
            'stop': float(stop) if stop is not None else None,
            'context': {'ema_trend': ema_trend, 'macd_hist': macd_hist, 'atr': atr_val}
        }
