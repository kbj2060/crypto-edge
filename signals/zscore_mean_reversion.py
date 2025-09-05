# zscore_mean_reversion_bot.py
from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

from indicators.global_indicators import get_vwap

@dataclass
class ZScoreConfig:
    window: int = 60
    use_log: bool = False
    z_thresh: float = 2.0
    exit_z: float = 0.5
    atr_period: int = 14
    stop_atr_mult: float = 2.0
    take_profit_atr_mult: Optional[float] = 2.5
    min_volume: float = 0.0
    min_history: int = 200
    mode: str = "price"         # or "vwap_residual"
    vwap_window: int = 390

class ZScoreMeanReversion:
    """Z-Score Mean Reversion Bot - 클래스 기반"""
    
    def __init__(self, config: ZScoreConfig = None):
        self.config = config or ZScoreConfig()
    
    @staticmethod
    def ensure_index(df: pd.DataFrame) -> pd.DataFrame:
        """DataFrame 인덱스를 DatetimeIndex로 변환"""
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        return df.sort_index()
    
    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range 계산"""
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        close = df['close'].astype(float)
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period, min_periods=1).mean()
    
    @staticmethod
    def compute_zscore(series: pd.Series, window: int) -> pd.Series:
        """Z-Score 계산"""
        mean = series.rolling(window, min_periods=1).mean().shift(1)
        std = series.rolling(window, min_periods=1).std(ddof=0).shift(1).replace(0, np.nan)
        z = (series - mean) / std
        return z.fillna(0.0)

    def _conf_bucket(self, v: float) -> str:
        if v >= 0.75: return "HIGH"
        if v >= 0.50: return "MEDIUM"
        return "LOW"

    def on_kline_close_3m(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Z-Score Mean Reversion 신호 생성
        
        Returns:
            {'name','action'('BUY'/'SELL'/'HOLD'),'score'(0..1),'confidence'(0..1),'entry','stop','tp','context'}
        """
        df = self.ensure_index(df)
        if len(df) < max(self.config.min_history, self.config.window + 5):
            print('insufficient_history')
            return {
                'name': 'ZSCORE_MEAN_REVERSION', 
                'action': 'HOLD', 
                'score': 0.0, 
                'confidence': 0.0, 
                'context': {'reason': 'insufficient_history', 'n': len(df)}
            }

        last_vol = float(df['quote_volume'].iloc[-1]) if 'quote_volume' in df.columns else 0.0
        if self.config.min_volume and last_vol < self.config.min_volume:
            print('low_volume')
            return {
                'name': 'ZSCORE_MEAN_REVERSION', 
                'action': 'HOLD', 
                'score': 0.0, 
                'confidence': 0.0, 
                'context': {'reason': 'low_volume', 'last_vol': last_vol}
            }

        if self.config.mode == "vwap_residual":
            full_vwap, _ = get_vwap()
            series = (df['close'].astype(float) - full_vwap).astype(float)
        else:
            series = df['close'].astype(float)
            if self.config.use_log:
                series = np.log(series.replace(0, np.nan)).fillna(method='ffill')

        z = self.compute_zscore(series, self.config.window)
        last_z = float(z.iloc[-1])

        atr_val = float(self.atr(df, self.config.atr_period).iloc[-1]) if len(df) > 0 else 0.0
        last_price = float(df['close'].iloc[-1])

        action = 'HOLD'; score = 0.0; conf = 0.0; entry = last_price; stop = None; tp = None
        if last_z >= self.config.z_thresh:
            action = 'SELL'
            score = min(1.0, abs(last_z) / (self.config.z_thresh * 2.0))
            conf = min(1.0, (abs(last_z) - self.config.z_thresh + 1.0) / (abs(last_z) + 1.0))
            entry = last_price
            stop = entry + max(1e-6, self.config.stop_atr_mult * atr_val)
            if self.config.take_profit_atr_mult is not None:
                tp = entry - self.config.take_profit_atr_mult * atr_val
        elif last_z <= -self.config.z_thresh:
            action = 'BUY'
            score = min(1.0, abs(last_z) / (self.config.z_thresh * 2.0))
            conf = min(1.0, (abs(last_z) - self.config.z_thresh + 1.0) / (abs(last_z) + 1.0))
            entry = last_price
            stop = entry - max(1e-6, self.config.stop_atr_mult * atr_val)
            if self.config.take_profit_atr_mult is not None:
                tp = entry + self.config.take_profit_atr_mult * atr_val

        return {
            'name': 'ZSCORE_MEAN_REVERSION',
            'action': action,
            'score': float(score),
            'confidence': self._conf_bucket(float(conf)),
            'entry': float(entry) if entry is not None else None,
            'stop': float(stop) if stop is not None else None,
            'tp': float(tp) if tp is not None else None,
            'context': {
                'last_z': last_z, 
                'z_threshold': self.config.z_thresh, 
                'atr': atr_val, 
                'mode': self.config.mode
            }
        }
