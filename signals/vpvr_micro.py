# vpvr_micro.py
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

@dataclass
class VPVRConfig:
    n_bins: int = 64
    lookback_bars: int = 240
    poc_tolerance: float = 0.002
    min_profile_volume: float = 1.0
    lookback_retest_bars: int = 3
    side_bias: Optional[str] = None  # 'LONG' or 'SHORT' or None

class VPVRMicro:
    """Volume Profile Volume Range Micro Bot - 클래스 기반"""
    
    def __init__(self, config: VPVRConfig = None):
        self.config = config or VPVRConfig()
    
    @staticmethod
    def ensure_index(df: pd.DataFrame) -> pd.DataFrame:
        """DataFrame 인덱스를 DatetimeIndex로 변환"""
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        return df.sort_index()
    
    @staticmethod
    def compute_vpvr(df: pd.DataFrame, n_bins: int = 64):
        """
        Volume Profile Volume Range 계산
        
        Returns:
            bins: 가격 구간 배열
            vol_hist: 각 구간별 거래량 히스토그램
            poc_price: Point of Control 가격
        """
        prices = df['close'].astype(float).values
        vols = df['volume'].astype(float).values
        if len(prices) == 0:
            return None, None, None
        p_min, p_max = float(np.min(prices)), float(np.max(prices))
        if p_max == p_min:
            return None, None, None
        bins = np.linspace(p_min, p_max, n_bins + 1)
        vol_hist = np.zeros(n_bins)
        bin_idx = np.digitize(prices, bins) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        for i, b in enumerate(bin_idx):
            vol_hist[b] += vols[i]
        poc_idx = int(np.argmax(vol_hist))
        poc_price = float((bins[poc_idx] + bins[poc_idx + 1]) / 2.0)
        return bins, vol_hist, poc_price
    
    def _conf_bucket(self, v: float) -> str:
        if v >= 0.75: return "HIGH"
        if v >= 0.50: return "MEDIUM"
        return "LOW"

    def on_kline_close_3m(self, df_3m: pd.DataFrame) -> Dict[str, Any]:
        """
        VPVR 기반 신호 생성
        
        Returns:
            {'name','action'('BUY'/'SELL'/'HOLD'),'score'(0..1),'confidence'(0..1),'entry','stop','context'}
        """
        df_3m = self.ensure_index(df_3m)
        if len(df_3m) < max(self.config.lookback_bars, 10):
            return {
                'name': 'VPVR_MICRO', 
                'action': 'HOLD', 
                'score': 0.0, 
                'confidence': 0.0, 
                'context': {'reason': 'insufficient_bars'}
            }

        profile_df = df_3m.iloc[-self.config.lookback_bars:]
        bins, vol_hist, poc = self.compute_vpvr(profile_df, n_bins=self.config.n_bins)
        if bins is None:
            return {
                'name': 'VPVR_MICRO', 
                'action': 'HOLD', 
                'score': 0.0, 
                'confidence': 0.0, 
                'context': {'reason': 'vpvr_fail'}
            }

        recent_close = float(df_3m['close'].iloc[-1])
        tol_price = poc * self.config.poc_tolerance
        within = abs(recent_close - poc) <= tol_price

        recent_prices = df_3m['close'].iloc[-self.config.lookback_retest_bars - 1:-1].astype(float).values
        retest = any(abs(p - poc) <= tol_price for p in recent_prices) if len(recent_prices) > 0 else False

        action = 'HOLD'; score = 0.0; conf = 0.0; entry = None; stop = None
        if within and retest and float(np.sum(vol_hist)) >= self.config.min_profile_volume:
            last_close = recent_close
            if self.config.side_bias == 'LONG':
                action, score, conf = 'BUY', 0.8, 0.7
            elif self.config.side_bias == 'SHORT':
                action, score, conf = 'SELL', 0.8, 0.7
            else:
                action = 'BUY' if last_close >= poc else 'SELL'
                score, conf = 0.75, 0.6
            entry = last_close
            bins_arr = np.array(bins)
            bin_w = bins_arr[1] - bins_arr[0]
            stop = poc - 1.5 * bin_w if action == 'BUY' else poc + 1.5 * bin_w

        return {
            'name': 'VPVR_MICRO',
            'action': action,
            'score': float(score),
            'confidence': self._conf_bucket(float(conf)),
            'entry': float(entry) if entry is not None else None,
            'stop': float(stop) if stop is not None else None,
            'context': {
                'poc': float(poc), 
                'within_tol': bool(within), 
                'retest': bool(retest), 
                'vol_sum': float(np.sum(vol_hist))
            }
        }
