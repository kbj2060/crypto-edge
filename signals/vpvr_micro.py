# vpvr_micro.py
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

@dataclass
class VPVRConfig:
    # 단기 전략 최적화 (3분봉 기준)
    n_bins: int = 32                    # 64 → 32 (더 세밀한 분석)
    lookback_bars: int = 60             # 180 → 60 (3시간, 단기용)
    poc_tolerance: float = 0.005        # 0.002 → 0.005 (더 넓은 범위)
    min_profile_volume: float = 3       # 5 → 3 (더 낮은 임계값)
    lookback_retest_bars: int = 2       # 3 → 2 (더 빠른 확인)
    side_bias: Optional[str] = None     # 'LONG' or 'SHORT' or None
    debug: bool = False                  # 디버깅 활성화

def _calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """3분봉 데이터에 맞는 ATR 계산"""
    try:
        if len(df) < period + 1:
            return 0.0
        
        high = pd.to_numeric(df['high'].astype(float))
        low = pd.to_numeric(df['low'].astype(float))
        close = pd.to_numeric(df['close'].astype(float))
        
        # True Range 계산
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR 계산 (지수이동평균 사용)
        atr = true_range.ewm(alpha=1/period, adjust=False).mean()
        
        return float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0
        
    except Exception as e:
        print(f"[VPVR_MICRO] ATR 계산 오류: {e}")
        return 0.0

class VPVRMicro:
    """Volume Profile Volume Range Micro Bot - 클래스 기반"""
    
    def __init__(self, config: VPVRConfig = VPVRConfig()):
        self.config = config or VPVRConfig()
        self.last_signal_time = None
    
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
        vols = df['quote_volume'].astype(float).values
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

    def on_kline_close_3m(self, df_3m: pd.DataFrame) -> Dict[str, Any]:
        """
        VPVR 기반 신호 생성 (단기 전략 최적화)
        
        Returns:
            {'name','action'('BUY'/'SELL'/'HOLD'),'score'(0..1),'entry','stop','context'}
        """
        df_3m = self.ensure_index(df_3m)
        if len(df_3m) < max(self.config.lookback_bars, 10):
            if self.config.debug:
                print(f"[VPVR_MICRO] 데이터 부족: {len(df_3m)} < {max(self.config.lookback_bars, 10)}")
            return {
                'name': 'VPVR_MICRO', 
                'action': 'HOLD', 
                'score': 0.0, 
                'context': {'reason': 'insufficient_bars'}
            }

        profile_df = df_3m.iloc[-self.config.lookback_bars:]
        bins, vol_hist, poc = self.compute_vpvr(profile_df, n_bins=self.config.n_bins)
        if bins is None:
            if self.config.debug:
                print("[VPVR_MICRO] VPVR 계산 실패")
            return {
                'name': 'VPVR_MICRO', 
                'action': 'HOLD', 
                'score': 0.0,
                'context': {'reason': 'vpvr_fail'}
            }

        recent_close = float(df_3m['close'].iloc[-1])
        tol_price = poc * self.config.poc_tolerance
        within = abs(recent_close - poc) <= tol_price

        # 리테스트 확인 (더 유연하게)
        recent_prices = df_3m['close'].iloc[-self.config.lookback_retest_bars - 1:-1].astype(float).values
        retest = any(abs(p - poc) <= tol_price for p in recent_prices) if len(recent_prices) > 0 else False

        # ATR 계산 (스탑로스용)
        atr = _calculate_atr(df_3m, period=14)
        
        action = 'HOLD'
        score = 0.0
        conf = 0.0
        entry = None
        stop = None
        
        # 신호 생성 조건 (완화됨)
        vol_sum = float(np.sum(vol_hist))
        if vol_sum >= self.config.min_profile_volume:
            # POC 근접 또는 리테스트 중 하나만 만족해도 신호 생성
            if within or retest:
                last_close = recent_close
                
                # 방향 결정
                if self.config.side_bias == 'LONG':
                    action, score, conf = 'BUY', 0.8, 0.7
                elif self.config.side_bias == 'SHORT':
                    action, score, conf = 'SELL', 0.8, 0.7
                else:
                    # POC 대비 현재가 위치로 방향 결정
                    if last_close >= poc:
                        action, score, conf = 'BUY', 0.75, 0.6
                    else:
                        action, score, conf = 'SELL', 0.75, 0.6
                
                entry = last_close
                
                # ATR 기반 스탑로스 (더 정확함)
                if atr > 0:
                    stop = poc - (atr * 1.5) if action == 'BUY' else poc + (atr * 1.5)
                else:
                    # ATR 실패 시 기존 방식 사용
                    bins_arr = np.array(bins)
                    bin_w = bins_arr[1] - bins_arr[0]
                    stop = poc - (bin_w * 1.5) if action == 'BUY' else poc + (bin_w * 1.5)
                
                if self.config.debug:
                    print(f"[VPVR_MICRO] 신호 생성: {action} at {entry:.4f}, POC={poc:.4f}, "
                          f"within={within}, retest={retest}, vol_sum={vol_sum:.1f}, ATR={atr:.4f}")

        return {
            'name': 'VPVR_MICRO',
            'action': action,
            'score': float(score),
            'entry': float(entry) if entry is not None else None,
            'stop': float(stop) if stop is not None else None,
            'context': {
                'poc': float(poc), 
                'within_tol': bool(within), 
                'retest': bool(retest), 
                'vol_sum': float(vol_sum),
                'atr': float(atr),
                'tolerance': float(tol_price)
            }
        }
