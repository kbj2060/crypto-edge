# signals/macd_histogram_strategy.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

from data.data_manager import get_data_manager
from indicators.global_indicators import get_atr
from utils.time_manager import get_time_manager

def _clamp(x, a=0.0, b=1.0):
    try:
        return max(a, min(b, float(x)))
    except Exception:
        return a

@dataclass
class MACDHistogramCfg:
    fast_period: int = 8             # 12 → 8 (더 민감)
    slow_period: int = 18            # 26 → 18 (더 민감)
    signal_period: int = 6           # 9 → 6 (더 민감)
    lookback_bars: int = 60          # 100 → 60 (단축)
    min_histogram_change: float = 0.00005  # 최소 히스토그램 변화량
    divergence_lookback: int = 10
    momentum_threshold: float = 0.04         # 모멘텀 임계값
    atr_stop_mult: float = 1.2
    tp_R1: float = 2.0
    tp_R2: float = 3.0
    tick: float = 0.01
    debug: bool = False
    
    # 점수 구성 가중치
    w_histogram: float = 0.70    # 0.40 → 0.50
    w_momentum: float = 0.15     # 0.30 → 0.25  
    w_divergence: float = 0.10   # 0.20 → 0.15
    w_volume: float = 0.05

class MACDHistogramStrategy:
    """
    MACD 히스토그램 기반 모멘텀 가속도 전략
    - 히스토그램의 기울기 변화로 모멘텀 가속도 측정
    - 다이버전스 패턴 감지
    - RSI_DIV와 함께 사용하면 최강의 모멘텀 신호 조합
    """
    
    def __init__(self, cfg: MACDHistogramCfg = MACDHistogramCfg()):
        self.cfg = cfg
        self.time_manager = get_time_manager()
        
    def _calculate_macd(self, close: pd.Series) -> Dict[str, pd.Series]:
        """MACD 지표 계산"""
        fast_ema = close.ewm(span=self.cfg.fast_period, adjust=False).mean()
        slow_ema = close.ewm(span=self.cfg.slow_period, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=self.cfg.signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd_line': macd_line,
            'signal_line': signal_line,
            'histogram': histogram
        }
    
    def _detect_divergence(self, price: pd.Series, histogram: pd.Series) -> Dict[str, float]:
        """다이버전스 패턴 감지"""
        if len(price) < self.cfg.divergence_lookback:
            return {'bull_div': 0.0, 'bear_div': 0.0}
            
        # 최근 구간에서 고점/저점 찾기
        lookback_data = price.tail(self.cfg.divergence_lookback)
        hist_data = histogram.tail(self.cfg.divergence_lookback)
        
        # 단순한 고점/저점 감지 (롤링 윈도우 방식)
        window = 3
        price_highs = lookback_data.rolling(window, center=True).max() == lookback_data
        price_lows = lookback_data.rolling(window, center=True).min() == lookback_data
        
        bull_div_score = 0.0
        bear_div_score = 0.0
        
        try:
            # 강세 다이버전스: 가격은 더 낮은 저점, MACD는 더 높은 저점
            low_indices = price_lows[price_lows].index
            if len(low_indices) >= 2:
                last_low_idx = low_indices[-1]
                prev_low_idx = low_indices[-2]
                
                price_lower = lookback_data.loc[last_low_idx] < lookback_data.loc[prev_low_idx]
                hist_higher = hist_data.loc[last_low_idx] > hist_data.loc[prev_low_idx]
                
                if price_lower and hist_higher:
                    price_diff = abs(lookback_data.loc[last_low_idx] - lookback_data.loc[prev_low_idx])
                    hist_diff = abs(hist_data.loc[last_low_idx] - hist_data.loc[prev_low_idx])
                    bull_div_score = _clamp(hist_diff / (price_diff + 1e-9), 0.0, 1.0)
            
            # 약세 다이버전스: 가격은 더 높은 고점, MACD는 더 낮은 고점
            high_indices = price_highs[price_highs].index
            if len(high_indices) >= 2:
                last_high_idx = high_indices[-1]
                prev_high_idx = high_indices[-2]
                
                price_higher = lookback_data.loc[last_high_idx] > lookback_data.loc[prev_high_idx]
                hist_lower = hist_data.loc[last_high_idx] < hist_data.loc[prev_high_idx]
                
                if price_higher and hist_lower:
                    price_diff = abs(lookback_data.loc[last_high_idx] - lookback_data.loc[prev_high_idx])
                    hist_diff = abs(hist_data.loc[last_high_idx] - hist_data.loc[prev_high_idx])
                    bear_div_score = _clamp(hist_diff / (price_diff + 1e-9), 0.0, 1.0)
                    
        except Exception as e:
            if self.cfg.debug:
                print(f"[MACD_HISTOGRAM] 다이버전스 계산 오류: {e}")
        
        return {'bull_div': bull_div_score, 'bear_div': bear_div_score}
    
    def _calculate_momentum_acceleration(self, histogram: pd.Series) -> float:
        """히스토그램 기울기 변화로 모멘텀 가속도 계산"""
        if len(histogram) < 3:
            return 0.0
            
        # 히스토그램의 1차, 2차 차분 계산
        hist_diff1 = histogram.diff()  # 1차 차분 (속도)
        hist_diff2 = hist_diff1.diff()  # 2차 차분 (가속도)
        
        latest_acceleration = float(hist_diff2.iloc[-1])
        
        # NaN 값 체크
        if pd.isna(latest_acceleration):
            return 0.0
        
        # 가속도 정규화 (최근 변동성 기준)
        recent_std = float(hist_diff2.tail(10).std())
        if recent_std > 0:
            normalized_accel = latest_acceleration / recent_std
        else:
            normalized_accel = 0.0
            
        # 속도와 가속도를 결합한 모멘텀 점수
        momentum_score = _clamp(abs(normalized_accel), 0.0, 3.0) / 3.0

        if self.cfg.debug:
            print(f"[MACD_HISTOGRAM] 가속도: {latest_acceleration:.6f}, "
                f"정규화된 가속도: {normalized_accel:.3f}, 모멘텀 점수: {momentum_score:.3f}")

        return momentum_score
    
    def _calculate_volume_confirmation(self, df: pd.DataFrame) -> float:
        """거래량 확인"""
        try:
            vol_series = df['quote_volume'].astype(float)
                
            vol_ma = vol_series.rolling(20, min_periods=1).mean()
            current_vol = float(vol_series.iloc[-1])
            avg_vol = float(vol_ma.iloc[-1])
            
            if avg_vol > 0:
                vol_ratio = current_vol / avg_vol
                return _clamp((vol_ratio - 1.0) / 2.0, 0.0, 1.0)
            else:
                return 0.0
                
        except Exception:
            return 0.0
    def _no_signal_result(self,**kwargs):
        return {
            'name': 'MACD_HISTOGRAM',
            'action': 'HOLD',
            'score': 0.0,
            'timestamp': self.time_manager.get_current_time(),
            'context': kwargs
        }

    def on_kline_close_3m(self) -> Optional[Dict[str, Any]]:
        """3분봉 마감 시 MACD 히스토그램 전략 실행"""
        data_manager = get_data_manager()
        if data_manager is None:
            return self._no_signal_result()
            
        df = data_manager.get_latest_data(self.cfg.lookback_bars + 50)
        if df is None or len(df) < max(self.cfg.slow_period + self.cfg.signal_period + 5, 50):
            if self.cfg.debug:
                print(f"[MACD_HISTOGRAM] 데이터 부족: 필요={self.cfg.lookback_bars + 50}, 실제={len(df) if df is not None else 'None'}")
            return self._no_signal_result()
        
        close = pd.to_numeric(df['close'].astype(float))
        
        # MACD 계산
        macd_data = self._calculate_macd(close)
        histogram = macd_data['histogram']
        macd_line = macd_data['macd_line']
        signal_line = macd_data['signal_line']
        
        # 현재 값들
        current_hist = float(histogram.iloc[-1])
        prev_hist = float(histogram.iloc[-2])
        current_macd = float(macd_line.iloc[-1])
        current_signal = float(signal_line.iloc[-1])
        current_close = float(close.iloc[-1])
        
        # 히스토그램 변화량 체크
        hist_change = current_hist - prev_hist
        if abs(hist_change) < self.cfg.min_histogram_change:
            if self.cfg.debug:
                print(f"[MACD_HISTOGRAM] 히스토그램 변화량 부족: {hist_change:.6f}")
            return self._no_signal_result()
        
        # 각 컴포넌트 점수 계산
        momentum_score = self._calculate_momentum_acceleration(histogram)
        divergence_scores = self._detect_divergence(close, histogram)
        volume_score = self._calculate_volume_confirmation(df)
        
        # 히스토그램 기반 방향성 및 점수
        hist_score = 0.0
        action = "HOLD"
        
        # 강세 신호 조건
        if (current_hist > 0 and hist_change > 0) or \
           (current_hist < 0 and hist_change > 0 and current_macd > current_signal):
            action = "BUY"
            hist_score = min(1.0, abs(hist_change) / (abs(prev_hist) + 1e-9))
            div_score = divergence_scores['bull_div']
        
        # 약세 신호 조건  
        elif (current_hist < 0 and hist_change < 0) or \
             (current_hist > 0 and hist_change < 0 and current_macd < current_signal):
            action = "SELL"
            hist_score = min(1.0, abs(hist_change) / (abs(prev_hist) + 1e-9))
            div_score = divergence_scores['bear_div']
        else:
            div_score = 0.0
        
        if action == "HOLD":
            return self._no_signal_result()
        
        # 최종 점수 계산 (가중 평균)
        total_score = (
            self.cfg.w_histogram * hist_score +
            self.cfg.w_momentum * momentum_score +
            self.cfg.w_divergence * div_score +
            self.cfg.w_volume * volume_score
        )
        
        total_score = _clamp(total_score, 0.0, 1.0)
        
        # 모멘텀 임계값 체크
        if momentum_score < self.cfg.momentum_threshold:
            if self.cfg.debug:
                print(f"[MACD_HISTOGRAM] 모멘텀 부족: {momentum_score:.3f} < {self.cfg.momentum_threshold}")
            return self._no_signal_result()
        
        # 진입/손절/목표가 계산
        atr = get_atr()
        if atr is None:
            atr = float(close.pct_change().rolling(14).std() * close.iloc[-1])
        
        if action == "BUY":
            entry = current_close + self.cfg.tick
            stop = current_close - self.cfg.atr_stop_mult * float(atr)
        else:  # SELL
            entry = current_close - self.cfg.tick
            stop = current_close + self.cfg.atr_stop_mult * float(atr)
        
        R = abs(entry - stop)
        tp1 = entry + (self.cfg.tp_R1 * R if action == "BUY" else -self.cfg.tp_R1 * R)
        tp2 = entry + (self.cfg.tp_R2 * R if action == "BUY" else -self.cfg.tp_R2 * R)
        
        if self.cfg.debug:
            print(f"[MACD_HISTOGRAM] {action} 신호 - 점수: {total_score:.3f}, "
                  f"히스토그램 변화: {hist_change:.6f}, 모멘텀: {momentum_score:.3f}, "
                  f"다이버전스: {div_score:.3f}, 거래량: {volume_score:.3f}")
        
        return {
            'name': 'MACD_HISTOGRAM',
            'action': action,
            'score': float(total_score),
            'entry': float(entry),
            'stop': float(stop),
            'targets': [float(tp1), float(tp2)],
            'timestamp': self.time_manager.get_current_time(),
            'context': {
                'mode': 'MACD_HISTOGRAM_ACCELERATION',
                'histogram_current': float(current_hist),
                'histogram_change': float(hist_change),
                'momentum_score': float(momentum_score),
                'divergence_score': float(div_score),
                'volume_score': float(volume_score),
                'macd_line': float(current_macd),
                'signal_line': float(current_signal),
                'atr': float(atr)
            }
        }