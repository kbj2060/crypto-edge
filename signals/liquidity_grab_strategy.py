# signals/liquidity_grab_strategy.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
import numpy as np

from data.data_manager import get_data_manager
from indicators.global_indicators import get_atr, get_vwap
from utils.time_manager import get_time_manager

def _clamp(x, a=0.0, b=1.0):
    try:
        return max(a, min(b, float(x)))
    except Exception:
        return a

@dataclass
class LiquidityGrabCfg:
    lookback_bars: int = 100
    support_resistance_period: int = 50    # 지지/저항선 찾는 기간
    min_touches: int = 1                  # 최소 터치 횟수
    grab_threshold_pct: float =  0.05       # 0.1% 돌파로 간주
    recovery_threshold_pct: float = 0.1  # 0.15% 복귀로 간주
    max_grab_bars: int = 3               # 가짜돌파 후 최대 N봉 내 복귀
    volume_spike_threshold: float = 1.2   # 거래량 급증 임계값
    atr_stop_mult: float = 1.0
    tp_R1: float = 2.5
    tp_R2: float = 4.0
    tick: float = 0.01
    debug: bool = False
    
    # 점수 구성 가중치
    w_grab_quality: float = 0.35
    w_volume_spike: float = 0.25
    w_level_strength: float = 0.25
    w_recovery_speed: float = 0.15

class LiquidityGrabStrategy:
    """
    유동성 사냥 패턴 감지 전략
    - 중요 지지/저항선 위아래 가짜 돌파 감지
    - 유동성 사냥 후 빠른 반전 신호 포착
    - 스마트머니의 유동성 수집 패턴 추적
    """
    
    def __init__(self, cfg: LiquidityGrabCfg = LiquidityGrabCfg()):
        self.cfg = cfg
        self.time_manager = get_time_manager()
        self.cached_levels = []
        self.last_level_update = None
        
    def _find_support_resistance_levels(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """지지/저항선 찾기"""
        high = pd.to_numeric(df['high'].astype(float))
        low = pd.to_numeric(df['low'].astype(float))
        close = pd.to_numeric(df['close'].astype(float))
        
        levels = []
        window = 5  # 피벗 포인트를 찾기 위한 윈도우
        
        # 지지선 찾기 (저점)
        for i in range(window, len(low) - window):
            if all(low.iloc[i] <= low.iloc[i-j] for j in range(1, window+1)) and \
               all(low.iloc[i] <= low.iloc[i+j] for j in range(1, window+1)):
                
                level_price = float(low.iloc[i])
                level_index = low.index[i]
                
                # 해당 레벨 터치 횟수 계산
                touches = self._count_level_touches(df, level_price, 'support')
                
                if touches >= self.cfg.min_touches:
                    levels.append({
                        'price': level_price,
                        'type': 'support',
                        'touches': touches,
                        'last_touch_index': level_index,
                        'strength': min(1.0, touches / 5.0)  # 정규화된 강도
                    })
        
        # 저항선 찾기 (고점)
        for i in range(window, len(high) - window):
            if all(high.iloc[i] >= high.iloc[i-j] for j in range(1, window+1)) and \
               all(high.iloc[i] >= high.iloc[i+j] for j in range(1, window+1)):
                
                level_price = float(high.iloc[i])
                level_index = high.index[i]
                
                # 해당 레벨 터치 횟수 계산
                touches = self._count_level_touches(df, level_price, 'resistance')
                
                if touches >= self.cfg.min_touches:
                    levels.append({
                        'price': level_price,
                        'type': 'resistance',
                        'touches': touches,
                        'last_touch_index': level_index,
                        'strength': min(1.0, touches / 5.0)
                    })
        
        # 가격에 따라 정렬
        levels.sort(key=lambda x: x['price'])
        
        if self.cfg.debug:
            print(f"[LIQUIDITY_GRAB] 발견된 레벨 수: {len(levels)}")
            for level in levels[-5:]:  # 최근 5개만 출력
                print(f"  {level['type']} @ {level['price']:.2f} (터치: {level['touches']})")
        
        return levels
    
    def _count_level_touches(self, df: pd.DataFrame, level_price: float, level_type: str) -> int:
        """특정 레벨의 터치 횟수 계산"""
        tolerance = level_price * 0.002  # 0.2% 허용 오차
        touches = 0
        
        if level_type == 'support':
            # 저가가 레벨 근처에 온 횟수
            near_level = abs(df['low'].astype(float) - level_price) <= tolerance
        else:  # resistance
            # 고가가 레벨 근처에 온 횟수
            near_level = abs(df['high'].astype(float) - level_price) <= tolerance
            
        touches = near_level.sum()
        return int(touches)
    
    def _detect_liquidity_grab(self, df: pd.DataFrame, levels: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """유동성 사냥 패턴 감지"""
        if len(df) < self.cfg.max_grab_bars + 1:
            return None
            
        recent_bars = df.tail(self.cfg.max_grab_bars + 1)
        
        for level in levels:
            level_price = level['price']
            level_type = level['type']
            
            # 돌파 임계값 계산
            grab_distance = level_price * (self.cfg.grab_threshold_pct / 100.0)
            recovery_distance = level_price * (self.cfg.recovery_threshold_pct / 100.0)
            
            if level_type == 'support':
                # 지지선 하향 돌파 체크
                if self._check_support_grab(recent_bars, level_price, grab_distance, recovery_distance):
                    grab_info = self._analyze_grab_quality(recent_bars, level, 'support_grab')
                    if grab_info:
                        grab_info.update({
                            'level_price': level_price,
                            'level_type': level_type,
                            'level_strength': level['strength'],
                            'signal_direction': 'BUY'  # 지지선 가짜돌파 후 반등
                        })
                        return grab_info
                        
            else:  # resistance
                # 저항선 상향 돌파 체크
                if self._check_resistance_grab(recent_bars, level_price, grab_distance, recovery_distance):
                    grab_info = self._analyze_grab_quality(recent_bars, level, 'resistance_grab')
                    if grab_info:
                        grab_info.update({
                            'level_price': level_price,
                            'level_type': level_type,
                            'level_strength': level['strength'],
                            'signal_direction': 'SELL'  # 저항선 가짜돌파 후 하락
                        })
                        return grab_info
        
        return None
    
    def _check_support_grab(self, bars: pd.DataFrame, level_price: float, 
                           grab_distance: float, recovery_distance: float) -> bool:
        """지지선 유동성 사냥 체크"""
        # 최근 봉들 중에 하향 돌파가 있었는지 확인
        break_threshold = level_price - grab_distance
        recovery_threshold = level_price + recovery_distance
        
        lowest_low = float(bars['low'].min())
        current_close = float(bars['close'].iloc[-1])
        
        # 조건: 1) 하향 돌파했다가 2) 빠르게 회복
        broke_below = lowest_low < break_threshold
        recovered_above = current_close > recovery_threshold
        
        return broke_below and recovered_above
    
    def _check_resistance_grab(self, bars: pd.DataFrame, level_price: float,
                              grab_distance: float, recovery_distance: float) -> bool:
        """저항선 유동성 사냥 체크"""
        # 최근 봉들 중에 상향 돌파가 있었는지 확인
        break_threshold = level_price + grab_distance
        recovery_threshold = level_price - recovery_distance
        
        highest_high = float(bars['high'].max())
        current_close = float(bars['close'].iloc[-1])
        
        # 조건: 1) 상향 돌파했다가 2) 빠르게 회복
        broke_above = highest_high > break_threshold
        recovered_below = current_close < recovery_threshold
        
        return broke_above and recovered_below
    
    def _analyze_grab_quality(self, bars: pd.DataFrame, level: Dict[str, Any], 
                             grab_type: str) -> Optional[Dict[str, Any]]:
        """유동성 사냥의 품질 분석"""
        try:
            # 거래량 분석
            vol_series = bars['quote_volume'].astype(float)
            
            # 평균 거래량 대비 현재 거래량
            recent_avg_vol = float(vol_series.mean())
            peak_vol = float(vol_series.max())
            
            volume_spike = peak_vol / (recent_avg_vol + 1e-9)
            volume_score = _clamp((volume_spike - 1.0) / self.cfg.volume_spike_threshold, 0.0, 1.0)
            
            # 가격 변동성 분석
            price_range = float(bars['high'].max() - bars['low'].min())
            avg_range = float((bars['high'] - bars['low']).mean())
            volatility_ratio = price_range / (avg_range + 1e-9)
            
            # 회복 속도 분석 (빠를수록 좋음)
            recovery_bars = len(bars) - 1  # 마지막 봉까지의 시간
            recovery_speed = _clamp(1.0 - (recovery_bars / self.cfg.max_grab_bars), 0.0, 1.0)
            
            # 가짜돌파 품질 점수
            grab_quality = min(1.0, volatility_ratio * 0.5 + 0.5)
            
            # 최소 품질 체크
            if volume_score < 0.3 and grab_quality < 0.4:
                return None
            
            return {
                'grab_type': grab_type,
                'volume_spike': volume_spike,
                'volume_score': volume_score,
                'volatility_ratio': volatility_ratio,
                'recovery_speed': recovery_speed,
                'grab_quality': grab_quality
            }
            
        except Exception as e:
            if self.cfg.debug:
                print(f"[LIQUIDITY_GRAB] 품질 분석 오류: {e}")
            return self.default_result
    
    def _no_signal_result(self,**kwargs):
        return {
            'name': 'LIQUIDITY_GRAB',
            'action': 'HOLD',
            'score': 0.0,
            'timestamp': self.time_manager.get_current_time(),
            'context': kwargs
        }
    
    def on_kline_close_3m(self) -> Optional[Dict[str, Any]]:
        """3분봉 마감 시 유동성 사냥 전략 실행"""
        data_manager = get_data_manager()
        if data_manager is None:
            return self._no_signal_result()
            
        df = data_manager.get_latest_data(self.cfg.lookback_bars + 20)
        if df is None or len(df) < self.cfg.support_resistance_period + 10:
            if self.cfg.debug:
                print(f"[LIQUIDITY_GRAB] 데이터 부족: 필요={self.cfg.lookback_bars + 20}, "
                        f"실제={len(df) if df is not None else 'None'}")
            return self._no_signal_result()
        
        # 지지/저항선 찾기 (캐시 활용)
        now = self.time_manager.get_current_time()
        if (self.last_level_update is None or 
            (now - self.last_level_update).total_seconds() > 1800):  # 30분마다 업데이트
            
            levels = self._find_support_resistance_levels(df.iloc[:-10])  # 최근 10봉 제외하고 계산
            self.cached_levels = levels
            self.last_level_update = now
        else:
            levels = self.cached_levels
        
        if not levels:
            if self.cfg.debug:
                print("[LIQUIDITY_GRAB] 유효한 지지/저항선 없음")
            return self._no_signal_result()
        
        # 유동성 사냥 패턴 감지
        grab_result = self._detect_liquidity_grab(df, levels)
        if grab_result is None:
            return self._no_signal_result()
        
        # 추가 확인 지표들
        current_price = float(df['close'].iloc[-1])
        
        # VWAP과의 관계 체크
        vwap, vwap_std = get_vwap()
        vwap_score = 0.5
        if vwap is not None:
            vwap_distance = abs(current_price - float(vwap)) / float(vwap_std or current_price * 0.01)
            vwap_score = _clamp(1.0 - vwap_distance / 2.0, 0.0, 1.0)
        
        # 최종 점수 계산
        total_score = (
            self.cfg.w_grab_quality * grab_result['grab_quality'] +
            self.cfg.w_volume_spike * grab_result['volume_score'] +
            self.cfg.w_level_strength * grab_result['level_strength'] +
            self.cfg.w_recovery_speed * grab_result['recovery_speed']
        )
        
        # VWAP 보너스
        total_score = total_score * 0.9 + vwap_score * 0.1
        total_score = _clamp(total_score, 0.0, 1.0)
        
        # 최소 점수 체크
        if total_score < 0.5:
            if self.cfg.debug:
                print(f"[LIQUIDITY_GRAB] 점수 부족: {total_score:.3f}")
            return self._no_signal_result()
        
        # 진입/손절/목표가 계산
        action = grab_result['signal_direction']
        level_price = grab_result['level_price']
        
        atr = get_atr()
        if atr is None:
            close_series = pd.to_numeric(df['close'].astype(float))
            atr = float(close_series.pct_change().rolling(14).std() * current_price)
        
        if action == "BUY":
            entry = current_price + self.cfg.tick
            stop = min(current_price - self.cfg.atr_stop_mult * float(atr), 
                      level_price * 0.995)  # 레벨 아래 0.5%
        else:  # SELL
            entry = current_price - self.cfg.tick
            stop = max(current_price + self.cfg.atr_stop_mult * float(atr),
                      level_price * 1.005)  # 레벨 위 0.5%
        
        R = abs(entry - stop)
        tp1 = entry + (self.cfg.tp_R1 * R if action == "BUY" else -self.cfg.tp_R1 * R)
        tp2 = entry + (self.cfg.tp_R2 * R if action == "BUY" else -self.cfg.tp_R2 * R)
        
        if self.cfg.debug:
            print(f"[LIQUIDITY_GRAB] {action} 신호 - 점수: {total_score:.3f}, "
                  f"레벨: {level_price:.2f} ({grab_result['level_type']}), "
                  f"거래량 급증: {grab_result['volume_spike']:.2f}x")
        
        return {
            'name': 'LIQUIDITY_GRAB',
            'action': action,
            'score': float(total_score),
            'entry': float(entry),
            'stop': float(stop),
            'targets': [float(tp1), float(tp2)],
            'timestamp': self.time_manager.get_current_time(),
            'context': {
                'mode': 'LIQUIDITY_GRAB_PATTERN',
                'grab_type': grab_result['grab_type'],
                'level_price': float(level_price),
                'level_type': grab_result['level_type'],
                'level_strength': float(grab_result['level_strength']),
                'volume_spike': float(grab_result['volume_spike']),
                'grab_quality': float(grab_result['grab_quality']),
                'recovery_speed': float(grab_result['recovery_speed']),
                'vwap_score': float(vwap_score),
                'atr': float(atr)
            }
        }