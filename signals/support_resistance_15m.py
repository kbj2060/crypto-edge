# signals/support_resistance_15m.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
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
class SupportResistanceCfg:
    # 레벨 감지 설정 (15분봉 기준)
    level_lookback_bars: int = 96      # 24시간 (96개 15분봉)
    min_touches: int = 2               # 최소 터치 횟수
    level_tolerance: float = 0.003     # 레벨 허용 오차 (0.3%)
    swing_window: int = 3              # 스윙 감지 윈도우
    
    # 레벨 강도 계산
    max_level_age_bars: int = 48       # 최대 레벨 수명 (12시간)
    level_strength_decay: float = 0.1  # 시간당 강도 감소율
    min_level_strength: float = 0.3    # 최소 레벨 강도
    
    # 반발/돌파 감지
    bounce_confirmation_bars: int = 2   # 30분 반발 확인
    breakout_confirmation_bars: int = 3 # 45분 돌파 확인
    retest_wait_bars: int = 8          # 재시험 대기 시간 (2시간)
    retest_tolerance: float = 0.005    # 재시험 허용 오차 (0.5%)
    
    # 캔들 패턴 조건
    min_body_ratio: float = 0.3        # 최소 몸통 비율
    max_wick_ratio: float = 0.7        # 최대 꼬리 비율
    
    # 거래량 조건
    volume_confirmation_mult: float = 1.3  # 거래량 배수
    
    # 손익 설정
    atr_stop_mult: float = 1.5
    tp_R1: float = 2.0
    tp_R2: float = 3.5
    tick: float = 0.01
    debug: bool = False
    
    # 점수 구성 가중치
    w_level_strength: float = 0.35     # 레벨 강도
    w_pattern_quality: float = 0.25    # 패턴 품질
    w_volume: float = 0.20             # 거래량
    w_timing: float = 0.20             # 타이밍

class SupportResistance:
    """
    Support/Resistance 15분봉 전략
    - 지지/저항선 자동 감지
    - 레벨 반발 및 돌파 후 재시험 패턴 포착
    - 레벨 강도와 시간 가중치 적용
    """
    
    def __init__(self, cfg: SupportResistanceCfg = SupportResistanceCfg()):
        self.cfg = cfg
        self.time_manager = get_time_manager()
        self.cached_levels = []
        self.last_level_update = None
        
    def _find_swing_points(self, df: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
        """스윙 고점/저점 찾기"""
        high = pd.to_numeric(df['high'].astype(float))
        low = pd.to_numeric(df['low'].astype(float))
        
        swing_highs = []
        swing_lows = []
        
        window = self.cfg.swing_window
        
        for i in range(window, len(df) - window):
            # 스윙 고점 감지
            is_swing_high = all(high.iloc[i] >= high.iloc[i-j] for j in range(1, window+1))
            is_swing_high = is_swing_high and all(high.iloc[i] >= high.iloc[i+j] for j in range(1, window+1))
            
            if is_swing_high:
                swing_highs.append({
                    'index': i,
                    'price': float(high.iloc[i]),
                    'timestamp': df.index[i] if hasattr(df.index[i], 'timestamp') else i
                })
            
            # 스윙 저점 감지
            is_swing_low = all(low.iloc[i] <= low.iloc[i-j] for j in range(1, window+1))
            is_swing_low = is_swing_low and all(low.iloc[i] <= low.iloc[i+j] for j in range(1, window+1))
            
            if is_swing_low:
                swing_lows.append({
                    'index': i,
                    'price': float(low.iloc[i]),
                    'timestamp': df.index[i] if hasattr(df.index[i], 'timestamp') else i
                })
        
        return swing_highs, swing_lows
    
    def _identify_support_resistance_levels(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """지지/저항선 식별"""
        swing_highs, swing_lows = self._find_swing_points(df)
        
        levels = []
        
        # 저항선 처리 (스윙 고점들)
        for swing in swing_highs:
            level_price = swing['price']
            touches = self._count_level_touches(df, level_price, 'resistance')
            
            if touches >= self.cfg.min_touches:
                age_bars = len(df) - 1 - swing['index']
                strength = self._calculate_level_strength(touches, age_bars)
                
                if strength >= self.cfg.min_level_strength:
                    levels.append({
                        'price': level_price,
                        'type': 'resistance',
                        'touches': touches,
                        'strength': strength,
                        'age_bars': age_bars,
                        'last_touch_index': swing['index']
                    })
        
        # 지지선 처리 (스윙 저점들)
        for swing in swing_lows:
            level_price = swing['price']
            touches = self._count_level_touches(df, level_price, 'support')
            
            if touches >= self.cfg.min_touches:
                age_bars = len(df) - 1 - swing['index']
                strength = self._calculate_level_strength(touches, age_bars)
                
                if strength >= self.cfg.min_level_strength:
                    levels.append({
                        'price': level_price,
                        'type': 'support',
                        'touches': touches,
                        'strength': strength,
                        'age_bars': age_bars,
                        'last_touch_index': swing['index']
                    })
        
        # 강도순 정렬
        levels.sort(key=lambda x: x['strength'], reverse=True)
        
        if self.cfg.debug:
            print(f"[SR_15M] 감지된 레벨: {len(levels)}개")
            for level in levels[:5]:  # 상위 5개만 출력
                print(f"  {level['type']} @ {level['price']:.2f} (강도: {level['strength']:.3f}, 터치: {level['touches']})")
        
        return levels
    
    def _count_level_touches(self, df: pd.DataFrame, level_price: float, level_type: str) -> int:
        """레벨 터치 횟수 계산"""
        tolerance = level_price * self.cfg.level_tolerance
        touches = 0
        
        for _, row in df.iterrows():
            high_price = float(row['high'])
            low_price = float(row['low'])
            
            if level_type == 'resistance':
                # 고가가 레벨에 근접
                if abs(high_price - level_price) <= tolerance:
                    touches += 1
            else:  # support
                # 저가가 레벨에 근접
                if abs(low_price - level_price) <= tolerance:
                    touches += 1
        
        return touches
    
    def _calculate_level_strength(self, touches: int, age_bars: int) -> float:
        """레벨 강도 계산"""
        # 기본 강도 (터치 횟수 기반)
        base_strength = min(1.0, touches / 5.0)  # 5회 터치 시 최대
        
        # 시간 감쇠 적용
        if age_bars > self.cfg.max_level_age_bars:
            time_decay = 0.5  # 오래된 레벨은 50% 강도
        else:
            time_decay = 1.0 - (age_bars / self.cfg.max_level_age_bars) * self.cfg.level_strength_decay
        
        time_decay = max(0.1, time_decay)  # 최소 10% 강도 유지
        
        return base_strength * time_decay
    
    def _detect_bounce_pattern(self, df: pd.DataFrame, level: Dict[str, Any]) -> Dict[str, Any]:
        """반발 패턴 감지"""
        level_price = level['price']
        level_type = level['type']
        tolerance = level_price * self.cfg.level_tolerance
        
        # 최근 몇 개 봉에서 레벨 테스트 확인
        recent_bars = df.tail(self.cfg.bounce_confirmation_bars + 1)
        
        bounce_detected = False
        pattern_quality = 0.0
        
        for i, (_, row) in enumerate(recent_bars.iterrows()):
            open_price = float(row['open'])
            high_price = float(row['high'])
            low_price = float(row['low'])
            close_price = float(row['close'])
            
            body_size = abs(close_price - open_price)
            total_range = high_price - low_price
            
            if total_range == 0:
                continue
                
            body_ratio = body_size / total_range
            
            if level_type == 'support':
                # 지지선 반발: 저가가 레벨 터치 후 상승 마감
                level_touched = low_price <= level_price + tolerance
                bullish_close = close_price > open_price
                good_body = body_ratio >= self.cfg.min_body_ratio
                
                if level_touched and bullish_close and good_body:
                    bounce_detected = True
                    # 하락 꼬리 길이로 품질 측정
                    lower_wick = min(open_price, close_price) - low_price
                    wick_quality = min(1.0, lower_wick / (total_range * 0.5))
                    pattern_quality = max(pattern_quality, body_ratio * 0.6 + wick_quality * 0.4)
                    
            else:  # resistance
                # 저항선 반발: 고가가 레벨 터치 후 하락 마감
                level_touched = high_price >= level_price - tolerance
                bearish_close = close_price < open_price
                good_body = body_ratio >= self.cfg.min_body_ratio
                
                if level_touched and bearish_close and good_body:
                    bounce_detected = True
                    # 상승 꼬리 길이로 품질 측정
                    upper_wick = high_price - max(open_price, close_price)
                    wick_quality = min(1.0, upper_wick / (total_range * 0.5))
                    pattern_quality = max(pattern_quality, body_ratio * 0.6 + wick_quality * 0.4)
        
        return {
            'bounce_detected': bounce_detected,
            'pattern_quality': _clamp(pattern_quality, 0.0, 1.0),
            'signal_direction': 'BUY' if level_type == 'support' else 'SELL'
        }
    
    def _detect_breakout_retest(self, df: pd.DataFrame, level: Dict[str, Any]) -> Dict[str, Any]:
        """돌파 후 재시험 패턴 감지"""
        level_price = level['price']
        level_type = level['type']
        
        # 돌파 확인을 위한 최근 데이터
        lookback_bars = self.cfg.breakout_confirmation_bars + self.cfg.retest_wait_bars
        recent_data = df.tail(lookback_bars)
        
        if len(recent_data) < lookback_bars:
            return {'retest_detected': False, 'pattern_quality': 0.0}
        
        # 돌파 지점 찾기
        breakout_detected = False
        breakout_index = -1
        
        for i in range(self.cfg.retest_wait_bars, len(recent_data)):
            row = recent_data.iloc[i]
            close_price = float(row['close'])
            
            if level_type == 'resistance' and close_price > level_price * (1 + self.cfg.level_tolerance):
                breakout_detected = True
                breakout_index = i
                break
            elif level_type == 'support' and close_price < level_price * (1 - self.cfg.level_tolerance):
                breakout_detected = True
                breakout_index = i
                break
        
        if not breakout_detected:
            return {'retest_detected': False, 'pattern_quality': 0.0}
        
        # 돌파 후 재시험 확인
        retest_data = recent_data.iloc[breakout_index:]
        retest_tolerance = level_price * self.cfg.retest_tolerance
        
        retest_detected = False
        pattern_quality = 0.0
        
        for _, row in retest_data.iterrows():
            high_price = float(row['high'])
            low_price = float(row['low'])
            close_price = float(row['close'])
            
            if level_type == 'resistance':
                # 저항선 돌파 후 재시험: 다시 저항선 근처로 하락했다가 상승
                retest_touch = low_price <= level_price + retest_tolerance
                successful_retest = close_price > level_price
                
                if retest_touch and successful_retest:
                    retest_detected = True
                    pattern_quality = 0.8  # 재시험 성공
                    break
                    
            else:  # support
                # 지지선 돌파 후 재시험: 다시 지지선 근처로 상승했다가 하락
                retest_touch = high_price >= level_price - retest_tolerance
                successful_retest = close_price < level_price
                
                if retest_touch and successful_retest:
                    retest_detected = True
                    pattern_quality = 0.8  # 재시험 성공
                    break
        
        return {
            'retest_detected': retest_detected,
            'pattern_quality': pattern_quality,
            'signal_direction': 'BUY' if level_type == 'resistance' else 'SELL'
        }
    
    def _calculate_volume_confirmation(self, df: pd.DataFrame) -> float:
        """거래량 확인"""
        try:
            if 'quote_volume' in df.columns:
                vol_series = df['quote_volume'].astype(float)
            else:
                vol_series = df['volume'].astype(float) * df['close'].astype(float)
                
            vol_ma = vol_series.rolling(20, min_periods=1).mean()
            current_vol = float(vol_series.iloc[-1])
            avg_vol = float(vol_ma.iloc[-1])
            
            if avg_vol > 0:
                vol_ratio = current_vol / avg_vol
                return _clamp((vol_ratio - 1.0) / (self.cfg.volume_confirmation_mult - 1.0), 0.0, 1.0)
            else:
                return 0.0
                
        except Exception:
            return 0.5
    
    def _calculate_timing_score(self, level: Dict[str, Any]) -> float:
        """타이밍 점수 계산"""
        age_bars = level['age_bars']
        
        # 너무 오래되지 않은 레벨일수록 높은 점수
        if age_bars <= 12:  # 3시간 이내
            return 1.0
        elif age_bars <= 24:  # 6시간 이내
            return 0.8
        elif age_bars <= 48:  # 12시간 이내
            return 0.6
        else:
            return 0.3
    
    def _no_signal_result(self, **kwargs):
        return {
            'name': 'SUPPORT_RESISTANCE',
            'action': 'HOLD',
            'score': 0.0,
            'timestamp': self.time_manager.get_current_time(),
            'context': kwargs
        }
    
    def on_kline_close_15m(self) -> Optional[Dict[str, Any]]:
        """15분봉 마감 시 Support/Resistance 전략 실행"""
        data_manager = get_data_manager()
        if data_manager is None:
            return self._no_signal_result()
            
        df = data_manager.get_latest_data_15m(self.cfg.level_lookback_bars + 20)
        if df is None or len(df) < self.cfg.level_lookback_bars:
            if self.cfg.debug:
                print(f"[SR_15M] 데이터 부족: 필요={self.cfg.level_lookback_bars}, "
                      f"실제={len(df) if df is not None else 'None'}")
            return self._no_signal_result()
        
        # 지지/저항선 식별
        levels = self._identify_support_resistance_levels(df)
        
        if not levels:
            if self.cfg.debug:
                print("[SR_15M] 유효한 지지/저항선 없음")
            return self._no_signal_result()
        
        # 각 레벨에 대해 패턴 분석
        best_signal = None
        best_score = 0.0
        
        for level in levels:
            # 반발 패턴 체크
            bounce_result = self._detect_bounce_pattern(df, level)
            
            # 돌파 후 재시험 패턴 체크
            retest_result = self._detect_breakout_retest(df, level)
            
            # 신호가 있는 경우에만 점수 계산
            if bounce_result['bounce_detected'] or retest_result['retest_detected']:
                
                # 더 좋은 패턴 선택
                if bounce_result['pattern_quality'] >= retest_result['pattern_quality']:
                    pattern_quality = bounce_result['pattern_quality']
                    signal_direction = bounce_result['signal_direction']
                    pattern_type = 'BOUNCE'
                else:
                    pattern_quality = retest_result['pattern_quality']
                    signal_direction = retest_result['signal_direction']
                    pattern_type = 'RETEST'
                
                # 거래량 확인
                volume_score = self._calculate_volume_confirmation(df)
                
                # 타이밍 점수
                timing_score = self._calculate_timing_score(level)
                
                # 최종 점수 계산
                total_score = (
                    self.cfg.w_level_strength * level['strength'] +
                    self.cfg.w_pattern_quality * pattern_quality +
                    self.cfg.w_volume * volume_score +
                    self.cfg.w_timing * timing_score
                )
                
                total_score = _clamp(total_score, 0.0, 1.0)
                
                if total_score > best_score:
                    best_score = total_score
                    best_signal = {
                        'level': level,
                        'action': signal_direction,
                        'score': total_score,
                        'pattern_type': pattern_type,
                        'pattern_quality': pattern_quality,
                        'volume_score': volume_score,
                        'timing_score': timing_score
                    }
        
        if best_signal is None or best_score < 0.4:
            if self.cfg.debug:
                print(f"[SR_15M] 신호 없음 또는 점수 부족: {best_score:.3f}")
            return self._no_signal_result()
        
        # 진입/손절/목표가 계산
        level_price = best_signal['level']['price']
        current_price = float(df['close'].iloc[-1])
        
        atr = get_atr()
        if atr is None:
            close_series = pd.to_numeric(df['close'].astype(float))
            atr = float(close_series.pct_change().rolling(14).std() * current_price)
        
        action = best_signal['action']
        
        if action == "BUY":
            entry = current_price + self.cfg.tick
            if best_signal['pattern_type'] == 'BOUNCE':
                stop = level_price - self.cfg.atr_stop_mult * float(atr)
            else:  # RETEST
                stop = level_price - self.cfg.tick
        else:  # SELL
            entry = current_price - self.cfg.tick
            if best_signal['pattern_type'] == 'BOUNCE':
                stop = level_price + self.cfg.atr_stop_mult * float(atr)
            else:  # RETEST
                stop = level_price + self.cfg.tick
        
        R = abs(entry - stop)
        tp1 = entry + (self.cfg.tp_R1 * R if action == "BUY" else -self.cfg.tp_R1 * R)
        tp2 = entry + (self.cfg.tp_R2 * R if action == "BUY" else -self.cfg.tp_R2 * R)
        
        if self.cfg.debug:
            print(f"[SR_15M] {action} 신호 - 점수: {best_score:.3f}, "
                  f"패턴: {best_signal['pattern_type']}, 레벨: {level_price:.2f}, "
                  f"강도: {best_signal['level']['strength']:.3f}")
        
        return {
            'name': 'SUPPORT_RESISTANCE',
            'action': action,
            'score': float(best_score),
            'entry': float(entry),
            'stop': float(stop),
            'targets': [float(tp1), float(tp2)],
            'timestamp': self.time_manager.get_current_time(),
            'context': {
                'mode': 'SR_LEVEL_PATTERN',
                'pattern_type': best_signal['pattern_type'],
                'level_price': float(level_price),
                'level_type': best_signal['level']['type'],
                'level_strength': float(best_signal['level']['strength']),
                'level_touches': best_signal['level']['touches'],
                'level_age_bars': best_signal['level']['age_bars'],
                'pattern_quality': float(best_signal['pattern_quality']),
                'volume_score': float(best_signal['volume_score']),
                'timing_score': float(best_signal['timing_score']),
                'atr': float(atr)
            }
        }