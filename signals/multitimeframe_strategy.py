from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import pandas as pd

from data.data_manager import get_data_manager
from indicators.global_indicators import get_atr
from utils.time_manager import get_time_manager

def _clamp(x, a=0.0, b=1.0):
    try:
        return max(a, min(b, float(x)))
    except Exception:
        return a

@dataclass
class MultiTimeframeCfg:
    # 시간프레임 설정
    higher_timeframe: str = "15m"        # 상위 시간프레임 (추세 분석용)
    lower_timeframe: str = "3m"          # 하위 시간프레임 (진입 타이밍)
    
    # 데이터 요구량
    htf_lookback: int = 100              # 상위 시간프레임 데이터
    ltf_lookback: int = 200              # 하위 시간프레임 데이터
    
    # 레벨 감지 설정
    level_strength_min: int = 3          # 최소 터치 횟수
    level_tolerance: float = 0.002       # 레벨 허용 오차 (0.2%)
    breakout_confirmation: int = 2       # 돌파 확인 봉수
    
    # 패턴 감지 설정
    pattern_lookback: int = 20           # 패턴 감지 범위
    fakeout_retracement: float = 0.618   # 가짜돌파 되돌림 비율
    
    # 캔들 패턴 설정
    engulfing_body_ratio: float = 1.2    # Engulfing 몸통 비율
    doji_body_ratio: float = 0.1         # Doji 몸통 비율
    
    # 손익 설정
    atr_stop_mult: float = 1.5
    tp_R1: float = 2.0
    tp_R2: float = 3.5
    tick: float = 0.01
    debug: bool = False
    
    # 점수 가중치
    w_htf_trend: float = 0.40            # 상위 추세 강도
    w_level_quality: float = 0.30        # 레벨 품질
    w_pattern_strength: float = 0.20     # 패턴 강도
    w_ltf_confirmation: float = 0.10     # 하위 확인 신호

    level_strength_min: int = 3          # 최소 터치 횟수
    level_tolerance: float = 0.002       # 레벨 허용 오차 (0.2%)
    level_lookback: int = 100            # 레벨 감지용 히스토리 범위 ✅
    breakout_confirmation: int = 2       # 돌파 확인 봉수

class MultiTimeframeStrategy:
    """
    Multi-Timeframe Analysis 전략
    - Top-Down 접근법으로 상위→하위 시간프레임 분석
    - 5가지 전략: 레벨 돌파/반발, 가짜돌파, 캔들패턴, 차트패턴
    - 상위에서 바이어스, 하위에서 정확한 진입점 포착
    """
    
    def __init__(self, cfg: MultiTimeframeCfg = MultiTimeframeCfg()):
        self.cfg = cfg
        self.time_manager = get_time_manager()
        
    def _get_higher_timeframe_data(self) -> Optional[pd.DataFrame]:
        """상위 시간프레임 데이터 가져오기 (캐싱 적용)"""
        data_manager = get_data_manager()
        if data_manager is None:
            return None
        
        # 상위 시간프레임 데이터 구성 (3분봉을 15분봉으로 리샘플링)
        htf_data = data_manager.get_latest_data_15m(self.cfg.htf_lookback * 5)  # 15분 = 5개의 3분봉
        if htf_data is None or len(htf_data) < 50:
            return None

        if self.cfg.debug:
            print(f"[MTF] HTF 데이터 업데이트: {len(htf_data)}개 15분봉")
        
        return htf_data
        
    def _identify_support_resistance_levels(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """지지저항선 식별"""
        if len(df) < self.cfg.level_strength_min * 2:
            return {'support': [], 'resistance': []}
        
        highs = df['high'].values
        lows = df['low'].values
        
        # 피벗 포인트 찾기
        pivot_highs = []
        pivot_lows = []
        
        for i in range(2, len(df) - 2):
            # 피벗 하이
            if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and
                highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                pivot_highs.append(highs[i])
            
            # 피벗 로우
            if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and
                lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                pivot_lows.append(lows[i])
        
        # 레벨별 터치 횟수 계산
        resistance_levels = []
        support_levels = []
        
        for pivot_high in pivot_highs:
            touch_count = 0
            for high in highs:
                if abs(high - pivot_high) / pivot_high <= self.cfg.level_tolerance:
                    touch_count += 1
            
            if touch_count >= self.cfg.level_strength_min:
                resistance_levels.append(pivot_high)
        
        for pivot_low in pivot_lows:
            touch_count = 0
            for low in lows:
                if abs(low - pivot_low) / pivot_low <= self.cfg.level_tolerance:
                    touch_count += 1
            
            if touch_count >= self.cfg.level_strength_min:
                support_levels.append(pivot_low)
        
        return {
            'resistance': list(set(resistance_levels)),  # 중복 제거
            'support': list(set(support_levels))
        }
    
    def _identify_support_resistance_levels(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """지지저항선 식별 (더 많은 데이터로 정밀 분석)"""
        if len(df) < self.cfg.level_strength_min * 2:
            return {'support': [], 'resistance': []}
        
        # 레벨 분석용 데이터 범위 (더 많은 히스토리 활용)
        analysis_data = df.tail(min(len(df), self.cfg.level_lookback))
        highs = analysis_data['high'].values
        lows = analysis_data['low'].values
        
        # 피벗 포인트 찾기 (더 정밀한 기준)
        pivot_highs = []
        pivot_lows = []
        
        window_size = max(2, len(analysis_data) // 50)  # 동적 윈도우 크기
        
        for i in range(window_size, len(analysis_data) - window_size):
            # 피벗 하이 (더 엄격한 기준)
            left_max = max(highs[i-window_size:i])
            right_max = max(highs[i+1:i+window_size+1])
            if highs[i] > left_max and highs[i] > right_max:
                pivot_highs.append(highs[i])
            
            # 피벗 로우 (더 엄격한 기준)
            left_min = min(lows[i-window_size:i])
            right_min = min(lows[i+1:i+window_size+1])
            if lows[i] < left_min and lows[i] < right_min:
                pivot_lows.append(lows[i])
        
        # 레벨별 터치 횟수 계산 (전체 데이터에서)
        resistance_levels = []
        support_levels = []
        
        for pivot_high in pivot_highs:
            touch_count = 0
            for high in highs:
                if abs(high - pivot_high) / pivot_high <= self.cfg.level_tolerance:
                    touch_count += 1
            
            if touch_count >= self.cfg.level_strength_min:
                resistance_levels.append(pivot_high)
        
        for pivot_low in pivot_lows:
            touch_count = 0
            for low in lows:
                if abs(low - pivot_low) / pivot_low <= self.cfg.level_tolerance:
                    touch_count += 1
            
            if touch_count >= self.cfg.level_strength_min:
                support_levels.append(pivot_low)
        
        # 중복 제거 및 정렬
        resistance_levels = sorted(list(set(resistance_levels)), reverse=True)
        support_levels = sorted(list(set(support_levels)))
        
        return {
            'resistance': resistance_levels,
            'support': support_levels
        }
    
    def _detect_level_breakout(self, htf_df: pd.DataFrame, ltf_df: pd.DataFrame) -> Dict[str, Any]:
        """레벨 돌파 감지 (전략 #1)"""
        levels = self._identify_support_resistance_levels(htf_df)
        
        if not levels['resistance'] and not levels['support']:
            return {'signal': None, 'strength': 0.0}
        
        current_price = float(ltf_df['close'].iloc[-1])
        recent_high = float(htf_df['high'].tail(5).max())
        recent_low = float(htf_df['low'].tail(5).min())
        
        # 저항선 돌파 확인
        for resistance in levels['resistance']:
            if (current_price > resistance and 
                recent_high > resistance * (1 + self.cfg.level_tolerance)):
                
                # 하위 시간프레임에서 돌파 확인
                breakout_strength = (current_price - resistance) / resistance
                
                return {
                    'signal': 'BUY',
                    'strength': _clamp(breakout_strength * 100, 0.0, 1.0),
                    'level': resistance,
                    'type': 'RESISTANCE_BREAKOUT'
                }
        
        # 지지선 돌파 확인
        for support in levels['support']:
            if (current_price < support and 
                recent_low < support * (1 - self.cfg.level_tolerance)):
                
                breakout_strength = (support - current_price) / support
                
                return {
                    'signal': 'SELL',
                    'strength': _clamp(breakout_strength * 100, 0.0, 1.0),
                    'level': support,
                    'type': 'SUPPORT_BREAKOUT'
                }
        
        return {'signal': None, 'strength': 0.0}
    
    def _detect_level_bounce(self, htf_df: pd.DataFrame, ltf_df: pd.DataFrame) -> Dict[str, Any]:
        """레벨 반발 감지 (전략 #2)"""
        levels = self._identify_support_resistance_levels(htf_df)
        current_price = float(ltf_df['close'].iloc[-1])
        
        # 저항선에서 반발
        for resistance in levels['resistance']:
            if abs(current_price - resistance) / resistance <= self.cfg.level_tolerance:
                # 최근 캔들이 저항선 근처에서 반전 신호
                recent_candles = htf_df.tail(3)
                if (recent_candles['high'].max() >= resistance * 0.998 and
                    recent_candles['close'].iloc[-1] < recent_candles['high'].iloc[-1] * 0.99):
                    
                    bounce_strength = 1.0 - (current_price - resistance) / resistance
                    
                    return {
                        'signal': 'SELL',
                        'strength': _clamp(abs(bounce_strength), 0.0, 1.0),
                        'level': resistance,
                        'type': 'RESISTANCE_BOUNCE'
                    }
        
        # 지지선에서 반발
        for support in levels['support']:
            if abs(current_price - support) / support <= self.cfg.level_tolerance:
                recent_candles = htf_df.tail(3)
                if (recent_candles['low'].min() <= support * 1.002 and
                    recent_candles['close'].iloc[-1] > recent_candles['low'].iloc[-1] * 1.01):
                    
                    bounce_strength = (current_price - support) / support
                    
                    return {
                        'signal': 'BUY',
                        'strength': _clamp(bounce_strength * 50, 0.0, 1.0),
                        'level': support,
                        'type': 'SUPPORT_BOUNCE'
                    }
        
        return {'signal': None, 'strength': 0.0}
    
    def _detect_fakeout(self, htf_df: pd.DataFrame) -> Dict[str, Any]:
        """가짜돌파 감지 (전략 #3)"""
        if len(htf_df) < 10:
            return {'signal': None, 'strength': 0.0}
        
        # 최근 고점/저점 찾기
        recent_data = htf_df.tail(10)
        previous_high = float(recent_data['high'].iloc[-3])
        current_high = float(recent_data['high'].iloc[-1])
        current_close = float(recent_data['close'].iloc[-1])
        
        previous_low = float(recent_data['low'].iloc[-3])
        current_low = float(recent_data['low'].iloc[-1])
        
        # 상승 가짜돌파 (고점 넘었다가 다시 아래로)
        if (current_high > previous_high * 1.001 and  # 고점 돌파
            current_close < previous_high * 0.995):   # 다시 아래로
            
            fakeout_strength = (current_high - current_close) / current_high
            
            return {
                'signal': 'SELL',
                'strength': _clamp(fakeout_strength * 10, 0.0, 1.0),
                'level': previous_high,
                'type': 'BULL_TRAP_FAKEOUT'
            }
        
        # 하락 가짜돌파 (저점 깼다가 다시 위로)
        if (current_low < previous_low * 0.999 and   # 저점 돌파
            current_close > previous_low * 1.005):   # 다시 위로
            
            fakeout_strength = (current_close - current_low) / current_close
            
            return {
                'signal': 'BUY',
                'strength': _clamp(fakeout_strength * 10, 0.0, 1.0),
                'level': previous_low,
                'type': 'BEAR_TRAP_FAKEOUT'
            }
        
        return {'signal': None, 'strength': 0.0}
    
    def _detect_engulfing_candlestick(self, htf_df: pd.DataFrame) -> Dict[str, Any]:
        """Engulfing 캔들패턴 감지 (전략 #4)"""
        if len(htf_df) < 2:
            return {'signal': None, 'strength': 0.0}
        
        current = htf_df.iloc[-1]
        previous = htf_df.iloc[-2]
        
        curr_open = float(current['open'])
        curr_close = float(current['close'])
        curr_high = float(current['high'])
        curr_low = float(current['low'])
        
        prev_open = float(previous['open'])
        prev_close = float(previous['close'])
        
        curr_body = abs(curr_close - curr_open)
        prev_body = abs(prev_close - prev_open)
        
        # Bullish Engulfing
        if (prev_close < prev_open and  # 이전 캔들 음봉
            curr_close > curr_open and  # 현재 캔들 양봉
            curr_open < prev_close and  # 현재 시가가 이전 종가보다 낮음
            curr_close > prev_open and  # 현재 종가가 이전 시가보다 높음
            curr_body > prev_body * self.cfg.engulfing_body_ratio):  # 몸통 크기 조건
            
            engulfing_strength = curr_body / prev_body
            
            return {
                'signal': 'BUY',
                'strength': _clamp(engulfing_strength / 3.0, 0.0, 1.0),
                'type': 'BULLISH_ENGULFING'
            }
        
        # Bearish Engulfing
        if (prev_close > prev_open and  # 이전 캔들 양봉
            curr_close < curr_open and  # 현재 캔들 음봉
            curr_open > prev_close and  # 현재 시가가 이전 종가보다 높음
            curr_close < prev_open and  # 현재 종가가 이전 시가보다 낮음
            curr_body > prev_body * self.cfg.engulfing_body_ratio):  # 몸통 크기 조건
            
            engulfing_strength = curr_body / prev_body
            
            return {
                'signal': 'SELL',
                'strength': _clamp(engulfing_strength / 3.0, 0.0, 1.0),
                'type': 'BEARISH_ENGULFING'
            }
        
        return {'signal': None, 'strength': 0.0}
    
    def _detect_chart_patterns(self, ltf_df: pd.DataFrame) -> Dict[str, Any]:
        """차트 패턴 감지 (전략 #5) - 하위 시간프레임에서"""
        if len(ltf_df) < self.cfg.pattern_lookback:
            return {'signal': None, 'strength': 0.0}
        
        recent_data = ltf_df.tail(self.cfg.pattern_lookback)
        highs = recent_data['high'].values
        lows = recent_data['low'].values
        
        # Head and Shoulders 패턴 감지 (단순화된 버전)
        if len(highs) >= 9:
            # 3개의 주요 고점 찾기
            peak_indices = []
            for i in range(2, len(highs) - 2):
                if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                    peak_indices.append(i)
            
            if len(peak_indices) >= 3:
                # 가장 높은 고점 (헤드)와 양쪽 어깨
                sorted_peaks = sorted(peak_indices, key=lambda x: highs[x], reverse=True)
                head_idx = sorted_peaks[0]
                
                # 헤드 양쪽의 어깨 찾기
                left_shoulders = [i for i in sorted_peaks[1:] if i < head_idx]
                right_shoulders = [i for i in sorted_peaks[1:] if i > head_idx]
                
                if left_shoulders and right_shoulders:
                    left_shoulder = max(left_shoulders)
                    right_shoulder = min(right_shoulders)
                    
                    # 패턴 품질 확인
                    head_height = highs[head_idx]
                    shoulder_avg = (highs[left_shoulder] + highs[right_shoulder]) / 2
                    
                    if head_height > shoulder_avg * 1.02:  # 헤드가 어깨보다 2% 이상 높음
                        neckline = min(lows[left_shoulder:right_shoulder+1])
                        current_price = float(ltf_df['close'].iloc[-1])
                        
                        if current_price < neckline:  # 네크라인 하향 돌파
                            pattern_strength = (head_height - neckline) / head_height
                            
                            return {
                                'signal': 'SELL',
                                'strength': _clamp(pattern_strength * 2, 0.0, 1.0),
                                'type': 'HEAD_AND_SHOULDERS',
                                'neckline': neckline
                            }
        
        # Flag 패턴 감지
        if len(recent_data) >= 10:
            # 급격한 움직임 후 횡보 패턴 찾기
            price_change = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
            
            if abs(price_change) > 0.02:  # 2% 이상 움직임
                # 최근 절반 구간의 변동성 확인
                recent_half = recent_data.tail(5)
                volatility = (recent_half['high'].max() - recent_half['low'].min()) / recent_half['close'].mean()
                
                if volatility < 0.015:  # 1.5% 미만의 변동성 (횡보)
                    flag_strength = abs(price_change) * (1 / volatility)
                    
                    return {
                        'signal': 'BUY' if price_change > 0 else 'SELL',
                        'strength': _clamp(flag_strength / 5, 0.0, 1.0),
                        'type': 'FLAG_PATTERN'
                    }
        
        return {'signal': None, 'strength': 0.0}
    
    def _calculate_htf_trend_strength(self, htf_df: pd.DataFrame) -> float:
        """상위 시간프레임 추세 강도 계산"""
        if len(htf_df) < 20:
            return 0.0
        
        # EMA 기반 추세 확인
        close = htf_df['close']
        ema_20 = close.ewm(span=20).mean()
        ema_50 = close.ewm(span=50).mean() if len(close) >= 50 else ema_20
        
        current_price = float(close.iloc[-1])
        current_ema20 = float(ema_20.iloc[-1])
        current_ema50 = float(ema_50.iloc[-1])
        
        # EMA 배열과 기울기
        ema_alignment = 0.0
        if current_price > current_ema20 > current_ema50:
            ema_alignment = 1.0  # 강한 상승 추세
        elif current_price < current_ema20 < current_ema50:
            ema_alignment = -1.0  # 강한 하락 추세
        elif current_price > current_ema20:
            ema_alignment = 0.5  # 약한 상승 추세
        elif current_price < current_ema20:
            ema_alignment = -0.5  # 약한 하락 추세
        
        return ema_alignment
    
    def _no_signal_result(self, **kwargs):
        return {
            'name': 'MULTI_TIMEFRAME',
            'action': 'HOLD',
            'score': 0.0,
            'timestamp': self.time_manager.get_current_time(),
            'context': kwargs
        }
    
    def on_kline_close_3m(self) -> Optional[Dict[str, Any]]:
        """3분봉 마감 시 Multi-Timeframe 전략 실행"""
        # 상위 시간프레임 데이터 가져오기
        htf_data = self._get_higher_timeframe_data()
        if htf_data is None or len(htf_data) < 20:
            if self.cfg.debug:
                print("[MTF] HTF 데이터 부족")
            return self._no_signal_result()
        
        # 하위 시간프레임 데이터 가져오기
        data_manager = get_data_manager()
        if data_manager is None:
            return self._no_signal_result()
        
        ltf_data = data_manager.get_latest_data(self.cfg.ltf_lookback)
        if ltf_data is None or len(ltf_data) < 50:
            if self.cfg.debug:
                print("[MTF] LTF 데이터 부족")
            return self._no_signal_result()
        
        # 상위 시간프레임에서 바이어스 결정 (Top-Down 접근)
        htf_trend = self._calculate_htf_trend_strength(htf_data)
        
        # 5가지 전략별 신호 감지
        breakout_signal = self._detect_level_breakout(htf_data, ltf_data)
        bounce_signal = self._detect_level_bounce(htf_data, ltf_data)
        fakeout_signal = self._detect_fakeout(htf_data)
        engulfing_signal = self._detect_engulfing_candlestick(htf_data)
        pattern_signal = self._detect_chart_patterns(ltf_data)
        
        # 가장 강한 신호 선택
        signals = [breakout_signal, bounce_signal, fakeout_signal, engulfing_signal, pattern_signal]
        valid_signals = [s for s in signals if s['signal'] is not None]
        
        if not valid_signals:
            if self.cfg.debug:
                print("[MTF] 유효한 신호 없음")
            return self._no_signal_result()
        
        # 최고 강도 신호 선택
        best_signal = max(valid_signals, key=lambda x: x['strength'])
        
        # HTF 추세와의 일치성 확인
        trend_alignment = 0.0
        if best_signal['signal'] == 'BUY' and htf_trend > 0:
            trend_alignment = htf_trend
        elif best_signal['signal'] == 'SELL' and htf_trend < 0:
            trend_alignment = abs(htf_trend)
        else:
            trend_alignment = 0.3  # 추세와 반대지만 완전히 배제하지는 않음
        
        # 최종 점수 계산
        signal_strength = best_signal.get('strength', 0.0)
        pattern_bonus = 1.0 if 'PATTERN' in best_signal.get('type', '') else 0.5
        
        total_score = (
            self.cfg.w_htf_trend * abs(trend_alignment) +
            self.cfg.w_level_quality * signal_strength +
            self.cfg.w_pattern_strength * pattern_bonus +
            self.cfg.w_ltf_confirmation * 0.8  # LTF 확인 기본값
        )
        
        total_score = _clamp(total_score, 0.0, 1.0)
        
        # 최소 점수 체크
        if total_score < 0.4:
            if self.cfg.debug:
                print(f"[MTF] 점수 부족: {total_score:.3f}")
            return self._no_signal_result()
        
        # 진입/손절/목표가 계산
        current_price = float(ltf_data['close'].iloc[-1])
        atr = get_atr()
        if atr is None:
            close_series = pd.to_numeric(ltf_data['close'].astype(float))
            atr = float(close_series.pct_change().rolling(14).std() * current_price)
        
        action = best_signal['signal']
        
        if action == "BUY":
            entry = current_price + self.cfg.tick
            stop = current_price - self.cfg.atr_stop_mult * float(atr)
        else:  # SELL
            entry = current_price - self.cfg.tick
            stop = current_price + self.cfg.atr_stop_mult * float(atr)
        
        R = abs(entry - stop)
        tp1 = entry + (self.cfg.tp_R1 * R if action == "BUY" else -self.cfg.tp_R1 * R)
        tp2 = entry + (self.cfg.tp_R2 * R if action == "BUY" else -self.cfg.tp_R2 * R)
        
        if self.cfg.debug:
            print(f"[MTF] {action} 신호 - 점수: {total_score:.3f}, "
                  f"전략: {best_signal.get('type', 'UNKNOWN')}, 강도: {best_signal['strength']:.3f}, "
                  f"HTF 추세: {htf_trend:.2f}")
        
        return {
            'name': 'MULTI_TIMEFRAME',
            'action': action,
            'score': float(total_score),
            'entry': float(entry),
            'stop': float(stop),
            'targets': [float(tp1), float(tp2)],
            'timestamp': self.time_manager.get_current_time(),
            'context': {
                'mode': 'MTF_TOP_DOWN',
                'strategy_type': best_signal.get('type', 'UNKNOWN'),
                'signal_strength': float(best_signal['strength']),
                'htf_trend': float(htf_trend),
                'trend_alignment': float(trend_alignment),
                'key_level': float(best_signal.get('level', current_price)),
                'htf_candles': len(htf_data),
                'ltf_candles': len(ltf_data),
                'atr': float(atr)
            }
        }