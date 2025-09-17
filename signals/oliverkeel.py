# signals/oliver_keel_strategy.py
from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd

from managers.data_manager import get_data_manager
from indicators.global_indicators import get_atr
from utils.time_manager import get_time_manager

def _clamp(x, a=0.0, b=1.0):
    try:
        return max(a, min(b, float(x)))
    except Exception:
        return a

@dataclass
class OliverKeelCfg:
    ma_fast: int = 6                    # 빠른 이평선 (흰색)
    ma_medium: int = 20                 # 중간 이평선 (파란색)  
    ma_slow: int = 200                  # 느린 이평선 (녹색)
    lookback_bars: int = 250            # 200MA 계산을 위해 충분한 데이터
    
    # 떠있는 캔들 조건
    gap_threshold: float = 0.003        # 이평선과의 최소 간격 (0.3%)
    wick_ratio_limit: float = 0.4       # 아래꼬리/윗꼬리 최대 비율 (40%)
    floating_candle_lookback: int = 10  # 크로스 이후 떠있는 캔들 찾는 범위
    retracement_lookback: int = 5       # 되돌림 확인 범위
    
    # 손익 설정
    atr_stop_mult: float = 1.5
    tp_R1: float = 2.5                  # BTC 100-200불 수준
    tp_R2: float = 4.0                  # BTC 300-400불 수준
    tick: float = 0.01
    debug: bool = True
    
    # 점수 구성 가중치
    w_cross_strength: float = 0.40      # 크로스 강도
    w_floating_quality: float = 0.30    # 떠있는 캔들 품질
    w_ma200_filter: float = 0.20        # 200MA 필터 보너스
    w_volume_confirm: float = 0.10      # 거래량 확인

class OliverKeelStrategy:
    """
    Oliver Keel 단타 필승 전략 (1000% 수익률 달성)
    - 6, 20, 200 이평선 사용
    - 골든/데드크로스 → 떠있는 캔들 → 되돌림 진입
    - 200MA 위/아래 필터로 성공률 극대화
    """
    
    def __init__(self, cfg: OliverKeelCfg = OliverKeelCfg()):
        self.cfg = cfg
        self.time_manager = get_time_manager()
        
    def _calculate_moving_averages(self, close: pd.Series) -> Dict[str, pd.Series]:
        """이동평균선 계산"""
        ma_6 = close.rolling(window=self.cfg.ma_fast).mean()
        ma_20 = close.rolling(window=self.cfg.ma_medium).mean()  
        ma_200 = close.rolling(window=self.cfg.ma_slow).mean()
        
        return {
            'ma_6': ma_6,
            'ma_20': ma_20,
            'ma_200': ma_200
        }
    
    def _detect_cross_signal(self, ma_6: pd.Series, ma_20: pd.Series) -> Dict[str, float]:
        """골든/데드크로스 감지 및 강도 계산"""
        if len(ma_6) < 3:
            return {'golden_cross': 0.0, 'death_cross': 0.0}
        
        # 크로스 감지
        ma6_above_ma20_prev = ma_6.iloc[-2] <= ma_20.iloc[-2]
        ma6_above_ma20_curr = ma_6.iloc[-1] > ma_20.iloc[-1]
        golden_cross = ma6_above_ma20_prev and ma6_above_ma20_curr
        
        ma6_below_ma20_prev = ma_6.iloc[-2] >= ma_20.iloc[-2]
        ma6_below_ma20_curr = ma_6.iloc[-1] < ma_20.iloc[-1]
        death_cross = ma6_below_ma20_prev and ma6_below_ma20_curr
        
        # 크로스 강도 계산 (이평선 간격 기반)
        cross_strength = 0.0
        if golden_cross or death_cross:
            ma_gap = abs(ma_6.iloc[-1] - ma_20.iloc[-1])
            price_base = ma_20.iloc[-1]
            cross_strength = _clamp(ma_gap / (price_base * 0.001), 0.0, 1.0)
        
        return {
            'golden_cross': cross_strength if golden_cross else 0.0,
            'death_cross': cross_strength if death_cross else 0.0
        }
    
    def _identify_floating_candle(self, df: pd.DataFrame, ma_6: pd.Series, 
                                    start_idx: int, direction: str) -> Dict[str, Any]:
        """떠있는 캔들 패턴 감지"""
        floating_score = 0.0
        floating_idx = -1
        
        # 크로스 이후 최대 10봉까지 검색
        end_idx = min(start_idx + self.cfg.floating_candle_lookback, len(df))
        
        for i in range(start_idx + 1, end_idx):
            if i >= len(df):
                break
                
            candle = df.iloc[i]
            ma6_value = ma_6.iloc[i]
            
            # 캔들 정보
            open_price = float(candle['open'])
            high_price = float(candle['high'])
            low_price = float(candle['low'])
            close_price = float(candle['close'])
            
            body_size = abs(close_price - open_price)
            upper_wick = high_price - max(open_price, close_price)
            lower_wick = min(open_price, close_price) - low_price
            
            if direction == "long":
                # 롱: 윗꼬리 있고, 짧은 아래꼬리, 6MA 위에 떨어져 있음
                has_upper_wick = upper_wick > body_size * 0.1
                short_lower_wick = lower_wick <= body_size * self.cfg.wick_ratio_limit
                gap_from_ma6 = low_price > ma6_value * (1 + self.cfg.gap_threshold)
                
                if has_upper_wick and short_lower_wick and gap_from_ma6:
                    # 품질 점수 계산
                    gap_score = _clamp((low_price / ma6_value - 1) / self.cfg.gap_threshold, 0.0, 1.0)
                    wick_score = _clamp(upper_wick / body_size, 0.0, 1.0)
                    floating_score = (gap_score + wick_score) / 2.0
                    floating_idx = i
                    break
                    
            else:  # short
                # 숏: 아래꼬리 있고, 짧은 윗꼬리, 6MA 아래에 떨어져 있음
                has_lower_wick = lower_wick > body_size * 0.1
                short_upper_wick = upper_wick <= body_size * self.cfg.wick_ratio_limit
                gap_from_ma6 = high_price < ma6_value * (1 - self.cfg.gap_threshold)
                
                if has_lower_wick and short_upper_wick and gap_from_ma6:
                    # 품질 점수 계산
                    gap_score = _clamp((1 - high_price / ma6_value) / self.cfg.gap_threshold, 0.0, 1.0)
                    wick_score = _clamp(lower_wick / body_size, 0.0, 1.0)
                    floating_score = (gap_score + wick_score) / 2.0
                    floating_idx = i
                    break
        
        return {
            'score': floating_score,
            'index': floating_idx,
            'found': floating_score > 0.0
        }
    
    def _check_retracement_entry(self, df: pd.DataFrame, ma_dict: Dict[str, pd.Series],
                                floating_idx: int, direction: str) -> Dict[str, Any]:
        """되돌림 진입 조건 확인"""
        if floating_idx == -1 or floating_idx >= len(df) - 1:
            return {'found': False, 'entry_idx': -1, 'entry_level': None}
        
        floating_candle = df.iloc[floating_idx]
        floating_high = float(floating_candle['high'])
        floating_low = float(floating_candle['low'])
        
        # 떠있는 캔들 이후 최대 5개 봉 확인
        end_idx = min(floating_idx + self.cfg.retracement_lookback + 1, len(df))
        
        for i in range(floating_idx + 1, end_idx):
            current_candle = df.iloc[i]
            current_high = float(current_candle['high'])
            current_low = float(current_candle['low'])
            current_close = float(current_candle['close'])
            
            ma6_value = float(ma_dict['ma_6'].iloc[i])
            ma20_value = float(ma_dict['ma_20'].iloc[i])
            
            if direction == "long":
                # 롱: 고점이 떠있는 캔들보다 낮고, 6MA나 20MA 터치
                lower_high = current_high < floating_high
                
                touches_ma6 = (current_low <= ma6_value * 1.005 and 
                             current_close >= ma6_value * 0.995)
                touches_ma20 = (current_low <= ma20_value * 1.005 and 
                               current_close >= ma20_value * 0.995)
                
                if lower_high and (touches_ma6 or touches_ma20):
                    entry_level = ma6_value if touches_ma6 else ma20_value
                    return {
                        'found': True,
                        'entry_idx': i,
                        'entry_level': entry_level,
                        'touch_type': 'MA6' if touches_ma6 else 'MA20'
                    }
                    
            else:  # short
                # 숏: 저점이 떠있는 캔들보다 높고, 6MA나 20MA 터치
                higher_low = current_low > floating_low
                
                touches_ma6 = (current_high >= ma6_value * 0.995 and 
                             current_close <= ma6_value * 1.005)
                touches_ma20 = (current_high >= ma20_value * 0.995 and 
                               current_close <= ma20_value * 1.005)
                
                if higher_low and (touches_ma6 or touches_ma20):
                    entry_level = ma6_value if touches_ma6 else ma20_value
                    return {
                        'found': True,
                        'entry_idx': i,
                        'entry_level': entry_level,
                        'touch_type': 'MA6' if touches_ma6 else 'MA20'
                    }
        
        return {'found': False, 'entry_idx': -1, 'entry_level': None}
    
    def _calculate_ma200_filter(self, close_price: float, ma_200: float, direction: str) -> float:
        """200MA 필터 보너스 점수"""
        if direction == "long":
            # 롱: 200MA 위에서 진행시 보너스
            return 1.0 if close_price > ma_200 else 0.5
        else:  # short
            # 숏: 200MA 아래에서 진행시 보너스
            return 1.0 if close_price < ma_200 else 0.5
    
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
    
    def _no_signal_result(self, **kwargs):
        return {
            'name': 'OLIVER_KEEL',
            'action': 'HOLD',
            'score': 0.0,
            'timestamp': self.time_manager.get_current_time(),
            'context': kwargs
        }
    
    def on_kline_close_15m(self) -> Optional[Dict[str, Any]]:
        """3분봉 마감 시 Oliver Keel 전략 실행"""
        data_manager = get_data_manager()
        if data_manager is None:
            return self._no_signal_result()
            
        df = data_manager.get_latest_data_15m(self.cfg.lookback_bars + 50)
        if df is None or len(df) < self.cfg.ma_slow + 20:
            if self.cfg.debug:
                print(f"[OLIVER_KEEL] 데이터 부족: 필요={self.cfg.ma_slow + 20}, "
                      f"실제={len(df) if df is not None else 'None'}")
            return self._no_signal_result()
        
        close = pd.to_numeric(df['close'].astype(float))
        
        # 이동평균선 계산
        ma_dict = self._calculate_moving_averages(close)
        ma_6 = ma_dict['ma_6']
        ma_20 = ma_dict['ma_20']
        ma_200 = ma_dict['ma_200']
        
        # 현재 값들
        current_close = float(close.iloc[-1])
        current_ma200 = float(ma_200.iloc[-1])
        
        # 최근 10봉 내에서 크로스 검색
        cross_found = False
        cross_direction = ""
        cross_strength = 0.0
        
        for i in range(max(0, len(df) - 10), len(df)):
            cross_signals = self._detect_cross_signal(ma_6.iloc[:i+1], ma_20.iloc[:i+1])
            
            if cross_signals['golden_cross'] > 0:
                cross_found = True
                cross_direction = "long"
                cross_strength = cross_signals['golden_cross']
                cross_idx = i
                break
            elif cross_signals['death_cross'] > 0:
                cross_found = True
                cross_direction = "short"
                cross_strength = cross_signals['death_cross']
                cross_idx = i
                break
        
        if not cross_found:
            if self.cfg.debug:
                print("[OLIVER_KEEL] 최근 크로스 신호 없음")
            return self._no_signal_result()
        
        # 떠있는 캔들 찾기
        floating_result = self._identify_floating_candle(df, ma_6, cross_idx, cross_direction)
        
        if not floating_result['found']:
            if self.cfg.debug:
                print(f"[OLIVER_KEEL] 떠있는 캔들 없음 ({cross_direction})")
            return self._no_signal_result()
        
        # 되돌림 진입점 확인
        entry_result = self._check_retracement_entry(df, ma_dict, floating_result['index'], cross_direction)
        
        if not entry_result['found']:
            if self.cfg.debug:
                print(f"[OLIVER_KEEL] 되돌림 진입점 없음 ({cross_direction})")
            return self._no_signal_result()
        
        # 각 컴포넌트 점수 계산
        floating_score = floating_result['score']
        ma200_score = self._calculate_ma200_filter(current_close, current_ma200, cross_direction)
        volume_score = self._calculate_volume_confirmation(df)
        
        # 신호 방향 결정
        action = "BUY" if cross_direction == "long" else "SELL"
        
        # 최종 점수 계산
        total_score = (
            self.cfg.w_cross_strength * cross_strength +
            self.cfg.w_floating_quality * floating_score +
            self.cfg.w_ma200_filter * ma200_score +
            self.cfg.w_volume_confirm * volume_score
        )
        
        total_score = _clamp(total_score, 0.0, 1.0)
        
        # 최소 점수 체크
        min_score_threshold = 0.2
        if total_score < min_score_threshold:
            if self.cfg.debug:
                print(f"[OLIVER_KEEL] 점수 부족: {total_score:.3f} < {min_score_threshold}")
            return self._no_signal_result()
        
        # 진입/손절/목표가 계산
        entry_price = float(entry_result['entry_level'])
        atr = get_atr()
        if atr is None:
            atr = float(close.pct_change().rolling(14).std() * current_close)
        
        if action == "BUY":
            entry = entry_price + self.cfg.tick
            stop = entry_price - self.cfg.atr_stop_mult * float(atr)
        else:  # SELL
            entry = entry_price - self.cfg.tick
            stop = entry_price + self.cfg.atr_stop_mult * float(atr)
        
        R = abs(entry - stop)
        tp1 = entry + (self.cfg.tp_R1 * R if action == "BUY" else -self.cfg.tp_R1 * R)
        tp2 = entry + (self.cfg.tp_R2 * R if action == "BUY" else -self.cfg.tp_R2 * R)
        
        if self.cfg.debug:
            print(f"[OLIVER_KEEL] {action} 신호 - 점수: {total_score:.3f}, "
                  f"크로스 강도: {cross_strength:.3f}, 떠있는 캔들: {floating_score:.3f}, "
                  f"200MA 필터: {ma200_score:.3f}, 거래량: {volume_score:.3f}")
            print(f"[OLIVER_KEEL] 진입: {entry:.2f}, 손절: {stop:.2f}, "
                  f"익절1: {tp1:.2f}, 익절2: {tp2:.2f}")
        
        return {
            'name': 'OLIVER_KEEL',
            'action': action,
            'score': float(total_score),
            'entry': float(entry),
            'stop': float(stop),
            'targets': [float(tp1), float(tp2)],
            'timestamp': self.time_manager.get_current_time(),
            'context': {
                'mode': 'OLIVER_KEEL_PATTERN',
                'cross_direction': cross_direction,
                'cross_strength': float(cross_strength),
                'floating_score': float(floating_score),
                'ma200_filter': float(ma200_score),
                'volume_score': float(volume_score),
                'entry_level': float(entry_result['entry_level']),
                'touch_type': entry_result.get('touch_type', ''),
                'above_ma200': current_close > current_ma200,
                'ma_values': {
                    'ma_6': float(ma_6.iloc[-1]),
                    'ma_20': float(ma_20.iloc[-1]),
                    'ma_200': float(current_ma200)
                },
                'atr': float(atr)
            }
        }