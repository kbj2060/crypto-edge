# signals/EMA_CONFLUENCE.py
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np

from managers.data_manager import get_data_manager
from indicators.global_indicators import get_atr
from utils.time_manager import get_time_manager

def _clamp(x, a=0.0, b=1.0):
    try:
        return max(a, min(b, float(x)))
    except Exception:
        return a

@dataclass
class EMAConfluenceCfg:
    # EMA 설정 (15분봉 기준)
    ema_periods: List[int] = None  # [20, 50, 100] - 5시간, 12.5시간, 25시간
    lookback_bars: int = 120       # 30시간 데이터
    
    # 수렴/확산 감지 설정
    confluence_threshold: float = 0.015   # 1.5% 이내 수렴
    divergence_threshold: float = 0.025   # 2.5% 이상 확산
    slope_period: int = 6                 # 1.5시간 기울기 계산
    
    # 신호 조건
    price_ema_distance: float = 0.008     # 가격-EMA 거리 0.8%
    volume_confirmation_mult: float = 1.2 # 거래량 확인
    min_confluence_score: float = 0.3     # 최소 수렴 점수
    
    # 손익 설정
    atr_stop_mult: float = 1.8
    tp_R1: float = 2.5
    tp_R2: float = 4.0
    tick: float = 0.01
    debug: bool = False
    
    # 점수 구성 가중치
    w_confluence: float = 0.40       # 수렴 점수
    w_alignment: float = 0.25        # EMA 정렬 점수
    w_price_position: float = 0.20   # 가격 위치 점수
    w_slope: float = 0.15           # 기울기 점수

    def __post_init__(self):
        if self.ema_periods is None:
            self.ema_periods = [20, 50, 100]

class EMAConfluence:
    """
    EMA Confluence 전략 (15분봉 최적화)
    - 여러 EMA의 수렴/확산 패턴 감지
    - EMA 정렬 상태와 가격 위치 분석
    - 수렴 지점에서의 방향성 돌파 포착
    """
    
    def __init__(self, cfg: EMAConfluenceCfg = EMAConfluenceCfg()):
        self.cfg = cfg
        self.time_manager = get_time_manager()
        
    def _calculate_emas(self, close: pd.Series) -> Dict[str, pd.Series]:
        """EMA 계산"""
        emas = {}
        for period in self.cfg.ema_periods:
            emas[f'ema_{period}'] = close.ewm(span=period, adjust=False).mean()
        return emas
    
    def _detect_confluence(self, emas: Dict[str, pd.Series], current_price: float) -> Dict[str, float]:
        """EMA 수렴 상태 감지"""
        ema_values = []
        for key, ema_series in emas.items():
            if len(ema_series) > 0 and not pd.isna(ema_series.iloc[-1]):
                ema_values.append(float(ema_series.iloc[-1]))
        
        if len(ema_values) < 2:
            return {'confluence_score': 0.0, 'max_spread_pct': 0.0, 'avg_ema': current_price}
        
        # EMA 평균과 최대/최소 스프레드 계산
        avg_ema = np.mean(ema_values)
        max_ema = max(ema_values)
        min_ema = min(ema_values)
        
        # 스프레드를 평균 대비 퍼센트로 계산
        spread_pct = (max_ema - min_ema) / (avg_ema + 1e-9)
        
        # 수렴 점수: 스프레드가 작을수록 높은 점수
        if spread_pct <= self.cfg.confluence_threshold:
            confluence_score = 1.0 - (spread_pct / self.cfg.confluence_threshold)
        else:
            confluence_score = 0.0
        
        confluence_score = _clamp(confluence_score, 0.0, 1.0)
        
        if self.cfg.debug:
            print(f"[EMA_CONFLUENCE] 수렴 분석: 스프레드={spread_pct:.4f}, 점수={confluence_score:.3f}")
        
        return {
            'confluence_score': confluence_score,
            'max_spread_pct': spread_pct,
            'avg_ema': avg_ema
        }
    
    def _analyze_ema_alignment(self, emas: Dict[str, pd.Series]) -> Dict[str, Any]:
        """EMA 정렬 상태 분석"""
        if len(emas) < 2:
            return {'alignment_score': 0.0, 'trend_direction': 'NEUTRAL'}
        
        # 현재 EMA 값들을 정렬된 기간 순으로 가져오기
        sorted_periods = sorted(self.cfg.ema_periods)
        current_emas = []
        
        for period in sorted_periods:
            ema_key = f'ema_{period}'
            if ema_key in emas and len(emas[ema_key]) > 0:
                current_emas.append(float(emas[ema_key].iloc[-1]))
        
        if len(current_emas) < 2:
            return {'alignment_score': 0.0, 'trend_direction': 'NEUTRAL'}
        
        # 상승 정렬 체크 (빠른 EMA > 느린 EMA)
        bullish_alignment = all(current_emas[i] >= current_emas[i+1] for i in range(len(current_emas)-1))
        
        # 하락 정렬 체크 (빠른 EMA < 느린 EMA)  
        bearish_alignment = all(current_emas[i] <= current_emas[i+1] for i in range(len(current_emas)-1))
        
        if bullish_alignment:
            alignment_score = 1.0
            trend_direction = 'BULLISH'
        elif bearish_alignment:
            alignment_score = 1.0
            trend_direction = 'BEARISH'
        else:
            # 부분 정렬 점수 계산
            bullish_pairs = sum(1 for i in range(len(current_emas)-1) if current_emas[i] >= current_emas[i+1])
            bearish_pairs = sum(1 for i in range(len(current_emas)-1) if current_emas[i] <= current_emas[i+1])
            
            max_pairs = len(current_emas) - 1
            if bullish_pairs > bearish_pairs:
                alignment_score = bullish_pairs / max_pairs
                trend_direction = 'BULLISH'
            elif bearish_pairs > bullish_pairs:
                alignment_score = bearish_pairs / max_pairs
                trend_direction = 'BEARISH'
            else:
                alignment_score = 0.0
                trend_direction = 'NEUTRAL'
        
        if self.cfg.debug:
            print(f"[EMA_CONFLUENCE] 정렬 분석: 방향={trend_direction}, 점수={alignment_score:.3f}")
        
        return {
            'alignment_score': alignment_score,
            'trend_direction': trend_direction,
            'ema_values': current_emas
        }
    
    def _calculate_price_position_score(self, current_price: float, emas: Dict[str, pd.Series]) -> float:
        """가격과 EMA들의 위치 관계 점수"""
        ema_values = []
        for ema_series in emas.values():
            if len(ema_series) > 0 and not pd.isna(ema_series.iloc[-1]):
                ema_values.append(float(ema_series.iloc[-1]))
        
        if not ema_values:
            return 0.0
        
        avg_ema = np.mean(ema_values)
        
        # 가격과 EMA 평균 간의 거리
        distance_pct = abs(current_price - avg_ema) / (avg_ema + 1e-9)
        
        # 거리 점수: 가까울수록 높은 점수, 너무 멀면 낮은 점수
        if distance_pct <= self.cfg.price_ema_distance:
            position_score = 1.0 - (distance_pct / self.cfg.price_ema_distance)
        else:
            position_score = max(0.0, 1.0 - (distance_pct / (self.cfg.price_ema_distance * 3)))
        
        return _clamp(position_score, 0.0, 1.0)
    
    def _calculate_slope_score(self, emas: Dict[str, pd.Series]) -> Dict[str, float]:
        """EMA 기울기 점수 계산"""
        if not emas:
            return {'slope_score': 0.0, 'slope_direction': 'FLAT'}
        
        # 가장 빠른 EMA의 기울기 계산
        fastest_ema_key = f'ema_{min(self.cfg.ema_periods)}'
        if fastest_ema_key not in emas:
            return {'slope_score': 0.0, 'slope_direction': 'FLAT'}
        
        fastest_ema = emas[fastest_ema_key]
        if len(fastest_ema) < self.cfg.slope_period + 1:
            return {'slope_score': 0.0, 'slope_direction': 'FLAT'}
        
        # 기울기 계산 (최근 구간 vs 이전 구간)
        recent_avg = fastest_ema.tail(self.cfg.slope_period).mean()
        prev_avg = fastest_ema.iloc[-(self.cfg.slope_period*2):-self.cfg.slope_period].mean()
        
        if pd.isna(recent_avg) or pd.isna(prev_avg) or prev_avg == 0:
            return {'slope_score': 0.0, 'slope_direction': 'FLAT'}
        
        slope_pct = (recent_avg - prev_avg) / prev_avg
        
        # 기울기 점수와 방향
        slope_score = _clamp(abs(slope_pct) / 0.01, 0.0, 1.0)  # 1% 변화 시 최대 점수
        
        if slope_pct > 0.002:  # 0.2% 이상 상승
            slope_direction = 'UP'
        elif slope_pct < -0.002:  # 0.2% 이상 하락
            slope_direction = 'DOWN'
        else:
            slope_direction = 'FLAT'
        
        return {
            'slope_score': slope_score,
            'slope_direction': slope_direction
        }
    
    def _calculate_volume_confirmation(self, df: pd.DataFrame) -> float:
        """거래량 확인"""
        try:
            vol_series = df['quote_volume'].astype(float)
                
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
    
    def _no_signal_result(self, **kwargs):
        return {
            'name': 'EMA_CONFLUENCE',
            'action': 'HOLD',
            'score': 0.0,
            'timestamp': self.time_manager.get_current_time(),
            'context': kwargs
        }
    
    def on_kline_close_15m(self) -> Optional[Dict[str, Any]]:
        """15분봉 마감 시 EMA Confluence 전략 실행"""
        data_manager = get_data_manager()
        if data_manager is None:
            return self._no_signal_result()
            
        df = data_manager.get_latest_data_15m(self.cfg.lookback_bars + 50)
        if df is None or len(df) < max(self.cfg.ema_periods) + 20:
            if self.cfg.debug:
                print(f"[EMA_CONFLUENCE] 데이터 부족: 필요={max(self.cfg.ema_periods) + 20}, "
                      f"실제={len(df) if df is not None else 'None'}")
            return self._no_signal_result()
        
        close = pd.to_numeric(df['close'].astype(float))
        current_price = float(close.iloc[-1])
        
        # EMA 계산
        emas = self._calculate_emas(close)
        
        # 수렴 상태 분석
        confluence_analysis = self._detect_confluence(emas, current_price)
        confluence_score = confluence_analysis['confluence_score']
        
        # 최소 수렴 점수 체크
        if confluence_score < self.cfg.min_confluence_score:
            if self.cfg.debug:
                print(f"[EMA_CONFLUENCE] 수렴 부족: {confluence_score:.3f} < {self.cfg.min_confluence_score}")
            return self._no_signal_result()
        
        # EMA 정렬 분석
        alignment_analysis = self._analyze_ema_alignment(emas)
        alignment_score = alignment_analysis['alignment_score']
        trend_direction = alignment_analysis['trend_direction']
        
        # 중립 상태면 신호 없음
        if trend_direction == 'NEUTRAL':
            if self.cfg.debug:
                print("[EMA_CONFLUENCE] 중립 상태")
            return self._no_signal_result()
        
        # 가격 위치 점수
        position_score = self._calculate_price_position_score(current_price, emas)
        
        # 기울기 점수
        slope_analysis = self._calculate_slope_score(emas)
        slope_score = slope_analysis['slope_score']
        slope_direction = slope_analysis['slope_direction']
        
        # 거래량 확인
        volume_score = self._calculate_volume_confirmation(df)
        
        # 신호 방향 결정
        if trend_direction == 'BULLISH' and slope_direction in ['UP', 'FLAT']:
            action = "BUY"
        elif trend_direction == 'BEARISH' and slope_direction in ['DOWN', 'FLAT']:
            action = "SELL"
        else:
            if self.cfg.debug:
                print(f"[EMA_CONFLUENCE] 추세-기울기 불일치: {trend_direction} vs {slope_direction}")
            return self._no_signal_result()
        
        # 최종 점수 계산
        total_score = (
            self.cfg.w_confluence * confluence_score +
            self.cfg.w_alignment * alignment_score +
            self.cfg.w_price_position * position_score +
            self.cfg.w_slope * slope_score
        )
        
        total_score = _clamp(total_score, 0.0, 1.0)
        
        # 최소 점수 체크
        if total_score < 0.4:
            if self.cfg.debug:
                print(f"[EMA_CONFLUENCE] 점수 부족: {total_score:.3f} < 0.4")
            return self._no_signal_result()
        
        # 진입/손절/목표가 계산
        atr = get_atr()
        if atr is None:
            atr = float(close.pct_change().rolling(14).std() * current_price)
        
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
            print(f"[EMA_CONFLUENCE] {action} 신호 - 점수: {total_score:.3f}, "
                  f"수렴: {confluence_score:.3f}, 정렬: {alignment_score:.3f}, "
                  f"위치: {position_score:.3f}, 기울기: {slope_score:.3f}")
        
        return {
            'name': 'EMA_CONFLUENCE',
            'action': action,
            'score': float(total_score),
            'entry': float(entry),
            'stop': float(stop),
            'targets': [float(tp1), float(tp2)],
            'timestamp': self.time_manager.get_current_time(),
            'context': {
                'mode': 'EMA_CONFLUENCE_PATTERN',
                'confluence_score': float(confluence_score),
                'alignment_score': float(alignment_score),
                'trend_direction': trend_direction,
                'position_score': float(position_score),
                'slope_score': float(slope_score),
                'slope_direction': slope_direction,
                'volume_score': float(volume_score),
                'max_spread_pct': float(confluence_analysis['max_spread_pct']),
                'avg_ema': float(confluence_analysis['avg_ema']),
                'ema_periods': self.cfg.ema_periods,
                'atr': float(atr)
            }
        }