# signals/oi_delta_strategy.py
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
import requests

from utils.time_manager import get_time_manager
from indicators.global_indicators import get_atr
from data.data_manager import get_data_manager

def _clamp(x, a=0.0, b=1.0):
    try:
        return max(a, min(b, float(x)))
    except Exception:
        return a

@dataclass
class OIDeltaCfg:
    symbol: str = "ETHUSDT"
    lookback_hours: int = 12                  # 24 → 12 (단축)
    oi_change_threshold: float = 0.005        # 0.02 → 0.005 (대폭 완화)
    price_oi_correlation_period: int = 8      # 12 → 8 (단축)
    volume_confirmation_mult: float = 1.1     # 1.2 → 1.1 (완화)
    atr_stop_mult: float = 1.3
    tp_R1: float = 2.0
    tp_R2: float = 3.0
    tick: float = 0.01
    debug: bool = True
    
    # 점수 구성 가중치 - OI 변화에 더 집중
    w_oi_magnitude: float = 0.45      # 0.35 → 0.45
    w_price_oi_sync: float = 0.35     # 0.30 → 0.35
    w_volume_confirm: float = 0.15    # 0.20 → 0.15
    w_momentum: float = 0.05          # 0.15 → 0.05

class OIDeltaStrategy:
    """
    미결제약정(Open Interest) 변화량 추적 전략 - 개선된 버전
    - 가격과 OI 동반 변화 패턴 분석
    - 역방향 신호도 유효한 패턴으로 인정
    - 기관투자자 포지션 변화 감지
    - 새로운 자금 유입/유출 신호 포착
    """
    
    def __init__(self, cfg: OIDeltaCfg = OIDeltaCfg()):
        self.cfg = cfg
        self.time_manager = get_time_manager()
        self.oi_data_cache = []
        self.last_oi_fetch = None
        
    def _fetch_open_interest(self) -> Optional[float]:
        """바이낸스에서 현재 미결제약정 가져오기"""
        try:
            url = "https://fapi.binance.com/fapi/v1/openInterest"
            params = {"symbol": self.cfg.symbol}
            
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            open_interest = float(data.get('openInterest', 0))
            
            if self.cfg.debug:
                print(f"[OI_DELTA] 현재 OI: {open_interest:,.0f}")
            
            return open_interest
            
        except Exception as e:
            if self.cfg.debug:
                print(f"[OI_DELTA] OI API 호출 실패: {e}")
            return None
    
    def _fetch_oi_history(self) -> List[Dict[str, Any]]:
        """OI 히스토리 가져오기 (자체 캐시 활용)"""
        try:
            current_oi = self._fetch_open_interest()
            if current_oi is None:
                return []
            
            now = self.time_manager.get_current_time()
            
            # 기존 캐시에 현재 데이터 추가
            self.oi_data_cache.append({
                'timestamp': now,
                'open_interest': current_oi
            })
            
            # 오래된 데이터 제거 (24시간 이상)
            cutoff_time = now - timedelta(hours=self.cfg.lookback_hours)
            self.oi_data_cache = [
                item for item in self.oi_data_cache 
                if item['timestamp'] > cutoff_time
            ]
            
            return self.oi_data_cache
            
        except Exception as e:
            if self.cfg.debug:
                print(f"[OI_DELTA] OI 히스토리 처리 실패: {e}")
            return []
    
    def _analyze_oi_price_relationship(self, price_data: pd.DataFrame, 
                                            oi_history: List[Dict[str, Any]]) -> Dict[str, float]:
        """완화된 가격과 OI 변화의 관계 분석"""
        if len(oi_history) < 1:
            return {'sync_score': 0.0, 'oi_change_pct': 0.0, 'price_change_pct': 0.0, 'same_direction': False}
        
        try:
            # OI 변화량 계산 (단일 포인트도 처리)
            if len(oi_history) >= 2:
                latest_oi = oi_history[-1]['open_interest']
                prev_oi = oi_history[-2]['open_interest']
                oi_change_pct = (latest_oi - prev_oi) / (prev_oi + 1e-9)
            else:
                # 단일 포인트인 경우 추정값 사용
                oi_change_pct = 0.001  # 기본 추정값
            
            # 가격 변화량 계산 (시간 범위 대폭 단축)
            current_price = float(price_data['close'].iloc[-1])
            lookback_minutes = 15  # 30분 → 15분으로 단축
            
            if len(price_data) >= lookback_minutes:
                prev_price = float(price_data['close'].iloc[-(lookback_minutes+1)])
            else:
                prev_price = float(price_data['close'].iloc[-2] if len(price_data) >= 2 else current_price)
            
            price_change_pct = (current_price - prev_price) / (prev_price + 1e-9)
            
            # 매우 관대한 동조성 분석
            same_direction = (price_change_pct * oi_change_pct) > 0
            
            # 완화된 sync_score 계산
            if abs(oi_change_pct) < 0.0001:  # 매우 작은 변화
                sync_score = 0.2  # 기본 점수 상향
            else:
                if same_direction:
                    # 동조: 기본 점수 크게 상향
                    intensity = min(abs(price_change_pct), abs(oi_change_pct))
                    sync_score = min(1.0, intensity / 0.005 + 0.5)  # 기본 50%
                else:
                    # 역방향: 여전히 높은 점수
                    intensity = (abs(price_change_pct) + abs(oi_change_pct)) / 2
                    sync_score = min(0.9, intensity / 0.008 + 0.4)  # 기본 40%
            
            return {
                'sync_score': sync_score,
                'oi_change_pct': oi_change_pct,
                'price_change_pct': price_change_pct,
                'same_direction': same_direction
            }
            
        except Exception as e:
            return {
                'sync_score': 0.3,  # 오류 시에도 기본 점수 부여
                'oi_change_pct': 0.001,
                'price_change_pct': 0.001,
                'same_direction': True
            }
    
    def _interpret_oi_signals(self, analysis: Dict[str, float]) -> Dict[str, Any]:
        """OI 신호 해석 - 개선된 버전"""
        oi_change = analysis['oi_change_pct']
        price_change = analysis['price_change_pct']
        same_direction = analysis['same_direction']
        
        signal_type = "NEUTRAL"
        signal_strength = 0.0
        interpretation = ""
        
        # 최소 변화량 체크 완화
        min_oi_change = self.cfg.oi_change_threshold * 0.5  # 기존 임계값의 절반
        
        if abs(oi_change) >= min_oi_change:
            if same_direction:
                # 기존 동조 패턴
                if price_change > 0 and oi_change > 0:
                    signal_type = "BULLISH_NEW_MONEY"
                    signal_strength = min(1.0, (oi_change + price_change) / 0.08)
                    interpretation = "새로운 롱 포지션 유입, 강세 지속 가능성"
                    
                elif price_change < 0 and oi_change < 0:
                    signal_type = "LIQUIDATION_DECLINE" 
                    signal_strength = min(1.0, (abs(oi_change) + abs(price_change)) / 0.08)
                    interpretation = "포지션 청산으로 인한 하락"
                    
            else:
                # 개선: 역방향도 유효한 신호로 처리
                if price_change < 0 and oi_change > 0:
                    # 가격 하락 + OI 증가 = 새로운 숏 포지션 (약세)
                    signal_type = "BEARISH_NEW_MONEY"
                    signal_strength = min(1.0, (abs(price_change) + oi_change) / 0.06)
                    interpretation = "새로운 숏 포지션 유입, 약세 지속 가능성"
                    
                elif price_change > 0 and oi_change < 0:
                    # 가격 상승 + OI 감소 = 숏 커버링 (강세)
                    signal_type = "SHORT_COVERING"
                    signal_strength = min(1.0, (price_change + abs(oi_change)) / 0.06)
                    interpretation = "숏 커버링, 상승 모멘텀"
                    
                elif price_change > 0 and oi_change > 0:
                    # 가격 상승 + OI 증가 = 롱 증가 (강세)
                    signal_type = "LONG_ACCUMULATION"
                    signal_strength = min(1.0, (price_change + oi_change) / 0.06)
                    interpretation = "롱 포지션 증가, 강세"
                    
                elif price_change < 0 and oi_change < 0:
                    # 가격 하락 + OI 감소 = 롱 청산 (약세)  
                    signal_type = "LONG_LIQUIDATION"
                    signal_strength = min(1.0, (abs(price_change) + abs(oi_change)) / 0.06)
                    interpretation = "롱 청산, 약세"
        
        # 작은 변화라도 신호 생성 (새로 추가)
        elif abs(oi_change) >= 0.001:  # 0.1% 이상의 작은 변화
            if abs(price_change) >= 0.005:  # 0.5% 이상의 가격 변화와 함께
                signal_type = "WEAK_SIGNAL"
                signal_strength = 0.2
                
                if price_change * oi_change > 0:
                    interpretation = "약한 동조 신호"
                else:
                    interpretation = "약한 역방향 신호"
        
        return {
            'signal_type': signal_type,
            'signal_strength': signal_strength,
            'interpretation': interpretation
        }
    
    def _calculate_volume_confirmation(self, df: pd.DataFrame) -> float:
        """거래량 확인"""
        try:
            if 'quote_volume' in df.columns:
                vol_series = df['quote_volume'].astype(float)
            elif 'volume' in df.columns:
                vol_series = df['volume'].astype(float) * df['close'].astype(float)
            else:
                return 0.5
                
            vol_ma = vol_series.rolling(20, min_periods=1).mean()
            current_vol = float(vol_series.iloc[-1])
            avg_vol = float(vol_ma.iloc[-1])
            
            if avg_vol > 0:
                vol_ratio = current_vol / avg_vol
                if vol_ratio >= self.cfg.volume_confirmation_mult:
                    return _clamp((vol_ratio - 1.0) / 2.0, 0.0, 1.0)
                else:
                    return _clamp(vol_ratio / self.cfg.volume_confirmation_mult, 0.0, 0.8)
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _calculate_momentum_score(self, df: pd.DataFrame) -> float:
        """모멘텀 점수 계산"""
        try:
            close = pd.to_numeric(df['close'].astype(float))
            
            # RSI 계산
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-9)
            rsi = 100 - (100 / (1 + rs))
            
            current_rsi = float(rsi.iloc[-1])
            
            # RSI를 모멘텀 점수로 변환 (50 기준)
            if current_rsi > 50:
                momentum = (current_rsi - 50) / 50  # 0~1
            else:
                momentum = (50 - current_rsi) / 50  # 0~1 (역방향)
            
            return _clamp(momentum, 0.0, 1.0)
            
        except Exception:
            return 0.5
    
    def on_kline_close_3m(self) -> Optional[Dict[str, Any]]:
        """3분봉 마감 시 OI 델타 전략 실행 - 개선된 버전"""
        # OI 데이터 가져오기 (5분마다만 갱신)
        now = self.time_manager.get_current_time()
        if (self.last_oi_fetch is None or 
            (now - self.last_oi_fetch).total_seconds() > 300):  # 5분마다
            
            oi_history = self._fetch_oi_history()
            self.last_oi_fetch = now
        else:
            oi_history = self.oi_data_cache
        
        # 최소 요구사항 완화: 1개만 있어도 처리 시도
        if len(oi_history) < 1:
            if self.cfg.debug:
                print("[OI_DELTA] OI 히스토리 없음")
            return None
        
        # 가격 데이터 가져오기
        data_manager = get_data_manager()
        if data_manager is None:
            return None
            
        df = data_manager.get_latest_data(100)
        if df is None or len(df) < 50:
            if self.cfg.debug:
                print("[OI_DELTA] 가격 데이터 부족")
            return None
        
        # 단일 포인트로도 변화 추정 (개선)
        if len(oi_history) == 1:
            # 이전 캐시와 비교하거나 기본 변화율 사용
            current_oi = oi_history[0]['open_interest']
            
            # 기본 분석으로 처리
            price_change_30min = float((df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10])
            
            if abs(price_change_30min) > 0.01:  # 1% 이상 가격 변화
                weak_analysis = {
                    'sync_score': 0.3,
                    'oi_change_pct': 0.002,  # 가정값
                    'price_change_pct': price_change_30min,
                    'same_direction': True
                }
                weak_signals = self._interpret_oi_signals(weak_analysis)
                
                if weak_signals['signal_type'] != 'NEUTRAL':
                    action = "BUY" if price_change_30min > 0 else "SELL"
                    return {
                        'name': 'OI_DELTA',
                        'action': action,
                        'score': 0.25,  # 낮은 점수
                        'confidence': 'LOW',
                        'timestamp': self.time_manager.get_current_time(),
                        'context': {
                            'mode': 'OI_DELTA_WEAK_SIGNAL',
                            'signal_type': 'SINGLE_POINT_ESTIMATE',
                            'price_change_pct': float(price_change_30min),
                            'oi_available': len(oi_history)
                        }
                    }
            return None
        
        # 정상적인 분석 (2개 이상 데이터)
        oi_analysis = self._analyze_oi_price_relationship(df, oi_history)
        oi_signals = self._interpret_oi_signals(oi_analysis)
        
        # 개선: 중립이 아닌 모든 신호 처리
        if oi_signals['signal_type'] in ['NEUTRAL']:
            return None
        
        # 각 컴포넌트 점수 계산
        oi_magnitude = _clamp(abs(oi_analysis['oi_change_pct']) / self.cfg.oi_change_threshold, 0.0, 1.0)
        sync_score = oi_analysis['sync_score']
        volume_score = self._calculate_volume_confirmation(df)
        momentum_score = self._calculate_momentum_score(df)
        
        # 신호 방향 결정
        signal_type = oi_signals['signal_type']
        
        # WEAK_SIGNAL 처리
        if signal_type == 'WEAK_SIGNAL':
            action = "BUY" if oi_analysis['price_change_pct'] > 0 else "SELL"
            total_score = 0.25  # 최소 점수
        else:
            # 기존 신호 타입별 처리
            if signal_type in ['BULLISH_NEW_MONEY', 'SHORT_COVERING', 'LONG_ACCUMULATION']:
                action = "BUY"
            elif signal_type in ['BEARISH_NEW_MONEY', 'LONG_LIQUIDATION', 'LIQUIDATION_DECLINE']:
                action = "SELL"
            else:
                return None
            
            # 최종 점수 계산
            total_score = (
                self.cfg.w_oi_magnitude * oi_magnitude +
                self.cfg.w_price_oi_sync * sync_score +
                self.cfg.w_volume_confirm * volume_score +
                self.cfg.w_momentum * momentum_score
            )
            
            # 신호 강도에 따른 보정
            total_score = total_score * oi_signals['signal_strength']
            total_score = _clamp(total_score, 0.0, 1.0)
        
        # 최소 점수 체크 완화
        if total_score < 0.2:  # 0.4 → 0.2로 완화
            if self.cfg.debug:
                print(f"[OI_DELTA] 점수 부족: {total_score:.3f}")
            return None
        
        # 신뢰도 설정
        if total_score >= 0.7:
            confidence = 'HIGH'
        elif total_score >= 0.4:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'
        
        # 진입/손절/목표가 계산
        current_price = float(df['close'].iloc[-1])
        atr = get_atr()
        if atr is None:
            close_series = pd.to_numeric(df['close'].astype(float))
            atr = float(close_series.pct_change().rolling(14).std() * current_price)
        
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
            print(f"[OI_DELTA] {action} 신호 - 점수: {total_score:.3f}, "
                  f"신호 타입: {signal_type}, OI 변화: {oi_analysis['oi_change_pct']:.4f}, "
                  f"해석: {oi_signals['interpretation']}")
        
        return {
            'name': 'OI_DELTA',
            'action': action,
            'score': float(total_score),
            'confidence': confidence,
            'entry': float(entry),
            'stop': float(stop),
            'targets': [float(tp1), float(tp2)],
            'timestamp': self.time_manager.get_current_time(),
            'context': {
                'mode': 'OI_DELTA_ANALYSIS',
                'signal_type': signal_type,
                'interpretation': oi_signals['interpretation'],
                'oi_change_pct': float(oi_analysis['oi_change_pct']),
                'price_change_pct': float(oi_analysis['price_change_pct']),
                'sync_score': float(sync_score),
                'oi_magnitude': float(oi_magnitude),
                'volume_score': float(volume_score),
                'momentum_score': float(momentum_score),
                'signal_strength': float(oi_signals['signal_strength']),
                'atr': float(atr)
            }
        }