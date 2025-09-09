# signals/funding_rate_strategy.py
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
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
class FundingRateCfg:
    symbol: str = "ETHUSDT"
    extreme_funding_threshold: float = 0.002    # 0.005 → 0.002 (완화)
    moderate_funding_threshold: float = 0.0005  # 0.001 → 0.0005 (완화)
    funding_ma_period: int = 12                 # 24 → 12 (더 민감하게)
    lookback_hours: int = 48                    # 72 → 48 (단축)
    sentiment_multiplier: float = 1.5           # 2.0 → 1.5 (완화)
    atr_stop_mult: float = 1.5
    tp_R1: float = 2.0
    tp_R2: float = 3.5
    tick: float = 0.01
    debug: bool = True
    
    # 점수 구성 가중치 - 펀딩비율에 더 집중
    w_funding_extreme: float = 0.50    # 0.40 → 0.50
    w_funding_trend: float = 0.30      # 0.25 → 0.30
    w_volume_confirm: float = 0.15     # 0.20 → 0.15
    w_price_momentum: float = 0.05     # 0.15 → 0.05
class FundingRateStrategy:
    """
    펀딩비율 기반 시장 심리 전략
    - 극단적 펀딩비에서 반전 신호 포착
    - 롱/숏 과열 상태 감지
    - 시장 심리 역추세 매매
    """
    
    def __init__(self, cfg: FundingRateCfg = FundingRateCfg()):
        self.cfg = cfg
        self.time_manager = get_time_manager()
        self.funding_data_cache = []
        self.last_fetch_time = None
        
    def _fetch_funding_rate(self) -> Optional[float]:
        """바이낸스에서 현재 펀딩비율 가져오기"""
        try:
            # 바이낸스 Futures API
            url = "https://fapi.binance.com/fapi/v1/premiumIndex"
            params = {"symbol": self.cfg.symbol}
            
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            funding_rate = float(data.get('lastFundingRate', 0))
            
            if self.cfg.debug:
                print(f"[FUNDING_RATE] 현재 펀딩비율: {funding_rate:.6f} ({funding_rate*100:.4f}%)")
            
            return funding_rate
            
        except Exception as e:
            if self.cfg.debug:
                print(f"[FUNDING_RATE] API 호출 실패: {e}")
            return None
    
    def _fetch_funding_history(self) -> List[Dict[str, Any]]:
        """펀딩비율 히스토리 가져오기"""
        try:
            url = "https://fapi.binance.com/fapi/v1/fundingRate"
            
            # 시간 범위 설정
            end_time = int(self.time_manager.get_current_time().timestamp() * 1000)
            start_time = end_time - (self.cfg.lookback_hours * 3600 * 1000)
            
            params = {
                "symbol": self.cfg.symbol,
                "startTime": start_time,
                "endTime": end_time,
                "limit": 100
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            funding_history = []
            
            for item in data:
                funding_history.append({
                    'timestamp': datetime.fromtimestamp(int(item['fundingTime']) / 1000),
                    'rate': float(item['fundingRate'])
                })
            
            return sorted(funding_history, key=lambda x: x['timestamp'])
            
        except Exception as e:
            if self.cfg.debug:
                print(f"[FUNDING_RATE] 히스토리 가져오기 실패: {e}")
            return []
    
    def _calculate_funding_sentiment(self, current_rate: float, history: List[Dict[str, Any]]) -> Dict[str, float]:
        """완화된 펀딩비율 기반 시장 심리 계산"""
        if not history:
            return {'extreme_score': 0.0, 'trend_score': 0.0, 'sentiment': 'NEUTRAL'}
        
        rates = [item['rate'] for item in history]
        rates_series = pd.Series(rates)
        
        # 이동평균 계산 (기간 단축)
        funding_ma = float(rates_series.rolling(min(len(rates), self.cfg.funding_ma_period)).mean().iloc[-1])
        
        # 극단값 점수 (완화된 기준)
        abs_current = abs(current_rate)
        if abs_current >= self.cfg.extreme_funding_threshold:
            extreme_score = 1.0
        elif abs_current >= self.cfg.moderate_funding_threshold:
            extreme_score = (abs_current - self.cfg.moderate_funding_threshold) / \
                        (self.cfg.extreme_funding_threshold - self.cfg.moderate_funding_threshold)
        else:
            # 완화: 매우 작은 값도 일부 점수 부여
            extreme_score = abs_current / self.cfg.moderate_funding_threshold * 0.3
        
        # 트렌드 점수 (완화)
        if len(rates) >= 2:  # 3 → 2로 완화
            recent_rates = rates[-2:]
            if recent_rates[-1] > recent_rates[0]:
                trend_score = 0.6  # 기본 트렌드 점수
            else:
                trend_score = 0.6
        else:
            trend_score = 0.3
        
        # 시장 심리 판단 (완화된 기준)
        threshold = self.cfg.moderate_funding_threshold * 0.5  # 기준을 절반으로 완화
        if current_rate > threshold:
            sentiment = 'LONG_OVERHEATED'
        elif current_rate < -threshold:
            sentiment = 'SHORT_OVERHEATED' 
        else:
            sentiment = 'NEUTRAL'
        
        return {
            'extreme_score': extreme_score,
            'trend_score': trend_score,
            'sentiment': sentiment,
            'funding_ma': funding_ma,
            'current_vs_ma': current_rate - funding_ma
        }
    
    def _calculate_volume_confirmation(self, df: pd.DataFrame) -> float:
        """거래량 확인 (변동성 증가 시점)"""
        try:
            if 'quote_volume' in df.columns:
                vol_series = df['quote_volume'].astype(float)
            elif 'volume' in df.columns:
                vol_series = df['volume'].astype(float) * df['close'].astype(float)
            else:
                return 0.5
                
            # 최근 거래량이 평균보다 높은지 확인
            vol_ma = vol_series.rolling(20, min_periods=1).mean()
            current_vol = float(vol_series.iloc[-1])
            avg_vol = float(vol_ma.iloc[-1])
            
            if avg_vol > 0:
                vol_ratio = current_vol / avg_vol
                return _clamp((vol_ratio - 1.0) / 1.5, 0.0, 1.0)
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _calculate_price_momentum(self, df: pd.DataFrame) -> float:
        """가격 모멘텀 계산"""
        try:
            close = pd.to_numeric(df['close'].astype(float))
            
            # 단기 vs 장기 EMA 비교
            ema_short = close.ewm(span=12).mean()
            ema_long = close.ewm(span=26).mean()
            
            current_short = float(ema_short.iloc[-1])
            current_long = float(ema_long.iloc[-1])
            
            momentum = (current_short - current_long) / current_long
            return _clamp(abs(momentum) * 50, 0.0, 1.0)  # 정규화
            
        except Exception:
            return 0.0
    
    def on_kline_close_3m(self) -> Optional[Dict[str, Any]]:
        """3분봉 마감 시 펀딩비율 전략 실행"""
        # 현재 시간 체크 (펀딩비 업데이트는 8시간마다)
        now = self.time_manager.get_current_time()
        
        # 캐시된 데이터가 있고 1시간 이내라면 API 호출 생략
        if (self.last_fetch_time and 
            (now - self.last_fetch_time).total_seconds() < 3600 and 
            self.funding_data_cache):
            current_funding = self.funding_data_cache[-1]['rate']
            funding_history = self.funding_data_cache
        else:
            # 새로운 데이터 가져오기
            current_funding = self._fetch_funding_rate()
            if current_funding is None:
                if self.cfg.debug:
                    print("[FUNDING_RATE] 펀딩비율 데이터 없음")
                return None
            
            funding_history = self._fetch_funding_history()
            if not funding_history:
                if self.cfg.debug:
                    print("[FUNDING_RATE] 펀딩비율 히스토리 없음")
                return None
                
            self.funding_data_cache = funding_history
            self.last_fetch_time = now
        
        # 가격 데이터 가져오기
        data_manager = get_data_manager()
        if data_manager is None:
            return None
            
        df = data_manager.get_latest_data(50)
        if df is None or len(df) < 30:
            if self.cfg.debug:
                print("[FUNDING_RATE] 가격 데이터 부족")
            return None
        
        # 펀딩비율 분석
        funding_analysis = self._calculate_funding_sentiment(current_funding, funding_history)
        
        # 중립 상태라면 신호 없음
        if funding_analysis['sentiment'] == 'NEUTRAL':
            return None
        
        # 각 컴포넌트 점수 계산
        extreme_score = funding_analysis['extreme_score']
        trend_score = funding_analysis['trend_score']
        volume_score = self._calculate_volume_confirmation(df)
        momentum_score = self._calculate_price_momentum(df)
        
        # 신호 방향 결정
        if funding_analysis['sentiment'] == 'LONG_OVERHEATED':
            action = "SELL"  # 롱 과열 → 숏 진입
        elif funding_analysis['sentiment'] == 'SHORT_OVERHEATED':
            action = "BUY"   # 숏 과열 → 롱 진입
        else:
            return None
        
        # 최종 점수 계산
        total_score = (
            self.cfg.w_funding_extreme * extreme_score +
            self.cfg.w_funding_trend * trend_score +
            self.cfg.w_volume_confirm * volume_score +
            self.cfg.w_price_momentum * momentum_score
        )
        
        total_score = _clamp(total_score, 0.0, 1.0)
        
        # 최소 점수 체크
        if total_score < 0.4:
            if self.cfg.debug:
                print(f"[FUNDING_RATE] 점수 부족: {total_score:.3f}")
            return None
        
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
            print(f"[FUNDING_RATE] {action} 신호 - 점수: {total_score:.3f}, "
                  f"펀딩비율: {current_funding:.6f}, 심리: {funding_analysis['sentiment']}, "
                  f"극단: {extreme_score:.3f}, 트렌드: {trend_score:.3f}")
        
        return {
            'name': 'FUNDING_RATE',
            'action': action,
            'score': float(total_score),
            'entry': float(entry),
            'stop': float(stop),
            'targets': [float(tp1), float(tp2)],
            'timestamp': self.time_manager.get_current_time(),
            'context': {
                'mode': 'FUNDING_RATE_SENTIMENT',
                'current_funding_rate': float(current_funding),
                'funding_rate_pct': float(current_funding * 100),
                'sentiment': funding_analysis['sentiment'],
                'extreme_score': float(extreme_score),
                'trend_score': float(trend_score),
                'volume_score': float(volume_score),
                'momentum_score': float(momentum_score),
                'funding_ma': float(funding_analysis['funding_ma']),
                'atr': float(atr)
            }
        }