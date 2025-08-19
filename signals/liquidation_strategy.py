from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np

@dataclass
class LiquidationConfig:
    """청산 기반 전략 설정"""
    # 청산 임계값
    min_liquidation_count: int = 3  # 최소 청산 발생 수
    min_liquidation_quantity: float = 100.0  # 최소 청산 수량 (ETH)
    min_liquidation_value: float = 100000.0  # 최소 청산 가치 (USDT)
    
    # 시간 윈도우
    liquidation_window_minutes: int = 5  # 청산 분석 시간 윈도우
    volume_window_minutes: int = 3  # 거래량 분석 시간 윈도우
    
    # 신호 임계값
    buy_liquidation_ratio: float = 0.7  # BUY 청산 비율 임계값
    sell_liquidation_ratio: float = 0.7  # SELL 청산 비율 임계값
    volume_spike_threshold: float = 2.0  # 거래량 급증 임계값
    
    # 리스크 관리
    max_position_size: float = 0.1  # 최대 포지션 크기 (10%)
    stop_loss_atr: float = 2.0  # ATR 기반 손절
    take_profit_atr: float = 3.0  # ATR 기반 익절

class LiquidationStrategy:
    """청산 데이터 기반 신호 전략"""
    
    def __init__(self, config: LiquidationConfig):
        self.config = config
        self.last_signal_time = None
        self.signal_cooldown = timedelta(minutes=2)  # 신호 간 최소 대기 시간
    
    def analyze_liquidation_signal(self, 
                                    liquidation_stats: Dict, 
                                    volume_analysis: Dict,
                                    current_price: float,
                                    atr: float) -> Optional[Dict]:
        """청산 데이터 기반 신호 분석"""
        
        # 신호 쿨다운 체크
        if (self.last_signal_time and 
            datetime.now() - self.last_signal_time < self.signal_cooldown):
            return None
        
        # 기본 조건 확인
        if not self._check_basic_conditions(liquidation_stats, volume_analysis):
            return None
        
        # 신호 생성
        signal = self._generate_signal(liquidation_stats, volume_analysis, current_price, atr)
        
        if signal:
            self.last_signal_time = datetime.now()
        
        return signal
    
    def _check_basic_conditions(self, liquidation_stats: Dict, volume_analysis: Dict) -> bool:
        """기본 조건 확인"""
        # 청산 수량 조건
        if liquidation_stats['total_count'] < self.config.min_liquidation_count:
            return False
        
        # 청산 가치 조건
        if liquidation_stats['total_value'] < self.config.min_liquidation_value:
            return False
        
        # 거래량 급증 조건
        if volume_analysis['volume_ratio'] < self.config.volume_spike_threshold:
            return False
        
        return True
    
    def _generate_signal(self, 
                        liquidation_stats: Dict, 
                        volume_analysis: Dict,
                        current_price: float,
                        atr: float) -> Optional[Dict]:
        """신호 생성"""
        
        buy_ratio = liquidation_stats['buy_ratio']
        sell_ratio = liquidation_stats['sell_ratio']
        total_count = liquidation_stats['total_count']
        total_value = liquidation_stats['total_value']
        
        # 신뢰도 계산
        confidence = self._calculate_confidence(liquidation_stats, volume_analysis)
        
        # 최소 신뢰도 체크 (추가)
        if confidence < 0.3:  # 기본 최소 신뢰도 30%
            return None
        
        # BUY 신호 조건
        if (buy_ratio > self.config.buy_liquidation_ratio and 
            total_count >= self.config.min_liquidation_count):
            
            signal = self._create_buy_signal(
                current_price, atr, confidence, liquidation_stats, volume_analysis
            )
            return signal
        
        # SELL 신호 조건
        elif (sell_ratio > self.config.sell_liquidation_ratio and 
              total_count >= self.config.min_liquidation_count):
            
            signal = self._create_sell_signal(
                current_price, atr, confidence, liquidation_stats, volume_analysis
            )
            return signal
        
        return None
    
    def _calculate_confidence(self, liquidation_stats: Dict, volume_analysis: Dict) -> float:
        """신뢰도 계산"""
        confidence = 0.0
        
        # 청산 강도 (0-0.4)
        liquidation_intensity = min(liquidation_stats['total_count'] / 10.0, 1.0)
        confidence += liquidation_intensity * 0.4
        
        # 거래량 급증 강도 (0-0.3)
        volume_intensity = min(volume_analysis['volume_ratio'] / 3.0, 1.0)
        confidence += volume_intensity * 0.3
        
        # 청산 가치 강도 (0-0.3)
        value_intensity = min(liquidation_stats['total_value'] / 1000000.0, 1.0)
        confidence += value_intensity * 0.3
        
        return min(confidence, 1.0)
    
    def _create_buy_signal(self, 
                          current_price: float, 
                          atr: float, 
                          confidence: float,
                          liquidation_stats: Dict,
                          volume_analysis: Dict) -> Dict:
        """BUY 신호 생성"""
        
        # 리스크 관리 레벨 계산
        stop_loss = current_price - (atr * self.config.stop_loss_atr)
        take_profit1 = current_price + (atr * self.config.take_profit_atr)
        take_profit2 = current_price + (atr * self.config.take_profit_atr * 1.5)
        
        # 리스크/보상 비율
        risk = current_price - stop_loss
        reward = take_profit1 - current_price
        risk_reward = reward / risk if risk > 0 else 0
        
        return {
            'action': 'BUY',
            'bias': 'LONG',
            'timestamp': datetime.now(),
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'take_profit1': take_profit1,
            'take_profit2': take_profit2,
            'confidence': confidence,
            'risk_reward': risk_reward,
            'atr': atr,
            'signal_type': 'liquidation',
            'liquidation_stats': liquidation_stats,
            'volume_analysis': volume_analysis,
            'reason': f"BUY 청산 급증: {liquidation_stats['buy_count']}/{liquidation_stats['total_count']} | 거래량: {volume_analysis['volume_ratio']:.1f}x"
        }
    
    def _create_sell_signal(self, 
                           current_price: float, 
                           atr: float, 
                           confidence: float,
                           liquidation_stats: Dict,
                           volume_analysis: Dict) -> Dict:
        """SELL 신호 생성"""
        
        # 리스크 관리 레벨 계산
        stop_loss = current_price + (atr * self.config.stop_loss_atr)
        take_profit1 = current_price - (atr * self.config.take_profit_atr)
        take_profit2 = current_price - (atr * self.config.take_profit_atr * 1.5)
        
        # 리스크/보상 비율
        risk = stop_loss - current_price
        reward = current_price - take_profit1
        risk_reward = reward / risk if risk > 0 else 0
        
        return {
            'action': 'SELL',
            'bias': 'SHORT',
            'timestamp': datetime.now(),
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'take_profit1': take_profit1,
            'take_profit2': take_profit2,
            'confidence': confidence,
            'risk_reward': risk_reward,
            'atr': atr,
            'signal_type': 'liquidation',
            'liquidation_stats': liquidation_stats,
            'volume_analysis': volume_analysis,
            'reason': f"SELL 청산 급증: {liquidation_stats['sell_count']}/{liquidation_stats['total_count']} | 거래량: {volume_analysis['volume_ratio']:.1f}x"
        }
    
    def get_market_sentiment(self, liquidation_stats: Dict, volume_analysis: Dict) -> Dict:
        """시장 심리 분석"""
        
        buy_ratio = liquidation_stats['buy_ratio']
        sell_ratio = liquidation_stats['sell_ratio']
        volume_trend = volume_analysis['volume_trend']
        
        # 청산 기반 심리
        if buy_ratio > 0.7:
            liquidation_sentiment = 'bullish'
        elif sell_ratio > 0.7:
            liquidation_sentiment = 'bearish'
        else:
            liquidation_sentiment = 'neutral'
        
        # 거래량 기반 심리
        if volume_trend == 'increasing':
            volume_sentiment = 'bullish'
        elif volume_trend == 'decreasing':
            volume_sentiment = 'bearish'
        else:
            volume_sentiment = 'neutral'
        
        # 종합 심리
        if liquidation_sentiment == volume_sentiment:
            overall_sentiment = liquidation_sentiment
        else:
            overall_sentiment = 'mixed'
        
        return {
            'liquidation_sentiment': liquidation_sentiment,
            'volume_sentiment': volume_sentiment,
            'overall_sentiment': overall_sentiment,
            'buy_ratio': buy_ratio,
            'sell_ratio': sell_ratio,
            'volume_trend': volume_trend
        }
