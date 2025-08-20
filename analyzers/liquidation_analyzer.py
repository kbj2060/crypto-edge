#!/usr/bin/env python3
"""
청산 분석 엔진
"""

import datetime
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from data.binance_websocket import BinanceWebSocket


class LiquidationAnalyzer:
    """청산 분석 엔진"""
    
    def __init__(self, websocket: BinanceWebSocket):
        self.websocket = websocket
    
    def analyze_liquidation_with_technical(self, liquidations: List, density_analysis: Dict, 
                                          df_5m: pd.DataFrame, current_price: float) -> Optional[Dict]:
        """청산 데이터와 기술적 지표를 통합하여 강화된 신호 생성"""
        if not liquidations or df_5m.empty:
            return None
        
        try:
            # 최근 가격 데이터
            recent_close = df_5m['close'].iloc[-5:].values
            recent_volume = df_5m['volume'].iloc[-5:].values
            
            # 기술적 지표 계산
            price_momentum = (recent_close[-1] - recent_close[0]) / recent_close[0] * 100
            volume_trend = recent_volume[-1] / np.mean(recent_volume[:-1]) if len(recent_volume) > 1 else 1.0
            
            # EMA 기울기 계산
            ema_20 = df_5m['EMA_20'].iloc[-3:].values if 'EMA_20' in df_5m.columns else df_5m['close'].iloc[-3:].values
            ema_slope = (ema_20[-1] - ema_20[0]) / ema_20[0] * 100 if len(ema_20) > 1 else 0
            
            # RSI 확인
            rsi_k = df_5m['StochRSI_K'].iloc[-1] if 'StochRSI_K' in df_5m.columns else 50
            
            # 청산 패턴 분석
            # SELL side = 롱 포지션 청산 (매도), BUY side = 숏 포지션 청산 (매수)
            long_liquidations = [liq for liq in liquidations if liq.get('side') == 'SELL']
            short_liquidations = [liq for liq in liquidations if liq.get('side') == 'BUY']
            
            long_volume = sum(liq.get('quantity', 0) for liq in long_liquidations)
            short_volume = sum(liq.get('quantity', 0) for liq in short_liquidations)
            
            # 청산 밀도 정보
            max_density_price = density_analysis.get('max_density_price', current_price)
            max_density_volume = density_analysis.get('max_density_volume', 0)
            
            # 통합 신호 생성
            signal_strength = 0
            signal_bias = 'NEUTRAL'
            confidence = 0
            
            # 롱 청산 우세 + 기술적 하락 신호 (롱 청산 많음 = 가격 하락 = 숏 진입)
            if (long_volume > short_volume * 1.2 and 
                price_momentum < -0.05 and 
                ema_slope < -0.02 and 
                rsi_k > 20):
                
                signal_bias = 'SHORT'  # 롱 청산 많음 → 숏 진입
                signal_strength = min(0.8, (long_volume / max(short_volume, 1)) * 0.3 + abs(price_momentum) * 0.4)
                confidence = min(0.9, signal_strength + (volume_trend - 1) * 0.2)
                
            # 숏 청산 우세 + 기술적 상승 신호 (숏 청산 많음 = 가격 상승 = 롱 진입)
            elif (short_volume > long_volume * 1.2 and 
                  price_momentum > 0.05 and 
                  ema_slope > 0.02 and 
                  rsi_k < 80):
                
                signal_bias = 'LONG'  # 숏 청산 많음 → 롱 진입
                signal_strength = min(0.8, (short_volume / max(long_volume, 1)) * 0.3 + abs(price_momentum) * 0.4)
                confidence = min(0.9, signal_strength + (volume_trend - 1) * 0.2)
            
            # 청산 밀도가 높은 가격대 근처에서의 신호
            if max_density_volume > 0:
                density_distance = abs(max_density_price - current_price) / current_price * 100
                if density_distance < 0.5:  # 0.5% 이내
                    confidence = min(0.95, confidence + 0.1)  # 신뢰도 10% 증가
            
            # 손절가와 익절가 계산 (스캘핑용)
            atr = self._calculate_current_atr()
            if atr:
                if signal_bias == 'LONG':
                    stop_loss = current_price - (atr * 1.5)  # ATR 1.5배
                    take_profit1 = current_price + (atr * 2.0)  # ATR 2배
                    take_profit2 = current_price + (atr * 3.0)  # ATR 3배
                elif signal_bias == 'SHORT':
                    stop_loss = current_price + (atr * 1.5)  # ATR 1.5배
                    take_profit1 = current_price - (atr * 2.0)  # ATR 2배
                    take_profit2 = current_price - (atr * 3.0)  # ATR 3배
                else:
                    return None
            else:
                # ATR이 없을 때 기본값
                if signal_bias == 'LONG':
                    stop_loss = current_price * 0.995  # 0.5% 손절
                    take_profit1 = current_price * 1.008  # 0.8% 익절
                    take_profit2 = current_price * 1.015  # 1.5% 익절
                elif signal_bias == 'SHORT':
                    stop_loss = current_price * 1.005  # 0.5% 손절
                    take_profit1 = current_price * 0.992  # 0.8% 익절
                    take_profit2 = current_price * 0.985  # 1.5% 익절
                else:
                    return None
            
            return {
                'signal_type': 'ENHANCED_LIQUIDATION',
                'action': 'BUY' if signal_bias == 'LONG' else 'SELL' if signal_bias == 'SHORT' else 'NEUTRAL',
                'confidence': confidence,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit1': take_profit1,
                'take_profit2': take_profit2,
                'liquidation_volume': max(long_volume, short_volume),
                'price_momentum': price_momentum,
                'volume_trend': volume_trend,
                'ema_slope': ema_slope,
                'rsi_k': rsi_k,
                'timestamp': datetime.datetime.now()
            }
            
        except Exception as e:
            print(f"❌ 청산-기술 통합 신호 생성 오류: {e}")
            return None
    
    def analyze_liquidation_signal(self, liquidation_stats: Dict, volume_analysis: Dict, current_price: float) -> Optional[Dict]:
        """청산 데이터만 사용한 신호 생성 (기술적 지표 제거)"""
        try:
            # 청산 통계에서 신호 생성
            total_count = liquidation_stats.get('total_count', 0)
            buy_count = liquidation_stats.get('buy_count', 0)
            sell_count = liquidation_stats.get('sell_count', 0)
            buy_ratio = liquidation_stats.get('buy_ratio', 0.0)
            sell_ratio = liquidation_stats.get('sell_ratio', 0.0)
            total_value = liquidation_stats.get('total_value', 0.0)
            
            # 거래량 분석
            volume_ratio = volume_analysis.get('volume_ratio', 1.0)
            volume_trend = volume_analysis.get('volume_trend', 'stable')
            
            # 기본 조건 확인
            if total_count < 3:  # 최소 청산 수
                return None
            
            if total_value < 100000:  # 최소 청산 가치
                return None
            
            if volume_ratio < 1.5:  # 최소 거래량 급증
                return None
            
            # 신호 생성
            signal_bias = 'NEUTRAL'
            confidence = 0.0
            
            # BUY 신호 (SELL 청산 우세)
            if sell_ratio > 0.7 and total_count >= 3:
                signal_bias = 'LONG'
                confidence = min(0.9, sell_ratio * 0.6 + min(volume_ratio / 3.0, 1.0) * 0.4)
                
            # SELL 신호 (BUY 청산 우세)
            elif buy_ratio > 0.7 and total_count >= 3:
                signal_bias = 'SHORT'
                confidence = min(0.9, buy_ratio * 0.6 + min(volume_ratio / 3.0, 1.0) * 0.4)
            
            if signal_bias == 'NEUTRAL':
                return None
            
            # 고정 비율 기반 리스크 관리
            if signal_bias == 'LONG':
                stop_loss = current_price * 0.98  # 2% 손절
                take_profit1 = current_price * 1.04  # 4% 익절
                take_profit2 = current_price * 1.06  # 6% 익절
                action = 'BUY'
            else:  # SHORT
                stop_loss = current_price * 1.02  # 2% 손절
                take_profit1 = current_price * 0.96  # 4% 익절
                take_profit2 = current_price * 0.94  # 6% 익절
                action = 'SELL'
            
            # 리스크/보상 비율 계산
            risk = abs(current_price - stop_loss)
            reward = abs(take_profit1 - current_price)
            risk_reward = reward / risk if risk > 0 else 0
            
            return {
                'signal_type': 'ENHANCED_LIQUIDATION',
                'action': action,
                'confidence': confidence,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit1': take_profit1,
                'take_profit2': take_profit2,
                'risk_reward': risk_reward,
                'liquidation_stats': liquidation_stats,
                'volume_analysis': volume_analysis,
                'timestamp': datetime.datetime.now(),
                'reason': f"{action} 청산 급증: {buy_count if action == 'BUY' else sell_count}/{total_count} | 거래량: {volume_ratio:.1f}x | 청산가치: ${total_value:,.0f}"
            }
            
        except Exception as e:
            print(f"❌ 청산 신호 생성 오류: {e}")
            return None
    
    def _calculate_current_atr(self) -> Optional[float]:
        """현재 ATR 계산"""
        try:
            if len(self.websocket.price_history) >= 14:
                prices = [p['price'] for p in self.websocket.price_history[-14:]]
                price_changes = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
                atr = sum(price_changes) / len(price_changes)
                return atr
        except Exception:
            pass
        return None
