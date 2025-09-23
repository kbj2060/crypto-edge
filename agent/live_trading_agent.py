"""
실시간 트레이딩 에이전트 (Signal 기반 Feature 추출)
- Signal의 모든 indicator와 raw score 활용
- 80차원 상태 벡터로 확장
- 중복 계산 제거
"""

import numpy as np
import pandas as pd
import torch
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import json
import os
from pathlib import Path

try:
    from agent.rl_training_system import RLAgent, DuelingDQN
    print("✅ 올바른 모델 클래스 로드됨")
except ImportError:
    print("❌ 훈련 시스템 모듈을 찾을 수 없습니다.")
    from agent.rl_training_system import DuelingDQN, RLAgent

from utils.data_flattener import ensure_flattened_data

class SignalQualityAnalyzer:
    """신호 품질 분석기"""
    
    @staticmethod
    def analyze_signal_quality(flattened_signal_data: Dict[str, Any]) -> Dict[str, float]:
        """Signal 데이터 품질 분석 (Flatten 형태 지원)"""
        
        quality_metrics = {
            'high_confidence_signals': 0,
            'total_signals': 0,
            'agreement_score': 0.0,
            'overall_score': 0.0,
            'signal_strength': 0.0,
            'confidence_level': 0.0
        }
        
        actions = []
        confidences = []
        net_scores = []
        
        # Flatten 형태에서 시간대별 정보 추출
        for timeframe in ['short_term', 'medium_term', 'long_term']:
            action = flattened_signal_data.get(f'{timeframe}_action', 'HOLD')
            confidence = flattened_signal_data.get(f'{timeframe}_confidence', 'LOW')
            net_score = float(flattened_signal_data.get(f'{timeframe}_net_score', 0.0))
            
            if action != 'HOLD':
                quality_metrics['total_signals'] += 1
                actions.append(1 if action == 'LONG' else -1)
                net_scores.append(abs(net_score))
                
                if confidence == 'HIGH':
                    quality_metrics['high_confidence_signals'] += 1
                    confidences.append(1.0)
                elif confidence == 'MEDIUM':
                    confidences.append(0.5)
                else:
                    confidences.append(0.1)
        
        # 신호 일치도 계산
        if actions:
            action_agreement = 1.0 - (np.std(actions) if len(actions) > 1 else 0.0)
            avg_confidence = np.mean(confidences)
            avg_signal_strength = np.mean(net_scores)
            
            quality_metrics['agreement_score'] = action_agreement
            quality_metrics['confidence_level'] = avg_confidence
            quality_metrics['signal_strength'] = min(avg_signal_strength, 1.0)
            quality_metrics['overall_score'] = (
                action_agreement * 0.4 + 
                avg_confidence * 0.4 + 
                avg_signal_strength * 0.2
            )
        
        return quality_metrics

class EnhancedSignalStateBuilder:
    """Signal 기반 상태 벡터 구성기 (80차원)"""
    
    def __init__(self):
        self.price_indicator_keys = [
            'indicator_vwap', 'indicator_atr', 'indicator_poc', 
            'indicator_hvn', 'indicator_lvn', 'indicator_vwap_std',
            'indicator_prev_day_high', 'indicator_prev_day_low',
            'indicator_opening_range_high', 'indicator_opening_range_low'
        ]
    
    def build_state_vector(self, flattened_signal_data: Dict, current_candle: Dict, 
                          portfolio_state: Dict) -> np.ndarray:
        """Signal 정보를 최대한 활용한 80차원 상태 벡터 구성"""
        
        # 1. Price Indicator Features (20차원) - signal의 indicator 활용
        price_features = self._extract_price_indicators(flattened_signal_data, current_candle)
        
        # 2. Technical Score Features (25차원) - raw score들
        technical_features = self._extract_technical_scores(flattened_signal_data)
        
        # 3. Decision Features (25차원) - 기존 decision 로직
        decision_features = self._extract_decision_features(flattened_signal_data)
        
        # 4. Portfolio Features (10차원) - 포트폴리오 상태
        portfolio_features = self._extract_portfolio_features(portfolio_state)
        
        # 모든 특성 결합 (80차원)
        state = np.concatenate([price_features, technical_features, decision_features, portfolio_features])
        
        return state.astype(np.float32)
    
    def _extract_price_indicators(self, flattened_signal_data: Dict, current_candle: Dict) -> np.ndarray:
        """Signal의 indicator들을 price feature로 활용 (20차원)"""
        features = []
        current_price = current_candle['close']
        
        # 1. 가격 대비 지표 위치 (정규화)
        vwap = flattened_signal_data.get('indicator_vwap', current_price)
        poc = flattened_signal_data.get('indicator_poc', current_price)
        hvn = flattened_signal_data.get('indicator_hvn', current_price)
        lvn = flattened_signal_data.get('indicator_lvn', current_price)
        
        features.extend([
            (current_price - vwap) / current_price if current_price > 0 else 0.0,  # VWAP 상대 위치
            (current_price - poc) / current_price if current_price > 0 else 0.0,   # POC 상대 위치  
            (current_price - hvn) / current_price if current_price > 0 else 0.0,   # HVN 상대 위치
            (current_price - lvn) / current_price if current_price > 0 else 0.0,   # LVN 상대 위치
        ])
        
        # 2. 변동성 지표들
        atr = flattened_signal_data.get('indicator_atr', 0.0)
        vwap_std = flattened_signal_data.get('indicator_vwap_std', 0.0)
        
        features.extend([
            atr / current_price if current_price > 0 else 0.0,                     # ATR 정규화
            vwap_std / current_price if current_price > 0 else 0.0,                # VWAP 표준편차
        ])
        
        # 3. 일별 기준점들과의 관계
        prev_high = flattened_signal_data.get('indicator_prev_day_high', current_price)
        prev_low = flattened_signal_data.get('indicator_prev_day_low', current_price)
        or_high = flattened_signal_data.get('indicator_opening_range_high', current_price)
        or_low = flattened_signal_data.get('indicator_opening_range_low', current_price)
        
        # 전일 레인지에서의 위치
        prev_range = prev_high - prev_low
        if prev_range > 0:
            prev_day_position = (current_price - prev_low) / prev_range
        else:
            prev_day_position = 0.5
            
        # 오프닝 레인지에서의 위치
        or_range = or_high - or_low  
        if or_range > 0:
            or_position = (current_price - or_low) / or_range
        else:
            or_position = 0.5
        
        features.extend([
            prev_day_position,                                                      # 전일 레인지 위치
            or_position,                                                           # 오프닝 레인지 위치
            (current_price - prev_high) / current_price if current_price > 0 else 0.0,  # 전일고점 돌파도
            (prev_low - current_price) / current_price if current_price > 0 else 0.0,   # 전일저점 이탈도
        ])
        
        # 4. 현재 캔들 정보
        candle_features = self._extract_current_candle_features(current_candle)
        features.extend(candle_features[:8])  # 8개 더 추가해서 20개 맞춤
        
        return np.array(features[:20], dtype=np.float32)
    
    def _extract_current_candle_features(self, candle: Dict) -> List[float]:
        """현재 캔들의 기본 정보"""
        high, low, close, open_price = candle['high'], candle['low'], candle['close'], candle['open']
        volume = candle.get('volume', 0)
        
        return [
            (close - open_price) / open_price if open_price > 0 else 0.0,          # 캔들 수익률
            (high - low) / close if close > 0 else 0.0,                           # 캔들 변동성
            (high - close) / (high - low) if high > low else 0.5,                 # 상단 꼬리 비율
            (close - low) / (high - low) if high > low else 0.5,                  # 하단 꼬리 비율
            (close - open_price) / (high - low) if high > low else 0.0,           # 몸통 비율
            min(volume / 1000000, 2.0) if volume > 0 else 0.0,                   # 거래량 (정규화)
            1.0 if close > open_price else 0.0,                                   # 양봉/음봉
            (high - max(open_price, close)) / (high - low) if high > low else 0.0 # 위꼬리 비율
        ]
    
    def _extract_technical_scores(self, flattened_signal_data: Dict) -> np.ndarray:
        """각 전략의 raw score들을 특성으로 활용 (25차원)"""
        features = []
        
        # 모든 raw score 키들 수집
        all_raw_scores = []
        for key, value in flattened_signal_data.items():
            if '_raw_' in key and '_score' in key and value is not None:
                try:
                    all_raw_scores.append(float(value))
                except:
                    all_raw_scores.append(0.0)
        
        # 25개로 맞추기 (부족하면 0 패딩, 초과하면 상위 25개)
        if len(all_raw_scores) >= 25:
            # 절대값 기준으로 상위 25개 선택
            sorted_scores = sorted(all_raw_scores, key=abs, reverse=True)
            features = sorted_scores[:25]
        else:
            features = all_raw_scores + [0.0] * (25 - len(all_raw_scores))
        
        return np.array(features, dtype=np.float32)
    
    def _extract_decision_features(self, flattened_signal_data: Dict) -> np.ndarray:
        """기존 decision 특성들 (25차원)"""
        features = []
        
        # 각 시간대별 액션과 점수들 (3 × 6 = 18개)
        for timeframe in ['short_term', 'medium_term', 'long_term']:
            action = flattened_signal_data.get(f'{timeframe}_action', 'HOLD')
            action_strength = 1.0 if action == 'LONG' else (-1.0 if action == 'SHORT' else 0.0)
            
            net_score = float(flattened_signal_data.get(f'{timeframe}_net_score', 0.0))
            buy_score = float(flattened_signal_data.get(f'{timeframe}_buy_score', 0.0))
            sell_score = float(flattened_signal_data.get(f'{timeframe}_sell_score', 0.0))
            
            confidence = flattened_signal_data.get(f'{timeframe}_confidence', 'LOW')
            confidence_val = 1.0 if confidence == 'HIGH' else (0.5 if confidence == 'MEDIUM' else 0.0)
            
            leverage = min(float(flattened_signal_data.get(f'{timeframe}_leverage', 1.0)) / 20.0, 1.0)
            
            features.extend([action_strength, net_score, buy_score, sell_score, confidence_val, leverage])
        
        # 추가 메타 정보 (7개)
        signals_used_short = min(float(flattened_signal_data.get('short_term_signals_used', 0)) / 10.0, 1.0)
        signals_used_medium = min(float(flattened_signal_data.get('medium_term_signals_used', 0)) / 10.0, 1.0)  
        signals_used_long = min(float(flattened_signal_data.get('long_term_signals_used', 0)) / 10.0, 1.0)
        
        market_context_short = 1.0 if flattened_signal_data.get('short_term_market_context') == 'TRENDING' else 0.0
        market_context_medium = 1.0 if flattened_signal_data.get('medium_term_market_context') == 'TRENDING' else 0.0
        
        institutional_bias = flattened_signal_data.get('long_term_institutional_bias', 'NEUTRAL')
        bias_val = 1.0 if institutional_bias == 'BULLISH' else (-1.0 if institutional_bias == 'BEARISH' else 0.0)
        
        macro_strength = flattened_signal_data.get('long_term_macro_trend_strength', 'MEDIUM')
        strength_val = 1.0 if macro_strength == 'HIGH' else (0.5 if macro_strength == 'MEDIUM' else 0.0)
        
        features.extend([
            signals_used_short, signals_used_medium, signals_used_long,
            market_context_short, market_context_medium, bias_val, strength_val
        ])
        
        return np.array(features[:25], dtype=np.float32)
    
    def _extract_portfolio_features(self, portfolio_state: Dict) -> np.ndarray:
        """포트폴리오 상태 특성 (10차원)"""
        features = [
            portfolio_state.get('current_position', 0.0),
            portfolio_state.get('current_leverage', 1.0) / 20.0,
            portfolio_state.get('balance_ratio', 0.0),
            portfolio_state.get('unrealized_pnl_ratio', 0.0),
            min(portfolio_state.get('total_trades', 0) / 100.0, 1.0),
            portfolio_state.get('win_rate', 0.0),
            portfolio_state.get('max_drawdown', 0.0),
            min(portfolio_state.get('consecutive_losses', 0) / 10.0, 1.0),
            min(portfolio_state.get('holding_time', 0) / 1440.0, 1.0),
            1.0 if portfolio_state.get('in_position', False) else 0.0
        ]
        return np.array(features, dtype=np.float32)

class RiskManager:
    """리스크 관리자"""
    
    def __init__(self, initial_balance: float):
        self.initial_balance = initial_balance
        self.max_drawdown_limit = 0.12
        self.max_position_limit = 0.15
        self.max_leverage = 5.0
        self.consecutive_loss_limit = 3
    
    def check_risk_limits(self, decision: Dict[str, Any], portfolio_state: Dict) -> Dict[str, Any]:
        """리스크 한도 체크 및 거래 결정 조정"""
        
        # 최대 드로우다운 체크
        current_drawdown = portfolio_state.get('max_drawdown', 0.0)
        if current_drawdown > self.max_drawdown_limit:
            decision['action'] = 'HOLD'
            decision['reason'] = f"리스크 관리: 최대 손실 한도 ({current_drawdown:.1%})"
            decision['quantity'] = 0.0
            return decision
        
        # 연속 손실 체크
        consecutive_losses = portfolio_state.get('consecutive_losses', 0)
        if consecutive_losses > self.consecutive_loss_limit:
            decision['quantity'] = decision.get('quantity', 0.0) * 0.5
            decision['reason'] = decision.get('reason', '') + f" (연속손실 {consecutive_losses}회, 크기 감소)"
        
        # 포지션 크기 제한
        current_balance = portfolio_state.get('current_balance', self.initial_balance)
        max_position_usd = current_balance * self.max_position_limit
        
        if decision.get('quantity', 0.0) > max_position_usd:
            decision['quantity'] = max_position_usd
            decision['reason'] = decision.get('reason', '') + " (포지션 크기 제한)"
        
        # 레버리지 제한
        if decision.get('target_leverage', 1.0) > self.max_leverage:
            decision['target_leverage'] = self.max_leverage
            decision['reason'] = decision.get('reason', '') + f" (레버리지 {self.max_leverage}배 제한)"
        
        return decision
    
    def calculate_position_size(self, signal_strength: float, confidence: float, 
                              current_balance: float) -> float:
        """신호 강도와 신뢰도 기반 포지션 크기 계산"""
        
        # 기본 리스크 (잔고의 1-3%)
        base_risk_pct = 0.01 + (confidence * 0.02)
        base_risk = current_balance * base_risk_pct
        
        # 신호 강도 반영
        signal_multiplier = min(signal_strength, 1.0)
        
        # 최종 포지션 크기
        position_usd = base_risk * signal_multiplier
        
        # 최대 한도 적용
        max_position = current_balance * self.max_position_limit
        
        return min(position_usd, max_position)

class LiveTradingAgent:
    """실시간 트레이딩 에이전트 (Signal 기반 80차원)"""
    
    def __init__(self, model_path: str, initial_balance: float = 10000.0):
        """
        Args:
            model_path: 훈련된 모델 파일 경로
            initial_balance: 시작 잔고
        """
        self.model_path = model_path
        self.initial_balance = initial_balance
        
        # 포트폴리오 상태
        self.current_balance = initial_balance
        self.current_position = 0.0
        self.current_leverage = 1.0
        self.entry_price = 0.0
        self.holding_time = 0
        self.in_position = False
        
        # 거래 통계
        self.total_trades = 0
        self.winning_trades = 0
        self.max_drawdown = 0.0
        self.peak_balance = initial_balance
        self.consecutive_losses = 0
        
        # 헬퍼 클래스들
        self.signal_analyzer = SignalQualityAnalyzer()
        self.state_builder = EnhancedSignalStateBuilder()  # 80차원 상태 구성기
        self.risk_manager = RiskManager(initial_balance)
        
        # 훈련된 에이전트 로드
        self.agent = self._load_trained_agent()
        
        print(f"실시간 트레이딩 에이전트 초기화 완료 (80차원 상태 공간)")
        print(f"   모델: {model_path}")
        print(f"   초기 잔고: ${initial_balance:,.2f}")
    
    def _load_trained_agent(self):
        """훈련된 에이전트 로드"""
        try:
            # 모델 호환성 체크
            if os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
                expected_state_size = checkpoint.get('state_size', 60)
                
                if expected_state_size == 80:
                    # 80차원 모델
                    agent = RLAgent(state_size=80)
                    print("✅ 80차원 모델 감지")
                else:
                    # 기존 60차원 모델 - 80차원으로 업그레이드 필요
                    print("❌ 60차원 모델 감지됨. 80차원으로 재훈련 필요")
                    agent = RLAgent(state_size=80)  # 새 모델 생성
                    return agent
                
                if agent.load_model_with_compatibility(self.model_path):
                    agent.epsilon = 0.0
                    print(f"✅ 모델 로드 성공 ({expected_state_size}차원)")
                    return agent
                else:
                    raise Exception("모델 로드 실패")
            else:
                print("❌ 모델 파일이 없습니다. 새 모델을 생성합니다.")
                return RLAgent(state_size=80)
                
        except Exception as e:
            print(f"❌ 에이전트 로드 실패: {e}")
            print("새 80차원 에이전트를 생성합니다.")
            return RLAgent(state_size=80)
    
    def make_trading_decision(self, signal_data: Dict[str, Any], 
                            current_candle: Dict[str, float]) -> Dict[str, Any]:
        """
        Signal을 최대한 활용한 실시간 거래 결정 생성
        
        Args:
            signal_data: 신호 데이터 (flatten된 형태이거나 중첩된 형태)
            current_candle: 현재 캔들 데이터
            
        Returns:
            거래 결정 딕셔너리
        """
        
        if self.agent is None:
            return self._get_default_decision("에이전트 로드 실패")
        
        try:
            # 1. 데이터가 평면화되었는지 확인하고 필요시 평면화
            flattened_signal_data = ensure_flattened_data(signal_data)
            
            # 2. 신호 품질 분석
            signal_quality = self.signal_analyzer.analyze_signal_quality(flattened_signal_data)
            
            # 3. 포트폴리오 상태 구성
            portfolio_state = self._get_portfolio_state()
            
            # 4. 80차원 상태 벡터 구성 (Signal 정보 최대 활용)
            state_vector = self.state_builder.build_state_vector(
                flattened_signal_data, current_candle, portfolio_state
            )
            
            # 5. AI 에이전트의 액션 예측
            ai_action = self.agent.act(state_vector)
            
            # 6. 액션을 거래 결정으로 변환
            trading_decision = self._convert_action_to_decision(
                ai_action, current_candle, signal_quality, flattened_signal_data
            )
            
            # 6. 리스크 체크 및 최종 결정
            final_decision = self.risk_manager.check_risk_limits(
                trading_decision, portfolio_state
            )
            
            return final_decision
            
        except Exception as e:
            print(f"거래 결정 생성 오류: {e}")
            import traceback
            traceback.print_exc()
            return self._get_default_decision(f"오류: {str(e)}")
    
    def _get_portfolio_state(self) -> Dict:
        """현재 포트폴리오 상태"""
        return {
            'current_position': self.current_position,
            'current_leverage': self.current_leverage,
            'current_balance': self.current_balance,
            'balance_ratio': (self.current_balance - self.initial_balance) / self.initial_balance,
            'unrealized_pnl_ratio': 0.0,
            'total_trades': self.total_trades,
            'win_rate': self.winning_trades / max(self.total_trades, 1),
            'max_drawdown': self.max_drawdown,
            'consecutive_losses': self.consecutive_losses,
            'holding_time': self.holding_time,
            'in_position': self.in_position
        }
    
    def _convert_action_to_decision(self, ai_action: int, 
                                  current_candle: Dict, signal_quality: Dict,
                                  flattened_signal_data: Dict) -> Dict[str, Any]:
        """AI 액션을 실제 거래 결정으로 변환 (단순화된 액션)"""
        
        # 단순한 액션 처리: 0=Hold, 1=Buy, 2=Sell
        if ai_action == 0:  # Hold
            position_change = 0.0
        elif ai_action == 1:  # Buy
            position_change = 1.0  # 고정된 매수 크기
        else:  # ai_action == 2, Sell
            position_change = -1.0  # 고정된 매도 크기
        
        # 단타용 고정값
        leverage = 5.0  # 고정 레버리지
        holding_minutes = 30.0  # 30분 홀딩 (단타)
        
        current_price = current_candle['close']
        
        # AI 신뢰도 계산
        ai_confidence = self._calculate_ai_confidence(ai_action, signal_quality)
        
        # Signal 기반 추가 신뢰도 계산
        signal_confidence = self._calculate_signal_confidence(flattened_signal_data)
        
        # 거래 결정 생성
        decision = {
            'timestamp': datetime.now(),
            'current_price': current_price,
            'ai_confidence': ai_confidence,
            'signal_confidence': signal_confidence,
            'signal_quality': signal_quality,
            'position_change': position_change,
            'target_leverage': min(leverage, 5.0),
            'target_holding_minutes': holding_minutes,
            'action': 'HOLD',
            'reason': '',
            'quantity': 0.0,
            'stop_loss': None,
            'take_profit': None
        }
        
        # 액션 해석 (단순화된 액션)
        min_quality = 0.3
        min_confidence = 0.4
        
        combined_confidence = (ai_confidence + signal_confidence) / 2
        
        if (signal_quality['overall_score'] > min_quality and
            combined_confidence > min_confidence):
            
            if ai_action == 1:  # Buy
                decision['action'] = 'BUY'
                decision['reason'] = (f"AI+Signal 추천: Long "
                                    f"(신호품질: {signal_quality['overall_score']:.2f}, "
                                    f"AI신뢰도: {ai_confidence:.2f}, "
                                    f"Signal신뢰도: {signal_confidence:.2f})")
            elif ai_action == 2:  # Sell
                decision['action'] = 'SELL'
                decision['reason'] = (f"AI+Signal 추천: Short "
                                    f"(신호품질: {signal_quality['overall_score']:.2f}, "
                                    f"AI신뢰도: {ai_confidence:.2f}, "
                                    f"Signal신뢰도: {signal_confidence:.2f})")
            
            # 포지션 크기 계산
            decision['quantity'] = self.risk_manager.calculate_position_size(
                signal_quality['signal_strength'], 
                combined_confidence, 
                self.current_balance
            )
            
            # 스탑 설정
            decision['stop_loss'], decision['take_profit'] = self._calculate_stops(
                current_price, decision['action'], holding_minutes, signal_quality, flattened_signal_data
            )
        else:
            action_name = 'Hold' if ai_action == 0 else ('Buy' if ai_action == 1 else 'Sell')
            decision['reason'] = (f"임계값 미달 (액션: {action_name}, "
                                f"신호품질: {signal_quality['overall_score']:.2f}, "
                                f"종합신뢰도: {combined_confidence:.2f})")
        
        return decision
    
    def _calculate_ai_confidence(self, ai_action: int, signal_quality: Dict) -> float:
        """AI와 신호 품질을 결합한 신뢰도 계산 (단순화된 액션)"""
        
        # 단순한 액션에서는 액션 자체로 신뢰도 계산
        if ai_action == 0:  # Hold
            ai_confidence = 0.5  # 중간 신뢰도
        else:  # Buy 또는 Sell
            ai_confidence = 0.8  # 높은 신뢰도 (실제 거래 결정)
        
        signal_confidence = signal_quality['overall_score']
        
        combined_confidence = (ai_confidence * 0.6) + (signal_confidence * 0.4)
        
        return min(combined_confidence, 1.0)
    
    def _calculate_signal_confidence(self, flattened_signal_data: Dict) -> float:
        """Signal 데이터 기반 신뢰도 계산"""
        confidence_factors = []
        
        # 각 시간대별 신뢰도
        for timeframe in ['short_term', 'medium_term', 'long_term']:
            confidence = flattened_signal_data.get(f'{timeframe}_confidence', 'LOW')
            if confidence == 'HIGH':
                confidence_factors.append(1.0)
            elif confidence == 'MEDIUM':
                confidence_factors.append(0.5)
            else:
                confidence_factors.append(0.1)
        
        # 신호 사용 개수 (더 많은 신호 = 더 신뢰)
        signals_used = []
        for timeframe in ['short_term', 'medium_term', 'long_term']:
            used = flattened_signal_data.get(f'{timeframe}_signals_used', 0)
            signals_used.append(min(used / 5.0, 1.0))
        
        # 종합 신뢰도
        avg_confidence = np.mean(confidence_factors)
        avg_signals = np.mean(signals_used)
        
        return (avg_confidence * 0.7) + (avg_signals * 0.3)
    
    def _calculate_stops(self, current_price: float, action: str, 
                        holding_minutes: float, signal_quality: Dict, 
                        flattened_signal_data: Dict) -> Tuple[Optional[float], Optional[float]]:
        """Signal 정보를 활용한 스탑로스와 익절가 계산"""
        
        if action == 'HOLD':
            return None, None
        
        # ATR 기반 변동성 (Signal에서 직접 가져옴)
        atr = flattened_signal_data.get('indicator_atr', current_price * 0.02)
        volatility_estimate = atr / current_price
        
        # 신호 품질에 따른 조정
        quality_score = signal_quality['overall_score']
        stop_multiplier = 1.5 + (1.0 - quality_score)
        profit_multiplier = 1.0 + quality_score
        
        # 홀딩 시간 조정
        if holding_minutes < 120:
            stop_multiplier *= 0.8
            profit_multiplier *= 0.9
        
        # 시간대별 신호 강도 반영
        timeframe_strength = 0.0
        for timeframe in ['short_term', 'medium_term', 'long_term']:
            net_score = abs(float(flattened_signal_data.get(f'{timeframe}_net_score', 0.0)))
            timeframe_strength += net_score
        
        if timeframe_strength > 1.0:  # 강한 신호
            stop_multiplier *= 0.9
            profit_multiplier *= 1.2
        
        if action == 'BUY':
            stop_loss = current_price * (1 - volatility_estimate * stop_multiplier)
            take_profit = current_price * (1 + volatility_estimate * profit_multiplier)
        else:  # SELL
            stop_loss = current_price * (1 + volatility_estimate * stop_multiplier)
            take_profit = current_price * (1 - volatility_estimate * profit_multiplier)
        
        return stop_loss, take_profit
    
    def _get_default_decision(self, reason: str) -> Dict[str, Any]:
        """기본 결정 (거래 안함)"""
        return {
            'timestamp': datetime.now(),
            'action': 'HOLD',
            'reason': reason,
            'quantity': 0.0,
            'ai_confidence': 0.0,
            'signal_confidence': 0.0,
            'signal_quality': {'overall_score': 0.0},
            'stop_loss': None,
            'take_profit': None
        }
    
    def execute_decision(self, decision: Dict[str, Any]) -> bool:
        """거래 결정 실행"""
        
        if decision['action'] == 'HOLD':
            print(f"거래 없음: {decision['reason']}")
            return True
        
        print(f"\nAI 거래 결정:")
        print(f"   액션: {decision['action']}")
        print(f"   수량: ${decision['quantity']:.2f}")
        print(f"   AI 신뢰도: {decision['ai_confidence']:.2f}")
        print(f"   Signal 신뢰도: {decision['signal_confidence']:.2f}")
        print(f"   신호 품질: {decision['signal_quality']['overall_score']:.2f}")
        print(f"   스탑로스: ${decision['stop_loss']:.2f}" if decision['stop_loss'] else "   스탑로스: 없음")
        print(f"   익절가: ${decision['take_profit']:.2f}" if decision['take_profit'] else "   익절가: 없음")
        print(f"   이유: {decision['reason']}")
        
        return True
    
    def update_trade_result(self, trade_pnl: float):
        """거래 결과 업데이트"""
        self.total_trades += 1
        
        if trade_pnl > 0:
            self.winning_trades += 1
            self.consecutive_losses = 0
            print(f"수익 거래: +${trade_pnl:.2f}")
        else:
            self.consecutive_losses += 1
            print(f"손실 거래: ${trade_pnl:.2f}")
        
        self.current_balance += trade_pnl
        
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
        else:
            drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
            self.max_drawdown = max(self.max_drawdown, drawdown)
        
        win_rate = self.winning_trades / self.total_trades
        total_return = (self.current_balance - self.initial_balance) / self.initial_balance
        
        print(f"현재 통계: 승률 {win_rate:.1%}, 수익률 {total_return:.1%}, 잔고 ${self.current_balance:.2f}")

class IntegrationHelper:
    """기존 시스템과의 통합 도우미"""
    
    @staticmethod
    def create_live_agent_from_config(config_path: str = "config/live_trading.json") -> LiveTradingAgent:
        """설정 파일로부터 라이브 에이전트 생성"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            model_path = config.get('model_path', 'agent/final_optimized_model_80d.pth')
            initial_balance = config.get('initial_balance', 10000.0)
            
            return LiveTradingAgent(model_path, initial_balance)
            
        except FileNotFoundError:
            print(f"설정 파일이 없습니다: {config_path}")
            print("기본 설정으로 에이전트를 생성합니다.")
            return LiveTradingAgent('agent/final_optimized_model_80d.pth')
    
    @staticmethod
    def integrate_with_main_loop(live_agent: LiveTradingAgent):
        """메인 루프와의 통합 가이드"""
        
        integration_code = '''
# main.py에서 80차원 Signal 기반 통합 방법:

from agent.live_trading_agent import LiveTradingAgent

# 1. AI 에이전트 초기화 (80차원)
live_agent = LiveTradingAgent('agent/final_optimized_model_80d.pth')

# 2. 메인 루프에서
while True:
    try:
        # 기존 전략 실행으로 Flatten Signal 생성
        strategy_executor.execute_all_strategies()
        flattened_signal = strategy_executor.get_flattened_signals()  # 이미 Flatten됨
        
        # 현재 캔들 정보
        current_candle = {
            'open': latest_candle.open,
            'high': latest_candle.high,
            'low': latest_candle.low,
            'close': latest_candle.close,
            'volume': latest_candle.volume
        }
        
        # AI 최종 결정 (Signal의 모든 정보 활용)
        ai_decision = live_agent.make_trading_decision(flattened_signal, current_candle)
        
        # 거래 실행
        if ai_decision['action'] != 'HOLD':
            result = execute_trade(ai_decision)
            if result:
                live_agent.update_trade_result(result['pnl'])
        
        time.sleep(180)  # 3분 대기
        
    except Exception as e:
        print(f"오류 발생: {e}")
        time.sleep(60)
        '''
        
        print("80차원 Signal 기반 통합 방법:")
        print(integration_code)

def example_usage():
    """80차원 Signal 기반 사용 예시"""
    
    # AI 에이전트 초기화
    agent = LiveTradingAgent('agent/final_optimized_model_80d.pth')
    
    # 실제 Signal 데이터 예시 (Flatten 형태)
    signal_data = {
        "long_term_raw_vpvr_score": 0.7326526611001036,
        "medium_term_raw_bollinger_squeeze_score": 0.12975921344566924,
        "indicator_lvn": 2291.8958399999997,
        "short_term_raw_vwap_pinball_action": "SELL",
        "medium_term_raw_bollinger_squeeze_entry": 2295.6400000000003,
        "short_term_signals_used": 3,
        "indicator_vwap": 2282.330457020576,
        "long_term_raw_oi_delta_score": 0,
        "short_term_raw_session_score": 0.7,
        "short_term_raw_vwap_pinball_entry": 2297.02,
        "short_term_reason": "synergy SELL: net_score=-0.254, confidence=MEDIUM",
        "medium_term_raw_bollinger_squeeze_action": "BUY",
        "medium_term_buy_score": 0.8839643640350829,
        "long_term_raw_vpvr_entry": 2294.9300000000003,
        "short_term_raw_vol_spike_score": 0,
        "long_term_signals_used": 2,
        "medium_term_raw_support_resistance_score": 0.6799725877192946,
        "short_term_market_context": "RANGING",
        "long_term_risk_multiplier": 0.5,
        "long_term_max_holding_minutes": 1440,
        "medium_term_signals_used": 2,
        "long_term_stop_used": 2301.71,
        "medium_term_raw_htf_trend_score": 0.2,
        "medium_term_raw_support_resistance_action": "BUY",
        "indicator_hvn": 2285.09328,
        "long_term_qty": 18.436578171092137,
        "long_term_action": "SHORT",
        "short_term_risk_multiplier": 1,
        "short_term_raw_session_action": "BUY",
        "short_term_sell_score": 0.9544264126713916,
        "medium_term_market_context": "NEUTRAL",
        "indicator_prev_day_high": 2335.23,
        "short_term_confidence": "MEDIUM",
        "short_term_entry_used": 2297.02,
        "indicator_opening_range_low": 2271.42,
        "short_term_raw_liquidity_grab_action": "HOLD",
        "long_term_institutional_bias": "BEARISH",
        "medium_term_sell_score": 0.7729280342708474,
        "medium_term_raw_htf_trend_action": "HOLD",
        "short_term_action": "SHORT",
        "short_term_raw_orderflow_cvd_score": 0,
        "short_term_raw_liquidity_grab_score": 0,
        "timestamp": 1726531560000,
        "long_term_raw_funding_rate_score": 0.264457121888191,
        "long_term_raw_ichimoku_action": "SELL",
        "long_term_raw_vpvr_action": "SELL",
        "medium_term_raw_multi_timeframe_action": "HOLD",
        "short_term_stop_used": 2299.1609275355754,
        "long_term_timeframe": "1h+",
        "long_term_net_score": -1.3954,
        "long_term_reason": "long synergy SELL: net_score=-1.395, confidence=HIGH",
        "medium_term_raw_bollinger_squeeze_stop": 2279.3600955143643,
        "short_term_risk_usd": 50,
        "long_term_confidence": "HIGH",
        "long_term_raw_vpvr_stop": 2301.71,
        "short_term_raw_zscore_mean_reversion_action": "HOLD",
        "long_term_market_context": "NEUTRAL",
        "indicator_poc": 2306.63472,
        "short_term_leverage": 20,
        "short_term_raw_zscore_mean_reversion_score": 0.402117330177208,
        "short_term_timeframe": "3m",
        "long_term_entry_used": 2294.9300000000003,
        "short_term_buy_score": 0.7,
        "long_term_buy_score": 0,
        "medium_term_leverage": 10,
        "medium_term_reason": "medium synergy HOLD: net_score=0.111, confidence=LOW",
        "medium_term_risk_usd": 0,
        "short_term_raw_vwap_pinball_stop": 2299.1609275355754,
        "indicator_atr": 1.9818550711503902,
        "long_term_raw_oi_delta_action": "HOLD",
        "short_term_raw_session_stop": 2285.3972173932743,
        "medium_term_max_holding_minutes": 240,
        "indicator_prev_day_low": 2252.76,
        "long_term_sell_score": 1.3954282353240868,
        "indicator_opening_range_high": 2288.38,
        "short_term_raw_orderflow_cvd_action": "HOLD",
        "medium_term_raw_ema_confluence_action": "SELL",
        "medium_term_raw_ema_confluence_score": 0.6441066952257062,
        "short_term_raw_vol_spike_action": "HOLD",
        "medium_term_raw_multi_timeframe_score": 0,
        "medium_term_net_score": 0.111,
        "short_term_net_score": -0.2544,
        "short_term_raw_vwap_pinball_score": 0.6555126460655163,
        "medium_term_confidence": "LOW",
        "indicator_vwap_std": 7.909989276587904,
        "long_term_macro_trend_strength": "MEDIUM",
        "medium_term_action": "HOLD",
        "medium_term_timeframe": "15m",
        "short_term_max_holding_minutes": 30,
        "long_term_leverage": 5,
        "short_term_raw_session_entry": 2298.1800000000003,
        "long_term_raw_funding_rate_action": "SELL",
        "long_term_raw_ichimoku_score": 1,
        "short_term_qty": 467.0872709996812,
        "long_term_risk_usd": 25
    }
    
    current_candle = {
        'open': 3000,
        'high': 3010,
        'low': 2995,
        'close': 3005,
        'volume': 1000000
    }
    
    # 거래 결정 (Signal의 모든 정보 활용)
    decision = agent.make_trading_decision(signal_data, current_candle)
    print(f"AI 결정: {decision}")
    
    # 거래 실행
    agent.execute_decision(decision)

def create_enhanced_config():
    """80차원 Signal 기반 설정 파일 생성"""
    config = {
        "model_path": "agent/final_optimized_model_80d.pth",
        "initial_balance": 10000.0,
        "state_dimensions": 80,
        "feature_extraction": {
            "price_indicators": 20,
            "technical_scores": 25,
            "decision_features": 25,
            "portfolio_features": 10
        },
        "risk_settings": {
            "max_drawdown_limit": 0.12,
            "max_position_limit": 0.15,
            "max_leverage": 5.0,
            "consecutive_loss_limit": 3
        },
        "trading_settings": {
            "min_confidence_threshold": 0.4,
            "min_signal_quality": 0.3,
            "min_position_change": 0.2,
            "use_signal_indicators": True,
            "use_raw_scores": True
        }
    }
    
    os.makedirs("config", exist_ok=True)
    with open("config/live_trading_80d.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("80차원 Signal 기반 설정 파일이 생성되었습니다: config/live_trading_80d.json")

if __name__ == "__main__":
    print("80차원 Signal 기반 실시간 트레이딩 에이전트")
    
    # 설정 파일 생성
    create_enhanced_config()
    
    # 통합 가이드 출력
    IntegrationHelper.integrate_with_main_loop(None)
    
    # 사용 예시
    example_usage()