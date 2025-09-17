"""
실시간 트레이딩 에이전트
- 기존 신호 시스템과 완전 통합
- 훈련된 RL 모델을 활용한 최종 거래 결정
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
    # 대안: 직접 DuelingDQN 클래스 정의
    from agent.rl_training_system import DuelingDQN, RLAgent

class SignalQualityAnalyzer:
    """신호 품질 분석기"""
    
    @staticmethod
    def analyze_signal_quality(signal_data: Dict[str, Any]) -> Dict[str, float]:
        """신호 데이터 품질 분석"""
        decisions = SignalQualityAnalyzer._normalize_signal_data(signal_data)

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
        
        for category in ['SHORT_TERM', 'MEDIUM_TERM', 'LONG_TERM']:
            if category in decisions:
                decision = decisions[category]
                action = decision.get('action', 'HOLD')
                confidence = decision.get('meta', {}).get('synergy_meta', {}).get('confidence', 'LOW')
                net_score = float(decision.get('net_score', 0.0))
                
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
    
    @staticmethod
    def _normalize_signal_data(signal_data: Dict) -> Dict:
        """신호 데이터 정규화"""
        if 'decisions' in signal_data:
            return signal_data['decisions']
        
        # parquet 평면화된 형태 처리
        decisions = {}
        for category in ['SHORT_TERM', 'MEDIUM_TERM', 'LONG_TERM']:
            prefix = f"{category.lower()}_"
            decisions[category] = {
                'action': signal_data.get(f'{prefix}action', 'HOLD'),
                'net_score': float(signal_data.get(f'{prefix}net_score', 0.0)),
                'leverage': int(signal_data.get(f'{prefix}leverage', 1)),
                'max_holding_minutes': int(signal_data.get(f'{prefix}max_holding_minutes', 60)),
                'meta': {
                    'synergy_meta': {
                        'confidence': signal_data.get(f'{prefix}confidence', 'LOW'),
                        'buy_score': float(signal_data.get(f'{prefix}buy_score', 0.0)),
                        'sell_score': float(signal_data.get(f'{prefix}sell_score', 0.0)),
                        'conflicts_detected': []
                    }
                }
            }
        
        return decisions

class StateVectorBuilder:
    """상태 벡터 구성기"""
    
    def __init__(self):
        self.last_candle = None
    
    def build_state_vector(self, signal_data: Dict, current_candle: Dict, 
                          portfolio_state: Dict) -> np.ndarray:
        """상태 벡터 구성 (훈련 시와 동일한 60차원)"""
        
        self.last_candle = current_candle
        
        # 1. 가격 특성 (20개) - 현재 캔들 기반
        price_features = self._extract_price_features(current_candle)
        
        # 2. 신호 특성 (30개)
        signal_features = self._extract_signal_features(signal_data)
        
        # 3. 포트폴리오 특성 (10개)
        portfolio_features = self._extract_portfolio_features(portfolio_state)
        
        # 모든 특성 결합
        state = np.concatenate([price_features, signal_features, portfolio_features])
        
        return state.astype(np.float32)
    
    def _extract_price_features(self, candle: Dict) -> np.ndarray:
        """현재 캔들에서 가격 특성 추출"""
        
        high = candle['high']
        low = candle['low'] 
        close = candle['close']
        open_price = candle['open']
        volume = candle.get('volume', 0)
        
        # 현재 캔들 기반 특성
        price_change = (close - open_price) / open_price if open_price > 0 else 0.0
        price_range = (high - low) / close if close > 0 else 0.0
        
        # 20개 특성 구성 (실제 지표값이 있으면 사용, 없으면 추정값)
        features = [
            price_change,        # 현재 캔들 수익률
            price_range,         # 현재 변동성
            0.0,                 # returns_mean (중립)
            price_range,         # returns_std 대신
            0.5,                 # RSI (중립값)
            0.5,                 # BB position (중립값)
            0.0, 0.0, 0.0,       # MA ratios (중립값)
            0.0,                 # volume ratio
            price_range,         # volatility
            0.5,                 # price position
            0.0, 0.0, 0.0, 0.0,  # 추가 기술적 지표들
            0.0, 0.0, 0.0, 0.0, 0.0  # 나머지 패딩
        ]
        
        return np.array(features[:20], dtype=np.float32)
    
    def _extract_signal_features(self, signal_data: Dict) -> np.ndarray:
        """신호 특성 추출"""
        features = []
        
        # 신호 데이터 정규화
        decisions = SignalQualityAnalyzer._normalize_signal_data(signal_data)
        
        # 각 시간대별 신호 특성 (3개 × 8개 = 24개)
        for category in ['SHORT_TERM', 'MEDIUM_TERM', 'LONG_TERM']:
            if category in decisions:
                decision = decisions[category]
                
                action = decision.get('action', 'HOLD')
                action_strength = 1.0 if action == 'LONG' else (-1.0 if action == 'SHORT' else 0.0)
                
                features.extend([
                    action_strength,
                    float(decision.get('net_score', 0.0)),
                    min(float(decision.get('leverage', 1)) / 10.0, 2.0),
                    min(float(decision.get('max_holding_minutes', 60)) / 1440.0, 1.0),
                ])
                
                meta = decision.get('meta', {}).get('synergy_meta', {})
                confidence = meta.get('confidence', 'LOW')
                confidence_score = 1.0 if confidence == 'HIGH' else (0.5 if confidence == 'MEDIUM' else 0.0)
                
                features.extend([
                    confidence_score,
                    float(meta.get('buy_score', 0.0)),
                    float(meta.get('sell_score', 0.0)),
                    len(meta.get('conflicts_detected', [])) / 5.0
                ])
            else:
                features.extend([0.0] * 8)
        
        # 갈등 및 메타 정보 (6개)
        if 'conflicts' in signal_data:
            conflicts = signal_data['conflicts']
            features.extend([
                1.0 if conflicts.get('has_conflicts', False) else 0.0,
                len(conflicts.get('long_categories', [])) / 3.0,
                len(conflicts.get('short_categories', [])) / 3.0,
                float(signal_data.get('meta', {}).get('active_positions', 0)) / 3.0,
                0.0, 0.0
            ])
        else:
            features.extend([0.0] * 6)
        
        return np.array(features[:30], dtype=np.float32)
    
    def _extract_portfolio_features(self, portfolio_state: Dict) -> np.ndarray:
        """포트폴리오 상태 특성"""
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
        self.max_drawdown_limit = 0.12  # 12%
        self.max_position_limit = 0.15  # 15%
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
    """실시간 트레이딩 에이전트"""
    
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
        self.state_builder = StateVectorBuilder()
        self.risk_manager = RiskManager(initial_balance)
        
        # 훈련된 에이전트 로드
        self.agent = self._load_trained_agent()
        
        print(f"실시간 트레이딩 에이전트 초기화 완료")
        print(f"   모델: {model_path}")
        print(f"   초기 잔고: ${initial_balance:,.2f}")
    
            
    def _load_trained_agent(self):
        """훈련된 에이전트 로드"""
        try:
            # DuelingDQN을 사용하는 RLAgent 생성
            agent = RLAgent(state_size=60)
            
            # 모델 파일 존재 확인
            if not os.path.exists(self.model_path):
                print(f"❌ 모델 파일이 없습니다: {self.model_path}")
                return None
            
            # 모델 로드
            if agent.load_model_with_compatibility(self.model_path):
                agent.epsilon = 0.0
                print(f"✅ 훈련된 모델 로드 성공")
                return agent
            else:
                raise Exception("모델 로드 실패")
                
        except Exception as e:
            print(f"❌ 에이전트 로드 실패: {e}")
            return None
    
    def make_trading_decision(self, signal_data: Dict[str, Any], 
                            current_candle: Dict[str, float]) -> Dict[str, Any]:
        """
        실시간 거래 결정 생성
        
        Args:
            signal_data: 전략에서 생성된 신호
            current_candle: 현재 캔들 데이터
            
        Returns:
            거래 결정 딕셔너리
        """
        
        if self.agent is None:
            return self._get_default_decision("에이전트 로드 실패")
        
        try:
            # 1. 신호 품질 분석
            signal_quality = self.signal_analyzer.analyze_signal_quality(signal_data)
            
            # 2. 포트폴리오 상태 구성
            portfolio_state = self._get_portfolio_state()
            
            # 3. 상태 벡터 구성
            state_vector = self.state_builder.build_state_vector(
                signal_data, current_candle, portfolio_state
            )
            
            # 4. AI 에이전트의 액션 예측
            ai_action = self.agent.act(state_vector)
            
            # 5. 액션을 거래 결정으로 변환
            trading_decision = self._convert_action_to_decision(
                ai_action, current_candle, signal_quality
            )
            
            # 6. 리스크 체크 및 최종 결정
            final_decision = self.risk_manager.check_risk_limits(
                trading_decision, portfolio_state
            )
            
            return final_decision
            
        except Exception as e:
            print(f"거래 결정 생성 오류: {e}")
            return self._get_default_decision(f"오류: {str(e)}")
    
    def _get_portfolio_state(self) -> Dict:
        """현재 포트폴리오 상태"""
        return {
            'current_position': self.current_position,
            'current_leverage': self.current_leverage,
            'current_balance': self.current_balance,
            'balance_ratio': (self.current_balance - self.initial_balance) / self.initial_balance,
            'unrealized_pnl_ratio': 0.0,  # 실시간에서는 단순화
            'total_trades': self.total_trades,
            'win_rate': self.winning_trades / max(self.total_trades, 1),
            'max_drawdown': self.max_drawdown,
            'consecutive_losses': self.consecutive_losses,
            'holding_time': self.holding_time,
            'in_position': self.in_position
        }
    
    def _convert_action_to_decision(self, ai_action: np.ndarray, 
                                  current_candle: Dict, signal_quality: Dict) -> Dict[str, Any]:
        """AI 액션을 실제 거래 결정으로 변환"""
        
        position_change = ai_action[0]
        leverage = ai_action[1] 
        holding_minutes = ai_action[2]
        
        current_price = current_candle['close']
        
        # AI 신뢰도 계산
        ai_confidence = self._calculate_ai_confidence(ai_action, signal_quality)
        
        # 거래 결정 생성
        decision = {
            'timestamp': datetime.now(),
            'current_price': current_price,
            'ai_confidence': ai_confidence,
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
        
        # 액션 해석 (보수적 임계값)
        min_threshold = 0.2
        min_quality = 0.3
        
        if (abs(position_change) > min_threshold and 
            signal_quality['overall_score'] > min_quality and
            ai_confidence > 0.4):
            
            if position_change > min_threshold:
                decision['action'] = 'BUY'
                decision['reason'] = f"AI+신호 추천: Long {position_change:.2f} (품질: {signal_quality['overall_score']:.2f}, 신뢰도: {ai_confidence:.2f})"
            elif position_change < -min_threshold:
                decision['action'] = 'SELL'  
                decision['reason'] = f"AI+신호 추천: Short {abs(position_change):.2f} (품질: {signal_quality['overall_score']:.2f}, 신뢰도: {ai_confidence:.2f})"
            
            # 포지션 크기 계산
            decision['quantity'] = self.risk_manager.calculate_position_size(
                signal_quality['signal_strength'], 
                ai_confidence, 
                self.current_balance
            )
            
            # 스탑 설정
            decision['stop_loss'], decision['take_profit'] = self._calculate_stops(
                current_price, decision['action'], holding_minutes, signal_quality
            )
        else:
            decision['reason'] = (f"임계값 미달 (변경량: {position_change:.2f}, "
                                f"신호품질: {signal_quality['overall_score']:.2f}, "
                                f"AI신뢰도: {ai_confidence:.2f})")
        
        return decision
    
    def _calculate_ai_confidence(self, ai_action: np.ndarray, signal_quality: Dict) -> float:
        """AI와 신호 품질을 결합한 신뢰도 계산"""
        
        # AI 액션 강도 기반 신뢰도
        ai_confidence = min(abs(ai_action[0]) / 2.0, 1.0)
        
        # 신호 품질 신뢰도
        signal_confidence = signal_quality['overall_score']
        
        # 결합 신뢰도 (가중평균)
        combined_confidence = (ai_confidence * 0.6) + (signal_confidence * 0.4)
        
        return min(combined_confidence, 1.0)
    
    def _calculate_stops(self, current_price: float, action: str, 
                        holding_minutes: float, signal_quality: Dict) -> Tuple[Optional[float], Optional[float]]:
        """스탑로스와 익절가 계산"""
        
        if action == 'HOLD':
            return None, None
        
        # 기본 변동성 추정 (현재 캔들 기반)
        if self.state_builder.last_candle:
            candle = self.state_builder.last_candle
            price_range = (candle['high'] - candle['low']) / current_price
            volatility_estimate = max(price_range, 0.01)
        else:
            volatility_estimate = 0.02
        
        # 신호 품질에 따른 조정
        quality_score = signal_quality['overall_score']
        stop_multiplier = 1.5 + (1.0 - quality_score)  # 품질 낮으면 타이트하게
        profit_multiplier = 1.0 + quality_score  # 품질 높으면 더 큰 목표
        
        # 홀딩 시간 조정
        if holding_minutes < 120:  # 2시간 미만 단타
            stop_multiplier *= 0.8
            profit_multiplier *= 0.9
        
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
        print(f"   신호 품질: {decision['signal_quality']['overall_score']:.2f}")
        print(f"   스탑로스: ${decision['stop_loss']:.2f}" if decision['stop_loss'] else "   스탑로스: 없음")
        print(f"   익절가: ${decision['take_profit']:.2f}" if decision['take_profit'] else "   익절가: 없음")
        print(f"   이유: {decision['reason']}")
        
        # 실제 거래소 API 호출은 여기에 구현
        # result = exchange_api.place_order(...)
        
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
        
        # 잔고 및 통계 업데이트
        self.current_balance += trade_pnl
        
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
        else:
            drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
            self.max_drawdown = max(self.max_drawdown, drawdown)
        
        # 통계 출력
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
            
            model_path = config.get('model_path', 'agent/final_optimized_model.pth')
            initial_balance = config.get('initial_balance', 10000.0)
            
            return LiveTradingAgent(model_path, initial_balance)
            
        except FileNotFoundError:
            print(f"설정 파일이 없습니다: {config_path}")
            print("기본 설정으로 에이전트를 생성합니다.")
            return LiveTradingAgent('agent/final_optimized_model.pth')
    
    @staticmethod
    def integrate_with_main_loop(live_agent: LiveTradingAgent):
        """메인 루프와의 통합 가이드"""
        
        integration_code = '''
# main.py에서 통합 방법:

from live_trading_agent import LiveTradingAgent

# 1. AI 에이전트 초기화
live_agent = LiveTradingAgent('agent/final_optimized_model.pth')

# 2. 메인 루프에서
while True:
    try:
        # 기존 전략 실행
        strategy_executor.execute_all_strategies()
        signals = strategy_executor.get_signals()
        base_decision = decision_engine.decide_trade_realtime(signals)
        
        # 현재 캔들 정보
        current_candle = {
            'open': latest_candle.open,
            'high': latest_candle.high,
            'low': latest_candle.low,
            'close': latest_candle.close,
            'volume': latest_candle.volume
        }
        
        # AI 최종 결정
        ai_decision = live_agent.make_trading_decision(base_decision, current_candle)
        
        # 거래 실행
        if ai_decision['action'] != 'HOLD':
            # 실제 거래 실행
            result = execute_trade(ai_decision)
            if result:
                live_agent.update_trade_result(result['pnl'])
        
        time.sleep(180)  # 3분 대기
        
    except Exception as e:
        print(f"오류 발생: {e}")
        time.sleep(60)  # 1분 후 재시도
        '''
        
        print("기존 시스템과의 통합 방법:")
        print(integration_code)

def example_usage():
    """사용 예시"""
    
    # AI 에이전트 초기화
    agent = LiveTradingAgent('agent/final_optimized_model.pth')
    
    # 가상의 신호 데이터 (실제로는 strategy_executor에서)
    signal_data = {
        'decisions': {
            'SHORT_TERM': {
                'action': 'LONG',
                'net_score': 0.7,
                'leverage': 3,
                'max_holding_minutes': 120,
                'meta': {
                    'synergy_meta': {
                        'confidence': 'HIGH',
                        'buy_score': 0.8,
                        'sell_score': 0.1,
                        'conflicts_detected': []
                    }
                }
            },
            'MEDIUM_TERM': {'action': 'HOLD', 'net_score': 0.0, 'leverage': 1, 'max_holding_minutes': 240, 'meta': {'synergy_meta': {'confidence': 'LOW'}}},
            'LONG_TERM': {'action': 'HOLD', 'net_score': 0.0, 'leverage': 1, 'max_holding_minutes': 1440, 'meta': {'synergy_meta': {'confidence': 'LOW'}}}
        },
        'conflicts': {'has_conflicts': False},
        'meta': {'active_positions': 0}
    }
    
    current_candle = {
        'open': 3000,
        'high': 3010,
        'low': 2995,
        'close': 3005,
        'volume': 1000000
    }
    
    # 거래 결정
    decision = agent.make_trading_decision(signal_data, current_candle)
    print(f"AI 결정: {decision}")
    
    # 거래 실행
    agent.execute_decision(decision)

def create_default_config():
    """기본 설정 파일 생성"""
    config = {
        "model_path": "agent/final_optimized_model.pth",
        "initial_balance": 10000.0,
        "risk_settings": {
            "max_drawdown_limit": 0.12,
            "max_position_limit": 0.15,
            "max_leverage": 5.0,
            "consecutive_loss_limit": 3
        },
        "trading_settings": {
            "min_confidence_threshold": 0.4,
            "min_signal_quality": 0.3,
            "min_position_change": 0.2
        }
    }
    
    os.makedirs("config", exist_ok=True)
    with open("config/live_trading.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("기본 설정 파일이 생성되었습니다: config/live_trading.json")

if __name__ == "__main__":
    print("실시간 트레이딩 에이전트 예시")
    
    # 기본 설정 파일 생성
    create_default_config()
    
    # 통합 가이드 출력
    IntegrationHelper.integrate_with_main_loop(None)
    
    # 사용 예시
    example_usage()