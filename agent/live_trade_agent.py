# 수정된 실시간 트레이딩 시스템 - 기존 신호 데이터 활용

import numpy as np
import pandas as pd
import torch
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import json

from agent.ai import ImprovedCryptoRLAgent

class LiveTradingAgent:
    """실시간 거래를 위한 에이전트 - 기존 신호 데이터 활용"""
    
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
        
        # 가격 정보 (최소한만 유지)
        self.current_price = 0.0
        self.last_candle = None
        
        # 훈련된 에이전트 로드
        self.agent = self._load_trained_agent()
        
        print(f"✅ 실시간 트레이딩 에이전트 초기화 완료")
        print(f"   모델: {model_path}")
        print(f"   초기 잔고: ${initial_balance:,.2f}")
    
    def _load_trained_agent(self):
        """훈련된 에이전트 로드"""
        try:
            # 에이전트 생성 (상태 크기는 60으로 고정)
            agent = ImprovedCryptoRLAgent(state_size=60)
            
            # 모델 로드
            if agent.safe_load_model(self.model_path):
                agent.epsilon = 0.0  # 실거래에서는 탐험 비활성화
                print(f"✅ 훈련된 모델 로드 성공")
                return agent
            else:
                raise Exception("모델 로드 실패")
                
        except Exception as e:
            print(f"❌ 에이전트 로드 실패: {e}")
            return None
    
    def make_trading_decision(self, 
                            signal_data: Dict[str, Any], 
                            current_candle: Dict[str, float]) -> Dict[str, Any]:
        """
        실시간 거래 결정 생성
        
        Args:
            signal_data: 전략에서 생성된 신호 (parquet 형태 또는 중첩 딕셔너리)
            current_candle: 현재 캔들 데이터 {'open', 'high', 'low', 'close', 'volume'}
            
        Returns:
            거래 결정 딕셔너리
        """
        
        if self.agent is None:
            return self._get_default_decision("에이전트 로드 실패")
        
        self.current_price = current_candle['close']
        self.last_candle = current_candle
        
        try:
            # 1. 신호 데이터를 훈련된 형태로 변환
            state_vector = self._convert_signal_to_state(signal_data, current_candle)
            
            # 2. AI 에이전트의 액션 예측
            ai_action = self.agent.act(state_vector)
            
            # 3. 액션을 거래 결정으로 변환
            trading_decision = self._convert_action_to_decision(ai_action, signal_data)
            
            # 4. 리스크 체크 및 최종 결정
            final_decision = self._apply_risk_controls(trading_decision)
            
            return final_decision
            
        except Exception as e:
            print(f"❌ 거래 결정 생성 오류: {e}")
            return self._get_default_decision(f"오류: {str(e)}")
    
    def _convert_signal_to_state(self, signal_data: Dict, current_candle: Dict) -> np.ndarray:
        """신호 데이터를 훈련된 상태 벡터 형태로 변환"""
        
        # 1. 가격 특성 (20개) - 현재 캔들 정보 활용
        price_features = self._extract_price_features_simple(current_candle)
        
        # 2. 신호 특성 (30개) - 기존 신호 데이터 활용
        signal_features = self._extract_signal_features(signal_data)
        
        # 3. 포트폴리오 특성 (10개)
        portfolio_features = self._extract_portfolio_features()
        
        # 모든 특성 결합
        state = np.concatenate([price_features, signal_features, portfolio_features])
        
        return state.astype(np.float32)
    
    def _extract_price_features_simple(self, candle: Dict) -> np.ndarray:
        """현재 캔들에서 간단한 가격 특성 추출"""
        
        # 현재 캔들만으로 계산 가능한 특성들
        high = candle['high']
        low = candle['low'] 
        close = candle['close']
        open_price = candle['open']
        volume = candle.get('volume', 0)
        
        # 기본 특성들
        price_change = (close - open_price) / open_price if open_price > 0 else 0.0
        price_range = (high - low) / close if close > 0 else 0.0
        
        # 20개 특성 구성 (실제 계산된 값이 있으면 사용하고, 없으면 중립값)
        features = [
            price_change,        # 현재 캔들 수익률
            price_range,         # 변동성 대리값
            0.0,                 # returns_mean (중립)
            price_range,         # returns_std 대신 price_range
            0.5,                 # RSI (중립값 - 실제 계산은 signal_data에서)
            0.5,                 # BB position (중립값)
            0.0, 0.0, 0.0,       # MA ratios (중립값)
            0.0,                 # volume ratio (계산 복잡)
            price_range,         # volatility
            0.0,                 # price position
            0.0, 0.0, 0.0, 0.0,  # 추가 기술적 지표들
            0.0, 0.0, 0.0, 0.0, 0.0  # 나머지 패딩
        ]
        
        return np.array(features[:20], dtype=np.float32)
    
    def _extract_signal_features(self, signal_data: Dict) -> np.ndarray:
        """신호 데이터에서 특성 추출 (훈련 시와 동일한 로직)"""
        features = []
        
        # 신호 데이터 형태 확인 및 표준화
        decisions = self._normalize_signal_data(signal_data)
        
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
                
                # 메타 정보
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
        conflicts = signal_data.get('conflicts', {})
        features.extend([
            1.0 if conflicts.get('has_conflicts', False) else 0.0,
            len(conflicts.get('long_categories', [])) / 3.0,
            len(conflicts.get('short_categories', [])) / 3.0,
            float(signal_data.get('meta', {}).get('active_positions', 0)) / 3.0,
            0.0, 0.0  # 예비
        ])
        
        return np.array(features[:30], dtype=np.float32)
    
    def _normalize_signal_data(self, signal_data: Dict) -> Dict:
        """신호 데이터를 표준 형태로 변환"""
        
        # 이미 중첩 딕셔너리 형태인 경우 (ai.py와 동일)
        if 'decisions' in signal_data:
            return signal_data['decisions']
        
        # parquet 평면화된 형태인 경우 (agent.py와 동일)
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
    
    def _extract_portfolio_features(self) -> np.ndarray:
        """현재 포트폴리오 상태 특성"""
        features = [
            self.current_position,
            self.current_leverage / 20.0,
            (self.current_balance - self.initial_balance) / self.initial_balance,
            0.0,  # unrealized_pnl (단순화)
            min(self.total_trades / 100.0, 1.0),
            self.winning_trades / max(self.total_trades, 1),
            self.max_drawdown,
            min(self.consecutive_losses / 10.0, 1.0),
            min(self.holding_time / 1440.0, 1.0),
            1.0 if self.in_position else 0.0
        ]
        return np.array(features, dtype=np.float32)
    
    def _convert_action_to_decision(self, ai_action: np.ndarray, signal_data: Dict) -> Dict[str, Any]:
        """AI 액션을 실제 거래 결정으로 변환"""
        
        position_change = ai_action[0]
        leverage = ai_action[1] 
        holding_minutes = ai_action[2]
        
        # 신호 품질 분석
        signal_quality = self._analyze_signal_quality(signal_data)
        
        # 거래 결정 생성
        decision = {
            'timestamp': datetime.now(),
            'current_price': self.current_price,
            'ai_confidence': self._calculate_confidence(ai_action, signal_quality),
            'signal_quality': signal_quality,
            'position_change': position_change,
            'target_leverage': min(leverage, 5.0),  # 최대 5배로 제한
            'target_holding_minutes': holding_minutes,
            'action': 'HOLD',
            'reason': '',
            'quantity': 0.0,
            'stop_loss': None,
            'take_profit': None
        }
        
        # 액션 해석 (더 보수적으로)
        min_threshold = 0.2  # 최소 임계값 증가
        
        if abs(position_change) > min_threshold and signal_quality['overall_score'] > 0.3:
            if position_change > min_threshold:
                decision['action'] = 'BUY'
                decision['reason'] = f"AI+신호 추천: Long {position_change:.2f} (품질: {signal_quality['overall_score']:.2f})"
            elif position_change < -min_threshold:
                decision['action'] = 'SELL'  
                decision['reason'] = f"AI+신호 추천: Short {abs(position_change):.2f} (품질: {signal_quality['overall_score']:.2f})"
            
            # 포지션 크기 계산 (신호 품질 반영)
            decision['quantity'] = self._calculate_position_size(
                position_change, leverage, signal_quality['overall_score']
            )
            
            # 스탑 설정
            decision['stop_loss'], decision['take_profit'] = self._calculate_stops(
                decision['action'], holding_minutes, signal_quality
            )
        else:
            decision['reason'] = f"임계값 미달 (변경량: {position_change:.2f}, 신호품질: {signal_quality['overall_score']:.2f})"
        
        return decision
    
    def _analyze_signal_quality(self, signal_data: Dict) -> Dict:
        """신호 데이터 품질 분석"""
        decisions = self._normalize_signal_data(signal_data)
        
        quality_metrics = {
            'high_confidence_signals': 0,
            'total_signals': 0,
            'agreement_score': 0.0,
            'overall_score': 0.0
        }
        
        actions = []
        confidences = []
        
        for category in ['SHORT_TERM', 'MEDIUM_TERM', 'LONG_TERM']:
            if category in decisions:
                decision = decisions[category]
                action = decision.get('action', 'HOLD')
                confidence = decision.get('meta', {}).get('synergy_meta', {}).get('confidence', 'LOW')
                
                if action != 'HOLD':
                    quality_metrics['total_signals'] += 1
                    actions.append(1 if action == 'LONG' else -1)
                    
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
            
            quality_metrics['agreement_score'] = action_agreement
            quality_metrics['overall_score'] = (action_agreement + avg_confidence) / 2
        
        return quality_metrics
    
    def _calculate_confidence(self, ai_action: np.ndarray, signal_quality: Dict) -> float:
        """AI와 신호 품질을 결합한 신뢰도 계산"""
        
        # AI 신뢰도
        ai_confidence = min(abs(ai_action[0]) / 2.0, 1.0)
        
        # 신호 품질 신뢰도
        signal_confidence = signal_quality['overall_score']
        
        # 결합 신뢰도 (가중평균)
        combined_confidence = (ai_confidence * 0.6) + (signal_confidence * 0.4)
        
        return min(combined_confidence, 1.0)
    
    def _calculate_position_size(self, position_change: float, leverage: float, signal_quality: float) -> float:
        """포지션 크기 계산 (신호 품질 반영)"""
        
        # 기본 리스크 (잔고의 1-3%)
        base_risk_pct = 0.01 + (signal_quality * 0.02)  # 1-3%
        base_risk = self.current_balance * base_risk_pct
        
        # 포지션 변경량 반영
        position_multiplier = min(abs(position_change), 1.0)
        
        # 레버리지 제한
        safe_leverage = min(leverage, 3.0)  # 더 보수적
        
        # 최종 포지션 크기
        position_usd = base_risk * position_multiplier * safe_leverage
        
        # 최대 잔고의 15%로 제한
        max_position = self.current_balance * 0.15
        
        return min(position_usd, max_position)
    
    def _calculate_stops(self, action: str, holding_minutes: float, signal_quality: Dict) -> Tuple[Optional[float], Optional[float]]:
        """스탑로스와 익절가 계산"""
        
        if action == 'HOLD':
            return None, None
        
        # ATR 대신 캔들 정보 활용
        if self.last_candle:
            price_range = (self.last_candle['high'] - self.last_candle['low']) / self.current_price
            volatility_estimate = max(price_range, 0.01)  # 최소 1%
        else:
            volatility_estimate = 0.02  # 기본 2%
        
        # 신호 품질에 따른 스탑 조정
        stop_multiplier = 1.5 + (1.0 - signal_quality['overall_score'])  # 품질 낮으면 타이트하게
        profit_multiplier = 1.0 + signal_quality['overall_score']  # 품질 높으면 더 큰 목표
        
        if action == 'BUY':
            stop_loss = self.current_price * (1 - volatility_estimate * stop_multiplier)
            take_profit = self.current_price * (1 + volatility_estimate * profit_multiplier)
        else:  # SELL
            stop_loss = self.current_price * (1 + volatility_estimate * stop_multiplier)
            take_profit = self.current_price * (1 - volatility_estimate * profit_multiplier)
        
        return stop_loss, take_profit
    
    def _apply_risk_controls(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """최종 리스크 체크"""
        
        # 1. 최대 드로우다운 체크
        current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance if self.peak_balance > 0 else 0
        if current_drawdown > 0.12:  # 12% 이상 손실시
            decision['action'] = 'HOLD'
            decision['reason'] = f"리스크 관리: 최대 손실 한도 ({current_drawdown:.1%})"
            decision['quantity'] = 0.0
            return decision
        
        # 2. 연속 손실 체크
        if self.consecutive_losses > 3:
            decision['quantity'] *= 0.5
            decision['reason'] += f" (연속손실 {self.consecutive_losses}회, 크기 감소)"
        
        # 3. 신뢰도 체크
        if decision['ai_confidence'] < 0.4:
            decision['action'] = 'HOLD'
            decision['reason'] = f"신뢰도 부족 ({decision['ai_confidence']:.2f})"
            decision['quantity'] = 0.0
        
        # 4. 포지션 크기 최종 검증
        if decision['quantity'] > self.current_balance * 0.2:
            decision['quantity'] = self.current_balance * 0.2
            decision['reason'] += " (포지션 크기 제한)"
        
        return decision
    
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
            print(f"⏸️  거래 없음: {decision['reason']}")
            return True
        
        print(f"\n📊 AI 거래 결정:")
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
            print(f"✅ 수익 거래: +${trade_pnl:.2f}")
        else:
            self.consecutive_losses += 1
            print(f"❌ 손실 거래: ${trade_pnl:.2f}")
        
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
        
        print(f"📈 현재 통계: 승률 {win_rate:.1%}, 수익률 {total_return:.1%}, 잔고 ${self.current_balance:.2f}")

# =================================================================
# 사용 예시 (기존 시스템과 통합)
# =================================================================

def integrate_with_strategy_executor():
    """기존 strategy_executor와의 통합 예시"""
    
    print("""
    🔗 기존 시스템 통합 방법:

    1. main.py에서:
    ```python
    from agent.live_trade_agent import LiveTradingAgent
    
    # AI 에이전트 초기화
    live_agent = LiveTradingAgent('agent/final_optimized_model.pth')
    
    # 메인 루프에서
    while True:
        # 기존 전략 실행
        strategy_executor.execute_all_strategies()
        signals = strategy_executor.get_signals()
        decision = decision_engine.decide_trade_realtime(signals)
        
        # AI 에이전트 결정 추가
        current_candle = get_current_candle()
        ai_decision = live_agent.make_trading_decision(decision, current_candle)
        
        # 최종 거래 실행
        if ai_decision['action'] != 'HOLD':
            execute_trade(ai_decision)
            
        time.sleep(180)  # 3분 대기
    ```
    
    2. decision_engine.py에서 AI 결정 통합:
    ```python
    def decide_trade_with_ai(self, signals, ai_agent, current_candle):
        # 기존 결정
        base_decision = self.decide_trade_realtime(signals)
        
        # AI 결정
        ai_decision = ai_agent.make_trading_decision(base_decision, current_candle)
        
        # 결합 로직 (예: AI가 HOLD이면 기존 결정, AI가 거래면 AI 우선)
        if ai_decision['action'] != 'HOLD':
            return ai_decision
        else:
            return base_decision
    ```
    """)

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
