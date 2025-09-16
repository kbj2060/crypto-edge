# 실시간 트레이딩 시스템 - 훈련된 에이전트로 실제 거래 결정

import numpy as np
import pandas as pd
import torch
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import json

from agent.ai import ImprovedCryptoRLAgent

class LiveTradingAgent:
    """실시간 거래를 위한 에이전트 래퍼"""
    
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
        
        # 가격 히스토리 (상태 계산용)
        self.price_history = []
        
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
    
    def update_price(self, ohlcv_data: Dict[str, float]):
        """새로운 가격 데이터 업데이트"""
        self.price_history.append(ohlcv_data)
        
        # 최대 100개까지만 유지 (메모리 절약)
        if len(self.price_history) > 100:
            self.price_history.pop(0)
    
    def make_trading_decision(self, current_signals: Dict[str, Any], 
                            current_price: float) -> Dict[str, Any]:
        """
        실시간 거래 결정 생성
        
        Args:
            current_signals: 당신의 전략에서 생성된 최신 신호
            current_price: 현재 가격
            
        Returns:
            거래 결정 딕셔너리
        """
        
        if self.agent is None:
            return self._get_default_decision("에이전트 로드 실패")
        
        if len(self.price_history) < 20:
            return self._get_default_decision("가격 히스토리 부족")
        
        try:
            # 1. 현재 상태 구성
            current_state = self._build_current_state(current_signals, current_price)
            
            # 2. AI 에이전트의 액션 예측
            ai_action = self.agent.act(current_state)
            
            # 3. 액션을 거래 결정으로 변환
            trading_decision = self._convert_action_to_decision(ai_action, current_price, current_signals)
            
            # 4. 리스크 체크 및 최종 결정
            final_decision = self._apply_risk_controls(trading_decision, current_price)
            
            return final_decision
            
        except Exception as e:
            print(f"❌ 거래 결정 생성 오류: {e}")
            return self._get_default_decision(f"오류: {str(e)}")
    
    def _build_current_state(self, signals: Dict[str, Any], current_price: float) -> np.ndarray:
        """현재 상태 벡터 구성 (훈련 시와 동일한 형태로)"""
        
        # 1. 가격 특성 (20개)
        price_features = self._extract_price_features()
        
        # 2. 신호 특성 (30개)
        signal_features = self._extract_signal_features(signals)
        
        # 3. 포트폴리오 특성 (10개)
        portfolio_features = self._extract_portfolio_features()
        
        # 모든 특성 결합
        state = np.concatenate([price_features, signal_features, portfolio_features])
        
        return state.astype(np.float32)
    
    def _extract_price_features(self) -> np.ndarray:
        """가격 히스토리에서 특성 추출 (훈련 시와 동일)"""
        if len(self.price_history) < 20:
            return np.zeros(20, dtype=np.float32)
        
        # DataFrame으로 변환
        df = pd.DataFrame(self.price_history[-20:])
        
        features = []
        
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # 수익률 특성
        returns = close.pct_change().fillna(0)
        features.extend([
            returns.mean(),
            returns.std(),
            returns.iloc[-1],
            returns.tail(5).mean()
        ])
        
        # RSI
        if len(close) >= 14:
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / (loss + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            features.append(rsi.iloc[-1] / 100.0)
        else:
            features.append(0.5)
        
        # 볼린저 밴드 위치
        if len(close) >= 20:
            sma = close.rolling(window=20, min_periods=1).mean()
            std = close.rolling(window=20, min_periods=1).std()
            bb_upper = sma + (std * 2)
            bb_lower = sma - (std * 2)
            bb_width = bb_upper.iloc[-1] - bb_lower.iloc[-1]
            if bb_width > 0:
                bb_position = (close.iloc[-1] - bb_lower.iloc[-1]) / bb_width
            else:
                bb_position = 0.5
            features.append(bb_position)
        else:
            features.append(0.5)
        
        # 이동평균 비율
        for window in [5, 10, 20]:
            if len(close) >= window:
                ma = close.rolling(window=window, min_periods=1).mean()
                ma_ratio = (close.iloc[-1] / ma.iloc[-1] - 1) if ma.iloc[-1] > 0 else 0.0
                features.append(ma_ratio)
            else:
                features.append(0.0)
        
        # 나머지 특성들로 20개 맞추기
        while len(features) < 20:
            features.append(0.0)
        
        return np.array(features[:20], dtype=np.float32)
    
    def _extract_signal_features(self, signals: Dict[str, Any]) -> np.ndarray:
        """신호에서 특성 추출 (훈련 시와 동일)"""
        features = []
        
        # 각 시간대별 신호 특성
        for category in ['SHORT_TERM', 'MEDIUM_TERM', 'LONG_TERM']:
            if 'decisions' in signals and category in signals['decisions']:
                decision = signals['decisions'][category]
                
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
        
        # 갈등 및 메타 정보
        if 'conflicts' in signals:
            conflicts = signals['conflicts']
            features.extend([
                1.0 if conflicts.get('has_conflicts', False) else 0.0,
                len(conflicts.get('long_categories', [])) / 3.0,
                len(conflicts.get('short_categories', [])) / 3.0,
                float(signals.get('meta', {}).get('active_positions', 0)) / 3.0,
                0.0,
                0.0
            ])
        else:
            features.extend([0.0] * 6)
        
        return np.array(features[:30], dtype=np.float32)
    
    def _extract_portfolio_features(self) -> np.ndarray:
        """현재 포트폴리오 상태 특성"""
        features = [
            self.current_position,
            self.current_leverage / 20.0,
            (self.current_balance - self.initial_balance) / self.initial_balance,
            0.0,  # unrealized_pnl (실시간에서는 계산 복잡)
            min(self.total_trades / 100.0, 1.0),
            self.winning_trades / max(self.total_trades, 1),
            self.max_drawdown,
            0.0,  # consecutive_losses (단순화)
            min(self.holding_time / 1440.0, 1.0),
            1.0 if self.in_position else 0.0
        ]
        return np.array(features, dtype=np.float32)
    
    def _convert_action_to_decision(self, ai_action: np.ndarray, current_price: float, 
                                  signals: Dict[str, Any]) -> Dict[str, Any]:
        """AI 액션을 실제 거래 결정으로 변환"""
        
        position_change = ai_action[0]
        leverage = ai_action[1] 
        holding_minutes = ai_action[2]
        
        # 거래 결정 생성
        decision = {
            'timestamp': datetime.now(),
            'current_price': current_price,
            'ai_confidence': self._calculate_confidence(ai_action),
            'position_change': position_change,
            'target_leverage': leverage,
            'target_holding_minutes': holding_minutes,
            'action': 'HOLD',  # 기본값
            'reason': '',
            'quantity': 0.0,
            'stop_loss': None,
            'take_profit': None
        }
        
        # 액션 해석
        if abs(position_change) > 0.1:  # 의미있는 포지션 변경
            if position_change > 0.1:
                decision['action'] = 'BUY'
                decision['reason'] = f"AI 추천: Long 포지션 {position_change:.2f}"
            elif position_change < -0.1:
                decision['action'] = 'SELL'  
                decision['reason'] = f"AI 추천: Short 포지션 {abs(position_change):.2f}"
            
            # 포지션 크기 계산 (Kelly Criterion 기반)
            decision['quantity'] = self._calculate_position_size(position_change, leverage)
            
            # 스탑로스/익절 설정
            decision['stop_loss'], decision['take_profit'] = self._calculate_stops(
                current_price, decision['action'], holding_minutes
            )
        
        return decision
    
    def _calculate_confidence(self, ai_action: np.ndarray) -> float:
        """AI 결정의 신뢰도 계산"""
        # 포지션 변경량이 클수록 높은 신뢰도
        position_confidence = min(abs(ai_action[0]) / 2.0, 1.0)
        
        # 레버리지가 높을수록 높은 신뢰도 (단, 과도하면 감점)
        leverage_confidence = min(ai_action[1] / 5.0, 1.0) * (0.8 if ai_action[1] > 10 else 1.0)
        
        # 종합 신뢰도
        overall_confidence = (position_confidence + leverage_confidence) / 2
        
        return min(overall_confidence, 1.0)
    
    def _calculate_position_size(self, position_change: float, leverage: float) -> float:
        """포지션 크기 계산 (리스크 기반)"""
        # 기본 리스크: 잔고의 2%
        base_risk = self.current_balance * 0.02
        
        # 포지션 변경량에 따른 조정
        position_multiplier = min(abs(position_change), 1.0)
        
        # 레버리지 제한 (최대 5배)
        safe_leverage = min(leverage, 5.0)
        
        # 최종 포지션 크기 (USD)
        position_usd = base_risk * position_multiplier * safe_leverage
        
        # 최대 잔고의 20%로 제한
        max_position = self.current_balance * 0.2
        
        return min(position_usd, max_position)
    
    def _calculate_stops(self, current_price: float, action: str, 
                        holding_minutes: float) -> Tuple[Optional[float], Optional[float]]:
        """스탑로스와 익절가 계산"""
        
        # ATR 기반 (단순화: 가격의 2%)
        atr_estimate = current_price * 0.02
        
        if action == 'BUY':
            stop_loss = current_price - (atr_estimate * 1.5)  # 1.5 ATR
            take_profit = current_price + (atr_estimate * 1.0)  # 1.0 ATR (승률 우선)
        elif action == 'SELL':
            stop_loss = current_price + (atr_estimate * 1.5)
            take_profit = current_price - (atr_estimate * 1.0)
        else:
            return None, None
        
        # 홀딩 시간이 짧으면 더 타이트한 스탑
        if holding_minutes < 120:  # 2시간 미만
            stop_multiplier = 0.7
            profit_multiplier = 0.8
        else:
            stop_multiplier = 1.0
            profit_multiplier = 1.0
        
        if action == 'BUY':
            stop_loss = current_price - (atr_estimate * 1.5 * stop_multiplier)
            take_profit = current_price + (atr_estimate * 1.0 * profit_multiplier)
        elif action == 'SELL':
            stop_loss = current_price + (atr_estimate * 1.5 * stop_multiplier)
            take_profit = current_price - (atr_estimate * 1.0 * profit_multiplier)
        
        return stop_loss, take_profit
    
    def _apply_risk_controls(self, decision: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """최종 리스크 체크 및 거래 결정 조정"""
        
        # 1. 최대 드로우다운 체크
        current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        if current_drawdown > 0.15:  # 15% 이상 손실시
            decision['action'] = 'HOLD'
            decision['reason'] = f"리스크 관리: 최대 손실 한도 도달 ({current_drawdown:.1%})"
            decision['quantity'] = 0.0
            return decision
        
        # 2. 포지션 크기 재검증
        if decision['quantity'] > self.current_balance * 0.3:  # 30% 초과 금지
            decision['quantity'] = self.current_balance * 0.3
            decision['reason'] += " (포지션 크기 조정)"
        
        # 3. 연속 손실 후 보수적 진입
        if hasattr(self, 'recent_losses') and self.recent_losses > 3:
            decision['quantity'] *= 0.5  # 포지션 크기 반감
            decision['reason'] += " (연속 손실 후 보수적 진입)"
        
        # 4. 신뢰도가 낮으면 거래 금지
        if decision['ai_confidence'] < 0.3:
            decision['action'] = 'HOLD'
            decision['reason'] = f"신뢰도 부족 ({decision['ai_confidence']:.2f})"
            decision['quantity'] = 0.0
        
        return decision
    
    def _get_default_decision(self, reason: str) -> Dict[str, Any]:
        """기본 결정 (거래 안함)"""
        return {
            'timestamp': datetime.now(),
            'action': 'HOLD',
            'reason': reason,
            'quantity': 0.0,
            'ai_confidence': 0.0,
            'stop_loss': None,
            'take_profit': None
        }
    
    def execute_decision(self, decision: Dict[str, Any]) -> bool:
        """거래 결정 실행 (실제 거래소 연동은 여기서)"""
        
        if decision['action'] == 'HOLD':
            print(f"⏸️  거래 없음: {decision['reason']}")
            return True
        
        print(f"📊 AI 거래 결정:")
        print(f"   액션: {decision['action']}")
        print(f"   수량: ${decision['quantity']:.2f}")
        print(f"   신뢰도: {decision['ai_confidence']:.2f}")
        print(f"   스탑로스: ${decision['stop_loss']:.2f}" if decision['stop_loss'] else "   스탑로스: 없음")
        print(f"   익절가: ${decision['take_profit']:.2f}" if decision['take_profit'] else "   익절가: 없음")
        print(f"   이유: {decision['reason']}")
        
        # 실제 거래소 API 호출은 여기에 구현
        # exchange_api.place_order(...)
        
        return True
    
    def update_trade_result(self, trade_pnl: float):
        """거래 결과 업데이트"""
        self.total_trades += 1
        
        if trade_pnl > 0:
            self.winning_trades += 1
            print(f"✅ 수익 거래: +${trade_pnl:.2f}")
        else:
            print(f"❌ 손실 거래: ${trade_pnl:.2f}")
        
        # 잔고 업데이트
        self.current_balance += trade_pnl
        
        # 최대 낙폭 업데이트
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
# 사용 예시
# =================================================================

def example_live_trading():
    """실시간 트레이딩 사용 예시"""
    
    # 1. 훈련된 에이전트 로드
    live_agent = LiveTradingAgent(
        model_path='agent/final_optimized_model.pth',  # 훈련된 모델 경로
        initial_balance=10000.0
    )
    
    # 2. 실시간 거래 루프 시뮬레이션
    print("\n🚀 실시간 트레이딩 시뮬레이션 시작")
    
    # 가상의 실시간 데이터
    for i in range(10):  # 10번의 거래 기회
        
        # 현재 가격 데이터 (실제로는 거래소 API에서 가져옴)
        current_ohlcv = {
            'open': 3000 + i,
            'high': 3010 + i,
            'low': 2995 + i,
            'close': 3005 + i,
            'volume': 1000000
        }
        
        # 당신의 전략 신호 (실제로는 strategy_executor에서 가져옴)
        current_signals = {
            'decisions': {
                'SHORT_TERM': {
                    'action': 'LONG' if i % 3 == 0 else ('SHORT' if i % 3 == 1 else 'HOLD'),
                    'net_score': np.random.uniform(-1, 1),
                    'leverage': np.random.randint(1, 5),
                    'max_holding_minutes': np.random.randint(60, 240),
                    'meta': {
                        'synergy_meta': {
                            'confidence': np.random.choice(['HIGH', 'MEDIUM', 'LOW']),
                            'buy_score': np.random.uniform(0, 1),
                            'sell_score': np.random.uniform(0, 1),
                            'conflicts_detected': []
                        }
                    }
                },
                'MEDIUM_TERM': {'action': 'HOLD', 'net_score': 0.0, 'leverage': 1, 'max_holding_minutes': 240, 'meta': {'synergy_meta': {'confidence': 'LOW', 'buy_score': 0.0, 'sell_score': 0.0}}},
                'LONG_TERM': {'action': 'HOLD', 'net_score': 0.0, 'leverage': 1, 'max_holding_minutes': 1440, 'meta': {'synergy_meta': {'confidence': 'LOW', 'buy_score': 0.0, 'sell_score': 0.0}}}
            },
            'conflicts': {'has_conflicts': False, 'long_categories': [], 'short_categories': []},
            'meta': {'active_positions': 0}
        }
        
        print(f"\n⏰ 거래 기회 {i+1}")
        
        # 가격 데이터 업데이트
        live_agent.update_price(current_ohlcv)
        
        # AI 거래 결정
        decision = live_agent.make_trading_decision(current_signals, current_ohlcv['close'])
        
        # 거래 실행
        live_agent.execute_decision(decision)
        
        # 결과 시뮬레이션 (실제로는 거래소에서 받아옴)
        if decision['action'] != 'HOLD':
            simulated_pnl = np.random.uniform(-50, 100)  # 랜덤 손익
            live_agent.update_trade_result(simulated_pnl)

def integrate_with_your_system():
    """당신의 기존 시스템과 통합 방법"""
    
    print("""
    🔗 기존 시스템과 통합 방법:
    
    1. strategy_executor.py에서:
    ```python
    # 전략 실행 후
    signals = strategy_executor.get_signals()
    
    # AI 에이전트에 전달
    decision = live_agent.make_trading_decision(signals, current_price)
    
    # 거래 실행
    if decision['action'] != 'HOLD':
        execute_trade(decision)
    ```
    
    2. 실시간 루프에서:
    ```python
    while True:
        # 1. 새로운 캔들 데이터 받기
        new_candle = get_latest_candle()
        live_agent.update_price(new_candle)
        
        # 2. 전략 신호 생성
        signals = generate_strategy_signals()
        
        # 3. AI 결정 받기
        decision = live_agent.make_trading_decision(signals, new_candle['close'])
        
        # 4. 거래 실행
        if decision['action'] != 'HOLD':
            result = execute_real_trade(decision)
            live_agent.update_trade_result(result['pnl'])
        
        time.sleep(180)  # 3분 대기
    ```
    """)

if __name__ == "__main__":
    example_live_trading()
    integrate_with_your_system()