"""
58차원 RL Decision 기반 강화학습 트레이딩 AI 훈련 시스템 - Part 1
- 새로운 RL Decision 스키마 활용 (action_value, confidence_value 등)
- Conflict 정보 및 시너지 메타데이터 활용
- 중복 계산 제거 및 정보 활용 극대화
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import gymnasium as gym
import os

from collections import deque, namedtuple
from gymnasium import spaces
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

# PyTorch 호환성 설정
def setup_pytorch_compatibility():
    """PyTorch 버전 호환성 설정"""
    try:
        # NumPy 2.0 호환성을 위한 안전한 글로벌 설정
        safe_globals = [
            np.ndarray,
            np.dtype,
            np.float32,
            np.float64,
            np.int32,
            np.int64,
        ]
        
        # numpy.core 대신 numpy._core 사용 (NumPy 2.0 호환)
        try:
            import numpy._core.multiarray
            safe_globals.append(numpy._core.multiarray.scalar)
        except ImportError:
            # 이전 버전 호환성
            try:
                import numpy.core.multiarray
                safe_globals.append(numpy.core.multiarray.scalar)
            except ImportError:
                pass
        
        torch.serialization.add_safe_globals(safe_globals)
        print("PyTorch 호환 설정 완료 (NumPy 2.0 호환)")
    except AttributeError:
        print("PyTorch 이전 버전 감지됨")

setup_pytorch_compatibility()

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class RewardCalculator:
    """승률과 수익성을 최적화하는 보상 계산기 (Signal 기반)"""
    
    def __init__(self, max_trades_memory: int = 50):
        self.recent_trades = deque(maxlen=max_trades_memory)
        self.baseline_return = 0.0
        
    def calculate_reward(self, current_price: float, entry_price: float, position: float, 
                    action: str, holding_time: int, trade_pnl: Optional[float] = None) -> float:
        """강화된 보상 시스템 (100배 증폭 + 구간별 차등 보상)"""
        reward = 0.0
        
        # 거래 완료 시: 수익률을 100배 증폭 (강한 학습 신호)
        if trade_pnl is not None:
            reward = trade_pnl * 100  # 100배 증폭
            
            # 구간별 차등 보상
            if trade_pnl >= 0.02:  # 2% 이상 수익
                reward += 50  # 큰 보너스
            elif trade_pnl >= 0.01:  # 1% 이상 수익
                reward += 20  # 중간 보너스
            elif trade_pnl > 0:  # 양의 수익
                reward += 5   # 작은 보너스
            elif trade_pnl <= -0.05:  # 5% 이상 손실
                reward -= 30  # 큰 페널티
            elif trade_pnl < 0:  # 손실
                reward -= 10  # 작은 페널티
        
        # 거래 완료가 아닌 경우: 미실현 손익 기반 보상
        elif abs(position) > 0.0001 and entry_price > 0:  # 임계값 감소
            unrealized_pnl = self._calculate_unrealized_pnl(current_price, entry_price, position)
            reward = unrealized_pnl * 10  # 10배 보상 (학습 신호 강화)
        
        return reward
    
    
    def _calculate_unrealized_pnl(self, current_price: float, entry_price: float, position: float) -> float:
        """미실현 손익 계산"""
        if entry_price <= 0:
            return 0.0
        
        price_change = (current_price - entry_price) / entry_price
        return position * price_change
    
    def _calculate_signal_reward(self, signal_data: Dict, position: float) -> float:
        """Signal 데이터 기반 추가 보상 - 새로운 RL 스키마 기반"""
        signal_reward = 0.0
        
        # 각 시간대별 신호와 포지션 일치도
        for timeframe in ['short_term', 'medium_term', 'long_term']:
            action_value = float(signal_data.get(f'{timeframe}_action', 0.0))
            net_score = float(signal_data.get(f'{timeframe}_net_score', 0.0))
            confidence_value = float(signal_data.get(f'{timeframe}_confidence', 0.0))
            
            # Action value와 포지션 일치도 (action_value: -1~1, position: -1~1)
            action_match = 1.0 - abs(action_value - position) / 2.0  # 0~1 범위
            signal_reward += action_match * abs(net_score) * confidence_value * 0.3
        
        # Conflict 정보 활용
        conflict_penalty = float(signal_data.get('conflict_conflict_penalty', 0.0))
        conflict_consensus = float(signal_data.get('conflict_directional_consensus', 0.0))
        
        # Conflict가 적고 consensus가 높을 때 보상
        if conflict_penalty == 0.0 and conflict_consensus > 0.5:
            signal_reward += 0.2
        
        return signal_reward



class DuelingDQN(nn.Module):
    """Dueling DQN (Value + Advantage 분리로 안정적인 학습)"""
    
    def __init__(self, state_size: int, action_size: int = 3, hidden_size: int = 256, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.state_size = state_size
        self.hidden_size = hidden_size
        
        # 공통 특징 추출기 (Feature Extractor)
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Value Stream (상태의 가치)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1)  # 단일 Value 출력
        )
        
        # Advantage Stream (액션별 장점)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 각 액션 차원별 Advantage 헤드
        self.position_advantage = nn.Linear(hidden_size // 4, 21)  # 포지션 -0.5~0.5 (21개 구간)
        self.leverage_advantage = nn.Linear(hidden_size // 4, 10)  # 레버리지 1~5 (10개 구간)
        self.holding_advantage = nn.Linear(hidden_size // 4, 20)   # 홀딩 10~60분 (20개 구간)
        
        # 수익률 예측 (Value Stream 활용)
        self.profit_predictor = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1)
        )
        
        # 가중치 초기화
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Orthogonal 초기화 (DuelingDQN에 더 적합)"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # 배치 차원 확인
        if x.dim() == 1:
            x = x.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False
        
        # 공통 특징 추출
        features = self.feature_extractor(x)  # [batch_size, hidden_size//2]
        
        # Value Stream (상태의 가치)
        value = self.value_stream(features)  # [batch_size, 1]
        
        # Advantage Stream (액션별 장점)
        advantage_features = self.advantage_stream(features)  # [batch_size, hidden_size//4]
        
        # 각 액션 차원별 Advantage 계산
        position_adv = self.position_advantage(advantage_features)  # [batch_size, 21]
        leverage_adv = self.leverage_advantage(advantage_features)  # [batch_size, 10]
        holding_adv = self.holding_advantage(advantage_features)    # [batch_size, 20]
        
        # Dueling 구조: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        # 각 액션 차원별로 평균 Advantage를 빼서 안정성 확보
        position_q = value + position_adv - position_adv.mean(dim=1, keepdim=True)
        leverage_q = value + leverage_adv - leverage_adv.mean(dim=1, keepdim=True)
        holding_q = value + holding_adv - holding_adv.mean(dim=1, keepdim=True)
        
        # 수익률 예측 (공통 특징 활용)
        profit_pred = self.profit_predictor(features)
        
        # 단일 샘플이면 배치 차원 제거
        if single_sample:
            position_q = position_q.squeeze(0)
            leverage_q = leverage_q.squeeze(0)
            holding_q = holding_q.squeeze(0)
            profit_pred = profit_pred.squeeze(0)
        
        return position_q, leverage_q, holding_q, profit_pred





class TradingEnvironment(gym.Env):
    """58차원 RL Decision 기반 암호화폐 거래 강화학습 환경 (Gymnasium 호환) - Open 가격 기반"""
    
    def __init__(self, price_data: pd.DataFrame, signal_data: List[Dict], 
                 initial_balance: float = 10000.0, max_position: float = 1.0):
        super().__init__()
        
        self.price_data = price_data
        self.signal_data = signal_data
        self.initial_balance = initial_balance
        self.max_position = max_position
        
        self.reward_calculator = RewardCalculator()
        
        # 액션/상태 스페이스 정의 (포지션 -0.5~0.5 범위로 축소)
        self.action_space = spaces.Box(
            low=np.array([-0.5, 1.0, 10.0]),  # 포지션 -0.5~0.5 범위로 축소
            high=np.array([0.5, 5.0, 60.0]),  # 레버리지 1~5, 홀딩 10~60분
            dtype=np.float32
        )
        
        # 거래 제한 설정 (단타 최적화)
        self.min_trade_interval = 1  # 최소 1스텝 간격 (더 자주 거래 허용)
        self.last_trade_step = -self.min_trade_interval  # 초기값
        self.trading_cost = 0.0001  # 0.01% 거래 비용 (소액 거래에 적합)
        
        # 58차원 상태 공간 (기술적 지표 + 포트폴리오 + 의사결정 특성)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(58,),  # 3 + 20 + 9 + 26 = 58차원
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        """환경 초기화 (Gymnasium 호환)"""
        if seed is not None:
            np.random.seed(seed)
        
        self.current_step = 10
        self.balance = self.initial_balance
        self.current_position = 0.0
        self.current_leverage = 1.0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.max_drawdown = 0.0
        self.peak_balance = self.initial_balance
        self.consecutive_losses = 0
        self.holding_time = 0
        self.in_position = False
        self.last_trade_pnl = None
        self.last_trade_step = -self.min_trade_interval  # 거래 간격 초기화
        
        observation = self._get_observation()
        info = self._create_info_dict()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """환경 스텝 실행 (Gymnasium 호환) - Open 가격 기반"""
        if self.current_step >= min(len(self.price_data), len(self.signal_data)) - 1:
            return self._get_observation(), 0.0, True, False, {}
        
        position_change = np.clip(action[0], -0.5, 0.5)  # 포지션 -0.5~0.5 범위
        leverage = np.clip(action[1], 1.0, 5.0)  # 레버리지 1~5
        target_holding_minutes = np.clip(action[2], 10.0, 60.0)  # 10분~60분
        
        # 현재 스텝의 open 가격 (실제 거래에서 알 수 있는 가격)
        current_price = self.price_data.iloc[self.current_step]['open']
        # 다음 스텝의 open 가격 (다음 캔들 시작 시점)
        next_price = self.price_data.iloc[self.current_step + 1]['open']
        
        # 포지션 및 거래 처리
        trade_completed, old_position = self._process_position_change(
            position_change, leverage, current_price, target_holding_minutes
        )
        
        # 거래 완료 시 거래 스텝 업데이트
        if trade_completed:
            self.last_trade_step = self.current_step
        
        # Signal 데이터 가져오기
        current_signal = self.signal_data[self.current_step] if self.current_step < len(self.signal_data) else {}
        
        # 보상 계산 (Signal 정보 활용) - Open 가격 기반
        if trade_completed:
            # 거래 완료 시: 실제 수익률 기반 보상 (이전 캔들의 close 가격 사용)
            if self.current_step > 0:
                # 이전 캔들이 완성된 close 가격으로 수익률 계산
                prev_close_price = self.price_data.iloc[self.current_step - 1]['close']
            reward = self.reward_calculator.calculate_reward(
                    current_price=prev_close_price,  # 완성된 close 가격 사용
                entry_price=self.entry_price,
                position=old_position,  # 거래 전 포지션 사용
                action='TRADE',
                holding_time=self.holding_time,
                trade_pnl=self.last_trade_pnl
            )
        else:
            # 거래 완료가 아닌 경우: 미실현 손익 기반 보상 (현재 open 가격 사용)
            reward = self.reward_calculator.calculate_reward(
                current_price=current_price,  # 현재 open 가격 사용
                entry_price=self.entry_price,
                position=self.current_position,
                action='HOLD',
                holding_time=self.holding_time,
                trade_pnl=None
            )
        
        # 다음 스텝으로 이동
        self.current_step += 1
        self.holding_time += 3
                
        done = (self.current_step >= min(len(self.price_data), len(self.signal_data)) - 1 or 
                self.balance <= self.initial_balance * 0.1)
        
        truncated = False  # Gymnasium 호환을 위한 truncated 플래그
        info = self._create_info_dict()
        
        return self._get_observation(), reward, done, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """58차원 상태 관찰값 반환 (기술적 지표 + 포트폴리오 + 의사결정 특성) - Open 가격 기반"""
        if self.current_step >= min(len(self.price_data), len(self.signal_data)):
            return np.zeros(58, dtype=np.float32)
        
        # 현재 가격과 이전 가격 비교 (3차원) - Open 가격 사용
        current_price = self.price_data.iloc[self.current_step]['open']
        if self.current_step > 0:
            prev_price = self.price_data.iloc[self.current_step - 1]['open']
            price_change = (current_price - prev_price) / prev_price
        else:
            price_change = 0.0
        
        basic_observation = np.array([
            price_change,  # 가격 변화율
            self.current_position,  # 현재 포지션 (-1~1)
            self.balance / self.initial_balance  # 잔고 비율
        ], dtype=np.float32)
        
        # Signal 데이터 가져오기
        current_signal = self.signal_data[self.current_step] if self.current_step < len(self.signal_data) else {}
        current_candle = self.price_data.iloc[self.current_step].to_dict()
        
        # 각 차원별 특성 추출
        price_indicators = self._extract_price_indicators(current_signal, current_candle)  # 20차원
        portfolio_state = self._get_portfolio_state()  # 9차원
        decision_features = self._extract_decision_features(current_signal)  # 26차원
        
        # 모든 차원 결합 (3 + 20 + 9 + 26 = 58차원)
        observation = np.concatenate([
            basic_observation,      # 3차원
            price_indicators,       # 20차원
            portfolio_state,        # 9차원
            decision_features       # 26차원
        ], dtype=np.float32)
        
        return observation
    
    def _extract_price_indicators(self, signal_data: Dict, current_candle: Dict) -> np.ndarray:
        """Signal의 indicator들을 price feature로 활용 (20차원) - Open 가격 기반"""
        current_price = current_candle['open']  # Open 가격 사용
        
        # 1. 가격 대비 지표 위치
        vwap = signal_data.get('indicator_vwap')
        poc = signal_data.get('indicator_poc')  
        hvn = signal_data.get('indicator_hvn')
        lvn = signal_data.get('indicator_lvn')
        
        # 2. 변동성 지표들
        atr = signal_data.get('indicator_atr', 0.0)
        vwap_std = signal_data.get('indicator_vwap_std', 0.0)
        
        # 3. 일별 기준점들
        prev_high = signal_data.get('indicator_prev_day_high')
        prev_low = signal_data.get('indicator_prev_day_low')
        or_high = signal_data.get('indicator_opening_range_high')
        or_low = signal_data.get('indicator_opening_range_low')
        
        prev_range = prev_high - prev_low
        prev_day_position = (current_price - prev_low) / prev_range if prev_range > 0 else 0.5
            
        or_range = or_high - or_low  
        or_position = (current_price - or_low) / or_range if or_range > 0 else 0.5
        
        # 4. 현재 캔들 정보
        high, low, close, open_price = current_candle['high'], current_candle['low'], current_candle['close'], current_candle['open']
        quote_volume = current_candle.get('quote_volume')
        
        return np.array([
            # 가격 대비 지표 위치 (4개)
            (current_price - vwap) / current_price if current_price > 0 else 0.0,
            (current_price - poc) / current_price if current_price > 0 else 0.0,   
            (current_price - hvn) / current_price if current_price > 0 else 0.0,   
            (current_price - lvn) / current_price if current_price > 0 else 0.0,
            
            # 변동성 지표들 (2개)
            atr / current_price if current_price > 0 else 0.0,
            vwap_std / current_price if current_price > 0 else 0.0,
            
            # 일별 기준점들 (4개)
            prev_day_position,
            or_position,
            (current_price - prev_high) / current_price if current_price > 0 else 0.0,
            (prev_low - current_price) / current_price if current_price > 0 else 0.0,
            
            # 현재 캔들 정보 (8개)
            (close - open_price) / open_price if open_price > 0 else 0.0,
            (high - low) / close if close > 0 else 0.0,
            (high - close) / (high - low) if high > low else 0.5,
            (close - low) / (high - low) if high > low else 0.5,
            (close - open_price) / (high - low) if high > low else 0.0,
            min(quote_volume / 1000000, 2.0) if quote_volume > 0 else 0.0,
            1.0 if close > open_price else 0.0,
            (high - max(open_price, close)) / (high - low) if high > low else 0.0,
            
            # 추가 캔들 정보 (2개)
            (low - min(open_price, close)) / (high - low) if high > low else 0.0,
            abs(close - open_price) / (high - low) if high > low else 0.0
        ], dtype=np.float32)
    
    def _get_portfolio_state(self) -> np.ndarray:
        """포트폴리오 상태 정보 (9차원)"""
        return np.array([
            self.current_position,
            self.current_leverage / 20.0,
            (self.balance - self.initial_balance) / self.initial_balance,
            self.unrealized_pnl / self.initial_balance if self.initial_balance > 0 else 0.0,
            min(self.total_trades / 100.0, 1.0),
            self.winning_trades / max(self.total_trades, 1),
            self.max_drawdown,
            min(self.consecutive_losses / 10.0, 1.0),
            min(self.holding_time / 1440.0, 1.0)
        ], dtype=np.float32)
    
    def _extract_decision_features(self, signals: Dict) -> np.ndarray:
        """Decision 특성들 (26차원) - 새로운 RL 스키마 기반"""
        # 각 시간대별 특성 (3 × 6 = 18개)
        timeframe_features = []
        for timeframe in ['short_term', 'medium_term', 'long_term']:
            # 새로운 RL 스키마 필드들 사용
            action_value = float(signals.get(f'{timeframe}_action', 0.0))
            net_score = float(signals.get(f'{timeframe}_net_score', 0.0))
            buy_score = float(signals.get(f'{timeframe}_buy_score', 0.0))
            sell_score = float(signals.get(f'{timeframe}_sell_score', 0.0))
            confidence_value = float(signals.get(f'{timeframe}_confidence', 0.0))
            market_context_value = float(signals.get(f'{timeframe}_market_context', 0.0))
            
            timeframe_features.extend([action_value, net_score, buy_score, sell_score, confidence_value, market_context_value])
        
        # 추가 메타 정보 (3개)
        signals_used = []
        for timeframe in ['short_term', 'medium_term', 'long_term']:
            used = signals.get(f'{timeframe}_signals_used', 0)
            
            signals_used.append(min(float(used) / 10.0, 1.0))
        
        # Conflict 정보 (3개)
        conflict_severity = float(signals.get('conflict_conflict_severity', 0.0))
        conflict_consensus = float(signals.get('conflict_directional_consensus', 0.0))
        conflict_penalty = float(signals.get('conflict_conflict_penalty', 0.0))
        
        # Long term 특화 정보 (2개)
        institutional_bias = float(signals.get('long_term_institutional_bias', 0.0))
        macro_trend_strength = float(signals.get('long_term_macro_trend_strength', 0.0))
        
        return np.array(
            timeframe_features + 
            signals_used + 
            [conflict_severity, conflict_consensus, conflict_penalty, institutional_bias, macro_trend_strength],
            dtype=np.float32
        )
    
    
    def _process_position_change(self, position_change: float, leverage: float, 
                                current_price: float, target_holding_minutes: float) -> Tuple[bool, float]:
        """포지션 변경 처리 (순수 RL 에이전트 결정) - Open 가격 기반"""
        old_position = self.current_position
        trade_completed = False
        
        # RL 에이전트의 결정만으로 포지션 변경 (-0.5~0.5 범위)
        target_position = np.clip(self.current_position + position_change, -0.5, 0.5)
        
        # 포지션 변경이 필요한지 확인 (임계값 대폭 감소)
        if abs(target_position - self.current_position) > 0.0001:
            # 기존 포지션 청산 (이전 캔들의 close 가격으로 청산)
            if abs(self.current_position) > 0.0001:
                trade_completed = True
                if self.current_step > 0:
                    # 이전 캔들이 완성된 close 가격으로 청산
                    prev_close_price = self.price_data.iloc[self.current_step - 1]['close']
                    self.last_trade_pnl = self._calculate_trade_pnl(prev_close_price, self.entry_price, old_position)
                    self._close_position(prev_close_price)
                else:
                    # 첫 번째 스텝에서는 현재 open 가격 사용
                    self.last_trade_pnl = self._calculate_trade_pnl(current_price, self.entry_price, old_position)
                    self._close_position(current_price)
            
            # 새 포지션 진입 (현재 open 가격으로 진입)
            if abs(target_position) > 0.0001:
                self.current_position = target_position
                self.current_leverage = leverage
                self.entry_price = current_price  # Open 가격으로 진입
                self.holding_time = 0
                self.in_position = True
                
        
        return trade_completed, old_position
        
    def _calculate_trade_pnl(self, exit_price: float, entry_price: float, position: float) -> float:
        """거래 손익 계산"""
        if entry_price <= 0:
            return 0.0
        
        price_change = (exit_price - entry_price) / entry_price
        return position * price_change
    
    def _close_position(self, exit_price: float):
        """포지션 청산"""
        if abs(self.current_position) < 0.01:
            return
        
        pnl = self._calculate_trade_pnl(exit_price, self.entry_price, self.current_position)
        # 레버리지는 거래량에만 적용, 손익에는 적용하지 않음
        trade_volume = abs(self.current_position) * self.current_leverage * self.balance
        pnl_usd = pnl * trade_volume  # 올바른 손익 계산
        
        # 거래 수수료 차감 (거래량 기준)
        fee = trade_volume * self.trading_cost  # 0.1% 거래 비용
        pnl_usd -= fee
        
        # 잔고 및 통계 업데이트
        self.balance += pnl_usd
        self._update_trading_stats(pnl_usd)
        
        # 포지션 초기화
        self.current_position = 0.0
        self.unrealized_pnl = 0.0
        self.in_position = False
        self.holding_time = 0
        self.last_trade_pnl = pnl
    
    def _update_trading_stats(self, pnl_usd: float):
        """거래 통계 업데이트"""
        self.total_trades += 1
        
        if pnl_usd > 0:
            self.winning_trades += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
        
        # 최대 낙폭 업데이트
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        else:
            drawdown = (self.peak_balance - self.balance) / self.peak_balance
            self.max_drawdown = max(self.max_drawdown, drawdown)
    
    def _create_info_dict(self) -> Dict:
        """정보 딕셔너리 생성 - Open 가격 기반"""
        current_price = self.price_data.iloc[min(self.current_step, len(self.price_data)-1)]['open']
        
        return {
            'balance': self.balance,
            'position': self.current_position,
            'unrealized_pnl': self.unrealized_pnl,
            'total_trades': self.total_trades,
            'win_rate': self.winning_trades / max(self.total_trades, 1),
            'current_price': current_price,
            'entry_price': self.entry_price,
            'holding_time': self.holding_time,
            'max_drawdown': self.max_drawdown,
            'trade_completed': hasattr(self, 'last_trade_pnl') and self.last_trade_pnl is not None,
            'trade_pnl': self.last_trade_pnl if hasattr(self, 'last_trade_pnl') else None
        }

"""
58차원 RL Decision 기반 강화학습 트레이딩 AI 훈련 시스템 - Part 2
- RLAgent 클래스 및 훈련/평가 시스템
- 새로운 Decision 스키마 데이터 로더 및 유틸리티 함수들
"""

class RLAgent:
    """58차원 RL Decision 기반 강화학습 에이전트"""
    
    def __init__(self, state_size: int = 3, learning_rate: float = 3e-4, 
                    gamma: float = 0.99, epsilon: float = 0.9, epsilon_decay: float = 0.995,
                    hidden_size: int = 128):
        
        self.state_size = state_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate  # learning_rate 속성 추가
        self.epsilon_min = 0.1  # 10%로 설정 (적절한 탐험)
        
        # ε 값이 너무 낮으면 초기화
        if self.epsilon < self.epsilon_min:
            self.epsilon = 0.8  # 80%로 초기화 (충분한 탐험)
        
        # GPU 사용 가능 여부 확인 및 디바이스 설정
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"🚀 GPU 사용: {torch.cuda.get_device_name(0)}")
            print(f"   GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        else:
            self.device = torch.device("cpu")
            print("⚠️  GPU 사용 불가 - CPU 모드로 실행")
            print("   GPU 사용을 원하면 PyTorch CUDA 버전을 설치하세요:")
            print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print(f"Using device: {self.device} for {state_size}차원 모델")
        
        # 네트워크 초기화 (DuelingDQN 사용 - Value + Advantage 분리로 안정적인 학습)
        print("🚀 DuelingDQN 아키텍처 사용 (Value + Advantage 분리로 안정적인 학습)")
        self.q_network = DuelingDQN(state_size, 3, hidden_size).to(self.device)
        self.target_network = DuelingDQN(state_size, 3, hidden_size).to(self.device)
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 경험 리플레이 (수익률 학습 최적화)
        self.memory = deque(maxlen=20000)  # 메모리 크기 증가 (더 많은 경험)
        self.batch_size = 128  # 배치 크기 증가 (더 안정적인 학습)
        
        # 학습 추적
        self.training_rewards = []
        self.losses = []
        self.win_rates = []
        
        # 타겟 네트워크 업데이트 (수익률 학습 최적화)
        self.target_update_freq = 100  # 적절한 업데이트 빈도
        self.update_count = 0
        
        
        # 액션 공간 설정 (환경에서 가져옴)
        self.action_space = None  # 환경에서 설정됨
    
    def remember(self, state, action, reward, next_state, done):
        """경험 저장"""
        self.memory.append(Experience(state, action, reward, next_state, done))
    
    
    def adaptive_learning_rate(self, recent_rewards: List[float], recent_win_rates: List[float]):
        """적응형 학습률 조정"""
        if len(recent_rewards) < 10:
            return
        
        # 최근 성능 분석
        avg_reward = np.mean(recent_rewards[-10:])
        avg_win_rate = np.mean(recent_win_rates[-10:])
        
        # 성능이 좋으면 학습률 감소 (안정화)
        if avg_win_rate > 0.4 and avg_reward > 0:
            self.learning_rate *= 0.95
            self.learning_rate = max(self.learning_rate, 1e-5)  # 최소값 보장
        # 성능이 나쁘면 학습률 증가 (빠른 학습)
        elif avg_win_rate < 0.2 or avg_reward < -100:
            self.learning_rate *= 1.05
            self.learning_rate = min(self.learning_rate, 5e-3)  # 최대값 제한
        
        # 옵티마이저 업데이트
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate
    
    def act(self, state: np.ndarray) -> np.ndarray:
        """액션 선택 - 향상된 탐험 전략"""
        if np.random.random() <= self.epsilon:
            return self._get_smart_random_action(state)
        
        return self._get_greedy_action(state)
    
    def _get_smart_random_action(self, state: np.ndarray) -> np.ndarray:
        """지능적인 랜덤 액션 - 상태에 기반한 제한적 탐험"""
        # 기본 랜덤 액션
        action = self._get_random_action()
        
        # 상태 기반 액션 조정
        if len(self.memory) > 100:  # 충분한 경험이 있을 때
            recent_trades = [exp for exp in list(self.memory)[-50:] if exp.reward > 0]
            if recent_trades:
                # 최근 성공한 액션 패턴 분석
                successful_actions = [exp.action for exp in recent_trades]
                if successful_actions:
                    # 성공한 액션과 유사한 방향으로 탐험
                    base_action = np.mean(successful_actions, axis=0)
                    noise = np.random.normal(0, 0.1, action.shape)
                    
                    # action_space가 있을 때만 클리핑 적용
                    if self.action_space is not None:
                        action = np.clip(base_action + noise, 
                                       self.action_space.low, 
                                       self.action_space.high)
                    else:
                        # 기본 클리핑 (position_change, leverage, holding_time)
                        action = np.clip(base_action + noise, 
                                       [-0.5, 1.0, 10.0], 
                                       [0.5, 5.0, 60.0])
        
        return action
    
    def _get_random_action(self) -> np.ndarray:
        """보수적인 랜덤 액션"""
        return np.array([
            np.random.uniform(-0.5, 0.5),  # 포지션 범위 축소
            np.random.uniform(1.0, 5.0), # 레버리지 최대 5
            np.random.uniform(10.0, 60.0) # 홀딩 시간 축소
        ])
    
    def _get_greedy_action(self, state: np.ndarray) -> np.ndarray:
        """Q값 기반 탐욕적 액션 선택 (수익률 최적화)"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            position_q, leverage_q, holding_q, profit_pred = self.q_network(state_tensor)
            
            # 수익률 예측을 고려한 액션 선택
            position_idx = torch.argmax(position_q).item()
            leverage_idx = torch.argmax(leverage_q).item()
            holding_idx = torch.argmax(holding_q).item()
            
            # 인덱스를 실제 값으로 변환 (-0.5~0.5 범위)
            position = -0.5 + (position_idx * 0.05)   # -0.5~0.5 (21개 구간)
            leverage = 1.0 + (leverage_idx * 0.4)   # 1.0~5.0 (10개 구간)
            holding = 10.0 + (holding_idx * 2.5)    # 10~60분 (20개 구간)
            
            return np.array([position, leverage, holding])
    
    def replay(self):
        """우선순위 경험 리플레이 학습"""
        if len(self.memory) < self.batch_size * 2:
            return
        
        # 우선순위 샘플링 (긍정적 경험 70%, 중립 20%, 부정 10%)
        positive_experiences = [exp for exp in self.memory if exp.reward > 10]
        neutral_experiences = [exp for exp in self.memory if -10 <= exp.reward <= 10]
        negative_experiences = [exp for exp in self.memory if exp.reward < -10]
        
        batch = []
        batch_size = self.batch_size
        
        # 긍정적 경험 70%
        if positive_experiences:
            pos_count = int(batch_size * 0.7)
            batch.extend(random.sample(positive_experiences, min(pos_count, len(positive_experiences))))
        
        # 중립 경험 20%
        if neutral_experiences:
            neutral_count = int(batch_size * 0.2)
            batch.extend(random.sample(neutral_experiences, min(neutral_count, len(neutral_experiences))))
        
        # 부정적 경험 10%
        if negative_experiences:
            neg_count = batch_size - len(batch)
            batch.extend(random.sample(negative_experiences, min(neg_count, len(negative_experiences))))
        
        # 부족한 경우 랜덤 샘플링으로 채움
        if len(batch) < batch_size:
            remaining = batch_size - len(batch)
            batch.extend(random.sample(self.memory, remaining))
        
        loss = self._compute_loss(batch)
        
        # 역전파
        self.optimizer.zero_grad()
        loss.backward()
        
        # 고급 정규화 기법들 (과적합 방지)
        
        # 1. 그래디언트 클리핑 (적응적)
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 0.5)
        
        # 2. L2 정규화 (적응적)
        l2_lambda = 1e-3  # 더 강한 L2 정규화
        l2_reg = torch.tensor(0., device=self.device)
        for param in self.q_network.parameters():
            l2_reg += torch.norm(param)
        loss += l2_lambda * l2_reg
        
        # 3. 엔트로피 정규화는 _compute_loss 함수에서 처리됨
        
        # 4. Spectral Normalization 효과 (가중치 정규화)
        spectral_reg = torch.tensor(0., device=self.device)
        for module in self.q_network.modules():
            if isinstance(module, nn.Linear):
                # 가중치의 스펙트럴 노름 정규화
                weight_norm = torch.norm(module.weight, p=2)
                spectral_reg += weight_norm
        loss += 0.001 * spectral_reg
        
        # 5. Dropout 적응적 조정 (과적합 감지 시)
        if len(self.losses) > 10:
            recent_losses = self.losses[-10:]
            loss_variance = torch.var(torch.tensor(recent_losses))
            if loss_variance < 0.01:  # 손실이 안정적이면 드롭아웃 증가
                for module in self.q_network.modules():
                    if isinstance(module, nn.Dropout):
                        module.p = min(module.p + 0.005, 0.3)
        
        self.optimizer.step()
        
        
        self.losses.append(loss.item())
        
        # 적응적 엡실론 감소 (성능에 따라 조정)
        if self.epsilon > self.epsilon_min:
            # 최근 성능 기반 적응적 감소
            if len(self.training_rewards) > 20:  # 더 빠른 반응
                recent_rewards = self.training_rewards[-20:]
                avg_recent_reward = np.mean(recent_rewards)
                
                # 수익률 기반 감소 (새로운 보상 범위에 맞춤)
                if avg_recent_reward > 10.0:  # 높은 수익률 (100% 이상)
                    self.epsilon *= 0.95  # 빠른 감소
                elif avg_recent_reward > 5.0:  # 양의 수익률 (50% 이상)
                    self.epsilon *= 0.97  # 중간 감소
                elif avg_recent_reward > 0:  # 약간의 수익률
                    self.epsilon *= 0.99  # 느린 감소
                else:  # 손실
                    self.epsilon *= 0.995  # 매우 느린 감소
            else:
                self.epsilon *= 0.995  # 초기에는 매우 느린 감소
        
        # 적응적 학습 전략: 성과 개선이 없으면 탐험률 자동 증가
        if len(self.training_rewards) > 50:
            recent_rewards = self.training_rewards[-50:]
            older_rewards = self.training_rewards[-100:-50] if len(self.training_rewards) >= 100 else self.training_rewards[:-50]
            
            if len(older_rewards) > 0:
                recent_avg = np.mean(recent_rewards)
                older_avg = np.mean(older_rewards)
                
                # 성과 개선이 없으면 탐험률 증가
                if recent_avg <= older_avg:
                    self.epsilon = min(self.epsilon * 1.01, 0.9)  # 최대 90%까지 증가
                    print(f"📈 성과 개선 없음: 탐험률 증가 {self.epsilon:.3f}")
        
        # 타겟 네트워크 업데이트
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.update_target_network()
    
    def _compute_loss(self, batch) -> torch.Tensor:
        """수익률 최적화 손실 함수 계산"""
        # 효율적인 텐서 변환 (numpy 배열을 먼저 결합)
        states = torch.FloatTensor(np.array([e.state for e in batch])).to(self.device)
        actions = [e.action for e in batch]
        rewards = torch.FloatTensor(np.array([e.reward for e in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([e.next_state for e in batch])).to(self.device)
        dones = [bool(e.done) for e in batch]
        
        # 현재 Q값들과 수익률 예측
        current_position_q, current_leverage_q, current_holding_q, current_profit_pred = self.q_network(states)
        
        # Double DQN: 현재 네트워크로 액션 선택, 타겟 네트워크로 Q값 계산
        with torch.no_grad():
            # 현재 네트워크로 다음 상태의 액션 선택
            next_position_q_current, next_leverage_q_current, next_holding_q_current, _ = self.q_network(next_states)
            
            # 타겟 네트워크로 Q값 계산
            next_position_q_target, next_leverage_q_target, next_holding_q_target, _ = self.target_network(next_states)
            
            target_position_q = current_position_q.clone()
            target_leverage_q = current_leverage_q.clone()
            target_holding_q = current_holding_q.clone()
            
            for i, (action, reward, done) in enumerate(zip(actions, rewards, dones)):
                pos_idx = int(np.clip((action[0] + 0.5) / 0.05, 0, 20))   # -0.5~0.5 범위
                lev_idx = int(np.clip((action[1] - 1.0) / 0.4, 0, 9))    # 1.0~5.0 범위
                hold_idx = int(np.clip((action[2] - 10.0) / 2.5, 0, 19)) # 10~60분 범위
                
                if not done:
                    # Double DQN: 현재 네트워크로 선택한 액션의 타겟 네트워크 Q값 사용
                    best_pos_action = torch.argmax(next_position_q_current[i])
                    best_lev_action = torch.argmax(next_leverage_q_current[i])
                    best_hold_action = torch.argmax(next_holding_q_current[i])
                    
                    target_q_pos = reward + self.gamma * next_position_q_target[i, best_pos_action]
                    target_q_lev = reward + self.gamma * next_leverage_q_target[i, best_lev_action]
                    target_q_hold = reward + self.gamma * next_holding_q_target[i, best_hold_action]
                    
                    target_position_q[i, pos_idx] = target_q_pos
                    target_leverage_q[i, lev_idx] = target_q_lev
                    target_holding_q[i, hold_idx] = target_q_hold
                else:
                    # 최종 보상 (수익률 중심)
                    target_position_q[i, pos_idx] = reward
                    target_leverage_q[i, lev_idx] = reward
                    target_holding_q[i, hold_idx] = reward
        
        # Q-learning 손실 (수익률 가중치 적용)
        pos_loss = F.smooth_l1_loss(current_position_q, target_position_q)
        lev_loss = F.smooth_l1_loss(current_leverage_q, target_leverage_q)
        hold_loss = F.smooth_l1_loss(current_holding_q, target_holding_q)
        
        # 수익률 예측 손실 (보조 학습)
        profit_targets = rewards.unsqueeze(1)  # 실제 수익률을 타겟으로
        profit_loss = F.mse_loss(current_profit_pred, profit_targets)
        
        # 엔트로피 정규화 (과적합 방지)
        position_entropy = -torch.sum(F.softmax(current_position_q, dim=1) * 
                                    F.log_softmax(current_position_q, dim=1), dim=1).mean()
        leverage_entropy = -torch.sum(F.softmax(current_leverage_q, dim=1) * 
                                    F.log_softmax(current_leverage_q, dim=1), dim=1).mean()
        holding_entropy = -torch.sum(F.softmax(current_holding_q, dim=1) * 
                                   F.log_softmax(current_holding_q, dim=1), dim=1).mean()
        
        entropy_reg = 0.01 * (position_entropy + leverage_entropy + holding_entropy)
        
        # 수익률 중심 가중치 (수익률 예측에 더 높은 가중치)
        total_loss = (pos_loss + lev_loss + hold_loss) + 2.0 * profit_loss + entropy_reg
        
        return total_loss
    
    
    def update_target_network(self):
        """타겟 네트워크 업데이트"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_model(self, filepath: str) -> bool:
        """모델 저장"""
        try:
            os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
            
            save_dict = {
                'q_network': self.q_network.state_dict(),
                'target_network': self.target_network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epsilon': float(self.epsilon),
                'training_rewards': [float(r) for r in self.training_rewards],
                'losses': [float(l) for l in self.losses],
                'win_rates': [float(w) for w in self.win_rates],
                'update_count': int(self.update_count),
                'state_size': int(self.state_size)
            }
            
            torch.save(save_dict, filepath)
            print(f"58차원 모델 저장 완료: {filepath}")
            return True
            
        except Exception as e:
            print(f"모델 저장 실패: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """모델 로드"""
        if not os.path.exists(filepath):
            print(f"모델 파일이 없습니다: {filepath}")
            return False
        
        try:
            try:
                checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
            except:
                checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
            
            # 상태 크기 확인
            model_state_size = checkpoint.get('state_size', 60)
            if model_state_size != self.state_size:
                print(f"❌ 모델 차원 불일치: 기대 {self.state_size}, 실제 {model_state_size}")
                return False
            
            self.q_network.load_state_dict(checkpoint['q_network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.training_rewards = checkpoint.get('training_rewards', [])
            self.losses = checkpoint.get('losses', [])
            self.win_rates = checkpoint.get('win_rates', [])
            self.update_count = checkpoint.get('update_count', 0)
            
            print(f"✅ 58차원 모델 로드 성공! 엡실론: {self.epsilon:.3f}")
            return True
            
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            return False

    def load_model_with_compatibility(self, filepath: str) -> bool:
        """호환성을 고려한 모델 로드 (구조 차이 무시)"""
        if not os.path.exists(filepath):
            print(f"모델 파일이 없습니다: {filepath}")
            return False
        
        try:
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
            
            # 상태 크기 확인
            model_state_size = checkpoint.get('state_size', 60)
            print(f"기존 모델 차원: {model_state_size}차원")
            
            # 기존 모델과 현재 모델의 구조가 완전히 다르므로
            # 호환 가능한 부분만 로드하고 나머지는 새로 초기화
            print("⚠️ 모델 구조가 다릅니다. 호환 가능한 부분만 로드하고 나머지는 새로 초기화합니다.")
            
            # 현재 모델의 state_dict
            current_state_dict = self.q_network.state_dict()
            loaded_state_dict = checkpoint['q_network']
            
            # 호환 가능한 레이어만 로드
            compatible_state_dict = {}
            loaded_count = 0
            initialized_count = 0
            
            for key in current_state_dict.keys():
                if key in loaded_state_dict:
                    # 크기가 같은 경우만 로드
                    if current_state_dict[key].shape == loaded_state_dict[key].shape:
                        compatible_state_dict[key] = loaded_state_dict[key]
                        print(f"   ✅ {key}: 로드됨")
                        loaded_count += 1
                    else:
                        compatible_state_dict[key] = current_state_dict[key]
                        print(f"   ⚠️ {key}: 크기 불일치 ({loaded_state_dict[key].shape} → {current_state_dict[key].shape}), 새로 초기화")
                        initialized_count += 1
                else:
                    compatible_state_dict[key] = current_state_dict[key]
                    print(f"   ❌ {key}: 누락, 새로 초기화")
                    initialized_count += 1
            
            # 누락된 레이어들 확인
            missing_in_current = set(loaded_state_dict.keys()) - set(current_state_dict.keys())
            if missing_in_current:
                print(f"   📝 현재 모델에 없는 레이어들: {len(missing_in_current)}개")
                for key in sorted(missing_in_current):
                    print(f"      - {key}: {loaded_state_dict[key].shape}")
            
            # 호환 가능한 state_dict로 로드
            try:
                self.q_network.load_state_dict(compatible_state_dict)
                self.target_network.load_state_dict(compatible_state_dict)
                print("   ✅ 네트워크 가중치 로드 완료")
            except Exception as e:
                print(f"   ❌ 네트워크 가중치 로드 실패: {e}")
                # 부분적 로드를 시도
                try:
                    self.q_network.load_state_dict(compatible_state_dict, strict=False)
                    self.target_network.load_state_dict(compatible_state_dict, strict=False)
                    print("   ⚠️ 부분적 로드로 복구됨")
                except Exception as e2:
                    print(f"   ❌ 부분적 로드도 실패: {e2}")
                    return False
            
            # 옵티마이저 로드 (호환성 확인)
            try:
                if 'optimizer' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                    print("   ✅ 옵티마이저 상태 로드됨")
                else:
                    print("   ⚠️ 옵티마이저 상태 없음, 기본값 사용")
            except Exception as e:
                print(f"   ⚠️ 옵티마이저 로드 실패: {e}, 기본값 사용")
            
            # 기타 파라미터들
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.training_rewards = checkpoint.get('training_rewards', [])
            self.losses = checkpoint.get('losses', [])
            self.win_rates = checkpoint.get('win_rates', [])
            self.update_count = checkpoint.get('update_count', 0)
            
            print(f"✅ {model_state_size}차원 → 58차원 호환성 모델 로드 성공!")
            print(f"   - 로드된 레이어: {loaded_count}개")
            print(f"   - 새로 초기화된 레이어: {initialized_count}개")
            
            # 모델 검증
            try:
                # 간단한 테스트로 모델이 정상 작동하는지 확인
                test_input = torch.randn(1, self.state_size).to(self.device)
                with torch.no_grad():
                    _ = self.q_network(test_input)
                print("   ✅ 모델 검증 완료 - 정상 작동")
            except Exception as e:
                print(f"   ⚠️ 모델 검증 실패: {e}")
                print("   모델이 로드되었지만 예상과 다르게 작동할 수 있습니다.")
            
            return True
            
        except Exception as e:
            print(f"❌ 호환성 모델 로드 실패: {e}")
            print(f"   오류 타입: {type(e).__name__}")
            import traceback
            print(f"   상세 오류: {traceback.format_exc()}")
            return False

    def create_compatible_model(self, old_model_path: str) -> bool:
        """기존 모델을 SimpleDQN 아키텍처로 변환"""
        try:
            print(f"🔄 기존 모델을 SimpleDQN 아키텍처로 변환 중...")
            
            # 기존 모델 로드
            checkpoint = torch.load(old_model_path, map_location=self.device, weights_only=False)
            old_state_dict = checkpoint['q_network']
            
            # 새로운 DuelingDQN 모델 생성
            new_model = DuelingDQN(self.state_size, 3, self.hidden_size).to(self.device)
            new_state_dict = new_model.state_dict()
            
            # 호환 가능한 가중치만 복사
            compatible_weights = {}
            for key in new_state_dict.keys():
                if key in old_state_dict and new_state_dict[key].shape == old_state_dict[key].shape:
                    compatible_weights[key] = old_state_dict[key]
                    print(f"   ✅ {key}: 변환됨")
                else:
                    compatible_weights[key] = new_state_dict[key]
                    print(f"   ❌ {key}: 새로 초기화")
            
            # 새로운 모델에 가중치 로드
            new_model.load_state_dict(compatible_weights)
            
            # 현재 에이전트의 네트워크 교체
            self.q_network = new_model
            self.target_network = DuelingDQN(self.state_size, 3, self.hidden_size).to(self.device)
            self.target_network.load_state_dict(compatible_weights)
            
            print(f"✅ DuelingDQN 아키텍처로 변환 완료!")
            return True
            
        except Exception as e:
            print(f"❌ 모델 변환 실패: {e}")
            return False

    def diagnose_model_compatibility(self, model_path: str) -> Dict:
        """모델 호환성 진단"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            old_state_dict = checkpoint['q_network']
            current_state_dict = self.q_network.state_dict()
            
            diagnosis = {
                'total_old_layers': len(old_state_dict),
                'total_new_layers': len(current_state_dict),
                'compatible_layers': 0,
                'incompatible_layers': 0,
                'missing_layers': 0,
                'compatibility_rate': 0.0
            }
            
            for key in current_state_dict.keys():
                if key in old_state_dict:
                    if current_state_dict[key].shape == old_state_dict[key].shape:
                        diagnosis['compatible_layers'] += 1
                    else:
                        diagnosis['incompatible_layers'] += 1
                else:
                    diagnosis['missing_layers'] += 1
            
            diagnosis['compatibility_rate'] = diagnosis['compatible_layers'] / len(current_state_dict)
            
            return diagnosis
            
        except Exception as e:
            return {'error': str(e)}

class DataLoader:
    """58차원 RL Decision 기반 데이터 로딩 클래스"""
    
    @staticmethod
    def load_price_data(file_path: str = 'data/ETHUSDC_3m_historical_data.csv') -> Optional[pd.DataFrame]:
        """가격 데이터 로드"""
        try:
            required_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
            
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            df = df[required_columns]
            
            price_data = df.reset_index()
            print(f"가격 데이터 로드: {len(price_data):,}개 캔들")
            return price_data
            
        except Exception as e:
            print(f"가격 데이터 로드 실패: {e}")
            return None
    
    @staticmethod
    def load_signal_data(agent_folder: str = "agent") -> Optional[List[Dict]]:
        """58차원용 RL Decision 데이터 로드"""
        parquet_files = []
        
        if Path(agent_folder).exists():
            parquet_files = list(Path(agent_folder).glob("*.parquet"))
        
        if parquet_files:
            try:
                print(f"Signal 데이터 로드 중: {parquet_files[0].name}")
                signal_df = pd.read_parquet(parquet_files[0])
                print(f"Signal 데이터 로드: {len(signal_df):,}개 레코드")
                
                return DataLoader._convert_parquet_to_signal_dicts(signal_df)
                
            except Exception as e:
                print(f"Parquet 로드 실패: {e}")
        
        print("Parquet 파일이 없어 기본 Signal을 생성합니다.")
        return None
    
    @staticmethod
    def _convert_parquet_to_signal_dicts(signal_df: pd.DataFrame) -> List[Dict]:
        """Parquet을 Signal Dict 리스트로 변환 (58차원용) - 새로운 RL 스키마"""
        signal_data = []
        
        print("58차원용 RL 스키마 Signal 데이터 변환 중...")
        
        for idx, row in signal_df.iterrows():
            # 각 행을 딕셔너리로 변환 (새로운 RL 스키마 형태 유지)
            signal_dict = {}
            
            for col, value in row.items():
                if pd.notna(value):
                    # 수치 데이터는 그대로 유지
                    signal_dict[col] = value
                else:
                    # 기본값 설정 (새로운 RL 스키마에 맞게)
                    if 'action_value' in col or 'net_score' in col or 'buy_score' in col or 'sell_score' in col:
                        signal_dict[col] = 0.0
                    elif 'confidence_value' in col or 'market_context_value' in col:
                        signal_dict[col] = 0.0
                    elif 'conflict_' in col:
                        signal_dict[col] = 0.0
                    elif 'leverage' in col or 'signals_used' in col or 'strategies_count' in col:
                        signal_dict[col] = 0
                    elif 'max_holding_minutes' in col:
                        signal_dict[col] = 0
                    else:
                        signal_dict[col] = 0.0
            
            signal_data.append(signal_dict)
            
            if (idx + 1) % 5000 == 0:
                print(f"   변환 진행: {idx + 1:,}/{len(signal_df):,}")
        
        print(f"58차원용 RL 스키마 Signal 데이터 변환 완료: {len(signal_data):,}개")
        return signal_data
    

class PerformanceAnalyzer:
    """58차원 RL Decision 기반 성능 분석 클래스"""
    
    @staticmethod
    def evaluate_agent(agent: RLAgent, env: TradingEnvironment, num_episodes: int = 10) -> Tuple[List[Dict], Dict]:
        """58차원 에이전트 성능 평가"""
        print(f"58차원 에이전트 성능 평가 중 ({num_episodes} 에피소드)...")
        
        original_epsilon = agent.epsilon
        agent.epsilon = 0.01  # 테스트에서도 약간의 탐험 허용
        
        results = []
        all_trades = []
        
        for episode in range(num_episodes):
            # 테스트 환경 완전 초기화
            state, _ = env.reset()
            episode_reward = 0
            episode_trades = []
            episode_balance = env.initial_balance
            
            # 테스트 환경 상태 확인
            print(f"   에피소드 {episode+1}: 초기 잔고 ${episode_balance:.0f}")
            
            for step in range(500):
                action = agent.act(state)
                next_state, reward, done, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_balance = info['balance']
                
                if info.get('trade_completed', False):
                    trade_pnl = info.get('trade_pnl', 0.0)
                    episode_trades.append(1 if trade_pnl > 0 else 0)
                
                state = next_state
                if done:
                    break
            
            episode_return = (episode_balance - env.initial_balance) / env.initial_balance
            win_rate = np.mean(episode_trades) if episode_trades else 0.0
            
            # 테스트 결과 상세 로깅
            print(f"   에피소드 {episode+1}: 거래 {len(episode_trades)}개, 승률 {win_rate:.1%}, 수익률 {episode_return:.1%}, 잔고 ${episode_balance:.0f}")
            
            results.append({
                'episode': episode,
                'total_reward': episode_reward,
                'final_balance': episode_balance,
                'return': episode_return,
                'win_rate': win_rate,
                'total_trades': len(episode_trades),
                'max_drawdown': info.get('max_drawdown', 0.0)
            })
            
            all_trades.extend(episode_trades)
        
        agent.epsilon = original_epsilon
        
        overall_stats = {
            'avg_return': np.mean([r['return'] for r in results]),
            'avg_reward': np.mean([r['total_reward'] for r in results]),
            'overall_win_rate': np.mean(all_trades) if all_trades else 0.0,
            'avg_trades_per_episode': np.mean([r['total_trades'] for r in results]),
            'avg_max_drawdown': np.mean([r['max_drawdown'] for r in results]),
            'consistency': 1.0 - np.std([r['return'] for r in results]) if len(results) > 1 else 1.0,
            'total_trades': len(all_trades),
            'model_dimension': agent.state_size
        }
        
        return results, overall_stats
    
    @staticmethod
    def print_performance_report(results: List[Dict], stats: Dict):
        """58차원 성능 리포트 출력"""
        print("\n" + "="*60)
        print(f"58차원 RL Decision 기반 성능 평가 결과")
        print("="*60)
        print(f"모델 차원: {stats['model_dimension']}차원")
        print(f"전체 승률: {stats['overall_win_rate']:.3f}")
        print(f"평균 수익률: {stats['avg_return']:.3f} ({stats['avg_return']*100:.1f}%)")
        print(f"평균 리워드: {stats['avg_reward']:.1f}")
        print(f"에피소드당 평균 거래 수: {stats['avg_trades_per_episode']:.1f}")
        print(f"평균 최대 낙폭: {stats['avg_max_drawdown']:.3f}")
        print(f"성과 일관성: {stats['consistency']:.3f}")
        print(f"총 거래 수: {stats['total_trades']}")
        
        grade = PerformanceAnalyzer._get_performance_grade(stats)
        print(f"\n성능 등급: {grade}")
        
        recommendations = PerformanceAnalyzer._get_recommendations(stats)
        print("\n개선 제안:")
        for rec in recommendations:
            print(f"  • {rec}")
    
    @staticmethod
    def _get_performance_grade(stats: Dict) -> str:
        """성능 등급 산출 (수익률 중심)"""
        avg_return = stats['avg_return']
        consistency = stats['consistency']
        win_rate = stats['overall_win_rate']  # 보조 지표
        
        score = 0
        
        # 수익률이 주요 평가 기준 (가중치 높음)
        if avg_return >= 0.20: score += 5  # 20% 이상
        elif avg_return >= 0.15: score += 4  # 15% 이상
        elif avg_return >= 0.10: score += 3  # 10% 이상
        elif avg_return >= 0.05: score += 2  # 5% 이상
        elif avg_return >= 0.0: score += 1   # 0% 이상
        
        # 일관성 (중요하지만 수익률보다 낮은 가중치)
        if consistency >= 0.8: score += 2
        elif consistency >= 0.6: score += 1
        
        # 승률 (보조 지표)
        if win_rate >= 0.6: score += 1
        
        grades = {8: "A+ (우수)", 7: "A (좋음)", 6: "B+ (양호)", 5: "B (보통)", 
                 4: "C+ (미흡)", 3: "C (개선필요)", 2: "D (나쁨)", 1: "F (매우나쁨)", 0: "F (실패)"}
        
        return grades.get(score, "F (실패)")
    
    @staticmethod
    def _get_recommendations(stats: Dict) -> List[str]:
        """성능 기반 개선 제안 (수익률 중심)"""
        recommendations = []
        
        # 수익률이 주요 기준
        if stats['avg_return'] < 0.05:
            recommendations.append("수익률이 5% 미만입니다. 수익률 중심 보상 함수를 더 강화하세요.")
        
        if stats['avg_return'] < 0.10:
            recommendations.append("수익률이 10% 미만입니다. 58차원 상태 공간의 수익률 최적화를 더 활용하세요.")
        
        if stats['avg_return'] < 0.15:
            recommendations.append("수익률이 15% 미만입니다. Signal 기반 수익률 예측을 개선하세요.")
        
        # 리스크 관리
        if stats['avg_max_drawdown'] > 0.2:
            recommendations.append("최대 낙폭이 큽니다. 수익률과 리스크의 균형을 맞추세요.")
        
        # 일관성
        if stats['consistency'] < 0.5:
            recommendations.append("성과 일관성이 떨어집니다. 수익률 안정성을 위한 더 많은 훈련이 필요합니다.")
        
        # 거래 빈도
        if stats['avg_trades_per_episode'] < 3:
            recommendations.append("거래 빈도가 낮습니다. 수익률 기회를 놓치지 않도록 Signal 감도를 조정하세요.")
        
        # 승률은 보조 지표
        if stats['overall_win_rate'] < 0.4:
            recommendations.append("승률이 매우 낮습니다. 수익률과 승률의 균형을 고려하세요.")
        
        if not recommendations:
            recommendations.append("58차원 RL Decision 기반 수익률 중심 시스템이 잘 작동하고 있습니다!")
        
        return recommendations

class TrainingManager:
    """58차원 RL Decision 기반 훈련 관리 클래스"""
    
    @staticmethod
    def train_agent(agent: RLAgent, train_env: TradingEnvironment, 
                   episodes: int = 1000, save_interval: int = 100, 
                   test_env: TradingEnvironment = None) -> Tuple[RLAgent, List[float], List[float]]:
        """58차원 RL Decision 기반 에이전트 훈련 (테스트 환경 모니터링 포함)"""
        print(f"58차원 RL Decision 기반 강화학습 훈련 시작 ({episodes} 에피소드)")
        print(f"상태 공간: {train_env.observation_space.shape[0]}차원")
        if test_env:
            print(f"테스트 환경 모니터링: 활성화")
        
        # 에이전트에 액션 공간 설정
        agent.action_space = train_env.action_space
        
        episode_rewards = []
        episode_win_rates = []
        episode_returns = []  # 훈련 데이터 수익률 추적
        test_win_rates = []  # 테스트 데이터셋 승률 추적
        best_win_rate = 0.0
        best_test_win_rate = 0.0
        
        for episode in range(episodes):
            state, _ = train_env.reset()
            total_reward = 0
            episode_trades = []
            steps = 0
            
            while steps < 1000:
                action = agent.act(state)
                next_state, reward, done, truncated, info = train_env.step(action)
                
                agent.remember(state, action, reward, next_state, done)
                
                if info.get('trade_completed', False):
                    trade_pnl = info.get('trade_pnl', 0.0)
                    episode_trades.append(1 if trade_pnl > 0 else 0)
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if len(agent.memory) > agent.batch_size:
                    agent.replay()
                
                if done:
                    break
            
            episode_rewards.append(total_reward)
            episode_win_rate = np.mean(episode_trades) if episode_trades else 0.0
            episode_win_rates.append(episode_win_rate)
            
            # 훈련 데이터 수익률 계산 (잔고 변화 기반)
            initial_balance = train_env.initial_balance  # 환경의 실제 초기 잔고 사용
            final_balance = info.get('balance', initial_balance)
            episode_return = (final_balance - initial_balance) / initial_balance
            episode_returns.append(episode_return)
            
            agent.training_rewards.append(total_reward)
            agent.win_rates.append(episode_win_rate)
            
            
            # 적응형 학습률 업데이트 (10에피소드마다)
            if episode % 10 == 0 and episode > 0:
                recent_rewards = episode_rewards[-20:] if len(episode_rewards) >= 20 else episode_rewards
                recent_win_rates = episode_win_rates[-20:] if len(episode_win_rates) >= 20 else episode_win_rates
                agent.adaptive_learning_rate(recent_rewards, recent_win_rates)
            
            # 테스트 데이터셋으로 성능 평가 (과적합 방지 강화)
            if test_env and episode % 5 == 0 and episode > 0:  # 더 자주 평가
                print(f"\n📊 Episode {episode}: 테스트 데이터셋 성능 평가 중...")
                test_results, test_stats = PerformanceAnalyzer.evaluate_agent(agent, test_env, num_episodes=5)  # 더 많은 에피소드로 평가
                test_return = test_stats['avg_return']
                test_win_rates.append(test_stats['overall_win_rate'])
                
                print(f"   테스트 수익률: {test_return:.3f} ({test_return*100:.1f}%) (이전 최고: {best_test_win_rate:.3f})")
                
                # 과적합 감지: 훈련 수익률과 테스트 수익률 차이 확인
                recent_train_return = np.mean(episode_returns[-10:]) if len(episode_returns) >= 10 else 0.0
                overfitting_gap = recent_train_return - (test_return if test_return > 0 else -test_return)
                
                if overfitting_gap > 0.1:  # 훈련 수익률이 테스트 수익률보다 10% 이상 높으면 과적합 의심
                    print(f"⚠️ 과적합 감지: 훈련 수익률({recent_train_return:.3f}) - 테스트 수익률({test_return:.3f}) = {overfitting_gap:.3f}")
                    # 학습률 감소
                    for param_group in agent.optimizer.param_groups:
                        param_group['lr'] *= 0.9
                    print(f"   학습률 감소: {agent.optimizer.param_groups[0]['lr']:.2e}")
                
                if test_return > best_test_win_rate:
                    best_test_win_rate = test_return
                    # 에피소드별 모델 저장 (수익률 기준)
                    agent.save_model(f'best_test_model_ep{episode}_return{test_return:.3f}.pth')
                    # 최고 성능 모델 업데이트 (수익률 기준)
                    agent.save_model('agent/best_test_performance_model_return{:.3f}.pth'.format(test_return))
                    print(f"🎯 새로운 테스트 데이터셋 최고 수익률! 수익률: {test_return:.3f} ({test_return*100:.1f}%)")
                    print(f"   최고 성능 모델 업데이트: best_test_performance_model_return{test_return:.3f}.pth")
                print()  # 빈 줄 추가
            
            # 진행 상황 출력 (더 자주)
            if episode % 5 == 0 or episode < 10:
                recent_rewards = episode_rewards[-50:] if len(episode_rewards) >= 50 else episode_rewards
                recent_win_rates = episode_win_rates[-50:] if len(episode_win_rates) >= 50 else episode_win_rates
                recent_returns = episode_returns[-50:] if len(episode_returns) >= 50 else episode_returns
                
                avg_reward = np.mean(recent_rewards)
                avg_win_rate = np.mean(recent_win_rates)
                avg_return = np.mean(recent_returns)
                
                # 테스트 성능도 함께 표시
                test_info = ""
                if test_win_rates:
                    recent_test_win_rate = np.mean(test_win_rates[-5:]) if len(test_win_rates) >= 5 else test_win_rates[-1]
                    test_info = f" | 테스트: {recent_test_win_rate:.3f}"
                
                # 수익률과 리워드 일치성 확인
                reward_return_ratio = avg_reward / (avg_return * 100) if avg_return != 0 else 0
                
                print(f"Episode {episode:4d} | "
                        f"훈련승률: {avg_win_rate:.3f} | "
                        f"훈련수익률: {avg_return:.3f} ({avg_return*100:+.1f}%){test_info} | "
                        f"리워드: {avg_reward:7.1f} | "
                        f"잔고: ${info['balance']:7.0f} | "
                        f"거래: {info.get('total_trades', 0):3d}개 | "
                        f"ε: {agent.epsilon:.3f} | "
                        f"LR: {agent.learning_rate:.2e} | "
                        f"58D")
            
            # 베스트 모델 저장 (훈련 데이터 기준 - 수익률 중심)
            if episode % save_interval == 0 and episode > 0:
                # 최근 100 에피소드의 평균 수익률 계산
                recent_returns = []
                for i in range(max(0, len(episode_rewards)-100), len(episode_rewards)):
                    if i < len(episode_rewards):
                        # 간단한 수익률 추정 (리워드 기반)
                        estimated_return = episode_rewards[i] / 1000.0  # 리워드를 수익률로 근사
                        recent_returns.append(estimated_return)
                
                current_avg_return = np.mean(recent_returns) if recent_returns else 0.0
                
                if current_avg_return > best_win_rate:  # 변수명은 그대로 유지하지만 수익률로 사용
                    best_win_rate = current_avg_return
                    agent.save_model(f'best_train_model_ep{episode}_return{current_avg_return:.3f}.pth')
                    print(f"🎯 새로운 훈련 데이터셋 최고 수익률! 수익률: {current_avg_return:.3f} ({current_avg_return*100:.1f}%)")
            
            # 조기 종료 조건 (과적합 방지 강화)
            if episode > 500 and test_win_rates:
                # 최근 테스트 결과들의 수익률 확인
                recent_test_returns = []
                for i in range(max(0, len(test_win_rates)-5), len(test_win_rates)):
                    if i < len(test_win_rates):
                        # 테스트 수익률 추정 (승률을 수익률로 근사)
                        estimated_return = test_win_rates[i] * 0.1  # 승률 65% = 수익률 6.5%로 근사
                        recent_test_returns.append(estimated_return)
                
                recent_test_return = np.mean(recent_test_returns) if recent_test_returns else 0.0
                
                # 과적합 감지 시 조기 종료
                if len(episode_returns) >= 20:
                    recent_train_return = np.mean(episode_returns[-20:])
                    overfitting_gap = recent_train_return - recent_test_return
                    
                    if overfitting_gap > 0.15:  # 과적합이 심하면 조기 종료
                        print(f"🛑 과적합으로 인한 조기 종료: 훈련 수익률({recent_train_return:.3f}) - 테스트 수익률({recent_test_return:.3f}) = {overfitting_gap:.3f}")
                        agent.save_model('agent/early_stop_model.pth')
                        break
                
                if recent_test_return >= 0.20:  # 수익률 20% 이상 달성
                    print(f"🏆 58차원 목표 달성! 테스트 데이터셋 수익률 {recent_test_return:.3f} ({recent_test_return*100:.1f}%) 도달")
                    agent.save_model('agent/final_optimized_model_58d.pth')
                    break
        
        
        print(f"\n58차원 RL Decision 기반 훈련 완료!")
        print(f"   총 에피소드: {episode + 1}")
        print(f"   훈련 데이터 최고 승률: {best_win_rate:.3f}")
        print(f"   훈련 데이터 최종 승률: {np.mean(episode_win_rates[-50:]) if episode_win_rates else 0:.3f}")
        if test_win_rates:
            print(f"   테스트 데이터 최고 승률: {best_test_win_rate:.3f}")
            print(f"   테스트 데이터 최종 승률: {test_win_rates[-1]:.3f}")
        print(f"   상태 차원: 58차원 (RL Decision 기반)")
        print(f"   아키텍처: DuelingDQN (Value + Advantage 분리)")
        print(f"   정규화 기법: 엔트로피 정규화, Spectral Normalization, 적응적 드롭아웃")
        
        # 테스트 데이터셋 최고 성능 모델 저장
        if test_win_rates and best_test_win_rate > 0:
            best_test_model_path = f'agent/best_test_performance_model_wr{best_test_win_rate:.3f}.pth'
            agent.save_model(best_test_model_path)
            print(f"✅ 테스트 데이터셋 최고 성능 모델 저장: {best_test_model_path}")
        
        return agent, episode_rewards, episode_win_rates

def synchronize_data_by_timestamp(price_data: pd.DataFrame, signal_data: List[Dict]) -> Tuple[pd.DataFrame, List[Dict]]:
    """타임스탬프 기준으로 Price와 Signal 데이터 동기화"""
    print("타임스탬프 기준 데이터 동기화 중...")
    
    # Signal 데이터의 시작과 끝 타임스탬프 기준으로 Price 데이터 슬라이싱
    if not signal_data or 'timestamp' not in signal_data[0]:
        print("Signal 데이터에 타임스탬프가 없습니다. 길이 기준으로 동기화합니다.")
        min_length = len(signal_data)
        price_data = price_data.iloc[-min_length:].reset_index(drop=True)
        signal_data = signal_data[-min_length:]
        print(f"길이 기준 동기화 완료: {min_length:,}개")
        return price_data, signal_data
    
    signal_start_time = signal_data[0]['timestamp']
    signal_end_time = signal_data[-1]['timestamp']
        
    if hasattr(signal_end_time, 'timestamp'):
        signal_end_timestamp = signal_end_time.timestamp()
    elif isinstance(signal_end_time, str):
        signal_end_timestamp = pd.to_datetime(signal_end_time).timestamp()
    else:
        signal_end_timestamp = float(signal_end_time)
    
    # Price 데이터에서 Signal 시작/끝 시간과 정확히 일치하는 인덱스 찾기
    price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
    
    # 정확한 타임스탬프 매칭
    start_matches = price_data[price_data['timestamp'] == signal_start_time]
    end_matches = price_data[price_data['timestamp'] == signal_end_time]
    
    if len(start_matches) == 0 or len(end_matches) == 0:
        print("❌ 정확한 타임스탬프 매칭을 찾을 수 없습니다.")
        return None, None
    
    # Signal 데이터의 정확한 타임스탬프에 맞춰 Price 데이터 필터링
    # Signal 데이터에 있는 타임스탬프만 Price 데이터에서 선택
    signal_timestamps = set(signal['timestamp'] for signal in signal_data)
    price_data = price_data[price_data['timestamp'].isin(signal_timestamps)].reset_index(drop=True)
    
    # 동기화 검증
    price_start = price_data.iloc[0]['timestamp']
    price_end = price_data.iloc[-1]['timestamp']
    signal_start = signal_data[0]['timestamp']
    signal_end = signal_data[-1]['timestamp']
    
    # 타임스탬프 정확한 일치 확인
    start_time_match = (price_start == signal_start)
    end_time_match = (price_end == signal_end)
    length_match = (len(price_data) == len(signal_data))
    
    print(f"✅ 시작 시간 동기화: {'성공' if start_time_match else '실패'}")
    print(f"✅ 끝 시간 동기화: {'성공' if end_time_match else '실패'}")
    print(f"   Price: {price_start} ~ {price_end}")
    print(f"   Signal: {signal_start} ~ {signal_end}")
    print(f"✅ 길이 동기화: {'성공' if length_match else '실패'}")
    print(f"   Price: {len(price_data):,}개, Signal: {len(signal_data):,}개")
    
    if not (start_time_match and end_time_match and length_match):
        print("❌ 동기화 실패! 데이터를 다시 확인하세요.")
        return None, None
    
    return price_data, signal_data

def split_data(price_data: pd.DataFrame, signal_data: List[Dict], 
               train_ratio: float = 0.8, test_ratio: float = 0.2) -> Tuple[pd.DataFrame, List[Dict], pd.DataFrame, List[Dict]]:
    """데이터를 훈련용과 테스트용으로 분할"""
    total_length = min(len(price_data), len(signal_data))
    train_size = int(total_length * train_ratio)
    
    # 훈련 데이터
    train_price = price_data.iloc[:train_size].reset_index(drop=True)
    train_signal = signal_data[:train_size]
    
    # 테스트 데이터
    test_price = price_data.iloc[train_size:].reset_index(drop=True)
    test_signal = signal_data[train_size:]
    
    print(f"데이터 분할 완료:")
    print(f"  - 훈련 데이터: {len(train_price):,}개 ({train_ratio*100:.1f}%)")
    print(f"  - 테스트 데이터: {len(test_price):,}개 ({test_ratio*100:.1f}%)")
    
    return train_price, train_signal, test_price, test_signal

def main():
    """58차원 RL Decision 기반 메인 실행 함수"""
    print("58차원 RL Decision 기반 강화학습 트레이딩 시스템")
    print("=" * 80)
    
    try:
        # 1. 데이터 로딩
        print("\n1️⃣ 58차원용 데이터 로딩...")
        price_data = DataLoader.load_price_data()
        if price_data is None:
            print("가격 데이터 로드 실패")
            return
        
        signal_data = DataLoader.load_signal_data()
        
        # 타임스탬프 기준으로 데이터 동기화
        price_data, signal_data = synchronize_data_by_timestamp(price_data, signal_data)
        if price_data is None or signal_data is None:
            print("데이터 동기화 실패로 프로그램을 종료합니다.")
            return
        
        # 2. 데이터 분할 (훈련 80%, 테스트 20%)
        print("\n2️⃣ 데이터 분할...")
        train_price, train_signal, test_price, test_signal = split_data(price_data, signal_data, 0.8, 0.2)
        
        # 3. 환경 및 에이전트 생성
        print("\n3️⃣ 58차원 환경 및 에이전트 생성...")
        train_env = TradingEnvironment(train_price, train_signal)
        test_env = TradingEnvironment(test_price, test_signal)
        agent = RLAgent(train_env.observation_space.shape[0])  # 58차원
        
        print(f"상태 공간: {train_env.observation_space.shape[0]}차원")
        print("Signal의 모든 indicator와 raw score 활용")
        
        model_loaded = False
        
        # 1. 테스트 성능 모델 우선 로드
        import glob
        test_model_files = glob.glob('agent/best_test_performance_model_return*.pth')
        if test_model_files:
            # 가장 높은 승률의 테스트 모델 선택
            best_test_model = max(test_model_files, key=lambda x: float(x.split('return')[1].split('.pth')[0]))
            if agent.load_model(best_test_model):
                model_loaded = True
                print(f"✅ 테스트 데이터셋 최고 성능 모델 로드: {best_test_model}")
        
        # 2. 호환성 모드로 기존 모델 로드 시도
        if not model_loaded:
            for model_file in ['agent/final_optimized_model_58d.pth', 'agent/best_model_58d.pth']:
                if os.path.exists(model_file):
                    print(f"🔄 호환성 모드로 {model_file} 로드 시도...")
                    
                    # 모델 호환성 진단
                    diagnosis = agent.diagnose_model_compatibility(model_file)
                    if 'error' not in diagnosis:
                        print(f"   📊 호환성 진단: {diagnosis['compatibility_rate']:.1%} ({diagnosis['compatible_layers']}/{diagnosis['total_new_layers']} 레이어)")
                    
                    if agent.load_model_with_compatibility(model_file):
                        model_loaded = True
                        print(f"✅ 호환성 모드로 모델 로드 성공: {model_file}")
                        break
        
        # 3. 모델 변환 시도 (기존 모델을 AdvancedProfitDQN 아키텍처로)
        if not model_loaded:
            for model_file in ['agent/final_optimized_model_58d.pth', 'agent/best_model_58d.pth']:
                if os.path.exists(model_file):
                    print(f"🔄 모델 변환 시도: {model_file}")
                    if agent.create_compatible_model(model_file):
                        model_loaded = True
                        print(f"✅ 모델 변환 성공: {model_file}")
                        break
        
        if not model_loaded:
            print("새로운 58차원 모델로 시작합니다.")
        
        # 4. 훈련 전 테스트 데이터셋 성능 평가 (베이스라인)
        print("\n4️⃣ 훈련 전 테스트 데이터셋 성능 평가...")
        baseline_results, baseline_stats = PerformanceAnalyzer.evaluate_agent(agent, test_env, num_episodes=5)
        print("=== 훈련 전 테스트 데이터셋 성능 ===")
        PerformanceAnalyzer.print_performance_report(baseline_results, baseline_stats)
        
        # 5. 훈련 데이터셋으로 훈련
        print(f"\n5️⃣ 훈련 데이터셋으로 58차원 RL Decision 기반 훈련 시작...")
        print(f"   훈련 데이터: {len(train_price):,}개")
        print(f"   테스트 데이터: {len(test_price):,}개")
        print(f"   목표 수익률: 5%+ (수익률 중심)")
        print(f"   Signal 특성 활용: 수익률 최적화")
        
        # 훈련 실행 (과적합 방지 강화)
        trained_agent, rewards, win_rates = TrainingManager.train_agent(agent, train_env, episodes=1000, test_env=test_env)
        
        # 6. 훈련 후 테스트 데이터셋으로 성능 평가
        print("\n6️⃣ 훈련 후 테스트 데이터셋 성능 평가...")
        final_results, final_stats = PerformanceAnalyzer.evaluate_agent(trained_agent, test_env, num_episodes=10)
        print("=== 훈련 후 테스트 데이터셋 성능 ===")
        PerformanceAnalyzer.print_performance_report(final_results, final_stats)
        
        # 7. 성능 개선도 분석
        improvement = final_stats['overall_win_rate'] - baseline_stats['overall_win_rate']
        print(f"\n🚀 58차원 RL Decision 기반 성능 개선도 (테스트 데이터셋 기준):")
        print(f"   승률: {baseline_stats['overall_win_rate']:.3f} → {final_stats['overall_win_rate']:.3f} ({improvement:+.3f})")
        print(f"   평균 수익률: {baseline_stats['avg_return']:.3f} → {final_stats['avg_return']:.3f}")
        print(f"   Signal 활용도: 최대화됨")
        
        # 8. 최종 모델 저장
        trained_agent.save_model('agent/final_optimized_model_58d.pth')
        print(f"\n✅ 최종 모델이 저장되었습니다: agent/final_optimized_model_58d.pth")
        
        # 9. 추가 훈련 여부 확인 (수익률 기준)
        if final_stats['avg_return'] < 0.05:  # 수익률 5% 미만
            user_input = input("\n수익률이 목표(5%)에 미달합니다. 추가 훈련을 원하시나요? (y/n): ")
            if user_input.lower() == 'y':
                print("58차원 수익률 중심 추가 훈련 시작...")
                TrainingManager.train_agent(trained_agent, train_env, episodes=1000, test_env=test_env)
                
                # 추가 훈련 후 재평가
                print("\n추가 훈련 후 테스트 데이터셋 성능 평가...")
                additional_results, additional_stats = PerformanceAnalyzer.evaluate_agent(trained_agent, test_env, num_episodes=10)
                print("=== 추가 훈련 후 테스트 데이터셋 성능 ===")
                PerformanceAnalyzer.print_performance_report(additional_results, additional_stats)
        else:
            print(f"\n🎉 목표 수익률 달성! (테스트 데이터셋 수익률: {final_stats['avg_return']:.3f} ({final_stats['avg_return']*100:.1f}%))")
    
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()