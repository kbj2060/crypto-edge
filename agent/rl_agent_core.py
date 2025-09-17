"""
강화학습 에이전트 코어 모듈 (80차원 Signal 기반)
- Signal의 모든 indicator와 raw score 활용
- 중복 계산 제거 및 정보 활용 극대화
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import gym
from gym import spaces
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple, Any, Optional
import logging

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class EnhancedTradingEnvironment(gym.Env):
    """
    80차원 Signal 기반 암호화폐 거래 환경
    - Signal의 모든 정보 활용
    - 중복 계산 제거
    """
    
    def __init__(self, price_data: pd.DataFrame, signal_data: List[Dict], 
                 initial_balance: float = 10000.0, max_position: float = 1.0):
        super().__init__()
        
        self.price_data = price_data
        self.signal_data = signal_data
        self.initial_balance = initial_balance
        self.max_position = max_position
        
        # 액션 스페이스: [포지션 변경량, 레버리지, 홀딩 시간]
        self.action_space = spaces.Box(
            low=np.array([-2.0, 1.0, 0.0]), 
            high=np.array([2.0, 20.0, 1440.0]), 
            dtype=np.float32
        )
        
        # 상태 스페이스: 80차원 (20 + 25 + 25 + 10)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(80,), 
            dtype=np.float32
        )
        
        # Signal 기반 상태 구성기
        self.state_builder = EnhancedSignalStateBuilder()
        
        self.reset()
    
    def _get_state_size(self) -> int:
        return 80
    
    def _extract_signal_features(self, signals: Dict) -> np.ndarray:
        """Signal에서 모든 특성 추출 (50차원: 25 technical + 25 decision)"""
        technical_features = self._extract_technical_scores(signals)
        decision_features = self._extract_decision_features(signals)
        
        return np.concatenate([technical_features, decision_features])
    
    def _extract_technical_scores(self, signals: Dict) -> np.ndarray:
        """각 전략의 raw score들을 특성으로 활용 (25차원)"""
        features = []
        
        # 모든 raw score 키들 수집
        all_raw_scores = []
        for key, value in signals.items():
            if '_raw_' in key and '_score' in key and value is not None:
                try:
                    all_raw_scores.append(float(value))
                except:
                    all_raw_scores.append(0.0)
        
        # 25개로 맞추기
        if len(all_raw_scores) >= 25:
            sorted_scores = sorted(all_raw_scores, key=abs, reverse=True)
            features = sorted_scores[:25]
        else:
            features = all_raw_scores + [0.0] * (25 - len(all_raw_scores))
        
        return np.array(features, dtype=np.float32)
    
    def _extract_decision_features(self, signals: Dict) -> np.ndarray:
        """Decision 특성들 (25차원)"""
        features = []
        
        # 각 시간대별 특성 (3 × 6 = 18개)
        for timeframe in ['short_term', 'medium_term', 'long_term']:
            action = signals.get(f'{timeframe}_action', 'HOLD')
            action_strength = 1.0 if action == 'LONG' else (-1.0 if action == 'SHORT' else 0.0)
            
            net_score = float(signals.get(f'{timeframe}_net_score', 0.0))
            buy_score = float(signals.get(f'{timeframe}_buy_score', 0.0))
            sell_score = float(signals.get(f'{timeframe}_sell_score', 0.0))
            
            confidence = signals.get(f'{timeframe}_confidence', 'LOW')
            confidence_val = 1.0 if confidence == 'HIGH' else (0.5 if confidence == 'MEDIUM' else 0.0)
            
            leverage = min(float(signals.get(f'{timeframe}_leverage', 1.0)) / 20.0, 1.0)
            
            features.extend([action_strength, net_score, buy_score, sell_score, confidence_val, leverage])
        
        # 추가 메타 정보 (7개)
        signals_used = []
        for timeframe in ['short_term', 'medium_term', 'long_term']:
            used = signals.get(f'{timeframe}_signals_used', 0)
            signals_used.append(min(float(used) / 10.0, 1.0))
        
        market_contexts = []
        for timeframe in ['short_term', 'medium_term']:
            context = signals.get(f'{timeframe}_market_context', 'NEUTRAL')
            context_val = 1.0 if context == 'TRENDING' else 0.0
            market_contexts.append(context_val)
        
        bias = signals.get('long_term_institutional_bias', 'NEUTRAL')
        bias_val = 1.0 if bias == 'BULLISH' else (-1.0 if bias == 'BEARISH' else 0.0)
        
        strength = signals.get('long_term_macro_trend_strength', 'MEDIUM')
        strength_val = 1.0 if strength == 'HIGH' else (0.5 if strength == 'MEDIUM' else 0.0)
        
        additional_features = signals_used + market_contexts + [bias_val, strength_val]
        features.extend(additional_features)
        
        return np.array(features[:25], dtype=np.float32)
    
    def _extract_price_features(self, idx: int) -> np.ndarray:
        """Signal의 indicator들을 활용한 가격 특성 (20차원)"""
        if idx >= len(self.signal_data):
            return np.zeros(20, dtype=np.float32)
        
        current_signal = self.signal_data[idx]
        current_candle = {
            'open': self.price_data.iloc[idx]['open'],
            'high': self.price_data.iloc[idx]['high'],
            'low': self.price_data.iloc[idx]['low'],
            'close': self.price_data.iloc[idx]['close'],
            'volume': self.price_data.iloc[idx]['volume'],
        }
        
        return self.state_builder._extract_price_indicators(current_signal, current_candle)
    
    def _get_portfolio_state(self) -> np.ndarray:
        """포트폴리오 상태 정보 (10차원)"""
        features = [
            self.current_position,
            self.current_leverage / 20.0,
            (self.balance - self.initial_balance) / self.initial_balance,
            self.unrealized_pnl / self.initial_balance if self.initial_balance > 0 else 0.0,
            min(self.total_trades / 100.0, 1.0),
            self.winning_trades / max(self.total_trades, 1),
            self.max_drawdown,
            min(self.consecutive_losses / 10.0, 1.0),
            min(self.holding_time / 1440.0, 1.0),
            1.0 if self.in_position else 0.0
        ]
        return np.array(features, dtype=np.float32)
    
    def reset(self):
        """환경 초기화"""
        self.current_step = 20
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
        self.last_trade_win = True
        self.holding_time = 0
        self.position_entry_step = 0
        self.in_position = False
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """80차원 상태 관찰값 반환 (차원 보장)"""
        if self.current_step >= min(len(self.price_data), len(self.signal_data)):
            return np.zeros(80, dtype=np.float32)
        
        # Signal과 현재 캔들 데이터
        current_signal = self.signal_data[self.current_step]
        current_candle = {
            'open': self.price_data.iloc[self.current_step]['open'],
            'high': self.price_data.iloc[self.current_step]['high'],
            'low': self.price_data.iloc[self.current_step]['low'],
            'close': self.price_data.iloc[self.current_step]['close'],
            'volume': self.price_data.iloc[self.current_step]['volume'],
        }
        
        # 1. Price Indicators (20차원)
        price_features = self.state_builder._extract_price_indicators(current_signal, current_candle)
        
        # 2. Technical Scores (25차원)  
        technical_features = self._extract_technical_scores(current_signal)
        
        # 3. Decision Features (25차원)
        decision_features = self._extract_decision_features(current_signal)
        
        # 4. Portfolio Features (10차원)
        portfolio_features = self._get_portfolio_state()
        
        # 모든 특성 결합
        observation = np.concatenate([price_features, technical_features, decision_features, portfolio_features])
        
        # 🔥 차원 보정 (78차원 → 80차원)
        current_dim = len(observation)
        if current_dim != 80:
            print(f"⚠️ 차원 불일치 감지: {current_dim} → 80차원으로 보정")
            
            if current_dim < 80:
                # 부족한 차원을 0으로 패딩
                padding = np.zeros(80 - current_dim, dtype=np.float32)
                observation = np.concatenate([observation, padding])
            else:
                # 초과한 차원을 자름
                observation = observation[:80]
        
        return observation.astype(np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """한 스텝 실행"""
        if self.current_step >= min(len(self.signal_data), len(self.price_data)) - 1:
            return self._get_observation(), 0.0, True, {}
        
        # 액션 해석
        position_change = np.clip(action[0], -2.0, 2.0)
        leverage = np.clip(action[1], 1.0, 20.0)
        target_holding_minutes = np.clip(action[2], 1.0, 1440.0)
        
        # 현재 가격
        current_price = self.price_data.iloc[self.current_step]['close']
        next_price = self.price_data.iloc[self.current_step + 1]['close']
        
        # 보상 계산 (Signal 정보 활용)
        reward = self._calculate_reward(position_change, leverage, current_price, next_price)
        
        # 포지션 업데이트
        self._update_position(position_change, leverage, current_price, target_holding_minutes)
        
        # 다음 스텝으로
        self.current_step += 1
        self.holding_time += 3
        
        # 포지션 홀딩 시간 체크
        if self.in_position and self.holding_time >= target_holding_minutes:
            self._close_position(next_price)
        
        # 다음 상태
        next_state = self._get_observation()
        
        # 종료 조건
        done = (self.current_step >= min(len(self.signal_data), len(self.price_data)) - 1 or 
                self.balance <= self.initial_balance * 0.1)
        
        info = self._create_info_dict()
        
        return next_state, reward, done, info
    
    def _calculate_reward(self, position_change: float, leverage: float, 
                         current_price: float, next_price: float) -> float:
        """Signal 정보를 활용한 보상 함수"""
        reward = 0.0
        
        # 1. PnL 기반 보상
        if abs(self.current_position) > 0.01:
            price_change = (next_price - current_price) / current_price
            position_pnl = self.current_position * price_change * self.current_leverage
            reward += position_pnl * 100
        
        # 2. Signal 정보를 활용한 신호 일치도 보상
        current_signal = self.signal_data[self.current_step]
        signal_alignment = self._calculate_signal_alignment(position_change, current_signal)
        reward += signal_alignment * 10
        
        # 3. 리스크 관리 보상
        risk_penalty = self._calculate_risk_penalty(leverage, self.current_position)
        reward -= risk_penalty
        
        # 4. 거래 빈도 패널티
        if abs(position_change) > 0.1:
            reward -= 0.5
        
        # 5. 연속 손실 패널티
        reward -= self.consecutive_losses * 0.2
        
        # 6. 홀딩 시간 최적화
        if self.in_position:
            holding_reward = self._calculate_holding_reward()
            reward += holding_reward
        
        return reward
    
    def _calculate_signal_alignment(self, position_change: float, signals: Dict) -> float:
        """Signal과 액션 일치도 계산"""
        alignment_score = 0.0
        
        # 각 시간대별 신호와의 일치도
        for timeframe in ['short_term', 'medium_term', 'long_term']:
            action = signals.get(f'{timeframe}_action', 'HOLD')
            net_score = float(signals.get(f'{timeframe}_net_score', 0.0))
            
            if action == 'LONG' and position_change > 0:
                alignment_score += abs(net_score)
            elif action == 'SHORT' and position_change < 0:
                alignment_score += abs(net_score)
            elif action == 'HOLD' and abs(position_change) < 0.1:
                alignment_score += 0.1
        
        return alignment_score / 3
    
    def _calculate_risk_penalty(self, leverage: float, position: float) -> float:
        """리스크 패널티 계산"""
        penalty = 0.0
        
        if leverage > 10:
            penalty += (leverage - 10) * 0.1
        
        if abs(position) > 0.8:
            penalty += (abs(position) - 0.8) * 5
        
        return penalty
    
    def _calculate_holding_reward(self) -> float:
        """홀딩 시간 최적화 보상"""
        if self.holding_time > 60:
            return -0.01 * (self.holding_time - 60) / 60
        return 0.0
    
    def _update_position(self, position_change: float, leverage: float, 
                        current_price: float, target_holding_minutes: float):
        """포지션 업데이트"""
        new_position = np.clip(self.current_position + position_change, -1.0, 1.0)
        
        if abs(new_position - self.current_position) > 0.01:
            # 기존 포지션 청산
            if abs(self.current_position) > 0.01:
                self._close_position(current_price)
            
            # 새 포지션 진입
            if abs(new_position) > 0.01:
                self.current_position = new_position
                self.current_leverage = leverage
                self.entry_price = current_price
                self.position_entry_step = self.current_step
                self.holding_time = 0
                self.in_position = True
    
    def _close_position(self, exit_price: float):
        """포지션 청산"""
        if abs(self.current_position) < 0.01:
            return
        
        # PnL 계산
        price_change = (exit_price - self.entry_price) / self.entry_price
        pnl = self.current_position * price_change * self.current_leverage * self.balance
        
        # 잔고 업데이트
        self.balance += pnl
        
        # 통계 업데이트
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1
            self.consecutive_losses = 0
            self.last_trade_win = True
        else:
            self.consecutive_losses += 1
            self.last_trade_win = False
        
        # 최대 낙폭 업데이트
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        else:
            drawdown = (self.peak_balance - self.balance) / self.peak_balance
            self.max_drawdown = max(self.max_drawdown, drawdown)
        
        # 포지션 초기화
        self.current_position = 0.0
        self.unrealized_pnl = 0.0
        self.in_position = False
        self.holding_time = 0
    
    def _create_info_dict(self) -> Dict:
        """정보 딕셔너리 생성"""
        current_price = self.price_data.iloc[min(self.current_step, len(self.price_data)-1)]['close']
        
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
            'step': self.current_step
        }

# EnhancedSignalStateBuilder를 import하기 위한 클래스 정의
class EnhancedSignalStateBuilder:
    """Signal 기반 상태 벡터 구성기 (live_trading_agent.py와 동일)"""
    
    def _extract_price_indicators(self, signal_data: Dict, current_candle: Dict) -> np.ndarray:
        """Signal의 indicator들을 price feature로 활용 (20차원)"""
        features = []
        current_price = current_candle['close']
        
        # 1. 가격 대비 지표 위치 (정규화)
        vwap = signal_data.get('indicator_vwap', current_price)
        poc = signal_data.get('indicator_poc', current_price)
        hvn = signal_data.get('indicator_hvn', current_price)
        lvn = signal_data.get('indicator_lvn', current_price)
        
        features.extend([
            (current_price - vwap) / current_price if current_price > 0 else 0.0,
            (current_price - poc) / current_price if current_price > 0 else 0.0,   
            (current_price - hvn) / current_price if current_price > 0 else 0.0,   
            (current_price - lvn) / current_price if current_price > 0 else 0.0,   
        ])
        
        # 2. 변동성 지표들
        atr = signal_data.get('indicator_atr', 0.0)
        vwap_std = signal_data.get('indicator_vwap_std', 0.0)
        
        features.extend([
            atr / current_price if current_price > 0 else 0.0,
            vwap_std / current_price if current_price > 0 else 0.0,
        ])
        
        # 3. 일별 기준점들과의 관계
        prev_high = signal_data.get('indicator_prev_day_high', current_price)
        prev_low = signal_data.get('indicator_prev_day_low', current_price)
        or_high = signal_data.get('indicator_opening_range_high', current_price)
        or_low = signal_data.get('indicator_opening_range_low', current_price)
        
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
            prev_day_position,
            or_position,
            (current_price - prev_high) / current_price if current_price > 0 else 0.0,
            (prev_low - current_price) / current_price if current_price > 0 else 0.0,
        ])
        
        # 4. 현재 캔들 정보
        high, low, close, open_price = current_candle['high'], current_candle['low'], current_candle['close'], current_candle['open']
        volume = current_candle.get('volume', 0)
        
        candle_features = [
            (close - open_price) / open_price if open_price > 0 else 0.0,
            (high - low) / close if close > 0 else 0.0,
            (high - close) / (high - low) if high > low else 0.5,
            (close - low) / (high - low) if high > low else 0.5,
            (close - open_price) / (high - low) if high > low else 0.0,
            min(volume / 1000000, 2.0) if volume > 0 else 0.0,
            1.0 if close > open_price else 0.0,
            (high - max(open_price, close)) / (high - low) if high > low else 0.0
        ]
        
        features.extend(candle_features[:8])
        
        return np.array(features[:20], dtype=np.float32)

class StandardDQN(nn.Module):
    """80차원 입력을 위한 DQN"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 512):
        super().__init__()
        
        self.feature_layers = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU()
        )
        
        # 각 액션 차원별로 별도 출력
        self.position_head = nn.Linear(hidden_size // 2, 21)  # -2.0 ~ 2.0
        self.leverage_head = nn.Linear(hidden_size // 2, 20)  # 1 ~ 20
        self.holding_head = nn.Linear(hidden_size // 2, 48)   # 30분 ~ 1440분
    
    def forward(self, x):
        features = self.feature_layers(x)
        
        position_q = self.position_head(features)
        leverage_q = self.leverage_head(features)
        holding_q = self.holding_head(features)
        
        return position_q, leverage_q, holding_q

class StandardRLAgent:
    """80차원 Signal 기반 강화학습 에이전트"""
    
    def __init__(self, state_size: int = 80, learning_rate: float = 0.001, 
                 gamma: float = 0.99, epsilon: float = 1.0, epsilon_decay: float = 0.995):
        
        self.state_size = state_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        
        # 네트워크
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = StandardDQN(state_size, 3).to(self.device)
        self.target_network = StandardDQN(state_size, 3).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 경험 리플레이
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        
        # 성능 추적
        self.training_rewards = []
        self.losses = []
    
    def remember(self, state, action, reward, next_state, done):
        """경험 저장"""
        self.memory.append(Experience(state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray) -> np.ndarray:
        """액션 선택"""
        if np.random.random() <= self.epsilon:
            return np.array([
                np.random.uniform(-2.0, 2.0),
                np.random.uniform(1.0, 20.0), 
                np.random.uniform(30.0, 1440.0)
            ])
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            position_q, leverage_q, holding_q = self.q_network(state_tensor)
            
            position_idx = torch.argmax(position_q).item()
            leverage_idx = torch.argmax(leverage_q).item()
            holding_idx = torch.argmax(holding_q).item()
            
            # 인덱스를 실제 값으로 변환
            position = -2.0 + (position_idx * 0.2)
            leverage = 1.0 + leverage_idx
            holding = 30.0 + (holding_idx * 30.0)
            
            return np.array([position, leverage, holding])
    
    def replay(self):
        """경험 리플레이 학습"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        loss = self._compute_loss(batch)
        
        # 역전파
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        self.losses.append(loss.item())
        
        # 엡실론 감소
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def _compute_loss(self, batch) -> torch.Tensor:
        """손실 함수 계산"""
        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = [e.action for e in batch]
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        dones = [bool(e.done) for e in batch]
        
        # 현재 Q값들
        current_position_q, current_leverage_q, current_holding_q = self.q_network(states)
        
        # 타겟 Q값들
        with torch.no_grad():
            next_position_q, next_leverage_q, next_holding_q = self.target_network(next_states)
            
            target_position_q = current_position_q.clone()
            target_leverage_q = current_leverage_q.clone()
            target_holding_q = current_holding_q.clone()
            
            for i, (action, reward, done) in enumerate(zip(actions, rewards, dones)):
                if not done:
                    pos_idx = int(np.clip((action[0] + 2.0) / 0.2, 0, 20))
                    lev_idx = int(np.clip(action[1] - 1, 0, 19))
                    hold_idx = int(np.clip((action[2] - 30.0) / 30.0, 0, 47))
                    
                    target_position_q[i, pos_idx] = reward + self.gamma * torch.max(next_position_q[i])
                    target_leverage_q[i, lev_idx] = reward + self.gamma * torch.max(next_leverage_q[i])
                    target_holding_q[i, hold_idx] = reward + self.gamma * torch.max(next_holding_q[i])
                else:
                    pos_idx = int(np.clip((action[0] + 2.0) / 0.2, 0, 20))
                    lev_idx = int(np.clip(action[1] - 1, 0, 19))
                    hold_idx = int(np.clip((action[2] - 30.0) / 30.0, 0, 47))
                    
                    target_position_q[i, pos_idx] = reward
                    target_leverage_q[i, lev_idx] = reward
                    target_holding_q[i, hold_idx] = reward
        
        # 손실 계산
        pos_loss = F.mse_loss(current_position_q, target_position_q)
        lev_loss = F.mse_loss(current_leverage_q, target_leverage_q)
        hold_loss = F.mse_loss(current_holding_q, target_holding_q)
        
        return pos_loss + lev_loss + hold_loss
    
    def update_target_network(self):
        """타겟 네트워크 업데이트"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_model(self, filepath: str):
        """모델 저장"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_rewards': self.training_rewards,
            'losses': self.losses,
            'state_size': self.state_size
        }, filepath)
    
    def load_model(self, filepath: str):
        """모델 로드"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.training_rewards = checkpoint['training_rewards']
        self.losses = checkpoint['losses']

def train_enhanced_agent(price_data: pd.DataFrame, signal_data: List[Dict], 
                        episodes: int = 1000, save_interval: int = 100):
    """80차원 Signal 기반 에이전트 훈련"""
    
    # 환경과 에이전트 초기화
    env = EnhancedTradingEnvironment(price_data, signal_data)
    agent = StandardRLAgent(env.observation_space.shape[0])
    
    episode_rewards = []
    best_reward = -float('inf')
    
    print(f"80차원 Signal 기반 훈련 시작 (환경: {env.observation_space.shape[0]}차원)")
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            # 액션 선택 및 실행
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            
            # 경험 저장
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # 학습
            if len(agent.memory) > agent.batch_size:
                agent.replay()
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        agent.training_rewards.append(total_reward)
        
        # 타겟 네트워크 업데이트 (매 10 에피소드)
        if episode % 10 == 0:
            agent.update_target_network()
        
        # 진행 상황 출력
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
            print(f"Episode {episode}, Average Reward: {avg_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}, Balance: ${info['balance']:.2f}, "
                  f"Win Rate: {info['win_rate']:.3f}")
        
        # 모델 저장
        if episode % save_interval == 0 and total_reward > best_reward:
            best_reward = total_reward
            agent.save_model(f'best_enhanced_rl_model_80d_ep{episode}.pth')
            print(f"New best 80d model saved at episode {episode} with reward {best_reward:.2f}")
    
    return agent, episode_rewards

def evaluate_enhanced_agent(agent: StandardRLAgent, price_data: pd.DataFrame, 
                           signal_data: List[Dict], episodes: int = 10):
    """80차원 에이전트 성능 평가"""
    env = EnhancedTradingEnvironment(price_data, signal_data)
    agent.epsilon = 0  # 탐험 비활성화
    
    results = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        trades = []
        
        while True:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            
            # 거래 기록
            if env.total_trades > len(trades):
                trades.append({
                    'step': env.current_step,
                    'price': info['current_price'],
                    'action': 'CLOSE',
                    'balance': info['balance'],
                    'pnl': info['balance'] - env.initial_balance
                })
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        results.append({
            'episode': episode,
            'total_reward': total_reward,
            'final_balance': info['balance'],
            'total_return': (info['balance'] - env.initial_balance) / env.initial_balance,
            'total_trades': info['total_trades'],
            'win_rate': info['win_rate'],
            'max_drawdown': env.max_drawdown,
            'trades': trades
        })
    
    return results

def load_signal_data_with_conversion(agent_folder: str = "agent") -> Optional[List[Dict]]:
    """Signal 데이터 로드 및 변환"""
    try:
        parquet_files = list(Path(agent_folder).glob("*.parquet"))
        
        if parquet_files:
            print(f"Signal 데이터 로드 중: {parquet_files[0].name}")
            signal_df = pd.read_parquet(parquet_files[0])
            print(f"Signal 데이터 로드: {len(signal_df):,}개 레코드")
            
            # DataFrame을 Dict 리스트로 변환
            signal_data = []
            for idx, row in signal_df.iterrows():
                signal_dict = row.to_dict()
                signal_data.append(signal_dict)
            
            print(f"Signal 데이터 변환 완료: {len(signal_data):,}개")
            return signal_data
        else:
            print("Parquet 파일이 없습니다.")
            return None
            
    except Exception as e:
        print(f"Signal 데이터 로드 실패: {e}")
        return None

def main_example():
    """80차원 Signal 기반 사용 예시"""
    
    print("80차원 Signal 기반 강화학습 트레이딩 AI")
    print("="*60)
    
    # 1. 데이터 로드
    try:
        df_3m = pd.read_csv('data/ETHUSDC_3m_historical_data.csv')
        df_3m['timestamp'] = pd.to_datetime(df_3m['timestamp'])
        df_3m = df_3m.set_index('timestamp')
        price_data = df_3m.reset_index()
        
        print(f"가격 데이터: {len(price_data)}개 캔들")
        
        # Signal 데이터 로드
        signal_data = load_signal_data_with_conversion()
        
        if signal_data is None:
            print("Signal 데이터를 찾을 수 없어 기본 데이터를 생성합니다.")
            return
        
        # 데이터 길이 맞추기
        min_length = min(len(price_data), len(signal_data))
        price_data = price_data.iloc[:min_length].reset_index(drop=True)
        signal_data = signal_data[:min_length]
        
        print(f"최종 데이터: {min_length:,}개")
        
        # 2. 환경 테스트
        print("\n80차원 환경 테스트...")
        env = EnhancedTradingEnvironment(price_data, signal_data)
        state = env.reset()
        print(f"상태 벡터 차원: {state.shape}")
        print(f"상태 벡터 샘플: {state[:10]}")
        
        # 3. 에이전트 테스트
        agent = StandardRLAgent(80)
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        print(f"액션 샘플: {action}")
        print(f"보상: {reward:.3f}")
        
        print("\n80차원 Signal 기반 시스템 준비 완료!")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main_example()