"""
강화학습 에이전트 코어 모듈
- 기존 신호 데이터를 활용한 실시간 거래 환경
- 훈련된 모델과 호환되는 상태 변환
"""

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

class StandardTradingEnvironment(gym.Env):
    """
    표준화된 암호화폐 거래 환경
    - parquet 형태와 중첩 딕셔너리 형태 모두 지원
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
        
        # 상태 스페이스: 가격(20) + 신호(30) + 포트폴리오(10) = 60
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(60,), 
            dtype=np.float32
        )
        
        self.reset()
    
    def _get_state_size(self) -> int:
        return 60
    
    def _normalize_signal_data(self, signals: Dict) -> Dict:
        """신호 데이터를 표준 형태로 변환"""
        # 이미 중첩 딕셔너리 형태인 경우
        if 'decisions' in signals:
            return signals['decisions']
        
        # parquet 평면화된 형태인 경우
        decisions = {}
        
        for category in ['SHORT_TERM', 'MEDIUM_TERM', 'LONG_TERM']:
            prefix = f"{category.lower()}_"
            
            decisions[category] = {
                'action': signals.get(f'{prefix}action', 'HOLD'),
                'net_score': float(signals.get(f'{prefix}net_score', 0.0)),
                'leverage': int(signals.get(f'{prefix}leverage', 1)),
                'max_holding_minutes': int(signals.get(f'{prefix}max_holding_minutes', 60)),
                'meta': {
                    'synergy_meta': {
                        'confidence': signals.get(f'{prefix}confidence', 'LOW'),
                        'buy_score': float(signals.get(f'{prefix}buy_score', 0.0)),
                        'sell_score': float(signals.get(f'{prefix}sell_score', 0.0)),
                        'conflicts_detected': []
                    }
                }
            }
        
        return decisions
    
    def _extract_signal_features(self, signals: Dict) -> np.ndarray:
        """신호 데이터에서 특성 추출"""
        features = []
        
        # 신호 데이터 정규화
        decisions = self._normalize_signal_data(signals)
        
        # 각 카테고리별 특성 추출
        for category_prefix in ['short_term_', 'medium_term_', 'long_term_']:
            category_name = category_prefix.rstrip('_').upper()
            
            # parquet 형태 처리
            if f'{category_prefix}action' in signals:
                action_key = f'{category_prefix}action'
                net_score_key = f'{category_prefix}net_score'
                leverage_key = f'{category_prefix}leverage'
                max_holding_key = f'{category_prefix}max_holding_minutes'
                buy_score_key = f'{category_prefix}buy_score'
                sell_score_key = f'{category_prefix}sell_score'
                confidence_key = f'{category_prefix}confidence'
                signals_used_key = f'{category_prefix}signals_used'
                
                action = signals.get(action_key, 'HOLD')
                action_val = 1.0 if action == 'LONG' else (-1.0 if action == 'SHORT' else 0.0)
                
                confidence = signals.get(confidence_key, 'LOW')
                confidence_val = 1.0 if confidence == 'HIGH' else (0.5 if confidence == 'MEDIUM' else 0.0)
                
                features.extend([
                    action_val,
                    float(signals.get(net_score_key, 0.0)),
                    float(signals.get(leverage_key, 1.0)) / 20.0,
                    float(signals.get(max_holding_key, 60.0)) / 1440.0,
                    float(signals.get(buy_score_key, 0.0)),
                    float(signals.get(sell_score_key, 0.0)),
                    confidence_val,
                    float(signals.get(signals_used_key, 0.0)) / 5.0,
                ])
            
            # 중첩 딕셔너리 형태 처리
            elif category_name in decisions:
                decision = decisions[category_name]
                
                action = decision.get('action', 'HOLD')
                action_strength = 1.0 if action == 'LONG' else (-1.0 if action == 'SHORT' else 0.0)
                
                meta = decision.get('meta', {}).get('synergy_meta', {})
                confidence = meta.get('confidence', 'LOW')
                confidence_score = 1.0 if confidence == 'HIGH' else (0.5 if confidence == 'MEDIUM' else 0.0)
                
                features.extend([
                    action_strength,
                    float(decision.get('net_score', 0.0)),
                    min(float(decision.get('leverage', 1)) / 10.0, 2.0),
                    min(float(decision.get('max_holding_minutes', 60)) / 1440.0, 1.0),
                    confidence_score,
                    float(meta.get('buy_score', 0.0)),
                    float(meta.get('sell_score', 0.0)),
                    len(meta.get('conflicts_detected', [])) / 5.0
                ])
            else:
                features.extend([0.0] * 8)
        
        # 갈등 정보 처리
        if 'conflicts' in signals:
            conflicts = signals['conflicts']
            features.extend([
                1.0 if conflicts.get('has_conflicts', False) else 0.0,
                len(conflicts.get('long_categories', [])) / 3.0,
                len(conflicts.get('short_categories', [])) / 3.0,
                float(signals.get('meta', {}).get('active_positions', 0)) / 3.0,
            ])
        else:
            # parquet에서 갈등 정보 추정
            long_actions = sum(1 for prefix in ['short_term_', 'medium_term_', 'long_term_'] 
                              if signals.get(f'{prefix}action') == 'LONG')
            short_actions = sum(1 for prefix in ['short_term_', 'medium_term_', 'long_term_'] 
                               if signals.get(f'{prefix}action') == 'SHORT')
            
            features.extend([
                1.0 if long_actions > 0 and short_actions > 0 else 0.0,
                long_actions / 3.0,
                short_actions / 3.0,
                0.0,
            ])
        
        # 개별 전략 점수 (나머지 6개)
        strategy_scores = []
        strategy_names = ['vwap_pinball', 'liquidity_grab', 'zscore_mean_reversion']
        
        for strategy_name in strategy_names:
            score_key = f'short_term_raw_{strategy_name}_score'
            action_key = f'short_term_raw_{strategy_name}_action'
            
            score = float(signals.get(score_key, 0.0))
            action = signals.get(action_key, 'HOLD')
            action_val = 1.0 if action == 'BUY' else (-1.0 if action == 'SELL' else 0.0)
            strategy_scores.extend([score, action_val])
        
        features.extend(strategy_scores)
        
        return np.array(features[:30], dtype=np.float32)
    
    def _extract_price_features(self, idx: int) -> np.ndarray:
        """가격 데이터에서 특성 추출"""
        if idx < 20:
            return np.zeros(20, dtype=np.float32)
        
        recent_data = self.price_data.iloc[idx-19:idx+1]
        
        features = []
        close = recent_data['close']
        high = recent_data['high']
        low = recent_data['low']
        volume = recent_data['volume']
        
        # 수익률 특성
        returns = close.pct_change().fillna(0)
        features.extend([
            returns.mean(),
            returns.std(),
            returns.iloc[-1],
            returns.iloc[-5:].mean(),
        ])
        
        # 기술적 지표
        features.extend([
            self._calculate_rsi(close),
            self._calculate_bb_position(close),
        ])
        
        # 이동평균 비율
        for window in [5, 10, 20]:
            ma = close.rolling(window=window, min_periods=1).mean()
            ma_ratio = (close.iloc[-1] / ma.iloc[-1] - 1) if ma.iloc[-1] > 0 else 0.0
            features.append(self._safe_float(ma_ratio))
        
        # 거래량 및 변동성
        volume_mean = volume.mean()
        volume_ratio = volume.iloc[-1] / volume_mean - 1 if volume_mean > 0 else 0.0
        volatility = (high.iloc[-1] - low.iloc[-1]) / close.iloc[-1] if close.iloc[-1] > 0 else 0.0
        
        features.extend([
            self._safe_float(volume_ratio),
            self._safe_float(volatility),
        ])
        
        # 가격 위치
        max_high = high.max()
        min_low = low.min()
        price_range = max_high - min_low
        price_position = (close.iloc[-1] - min_low) / price_range if price_range > 0 else 0.5
        features.append(self._safe_float(price_position))
        
        # 추가 기술적 지표들
        features.extend([
            self._safe_float((close.iloc[-1] - close.iloc[-5]) / close.iloc[-5]) if len(close) >= 5 else 0.0,
            self._safe_float((close.iloc[-1] - close.iloc[-10]) / close.iloc[-10]) if len(close) >= 10 else 0.0,
            self._safe_float((high.iloc[-5:].max() - close.iloc[-1]) / close.iloc[-1]) if len(high) >= 5 else 0.0,
            self._safe_float((close.iloc[-1] - low.iloc[-5:].min()) / close.iloc[-1]) if len(low) >= 5 else 0.0,
        ])
        
        return np.array(features[:20], dtype=np.float32)
    
    def _calculate_rsi(self, close_prices: pd.Series) -> float:
        """RSI 계산"""
        if len(close_prices) < 14:
            return 0.5
        
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        return self._safe_float(rsi.iloc[-1] / 100.0, default=0.5)
    
    def _calculate_bb_position(self, close_prices: pd.Series) -> float:
        """볼린저 밴드 위치"""
        if len(close_prices) < 20:
            return 0.5
        
        sma = close_prices.rolling(window=20, min_periods=1).mean()
        std = close_prices.rolling(window=20, min_periods=1).std()
        bb_upper = sma + (std * 2)
        bb_lower = sma - (std * 2)
        bb_width = bb_upper.iloc[-1] - bb_lower.iloc[-1]
        
        if bb_width > 0:
            bb_position = (close_prices.iloc[-1] - bb_lower.iloc[-1]) / bb_width
            return self._safe_float(bb_position, default=0.5)
        else:
            return 0.5
    
    def _safe_float(self, value, default=0.0):
        """안전한 float 변환"""
        if pd.isna(value) or np.isinf(value):
            return default
        return float(value)
    
    def _get_portfolio_state(self) -> np.ndarray:
        """포트폴리오 상태 정보"""
        features = [
            self.current_position,
            self.current_leverage / 20.0,
            (self.balance - self.initial_balance) / self.initial_balance,
            self.unrealized_pnl / self.initial_balance,
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
        """현재 상태 관찰값 반환"""
        if self.current_step >= len(self.signal_data):
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        # 가격 특성
        price_features = self._extract_price_features(self.current_step)
        
        # 신호 특성
        current_signal = self.signal_data[self.current_step]
        signal_features = self._extract_signal_features(current_signal)
        
        # 포트폴리오 상태
        portfolio_features = self._get_portfolio_state()
        
        # 모든 특성 결합
        observation = np.concatenate([price_features, signal_features, portfolio_features])
        
        return observation.astype(np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """한 스텝 실행"""
        if self.current_step >= len(self.signal_data) - 1:
            return self._get_observation(), 0.0, True, {}
        
        # 액션 해석
        position_change = np.clip(action[0], -2.0, 2.0)
        leverage = np.clip(action[1], 1.0, 20.0)
        target_holding_minutes = np.clip(action[2], 1.0, 1440.0)
        
        # 현재 가격
        current_price = self.price_data.iloc[self.current_step]['close']
        next_price = self.price_data.iloc[self.current_step + 1]['close']
        
        # 보상 계산
        reward = self._calculate_reward(position_change, leverage, current_price, next_price)
        
        # 포지션 업데이트
        self._update_position(position_change, leverage, current_price, target_holding_minutes)
        
        # 다음 스텝으로
        self.current_step += 1
        self.holding_time += 3  # 3분 단위
        
        # 포지션 홀딩 시간 체크
        if self.in_position and self.holding_time >= target_holding_minutes:
            self._close_position(next_price)
        
        # 다음 상태
        next_state = self._get_observation()
        
        # 종료 조건
        done = (self.current_step >= len(self.signal_data) - 1 or 
                self.balance <= self.initial_balance * 0.1)
        
        info = {
            'balance': self.balance,
            'position': self.current_position,
            'unrealized_pnl': self.unrealized_pnl,
            'total_trades': self.total_trades,
            'win_rate': self.winning_trades / max(self.total_trades, 1)
        }
        
        return next_state, reward, done, info
    
    def _calculate_reward(self, position_change: float, leverage: float, 
                         current_price: float, next_price: float) -> float:
        """보상 함수"""
        reward = 0.0
        
        # 1. PnL 기반 보상
        if abs(self.current_position) > 0.01:
            price_change = (next_price - current_price) / current_price
            position_pnl = self.current_position * price_change * self.current_leverage
            reward += position_pnl * 100
        
        # 2. 신호 방향성과 일치도 보상
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
        
        # 6. 홀딩 시간 최적화 보상
        if self.in_position:
            holding_reward = self._calculate_holding_reward()
            reward += holding_reward
        
        return reward
    
    def _calculate_signal_alignment(self, position_change: float, signals: Dict) -> float:
        """신호와 액션 일치도 계산"""
        alignment_score = 0.0
        
        # 신호 데이터 정규화
        decisions = self._normalize_signal_data(signals)
        
        for category in ['SHORT_TERM', 'MEDIUM_TERM', 'LONG_TERM']:
            if category in decisions:
                decision = decisions[category]
                action = decision.get('action', 'HOLD')
                net_score = float(decision.get('net_score', 0.0))
                
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

class StandardDQN(nn.Module):
    """표준 Deep Q-Network"""
    
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
    """표준 암호화폐 강화학습 에이전트"""
    
    def __init__(self, state_size: int, learning_rate: float = 0.001, 
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
        
        states = torch.FloatTensor(np.array([e.state for e in batch])).to(self.device)
        actions = [e.action for e in batch]
        rewards = torch.FloatTensor(np.array([e.reward for e in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([e.next_state for e in batch])).to(self.device)
        dones = torch.tensor(np.array([bool(e.done) for e in batch]), dtype=torch.bool).to(self.device)
        
        # 현재 Q값
        current_position_q, current_leverage_q, current_holding_q = self.q_network(states)
        
        # 타겟 Q값
        with torch.no_grad():
            next_position_q, next_leverage_q, next_holding_q = self.target_network(next_states)
            
            target_position_q = current_position_q.clone()
            target_leverage_q = current_leverage_q.clone()
            target_holding_q = current_holding_q.clone()
            
            for i, (action, reward, done) in enumerate(zip(actions, rewards, dones)):
                if not done:
                    pos_idx = int((action[0] + 2.0) / 0.2)
                    lev_idx = int(action[1] - 1)
                    hold_idx = int((action[2] - 30.0) / 30.0)
                    
                    pos_idx = np.clip(pos_idx, 0, 20)
                    lev_idx = np.clip(lev_idx, 0, 19)
                    hold_idx = np.clip(hold_idx, 0, 47)
                    
                    target_position_q[i, pos_idx] = reward + self.gamma * torch.max(next_position_q[i])
                    target_leverage_q[i, lev_idx] = reward + self.gamma * torch.max(next_leverage_q[i])
                    target_holding_q[i, hold_idx] = reward + self.gamma * torch.max(next_holding_q[i])
                else:
                    pos_idx = int((action[0] + 2.0) / 0.2)
                    lev_idx = int(action[1] - 1)
                    hold_idx = int((action[2] - 30.0) / 30.0)
                    
                    pos_idx = np.clip(pos_idx, 0, 20)
                    lev_idx = np.clip(lev_idx, 0, 19)
                    hold_idx = np.clip(hold_idx, 0, 47)
                    
                    target_position_q[i, pos_idx] = reward
                    target_leverage_q[i, lev_idx] = reward
                    target_holding_q[i, hold_idx] = reward
        
        # 손실 계산
        pos_loss = F.mse_loss(current_position_q, target_position_q)
        lev_loss = F.mse_loss(current_leverage_q, target_leverage_q)
        hold_loss = F.mse_loss(current_holding_q, target_holding_q)
        
        total_loss = pos_loss + lev_loss + hold_loss
        
        # 역전파
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        self.losses.append(total_loss.item())
        
        # 엡실론 감소
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
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
            'losses': self.losses
        }, filepath)
    
    def load_model(self, filepath: str):
        """모델 로드"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.training_rewards = checkpoint['training_rewards']
        self.losses = checkpoint['losses']
    
    

def train_standard_agent(price_data: pd.DataFrame, signal_data: List[Dict], 
                        episodes: int = 1000, save_interval: int = 100):
    """표준 강화학습 에이전트 훈련"""
    
    # 환경과 에이전트 초기화
    env = StandardTradingEnvironment(price_data, signal_data)
    agent = StandardRLAgent(env.observation_space.shape[0])
    
    episode_rewards = []
    best_reward = -float('inf')
    
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
                  f"Epsilon: {agent.epsilon:.3f}, Balance: {info['balance']:.2f}, "
                  f"Win Rate: {info['win_rate']:.3f}")
        
        # 모델 저장
        if episode % save_interval == 0 and total_reward > best_reward:
            best_reward = total_reward
            agent.save_model(f'best_standard_rl_model_ep{episode}.pth')
            print(f"New best model saved at episode {episode} with reward {best_reward:.2f}")
    
    return agent, episode_rewards

def evaluate_standard_agent(agent: StandardRLAgent, price_data: pd.DataFrame, 
                           signal_data: List[Dict], episodes: int = 10):
    """표준 에이전트 성능 평가"""
    env = StandardTradingEnvironment(price_data, signal_data)
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
                    'price': env.price_data.iloc[env.current_step]['close'],
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

class BacktestResults:
    """백테스트 결과 분석 클래스"""
    
    @staticmethod
    def calculate_metrics(results: List[Dict]) -> Dict:
        """성능 지표 계산"""
        returns = [r['total_return'] for r in results]
        
        metrics = {
            'avg_return': np.mean(returns),
            'std_return': np.std(returns),
            'sharpe_ratio': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
            'max_return': np.max(returns),
            'min_return': np.min(returns),
            'win_rate': np.mean([r['win_rate'] for r in results]),
            'avg_trades': np.mean([r['total_trades'] for r in results]),
            'max_drawdown': np.mean([r['max_drawdown'] for r in results]),
            'profit_episodes': sum(1 for r in returns if r > 0),
            'loss_episodes': sum(1 for r in returns if r < 0),
        }
        
        return metrics
    
    @staticmethod
    def print_report(results: List[Dict], metrics: Dict):
        """성능 리포트 출력"""
        print("\n" + "="*60)
        print("강화학습 트레이딩 AI 성능 리포트")
        print("="*60)
        
        print(f"평균 수익률: {metrics['avg_return']:.2%}")
        print(f"수익률 표준편차: {metrics['std_return']:.2%}")
        print(f"샤프 비율: {metrics['sharpe_ratio']:.3f}")
        print(f"최대 수익률: {metrics['max_return']:.2%}")
        print(f"최소 수익률: {metrics['min_return']:.2%}")
        print(f"평균 승률: {metrics['win_rate']:.2%}")
        print(f"평균 거래 횟수: {metrics['avg_trades']:.1f}")
        print(f"최대 낙폭: {metrics['max_drawdown']:.2%}")
        print(f"수익 에피소드: {metrics['profit_episodes']}개")
        print(f"손실 에피소드: {metrics['loss_episodes']}개")

# 데이터 로딩 함수들
def load_ethusdc_data():
    """ETHUSDC CSV 데이터 로드"""
    try:
        df_3m = pd.read_csv('data/ETHUSDC_3m_historical_data.csv')
        df_3m['timestamp'] = pd.to_datetime(df_3m['timestamp'])
        df_3m = df_3m.set_index('timestamp')
        
        df_15m = pd.read_csv('data/ETHUSDC_15m_historical_data.csv')
        df_15m['timestamp'] = pd.to_datetime(df_15m['timestamp'])
        df_15m = df_15m.set_index('timestamp')
        
        df_1h = pd.read_csv('data/ETHUSDC_1h_historical_data.csv')
        df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'])
        df_1h = df_1h.set_index('timestamp')
        
        print(f"ETHUSDC 3분봉: {len(df_3m)}개")
        print(f"ETHUSDC 15분봉: {len(df_15m)}개")
        print(f"ETHUSDC 1시간봉: {len(df_1h)}개")
        
        return df_3m, df_15m, df_1h

    except FileNotFoundError as e:
        print(f"데이터 파일을 찾을 수 없습니다: {e}")
        return None, None, None
    except Exception as e:
        print(f"데이터 로드 중 오류: {e}")
        return None, None, None

def generate_signal_data_with_indicators(price_data: pd.DataFrame, price_data_15m: pd.DataFrame, 
                                        price_data_1h: pd.DataFrame, max_periods: int = 1000):
    """실제 지표로 신호 데이터 생성"""
    try:
        from data.strategy_executor import StrategyExecutor
        from engines.trade_decision_engine import TradeDecisionEngine
        from indicators.global_indicators import get_global_indicator_manager
        from utils.time_manager import get_time_manager
        
        strategy_executor = StrategyExecutor()
        decision_engine = TradeDecisionEngine()
        global_manager = get_global_indicator_manager()
        time_manager = get_time_manager()
        
        signal_data = []
        
        print("실제 지표로 신호 생성 중...")
        
        start_idx = 500
        
        for i in range(start_idx, min(len(price_data), start_idx + max_periods)):
            try:
                series_3m = price_data.iloc[i][['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'timestamp']]
                
                global_manager.update_all_indicators(series_3m)
                strategy_executor.execute_all_strategies()
                signals = strategy_executor.get_signals()
                decision = decision_engine.decide_trade_realtime(signals)
                
                signal_data.append(decision)
                
                if (i - start_idx) % 100 == 0:
                    print(f"   진행률: {i - start_idx + 1}/{max_periods}")
                    
            except Exception as e:
                print(f"신호 생성 오류 (인덱스 {i}): {e}")
                # 기본 신호 생성
                default_signal = {
                    'decisions': {
                        'SHORT_TERM': {'action': 'HOLD', 'net_score': 0.0, 'leverage': 1, 'max_holding_minutes': 60, 'meta': {'synergy_meta': {'confidence': 'LOW', 'buy_score': 0.0, 'sell_score': 0.0}}},
                        'MEDIUM_TERM': {'action': 'HOLD', 'net_score': 0.0, 'leverage': 1, 'max_holding_minutes': 240, 'meta': {'synergy_meta': {'confidence': 'LOW', 'buy_score': 0.0, 'sell_score': 0.0}}},
                        'LONG_TERM': {'action': 'HOLD', 'net_score': 0.0, 'leverage': 1, 'max_holding_minutes': 1440, 'meta': {'synergy_meta': {'confidence': 'LOW', 'buy_score': 0.0, 'sell_score': 0.0}}}
                    },
                    'conflicts': {'has_conflicts': False, 'long_categories': [], 'short_categories': []},
                    'meta': {'active_positions': 0}
                }
                signal_data.append(default_signal)
        
        print(f"신호 데이터 생성 완료: {len(signal_data)}개")
        return signal_data
        
    except ImportError:
        print("전략 모듈을 찾을 수 없습니다. 기본 신호를 생성합니다.")
        return None

# def main_example():
#     """사용 예시"""
    
#     print("표준 강화학습 트레이딩 AI 예시")
#     print("="*50)
    
#     # 1. 데이터 로드
#     price_data_3m, price_data_15m, price_data_1h = load_ethusdc_data()
    
#     if price_data_3m is None:
#         print("데이터 로드 실패")
#         return
    
#     # 2. 가격 데이터 전처리
#     price_data = price_data_3m.reset_index()
#     price_data = price_data.rename(columns={'timestamp': 'timestamp'})
    
#     required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
#     price_data = price_data[required_columns]
    
#     print(f"가격 데이터: {len(price_data)}개 캔들")
    
#     # 3. 신호 데이터 생성
#     signal_data = generate_signal_data_with_indicators(
#         price_data, price_data_15m, price_data_1h, max_periods=1000
#     )
    
#     if not signal_data:
#         print("기본 신호 데이터 생성")
#         signal_data = []
#         for i in range(min(1000, len(price_data))):
#             signal_data.append({
#                 'decisions': {
#                     'SHORT_TERM': {'action': 'HOLD', 'net_score': 0.0, 'leverage': 1, 'max_holding_minutes': 60, 'meta': {'synergy_meta': {'confidence': 'LOW'}}},
#                     'MEDIUM_TERM': {'action': 'HOLD', 'net_score': 0.0, 'leverage': 1, 'max_holding_minutes': 240, 'meta': {'synergy_meta': {'confidence': 'LOW'}}},
#                     'LONG_TERM': {'action': 'HOLD', 'net_score': 0.0, 'leverage': 1, 'max_holding_minutes': 1440, 'meta': {'synergy_meta': {'confidence': 'LOW'}}}
#                 },
#                 'conflicts': {'has_conflicts': False},
#                 'meta': {'active_positions': 0}
#             })
    
#     print("에이전트 훈련 시작")
    
#     # 4. 에이전트 훈련
#     agent, rewards = train_standard_agent(price_data, signal_data, episodes=200)
    
#     print("성능 평가 중")
    
#     # 5. 성능 평가
#     eval_results = evaluate_standard_agent(agent, price_data, signal_data, episodes=10)
    
#     # 6. 결과 분석
#     metrics = BacktestResults.calculate_metrics(eval_results)
#     BacktestResults.print_report(eval_results, metrics)
    
#     # 7. 모델 저장
#     agent.save_model('standard_crypto_rl_model.pth')
#     print("모델 저장 완료: standard_crypto_rl_model.pth")
    
#     return agent, eval_results, metrics