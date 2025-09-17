# 완전 독립형 강화학습 트레이딩 시스템
# 한 파일 실행으로 데이터 로딩 -> 훈련 -> 평가까지 모든 기능 포함

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
import os
import warnings
from pathlib import Path

# PyTorch 2.6 호환성 설정
def setup_safe_torch_loading():
    """PyTorch 2.6에서 안전한 모델 로딩을 위한 설정"""
    try:
        torch.serialization.add_safe_globals([
            np.core.multiarray.scalar,
            np.ndarray,
            np.dtype,
            np.float32,
            np.float64,
            np.int32,
            np.int64,
        ])
        print("✅ PyTorch 2.6 호환 설정 완료")
    except AttributeError:
        print("ℹ️  PyTorch 이전 버전 감지됨")

setup_safe_torch_loading()

# 경험 튜플 정의
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

# =================================================================
# 1. 개선된 보상 함수
# =================================================================

class ImprovedRewardFunction:
    """승률과 수익성을 동시에 개선하는 보상 함수"""
    
    def __init__(self):
        self.recent_trades = deque(maxlen=50)
        self.baseline_return = 0.0
        
    def calculate_reward(self, current_price, entry_price, position, action, 
                        holding_time, volatility=0.02, volume_ratio=1.0, trade_pnl=None):
        """개선된 보상 계산"""
        reward = 0.0
        
        # 1. 포지션 보유 중 실시간 평가
        if abs(position) > 0.01:
            unrealized_pnl = self._calculate_unrealized_pnl(current_price, entry_price, position)
            
            if unrealized_pnl > 0:
                reward += min(unrealized_pnl * 10, 1.0)  # 수익시 보상
            else:
                reward += max(unrealized_pnl * 15, -2.0)  # 손실시 더 큰 패널티
            
            # 홀딩 시간 최적화
            if holding_time > 30:
                reward -= 0.1 * (holding_time - 30) / 30
        
        # 2. 거래 완료시 승률 중심 평가
        if trade_pnl is not None:  # 거래 완료
            self.recent_trades.append(1 if trade_pnl > 0 else 0)
            current_win_rate = np.mean(self.recent_trades) if self.recent_trades else 0.5
            
            if trade_pnl > 0:  # 수익 거래
                reward += 5.0  # 승률 향상을 위한 큰 보상
                if current_win_rate > 0.6:
                    reward += 2.0  # 연속 승률 보너스
            else:  # 손실 거래
                reward -= 3.0  # 상대적으로 작은 패널티
        
        # 3. 시장 컨텍스트 반영
        if volatility > 0.05:  # 고변동성
            if abs(position) < 0.3:
                reward += 0.5
            else:
                reward -= 1.0
        
        return reward
    
    def _calculate_unrealized_pnl(self, current_price, entry_price, position):
        """미실현 손익 계산"""
        if entry_price <= 0:
            return 0.0
        if position > 0:  # Long
            return (current_price - entry_price) / entry_price
        else:  # Short  
            return (entry_price - current_price) / entry_price

# =================================================================
# 2. 개선된 DQN 네트워크
# =================================================================

class ImprovedDQNNetwork(nn.Module):
    """개선된 Dueling DQN 아키텍처"""
    
    def __init__(self, state_size, action_size, hidden_size=256):
        super().__init__()
        
        # 입력 정규화
        self.input_norm = nn.LayerNorm(state_size)
        
        # 특성 추출 레이어
        self.feature_extraction = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_size)
        )
        
        # Dueling 구조
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # 각 액션 차원별 advantage stream
        self.position_advantage = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 21)  # -2.0 ~ 2.0 (0.2 간격)
        )
        
        self.leverage_advantage = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), 
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 20)  # 1 ~ 20
        )
        
        self.holding_advantage = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 48)  # 30분 ~ 1440분
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 입력 정규화
        x = self.input_norm(x)
        
        # 특성 추출
        features = self.feature_extraction(x)
        
        # Dueling 분리
        value = self.value_stream(features)
        
        position_adv = self.position_advantage(features)
        leverage_adv = self.leverage_advantage(features) 
        holding_adv = self.holding_advantage(features)
        
        # Dueling 결합
        position_q = value + position_adv - position_adv.mean(dim=1, keepdim=True)
        leverage_q = value + leverage_adv - leverage_adv.mean(dim=1, keepdim=True)
        holding_q = value + holding_adv - holding_adv.mean(dim=1, keepdim=True)
        
        return position_q, leverage_q, holding_q

# =================================================================
# 3. 개선된 거래 환경
# =================================================================

class ImprovedTradingEnvironment(gym.Env):
    """개선된 거래 환경"""
    
    def __init__(self, price_data: pd.DataFrame, signal_data: List[Dict], 
                 initial_balance: float = 10000.0, max_position: float = 1.0):
        super().__init__()
        
        self.price_data = price_data
        self.signal_data = signal_data
        self.initial_balance = initial_balance
        self.max_position = max_position
        
        # 개선된 보상 함수
        self.reward_function = ImprovedRewardFunction()
        
        # 액션 스페이스 (연속형)
        self.action_space = spaces.Box(
            low=np.array([-2.0, 1.0, 0.0]), 
            high=np.array([2.0, 20.0, 1440.0]), 
            dtype=np.float32
        )
        
        # 상태 스페이스
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(60,),  # 20(가격) + 30(신호) + 10(포트폴리오)
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self):
        """환경 초기화"""
        self.current_step = 20  # 충분한 히스토리 확보
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
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """상태 관찰값 반환"""
        if self.current_step >= min(len(self.price_data), len(self.signal_data)):
            return np.zeros(60, dtype=np.float32)
        
        # 1. 가격 특성 (20개)
        price_features = self._extract_price_features(self.current_step)
        
        # 2. 신호 특성 (30개)
        signal_features = self._extract_signal_features(self.signal_data[self.current_step])
        
        # 3. 포트폴리오 상태 (10개)
        portfolio_features = self._get_portfolio_state()
        
        # 모든 특성 결합
        observation = np.concatenate([
            price_features,      # 20개
            signal_features,     # 30개  
            portfolio_features   # 10개
        ])
        
        return observation.astype(np.float32)
    
    def _extract_price_features(self, idx: int) -> np.ndarray:
        """가격 특성 추출"""
        if idx < 20:
            return np.zeros(20, dtype=np.float32)
        
        recent_data = self.price_data.iloc[max(0, idx-19):idx+1]
        features = []
        
        if len(recent_data) == 0:
            return np.zeros(20, dtype=np.float32)
        
        close = recent_data['close']
        high = recent_data['high']
        low = recent_data['low']
        volume = recent_data['volume']
        
        # 수익률 특성 (4개)
        returns = close.pct_change().fillna(0)
        features.extend([
            returns.mean(),
            returns.std(),
            returns.iloc[-1] if len(returns) > 0 else 0.0,
            returns.tail(5).mean() if len(returns) >= 5 else 0.0,
        ])
        
        # RSI (1개)
        if len(close) >= 14:
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / (loss + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            features.append(rsi.iloc[-1] / 100.0 if not pd.isna(rsi.iloc[-1]) else 0.5)
        else:
            features.append(0.5)
        
        # 볼린저 밴드 위치 (1개)
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
        
        # 이동평균 비율 (3개)
        for window in [5, 10, 20]:
            if len(close) >= window:
                ma = close.rolling(window=window, min_periods=1).mean()
                ma_ratio = (close.iloc[-1] / ma.iloc[-1] - 1) if ma.iloc[-1] > 0 else 0.0
                features.append(ma_ratio)
            else:
                features.append(0.0)
        
        # 거래량 및 변동성 (2개)
        if len(volume) > 1:
            vol_ratio = (volume.iloc[-1] / volume.mean() - 1) if volume.mean() > 0 else 0.0
            features.append(vol_ratio)
        else:
            features.append(0.0)
            
        if len(high) > 0 and len(low) > 0 and len(close) > 0:
            price_volatility = (high.iloc[-1] - low.iloc[-1]) / close.iloc[-1] if close.iloc[-1] > 0 else 0.0
            features.append(price_volatility)
        else:
            features.append(0.0)
        
        # 나머지 특성들로 20개 맞추기
        while len(features) < 20:
            features.append(0.0)
        
        return np.array(features[:20], dtype=np.float32)
    
    def _extract_signal_features(self, signals: Dict) -> np.ndarray:
        """신호 특성 추출"""
        features = []
        
        # 각 시간대별 신호 (3개 × 8개 = 24개)
        for category in ['SHORT_TERM', 'MEDIUM_TERM', 'LONG_TERM']:
            if category in signals['decisions']:
                decision = signals['decisions'][category]
                
                # 액션 강도
                action = decision.get('action', 'HOLD')
                action_strength = 1.0 if action == 'LONG' else (-1.0 if action == 'SHORT' else 0.0)
                
                features.extend([
                    action_strength,
                    float(decision.get('net_score', 0.0)),
                    min(float(decision.get('leverage', 1)) / 10.0, 2.0),  # 정규화
                    min(float(decision.get('max_holding_minutes', 60)) / 1440.0, 1.0),  # 정규화
                ])
                
                # 신뢰도 및 점수
                meta = decision.get('meta', {}).get('synergy_meta', {})
                confidence = meta.get('confidence', 'LOW')
                confidence_score = 1.0 if confidence == 'HIGH' else (0.5 if confidence == 'MEDIUM' else 0.0)
                
                features.extend([
                    confidence_score,
                    float(meta.get('buy_score', 0.0)),
                    float(meta.get('sell_score', 0.0)),
                    len(meta.get('conflicts_detected', [])) / 5.0  # 정규화
                ])
            else:
                features.extend([0.0] * 8)
        
        # 갈등 및 메타 정보 (6개)
        conflicts = signals.get('conflicts', {})
        features.extend([
            1.0 if conflicts.get('has_conflicts', False) else 0.0,
            len(conflicts.get('long_categories', [])) / 3.0,
            len(conflicts.get('short_categories', [])) / 3.0,
            float(signals.get('meta', {}).get('active_positions', 0)) / 3.0,
            0.0,  # 예비
            0.0   # 예비
        ])
        
        return np.array(features[:30], dtype=np.float32)
    
    def _get_portfolio_state(self) -> np.ndarray:
        """포트폴리오 상태"""
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
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """스텝 실행"""
        if self.current_step >= min(len(self.price_data), len(self.signal_data)) - 1:
            return self._get_observation(), 0.0, True, {}
        
        # 액션 해석
        position_change = np.clip(action[0], -2.0, 2.0)
        leverage = np.clip(action[1], 1.0, 20.0)
        target_holding_minutes = np.clip(action[2], 1.0, 1440.0)
        
        # 현재 가격
        current_price = self.price_data.iloc[self.current_step]['close']
        next_price = self.price_data.iloc[self.current_step + 1]['close']
        
        # 포지션 변경사항 처리
        trade_completed = False
        old_position = self.current_position
        
        # 포지션 업데이트
        self._update_position(position_change, leverage, current_price, target_holding_minutes)
        
        # 거래 완료 확인
        if abs(old_position) > 0.01 and abs(self.current_position) < 0.01:
            trade_completed = True
            self.last_trade_pnl = self._calculate_trade_pnl(current_price, self.entry_price, old_position)
        
        # 개선된 보상 계산
        reward = self.reward_function.calculate_reward(
            current_price=next_price,
            entry_price=self.entry_price,
            position=self.current_position,
            action='TRADE' if abs(position_change) > 0.1 else 'HOLD',
            holding_time=self.holding_time,
            volatility=self._calculate_volatility(),
            volume_ratio=self._calculate_volume_ratio(),
            trade_pnl=self.last_trade_pnl if trade_completed else None
        )
        
        # 다음 스텝으로
        self.current_step += 1
        self.holding_time += 3  # 3분 증가
        
        # 홀딩 시간 초과시 강제 청산
        if self.in_position and self.holding_time >= target_holding_minutes:
            self._close_position(next_price)
        
        # 다음 상태
        next_state = self._get_observation()
        
        # 종료 조건
        done = (self.current_step >= min(len(self.price_data), len(self.signal_data)) - 1 or 
                self.balance <= self.initial_balance * 0.1)
        
        info = {
            'balance': self.balance,
            'position': self.current_position,
            'unrealized_pnl': self.unrealized_pnl,
            'total_trades': self.total_trades,
            'win_rate': self.winning_trades / max(self.total_trades, 1),
            'current_price': next_price,
            'entry_price': self.entry_price,
            'holding_time': self.holding_time,
            'volatility': self._calculate_volatility(),
            'volume_ratio': self._calculate_volume_ratio(),
            'trade_completed': trade_completed,
            'trade_pnl': self.last_trade_pnl if trade_completed else None
        }
        
        return next_state, reward, done, info
    
    def _calculate_volatility(self):
        """현재 변동성 계산"""
        if self.current_step < 20:
            return 0.02
        recent_data = self.price_data.iloc[max(0, self.current_step-20):self.current_step+1]
        if len(recent_data) < 2:
            return 0.02
        returns = recent_data['close'].pct_change().dropna()
        return returns.std() if len(returns) > 1 else 0.02
    
    def _calculate_volume_ratio(self):
        """거래량 비율 계산"""
        if self.current_step < 20:
            return 1.0
        recent_volume = self.price_data.iloc[max(0, self.current_step-20):self.current_step+1]['volume']
        if len(recent_volume) < 2:
            return 1.0
        current_volume = self.price_data.iloc[self.current_step]['volume']
        avg_volume = recent_volume.mean()
        return current_volume / avg_volume if avg_volume > 0 else 1.0
    
    def _calculate_trade_pnl(self, exit_price, entry_price, position):
        """거래 손익 계산"""
        if entry_price <= 0:
            return 0.0
        if position > 0:  # Long
            return (exit_price - entry_price) / entry_price
        else:  # Short
            return (entry_price - exit_price) / entry_price
    
    def _update_position(self, position_change, leverage, current_price, target_holding_minutes):
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
                self.holding_time = 0
                self.in_position = True
    
    def _close_position(self, exit_price):
        """포지션 청산"""
        if abs(self.current_position) < 0.01:
            return
        
        # PnL 계산
        pnl = self._calculate_trade_pnl(exit_price, self.entry_price, self.current_position)
        pnl_usd = pnl * self.current_leverage * self.balance
        
        # 거래 수수료 차감 (0.1%)
        fee = abs(pnl_usd) * 0.001
        pnl_usd -= fee
        
        # 잔고 업데이트
        self.balance += pnl_usd
        
        # 통계 업데이트
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
        
        # 포지션 초기화
        self.current_position = 0.0
        self.unrealized_pnl = 0.0
        self.in_position = False
        self.holding_time = 0
        self.last_trade_pnl = pnl

# =================================================================
# 4. 개선된 RL 에이전트
# =================================================================

class ImprovedCryptoRLAgent:
    """개선된 암호화폐 강화학습 에이전트"""
    
    def __init__(self, state_size: int, learning_rate: float = 5e-5, 
                 gamma: float = 0.995, epsilon: float = 0.9, epsilon_decay: float = 0.9995):
        
        self.state_size = state_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.05
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 네트워크
        self.q_network = ImprovedDQNNetwork(state_size, 3).to(self.device)
        self.target_network = ImprovedDQNNetwork(state_size, 3).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 경험 리플레이
        self.memory = deque(maxlen=100000)
        self.batch_size = 128
        
        # 학습 추적
        self.training_rewards = []
        self.losses = []
        self.win_rates = []
        
        # 타겟 네트워크 업데이트
        self.target_update_freq = 1000
        self.update_count = 0
    
    def remember(self, state, action, reward, next_state, done):
        """경험 저장"""
        self.memory.append(Experience(state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray) -> np.ndarray:
        """액션 선택"""
        if np.random.random() <= self.epsilon:
            # 스마트한 랜덤 액션
            return np.array([
                np.random.uniform(-1.0, 1.0),    # 포지션 변경
                np.random.uniform(1.0, 5.0),     # 레버리지 (보수적)
                np.random.uniform(30.0, 180.0)   # 홀딩 시간
            ])
        
        # Q-값 기반 액션 선택
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
        if len(self.memory) < self.batch_size * 2:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = [e.action for e in batch]
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        # numpy.bool_로 인한 타입 문제 방지: 파이썬 bool 리스트로 유지 (로직 불변)
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
                    # 액션을 인덱스로 변환
                    pos_idx = int(np.clip((action[0] + 2.0) / 0.2, 0, 20))
                    lev_idx = int(np.clip(action[1] - 1, 0, 19))
                    hold_idx = int(np.clip((action[2] - 30.0) / 30.0, 0, 47))
                    
                    # Double DQN 적용
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
        pos_loss = F.smooth_l1_loss(current_position_q, target_position_q)
        lev_loss = F.smooth_l1_loss(current_leverage_q, target_leverage_q)
        hold_loss = F.smooth_l1_loss(current_holding_q, target_holding_q)
        
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
        
        # 타겟 네트워크 업데이트
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.update_target_network()
    
    def update_target_network(self):
        """타겟 네트워크 업데이트"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def safe_save_model(self, filepath: str):
        """안전한 모델 저장"""
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
            print(f"✅ 모델 저장 완료: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ 모델 저장 실패: {e}")
            return False
    
    def safe_load_model(self, filepath: str):
        """안전한 모델 로드"""
        if not os.path.exists(filepath):
            print(f"⚠️  모델 파일이 없습니다: {filepath}")
            return False
        
        try:
            # PyTorch 2.6 호환 로딩
            try:
                checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
            except:
                checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
            
            self.q_network.load_state_dict(checkpoint['q_network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.training_rewards = checkpoint.get('training_rewards', [])
            self.losses = checkpoint.get('losses', [])
            self.win_rates = checkpoint.get('win_rates', [])
            self.update_count = checkpoint.get('update_count', 0)
            
            print(f"✅ 모델 로드 성공! 엡실론: {self.epsilon:.3f}")
            return True
            
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            return False

# =================================================================
# 5. 데이터 로딩 함수들
# =================================================================

def load_price_data():
    """가격 데이터 로드"""
    try:
        required_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
        
        df_3m = pd.read_csv('data/ETHUSDC_3m_historical_data.csv')
        df_3m['timestamp'] = pd.to_datetime(df_3m['timestamp'])
        df_3m = df_3m.set_index('timestamp')
        df_3m = df_3m[required_columns]
        
        price_data = df_3m.reset_index()
        print(f"✅ 가격 데이터 로드: {len(price_data):,}개 캔들")
        return price_data
        
    except Exception as e:
        print(f"❌ 가격 데이터 로드 실패: {e}")
        return None

def load_signal_data():
    """신호 데이터 로드 (Parquet 또는 생성)"""
    
    # 1. Parquet 파일 찾기
    agent_folder = Path("agent")
    parquet_files = []
    
    if agent_folder.exists():
        parquet_files = list(agent_folder.glob("*.parquet"))
    
    if parquet_files:
        try:
            print(f"📖 신호 데이터 로드 중: {parquet_files[0].name}")
            signal_df = pd.read_parquet(parquet_files[0])
            print(f"✅ 신호 데이터 로드: {len(signal_df):,}개 레코드")
            
            # 간단한 변환
            signal_data = convert_parquet_to_signals(signal_df)
            return signal_data
            
        except Exception as e:
            print(f"❌ Parquet 로드 실패: {e}")
    
    # 2. Parquet 파일이 없으면 기본 신호 생성
    print("⚠️  Parquet 파일이 없어 기본 신호를 생성합니다.")
    return None

def convert_parquet_to_signals(signal_df):
    """Parquet을 신호 리스트로 변환"""
    signal_data = []
    
    print("🔄 신호 데이터 변환 중...")
    
    for idx, row in signal_df.iterrows():
        signal_dict = {
            'decisions': {},
            'conflicts': {'has_conflicts': False, 'long_categories': [], 'short_categories': []},
            'meta': {'active_positions': 0}
        }
        
        # 각 시간대별 신호 추출
        for category in ['SHORT_TERM', 'MEDIUM_TERM', 'LONG_TERM']:
            prefix = f"{category.lower()}_"
            
            action = row.get(f'{prefix}action', 'HOLD')
            net_score = row.get(f'{prefix}net_score', 0.0)
            confidence = row.get(f'{prefix}confidence', 'LOW')
            leverage = row.get(f'{prefix}leverage', 1)
            max_holding = row.get(f'{prefix}max_holding_minutes', 60 if category == 'SHORT_TERM' else (240 if category == 'MEDIUM_TERM' else 1440))
            buy_score = row.get(f'{prefix}buy_score', 0.0)
            sell_score = row.get(f'{prefix}sell_score', 0.0)
            
            signal_dict['decisions'][category] = {
                'action': action,
                'net_score': float(net_score) if pd.notna(net_score) else 0.0,
                'leverage': int(leverage) if pd.notna(leverage) else 1,
                'max_holding_minutes': int(max_holding) if pd.notna(max_holding) else (60 if category == 'SHORT_TERM' else (240 if category == 'MEDIUM_TERM' else 1440)),
                'raw': {},
                'meta': {
                    'synergy_meta': {
                        'confidence': confidence if pd.notna(confidence) else 'LOW',
                        'buy_score': float(buy_score) if pd.notna(buy_score) else 0.0,
                        'sell_score': float(sell_score) if pd.notna(sell_score) else 0.0,
                        'conflicts_detected': []
                    }
                }
            }
            
            # 갈등 정보
            if action == 'LONG':
                signal_dict['conflicts']['long_categories'].append(category)
            elif action == 'SHORT':
                signal_dict['conflicts']['short_categories'].append(category)
        
        # 갈등 여부
        if (len(signal_dict['conflicts']['long_categories']) > 0 and 
            len(signal_dict['conflicts']['short_categories']) > 0):
            signal_dict['conflicts']['has_conflicts'] = True
        
        signal_data.append(signal_dict)
        
        if (idx + 1) % 5000 == 0:
            print(f"   변환 진행: {idx + 1:,}/{len(signal_df):,}")
    
    print(f"✅ 신호 데이터 변환 완료: {len(signal_data):,}개")
    return signal_data

def generate_basic_signals(length):
    """기본 신호 데이터 생성 (Parquet이 없을 때)"""
    print(f"🔄 기본 신호 데이터 생성 중: {length:,}개")
    
    signal_data = []
    for i in range(length):
        # RSI 기반 간단한 신호 생성
        rsi_value = 30 + (i % 40)  # 30~70 사이 순환
        
        if rsi_value > 60:
            short_action = 'SHORT'
            short_score = (rsi_value - 60) / 10
        elif rsi_value < 40:
            short_action = 'LONG' 
            short_score = (40 - rsi_value) / 10
        else:
            short_action = 'HOLD'
            short_score = 0.0
        
        signal_dict = {
            'decisions': {
                'SHORT_TERM': {
                    'action': short_action,
                    'net_score': short_score,
                    'leverage': 1 + int(short_score * 3),  # 1~4배
                    'max_holding_minutes': 60,
                    'raw': {},
                    'meta': {'synergy_meta': {'confidence': 'MEDIUM' if short_score > 0.5 else 'LOW', 'buy_score': short_score if short_action == 'LONG' else 0.0, 'sell_score': short_score if short_action == 'SHORT' else 0.0, 'conflicts_detected': []}}
                },
                'MEDIUM_TERM': {
                    'action': 'HOLD',
                    'net_score': 0.0,
                    'leverage': 1,
                    'max_holding_minutes': 240,
                    'raw': {},
                    'meta': {'synergy_meta': {'confidence': 'LOW', 'buy_score': 0.0, 'sell_score': 0.0, 'conflicts_detected': []}}
                },
                'LONG_TERM': {
                    'action': 'HOLD',
                    'net_score': 0.0,
                    'leverage': 1,
                    'max_holding_minutes': 1440,
                    'raw': {},
                    'meta': {'synergy_meta': {'confidence': 'LOW', 'buy_score': 0.0, 'sell_score': 0.0, 'conflicts_detected': []}}
                }
            },
            'conflicts': {'has_conflicts': False, 'long_categories': [], 'short_categories': []},
            'meta': {'active_positions': 0}
        }
        
        signal_data.append(signal_dict)
    
    print(f"✅ 기본 신호 데이터 생성 완료")
    return signal_data

# =================================================================
# 6. 성능 분석 클래스
# =================================================================

class PerformanceAnalyzer:
    """성능 분석 및 평가"""
    
    @staticmethod
    def evaluate_agent(agent, env, num_episodes=10):
        """에이전트 성능 평가"""
        print(f"🔍 에이전트 성능 평가 중 ({num_episodes} 에피소드)...")
        
        original_epsilon = agent.epsilon
        agent.epsilon = 0.0  # 탐험 비활성화
        
        results = []
        all_trades = []
        
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            episode_trades = []
            episode_balance = env.initial_balance
            
            for step in range(500):  # 에피소드당 최대 500 스텝
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                
                episode_reward += reward
                episode_balance = info['balance']
                
                if info.get('trade_completed', False):
                    trade_pnl = info.get('trade_pnl', 0.0)
                    episode_trades.append(1 if trade_pnl > 0 else 0)
                
                state = next_state
                if done:
                    break
            
            # 에피소드 결과
            episode_return = (episode_balance - env.initial_balance) / env.initial_balance
            win_rate = np.mean(episode_trades) if episode_trades else 0.0
            
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
        
        # 원래 epsilon 복원
        agent.epsilon = original_epsilon
        
        # 통합 성능 지표
        overall_stats = {
            'avg_return': np.mean([r['return'] for r in results]),
            'avg_reward': np.mean([r['total_reward'] for r in results]),
            'overall_win_rate': np.mean(all_trades) if all_trades else 0.0,
            'avg_trades_per_episode': np.mean([r['total_trades'] for r in results]),
            'avg_max_drawdown': np.mean([r['max_drawdown'] for r in results]),
            'consistency': 1.0 - np.std([r['return'] for r in results]) if len(results) > 1 else 1.0,
            'total_trades': len(all_trades)
        }
        
        return results, overall_stats
    
    @staticmethod
    def print_performance_report(results, stats):
        """성능 리포트 출력"""
        print("\n" + "="*60)
        print("📊 성능 평가 결과")
        print("="*60)
        print(f"🎯 전체 승률: {stats['overall_win_rate']:.3f}")
        print(f"💰 평균 수익률: {stats['avg_return']:.3f} ({stats['avg_return']*100:.1f}%)")
        print(f"🏆 평균 리워드: {stats['avg_reward']:.1f}")
        print(f"📈 에피소드당 평균 거래 수: {stats['avg_trades_per_episode']:.1f}")
        print(f"📉 평균 최대 낙폭: {stats['avg_max_drawdown']:.3f}")
        print(f"🎲 성과 일관성: {stats['consistency']:.3f}")
        print(f"🔢 총 거래 수: {stats['total_trades']}")
        
        # 성능 등급
        grade = PerformanceAnalyzer.get_performance_grade(stats)
        print(f"\n🎯 성능 등급: {grade}")
        
        # 개선 제안
        recommendations = PerformanceAnalyzer.get_recommendations(stats)
        print("\n💡 개선 제안:")
        for rec in recommendations:
            print(f"  • {rec}")
    
    @staticmethod
    def get_performance_grade(stats):
        """성능 등급 부여"""
        win_rate = stats['overall_win_rate']
        avg_return = stats['avg_return']
        consistency = stats['consistency']
        
        score = 0
        if win_rate >= 0.65: score += 3
        elif win_rate >= 0.6: score += 2
        elif win_rate >= 0.55: score += 1
        
        if avg_return >= 0.15: score += 3
        elif avg_return >= 0.05: score += 2
        elif avg_return >= 0.0: score += 1
        
        if consistency >= 0.8: score += 2
        elif consistency >= 0.6: score += 1
        
        grades = {8: "A+ (우수)", 7: "A (좋음)", 6: "B+ (양호)", 5: "B (보통)", 
                 4: "C+ (미흡)", 3: "C (개선필요)", 2: "D (나쁨)", 1: "F (매우나쁨)", 0: "F (실패)"}
        
        return grades.get(score, "F (실패)")
    
    @staticmethod
    def get_recommendations(stats):
        """성능 기반 개선 제안"""
        recommendations = []
        
        if stats['overall_win_rate'] < 0.55:
            recommendations.append("승률이 낮습니다. 더 긴 훈련이 필요합니다.")
        
        if stats['avg_return'] < 0.02:
            recommendations.append("수익률이 낮습니다. 리스크-리워드 비율을 개선하세요.")
        
        if stats['avg_max_drawdown'] > 0.2:
            recommendations.append("최대 낙폭이 큽니다. 포지션 크기를 줄이세요.")
        
        if stats['consistency'] < 0.5:
            recommendations.append("성과 일관성이 떨어집니다. 더 많은 훈련이 필요합니다.")
        
        if stats['avg_trades_per_episode'] < 3:
            recommendations.append("거래 빈도가 낮습니다. 신호 감도를 높여보세요.")
        
        if not recommendations:
            recommendations.append("전반적으로 좋은 성능입니다!")
        
        return recommendations

# =================================================================
# 7. 훈련 함수
# =================================================================

def train_agent(agent, env, episodes=500, save_interval=100):
    """에이전트 훈련"""
    print(f"🚀 강화학습 훈련 시작 ({episodes} 에피소드)")
    
    episode_rewards = []
    episode_win_rates = []
    best_win_rate = 0.0
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        episode_trades = []
        steps = 0
        
        while steps < 500:  # 에피소드당 최대 500 스텝
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            
            if info.get('trade_completed', False):
                trade_pnl = info.get('trade_pnl', 0.0)
                episode_trades.append(1 if trade_pnl > 0 else 0)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # 학습
            if len(agent.memory) > agent.batch_size * 2:
                agent.replay()
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        episode_win_rate = np.mean(episode_trades) if episode_trades else 0.0
        episode_win_rates.append(episode_win_rate)
        
        agent.training_rewards.append(total_reward)
        agent.win_rates.append(episode_win_rate)
        
        # 진행 상황 출력
        if episode % 10 == 0 or episode < 10:
            recent_rewards = episode_rewards[-50:] if len(episode_rewards) >= 50 else episode_rewards
            recent_win_rates = episode_win_rates[-50:] if len(episode_win_rates) >= 50 else episode_win_rates
            
            avg_reward = np.mean(recent_rewards)
            avg_win_rate = np.mean(recent_win_rates)
            
            print(f"Episode {episode:4d} | "
                  f"승률: {avg_win_rate:.3f} | "
                  f"리워드: {avg_reward:7.1f} | "
                  f"잔고: ${info['balance']:7.0f} | "
                  f"ε: {agent.epsilon:.3f}")
        
        # 베스트 모델 저장
        if episode % save_interval == 0 and episode > 0:
            current_avg_win_rate = np.mean(episode_win_rates[-100:]) if len(episode_win_rates) >= 100 else np.mean(episode_win_rates)
            
            if current_avg_win_rate > best_win_rate:
                best_win_rate = current_avg_win_rate
                agent.safe_save_model(f'best_model_ep{episode}_wr{current_avg_win_rate:.3f}.pth')
                print(f"✅ 새로운 최고 성능! 승률: {current_avg_win_rate:.3f}")
        
        # 조기 종료 조건
        if episode > 200:
            recent_100_win_rate = np.mean(episode_win_rates[-100:])
            if recent_100_win_rate >= 0.65:
                print(f"🎯 목표 달성! 승률 {recent_100_win_rate:.3f} 도달")
                agent.safe_save_model('agent/final_optimized_model.pth')
                break
    
    print(f"\n🎉 훈련 완료!")
    print(f"   총 에피소드: {episode + 1}")
    print(f"   최고 승률: {best_win_rate:.3f}")
    print(f"   최종 승률: {np.mean(episode_win_rates[-50:]) if episode_win_rates else 0:.3f}")
    
    return agent, episode_rewards, episode_win_rates

# =================================================================
# 8. 메인 실행 함수
# =================================================================

def main():
    """메인 실행 함수"""
    print("🎯 완전 독립형 강화학습 트레이딩 시스템")
    print("=" * 60)
    
    try:
        # 1. 데이터 로드
        print("\n1️⃣ 데이터 로딩...")
        price_data = load_price_data()
        if price_data is None:
            print("❌ 가격 데이터 로드 실패")
            return
        
        signal_data = load_signal_data()
        if signal_data is None:
            # 기본 신호 생성
            signal_data = generate_basic_signals(min(len(price_data), 10000))  # 최대 10,000개
        
        # 데이터 길이 맞추기
        min_length = min(len(price_data), len(signal_data))
        price_data = price_data.iloc[:min_length].reset_index(drop=True)
        signal_data = signal_data[:min_length]
        
        print(f"✅ 최종 데이터 준비 완료: {min_length:,}개")
        
        # 2. 환경 및 에이전트 생성
        print("\n2️⃣ 환경 및 에이전트 생성...")
        env = ImprovedTradingEnvironment(price_data, signal_data)
        agent = ImprovedCryptoRLAgent(env.observation_space.shape[0])
        
        # 기존 모델 로드 시도
        model_files = ['agent/final_optimized_model.pth', 'agent/improved_crypto_rl_model.pth']
        model_loaded = False
        
        for model_file in model_files:
            if agent.safe_load_model(model_file):
                model_loaded = True
                break
        
        if not model_loaded:
            print("ℹ️  새로운 모델로 시작합니다.")
        
        # 3. 현재 성능 평가
        print("\n3️⃣ 현재 성능 평가...")
        results, stats = PerformanceAnalyzer.evaluate_agent(agent, env, num_episodes=5)
        PerformanceAnalyzer.print_performance_report(results, stats)
        
        # 4. 훈련 여부 결정
        if stats['overall_win_rate'] < 0.55 or not model_loaded:
            print(f"\n4️⃣ 훈련 시작...")
            print(f"   현재 승률: {stats['overall_win_rate']:.3f}")
            print(f"   목표 승률: 0.65+")
            
            # 훈련 실행
            trained_agent, rewards, win_rates = train_agent(agent, env, episodes=500)
            
            # 훈련 후 성능 재평가
            print("\n5️⃣ 훈련 후 성능 평가...")
            final_results, final_stats = PerformanceAnalyzer.evaluate_agent(trained_agent, env, num_episodes=10)
            PerformanceAnalyzer.print_performance_report(final_results, final_stats)
            
            # 개선도 출력
            improvement = final_stats['overall_win_rate'] - stats['overall_win_rate']
            print(f"\n🎯 성능 개선도:")
            print(f"   승률: {stats['overall_win_rate']:.3f} → {final_stats['overall_win_rate']:.3f} ({improvement:+.3f})")
            print(f"   평균 수익률: {stats['avg_return']:.3f} → {final_stats['avg_return']:.3f}")
            
            # 최종 모델 저장
            trained_agent.safe_save_model('agent/improved_crypto_rl_model.pth')
            
        else:
            print(f"✅ 현재 성능이 양호합니다 (승률: {stats['overall_win_rate']:.3f})")
            
            # 추가 훈련 여부 묻기
            user_input = input("\n💫 추가 훈련을 원하시나요? (y/n): ")
            if user_input.lower() == 'y':
                print("🚀 추가 훈련 시작...")
                train_agent(agent, env, episodes=200)
    
    except KeyboardInterrupt:
        print("\n⚠️  사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()