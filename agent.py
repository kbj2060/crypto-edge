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

# 경험 튜플 정의
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class TradingEnvironment(gym.Env):
    """
    암호화폐 거래 강화학습 환경
    """
    
    def __init__(self, price_data: pd.DataFrame, signal_data: List[Dict], 
                 initial_balance: float = 10000.0, max_position: float = 1.0):
        """
        Args:
            price_data: OHLCV 가격 데이터
            signal_data: 전략 신호 데이터 리스트
            initial_balance: 초기 자본
            max_position: 최대 포지션 크기 (-1.0 ~ 1.0)
        """
        super().__init__()
        
        self.price_data = price_data
        self.signal_data = signal_data
        self.initial_balance = initial_balance
        self.max_position = max_position
        
        # 액션 스페이스: [포지션 변경량, 레버리지, 홀딩 시간]
        # 포지션: -1(풀숏) ~ 1(풀롱), 레버리지: 1~20, 홀딩: 0~1440분
        self.action_space = spaces.Box(
            low=np.array([-2.0, 1.0, 0.0]), 
            high=np.array([2.0, 20.0, 1440.0]), 
            dtype=np.float32
        )
        
        # 상태 스페이스: 가격 데이터 + 신호 데이터 + 포트폴리오 상태
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self._get_state_size(),), 
            dtype=np.float32
        )
        
        self.reset()
    
    def _get_state_size(self) -> int:
        """상태 벡터 크기 계산"""
        # 가격 특성(20) + 신호 특성(30) + 포트폴리오 상태(10) = 60
        # 하지만 실제로는 56이 나오므로 56으로 설정
        return 56
    
    def _extract_signal_features(self, signals: Dict) -> np.ndarray:
        """신호 데이터에서 특성 추출"""
        features = []
        
        # 각 카테고리별 특성 추출
        for category in ['SHORT_TERM', 'MEDIUM_TERM', 'LONG_TERM']:
            if category in signals['decisions']:
                decision = signals['decisions'][category]
                
                # 기본 특성
                features.extend([
                    1.0 if decision['action'] == 'LONG' else (-1.0 if decision['action'] == 'SHORT' else 0.0),
                    float(decision['net_score']),
                    decision['leverage'] / 20.0,  # 정규화
                    decision['max_holding_minutes'] / 1440.0,  # 정규화
                ])
                
                # 신호 강도 특성
                meta = decision['meta']['synergy_meta']
                features.extend([
                    float(meta['buy_score']) if 'buy_score' in meta else 0.0,
                    float(meta['sell_score']) if 'sell_score' in meta else 0.0,
                    1.0 if meta['confidence'] == 'HIGH' else (0.5 if meta['confidence'] == 'MEDIUM' else 0.0),
                    len(meta.get('conflicts_detected', [])) / 5.0,  # 정규화
                ])
            else:
                features.extend([0.0] * 8)
        
        # 전체 갈등 정보
        conflicts = signals['conflicts']
        features.extend([
            1.0 if conflicts['has_conflicts'] else 0.0,
            len(conflicts['long_categories']) / 3.0,
            len(conflicts['short_categories']) / 3.0,
            signals['meta']['active_positions'] / 3.0,
        ])
        
        # 개별 전략 점수 (상위 6개만)
        all_strategies = {}
        for category_data in signals['decisions'].values():
            all_strategies.update(category_data['raw'])
        
        strategy_scores = []
        for strategy_name in ['VWAP_PINBALL', 'LIQUIDITY_GRAB', 'ZSCORE_MEAN_REVERSION', 
                             'SUPPORT_RESISTANCE', 'EMA_CONFLUENCE', 'ICHIMOKU']:
            if strategy_name in all_strategies:
                score = all_strategies[strategy_name]['score']
                action = all_strategies[strategy_name]['action']
                action_val = 1.0 if action == 'BUY' else (-1.0 if action == 'SELL' else 0.0)
                strategy_scores.extend([score, action_val])
            else:
                strategy_scores.extend([0.0, 0.0])
        
        features.extend(strategy_scores[:12])  # 6개 전략 * 2 = 12개
        
        return np.array(features[:30], dtype=np.float32)  # 30개로 제한
    
    def _extract_price_features(self, idx: int) -> np.ndarray:
        """가격 데이터에서 특성 추출"""
        if idx < 20:
            # 초기 데이터 부족시 패딩
            return np.zeros(20, dtype=np.float32)
        
        # 최근 20개 가격 데이터
        recent_data = self.price_data.iloc[idx-19:idx+1]
        
        features = []
        
        # 가격 변화율
        returns = recent_data['close'].pct_change().fillna(0)
        features.extend([
            returns.mean(),
            returns.std(),
            returns.iloc[-1],  # 최근 수익률
            returns.iloc[-5:].mean(),  # 5기간 평균 수익률
        ])
        
        # 기술적 지표
        close = recent_data['close']
        high = recent_data['high']
        low = recent_data['low']
        volume = recent_data['volume']
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        features.append(rsi.iloc[-1] / 100.0 if not pd.isna(rsi.iloc[-1]) else 0.5)
        
        # 볼린저 밴드
        sma = close.rolling(window=20).mean()
        std = close.rolling(window=20).std()
        bb_upper = sma + (std * 2)
        bb_lower = sma - (std * 2)
        bb_position = (close.iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
        features.append(bb_position if not pd.isna(bb_position) else 0.5)
        
        # 이동평균
        for window in [5, 10, 20]:
            ma = close.rolling(window=window).mean()
            ma_ratio = close.iloc[-1] / ma.iloc[-1] - 1
            features.append(ma_ratio if not pd.isna(ma_ratio) else 0.0)
        
        # 거래량 지표
        features.extend([
            volume.iloc[-1] / volume.mean() - 1,  # 거래량 비율
            (high.iloc[-1] - low.iloc[-1]) / close.iloc[-1],  # 변동성
        ])
        
        # 가격 위치
        max_high = high.max()
        min_low = low.min()
        price_position = (close.iloc[-1] - min_low) / (max_high - min_low)
        features.append(price_position if not pd.isna(price_position) else 0.5)
        
        # 추가 기술적 지표들
        features.extend([
            (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5],  # 5기간 수익률
            (close.iloc[-1] - close.iloc[-10]) / close.iloc[-10],  # 10기간 수익률
            (high.iloc[-5:].max() - close.iloc[-1]) / close.iloc[-1],  # 최근 고점과 거리
            (close.iloc[-1] - low.iloc[-5:].min()) / close.iloc[-1],  # 최근 저점과 거리
        ])
        
        return np.array(features[:20], dtype=np.float32)
    
    def _get_portfolio_state(self) -> np.ndarray:
        """포트폴리오 상태 정보"""
        features = [
            self.current_position,
            self.current_leverage / 20.0,
            self.balance / self.initial_balance - 1,  # 수익률
            self.unrealized_pnl / self.initial_balance,
            self.total_trades / 100.0,  # 정규화
            self.winning_trades / max(self.total_trades, 1),
            self.max_drawdown,
            self.consecutive_losses / 10.0,  # 정규화
            min(self.holding_time / 1440.0, 1.0),  # 정규화
            1.0 if self.in_position else 0.0
        ]
        return np.array(features, dtype=np.float32)
    
    def reset(self):
        """환경 초기화"""
        self.current_step = 20  # 충분한 가격 히스토리 확보
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
        signal_features = self._extract_signal_features(self.signal_data[self.current_step])
        
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
                self.balance <= self.initial_balance * 0.1)  # 90% 손실시 종료
        
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
            reward += position_pnl * 100  # 스케일링
        
        # 2. 신호 방향성과 일치도 보상
        signals = self.signal_data[self.current_step]
        signal_alignment = self._calculate_signal_alignment(position_change, signals)
        reward += signal_alignment * 10
        
        # 3. 리스크 관리 보상
        risk_penalty = self._calculate_risk_penalty(leverage, self.current_position)
        reward -= risk_penalty
        
        # 4. 거래 빈도 패널티 (과도한 거래 방지)
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
        
        for category, decision in signals['decisions'].items():
            if decision['action'] == 'LONG' and position_change > 0:
                alignment_score += abs(decision['net_score'])
            elif decision['action'] == 'SHORT' and position_change < 0:
                alignment_score += abs(decision['net_score'])
            elif decision['action'] == 'HOLD' and abs(position_change) < 0.1:
                alignment_score += 0.1
        
        return alignment_score / 3  # 3개 카테고리로 정규화
    
    def _calculate_risk_penalty(self, leverage: float, position: float) -> float:
        """리스크 패널티 계산"""
        penalty = 0.0
        
        # 과도한 레버리지 패널티
        if leverage > 10:
            penalty += (leverage - 10) * 0.1
        
        # 과도한 포지션 크기 패널티
        if abs(position) > 0.8:
            penalty += (abs(position) - 0.8) * 5
        
        return penalty
    
    def _calculate_holding_reward(self) -> float:
        """홀딩 시간 최적화 보상"""
        # 단타의 경우 빠른 청산이 유리
        if self.holding_time > 60:  # 1시간 이상
            return -0.01 * (self.holding_time - 60) / 60
        return 0.0
    
    def _update_position(self, position_change: float, leverage: float, 
                        current_price: float, target_holding_minutes: float):
        """포지션 업데이트"""
        # 새로운 포지션 계산
        new_position = np.clip(self.current_position + position_change, -1.0, 1.0)
        
        # 포지션 변경이 있는 경우
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


class DQNNetwork(nn.Module):
    """Deep Q-Network"""
    
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
        self.position_head = nn.Linear(hidden_size // 2, 21)  # -2.0 ~ 2.0 (0.2 간격)
        self.leverage_head = nn.Linear(hidden_size // 2, 20)  # 1 ~ 20
        self.holding_head = nn.Linear(hidden_size // 2, 48)   # 30분 ~ 1440분 (30분 간격)
    
    def forward(self, x):
        features = self.feature_layers(x)
        
        position_q = self.position_head(features)
        leverage_q = self.leverage_head(features)
        holding_q = self.holding_head(features)
        
        return position_q, leverage_q, holding_q


class CryptoRLAgent:
    """암호화폐 강화학습 에이전트"""
    
    def __init__(self, state_size: int, learning_rate: float = 0.001, 
                 gamma: float = 0.99, epsilon: float = 1.0, epsilon_decay: float = 0.995):
        
        self.state_size = state_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        
        # 네트워크
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQNNetwork(state_size, 3).to(self.device)
        self.target_network = DQNNetwork(state_size, 3).to(self.device)
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
            # 랜덤 액션
            return np.array([
                np.random.uniform(-2.0, 2.0),
                np.random.uniform(1.0, 20.0), 
                np.random.uniform(30.0, 1440.0)
            ])
        
        # Q-값 기반 액션 선택
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            position_q, leverage_q, holding_q = self.q_network(state_tensor)
            
            # 각 차원별 최적 액션 선택
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
        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = [e.action for e in batch]
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        dones = torch.BoolTensor([e.done for e in batch]).to(self.device)
        
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
                    # 각 액션 차원별 타겟 계산
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


def train_rl_agent(price_data: pd.DataFrame, signal_data: List[Dict], 
                  episodes: int = 1000, save_interval: int = 100):
    """강화학습 에이전트 훈련"""
    
    # 환경과 에이전트 초기화
    env = TradingEnvironment(price_data, signal_data)
    agent = CryptoRLAgent(env.observation_space.shape[0])
    
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
        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode}, Average Reward: {avg_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}, Balance: {info['balance']:.2f}, "
                  f"Win Rate: {info['win_rate']:.3f}")
        
        # 모델 저장
        if episode % save_interval == 0 and total_reward > best_reward:
            best_reward = total_reward
            agent.save_model(f'best_crypto_rl_model_ep{episode}.pth')
            print(f"New best model saved at episode {episode} with reward {best_reward:.2f}")
    
    return agent, episode_rewards


def evaluate_agent(agent: CryptoRLAgent, price_data: pd.DataFrame, 
                  signal_data: List[Dict], episodes: int = 10):
    """에이전트 성능 평가"""
    env = TradingEnvironment(price_data, signal_data)
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


class BacktestAnalyzer:
    """백테스트 결과 분석기"""
    
    @staticmethod
    def calculate_performance_metrics(results: List[Dict]) -> Dict:
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
    def generate_report(results: List[Dict], metrics: Dict) -> str:
        """성능 리포트 생성"""
        report = f"""
=== 강화학습 트레이딩 AI 성능 리포트 ===

기본 통계:
- 평균 수익률: {metrics['avg_return']:.2%}
- 수익률 표준편차: {metrics['std_return']:.2%}
- 샤프 비율: {metrics['sharpe_ratio']:.3f}
- 최대 수익률: {metrics['max_return']:.2%}
- 최소 수익률: {metrics['min_return']:.2%}

거래 통계:
- 평균 승률: {metrics['win_rate']:.2%}
- 평균 거래 횟수: {metrics['avg_trades']:.1f}
- 최대 낙폭: {metrics['max_drawdown']:.2%}
- 수익 에피소드: {metrics['profit_episodes']}개
- 손실 에피소드: {metrics['loss_episodes']}개

위험 지표:
- 최대 낙폭: {metrics['max_drawdown']:.2%}
- 변동성: {metrics['std_return']:.2%}
- 손실 확률: {metrics['loss_episodes']/(metrics['profit_episodes']+metrics['loss_episodes']):.2%}
        """
        
        return report


# 실제 사용 예시
def load_ethusdc_data():
    """ETHUSDC CSV 데이터 로드 - 3분, 15분, 1시간봉"""
    try:
        # 3분봉 데이터 로드
        df_3m = pd.read_csv('data/ETHUSDC_3m_historical_data.csv')
        df_3m['timestamp'] = pd.to_datetime(df_3m['timestamp'])
        df_3m = df_3m.set_index('timestamp')
        
        df_15m = pd.read_csv('data/ETHUSDC_15m_historical_data.csv')
        df_15m['timestamp'] = pd.to_datetime(df_15m['timestamp'])
        df_15m = df_15m.set_index('timestamp')
        
        # 3분봉에서 1시간봉 생성
        df_1h = pd.read_csv('data/ETHUSDC_1h_historical_data.csv')
        df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'])
        df_1h = df_1h.set_index('timestamp')
        
        print(f"✅ ETHUSDC 3분봉 데이터 로드 완료: {len(df_3m)}개 캔들")
        print(f"✅ ETHUSDC 15분봉 데이터 생성 완료: {len(df_15m)}개 캔들")
        print(f"✅ ETHUSDC 1시간봉 데이터 생성 완료: {len(df_1h)}개 캔들")
        
        return df_3m, df_15m, df_1h

    except FileNotFoundError as e:
        print(f"❌ 데이터 파일을 찾을 수 없습니다: {e}")
        return None, None, None, None
    except Exception as e:
        print(f"❌ 데이터 로드 중 오류 발생: {e}")
        return None, None, None, None

def generate_signal_data_with_indicators(price_data: pd.DataFrame, price_data_15m: pd.DataFrame, 
                                        price_data_1h: pd.DataFrame, max_periods: int = 1000):
    """CSV 데이터로 실제 지표 업데이트 및 전략 실행 (3분, 15분, 1시간봉 사용)"""
    from data.strategy_executor import StrategyExecutor
    from engines.trade_decision_engine import TradeDecisionEngine
    from data.candle_creator import CandleCreator
    from data.data_manager import get_data_manager
    from indicators.global_indicators import get_global_indicator_manager
    from utils.time_manager import get_time_manager
    
    # 컴포넌트 초기화
    strategy_executor = StrategyExecutor()
    decision_engine = TradeDecisionEngine()
    global_manager = get_global_indicator_manager()
    time_manager = get_time_manager()
    
    signal_data = []
    
    print("🔄 CSV 데이터로 지표 업데이트 및 전략 실행 중...")
    print(f"   - 3분봉: {len(price_data)}개 캔들")
    print(f"   - 15분봉: {len(price_data_15m)}개 캔들")
    print(f"   - 1시간봉: {len(price_data_1h)}개 캔들")
    
    # 최근 데이터부터 처리 (최대 max_periods개)
    start_idx = 500
    
    for i in range(start_idx, len(price_data)):
        try:
            # 현재 캔들 데이터
            series_3m = price_data.iloc[i][['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'timestamp']]
            
            # 글로벌 지표 업데이트
            global_manager.update_all_indicators(series_3m)
            
            # 전략 실행
            strategy_executor.execute_all_strategies()
            
            # 신호 수집
            signals = strategy_executor.get_signals()
            
            # 거래 결정
            decision = decision_engine.decide_trade_realtime(signals)
            
            signal_data.append(decision)
            
            if (i - start_idx) % 100 == 0:
                print(f"   진행률: {i - start_idx + 1}/{max_periods} ({((i - start_idx + 1) / max_periods) * 100:.1f}%)")
                
        except Exception as e:
            print(f"❌ 신호 생성 중 오류 (인덱스 {i}): {e}")
            # 오류 발생 시 기본 신호 생성
            signal_data.append({
                'decisions': {
                    'SHORT_TERM': {'action': 'HOLD', 'net_score': 0.0, 'leverage': 1, 'max_holding_minutes': 60, 'raw': {}, 'meta': {'synergy_meta': {'confidence': 'LOW', 'buy_score': 0.0, 'sell_score': 0.0, 'conflicts_detected': []}}},
                    'MEDIUM_TERM': {'action': 'HOLD', 'net_score': 0.0, 'leverage': 1, 'max_holding_minutes': 240, 'raw': {}, 'meta': {'synergy_meta': {'confidence': 'LOW', 'buy_score': 0.0, 'sell_score': 0.0, 'conflicts_detected': []}}},
                    'LONG_TERM': {'action': 'HOLD', 'net_score': 0.0, 'leverage': 1, 'max_holding_minutes': 1440, 'raw': {}, 'meta': {'synergy_meta': {'confidence': 'LOW', 'buy_score': 0.0, 'sell_score': 0.0, 'conflicts_detected': []}}}
                },
                'conflicts': {'has_conflicts': False, 'long_categories': [], 'short_categories': []},
                'meta': {'active_positions': 0}
            })
    
    print(f"✅ 신호 데이터 생성 완료: {len(signal_data)}개")
    return signal_data

def main_example():
    """강화학습 트레이딩 AI 사용 예시 - 실제 바이낸스 데이터 사용"""
    
    print("=== 강화학습 트레이딩 AI 훈련 시작 (실제 데이터) ===")
    
    # 1. 실제 ETHUSDC 데이터 로드 (3분, 15분, 1시간봉)
    price_data_3m, price_data_15m, price_data_1h = load_ethusdc_data()
    
    if price_data_3m is None:
        print("❌ 데이터 로드 실패. 프로그램을 종료합니다.")
        return None, None, None
    
    # 2. 가격 데이터 전처리 (3분봉을 메인으로 사용)
    price_data = price_data_3m.reset_index()
    price_data = price_data.rename(columns={'timestamp': 'timestamp'})
    
    # 필요한 컬럼만 선택
    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    price_data = price_data[required_columns]
    
    print(f"📊 가격 데이터 정보:")
    print(f"   - 기간: {price_data['timestamp'].min()} ~ {price_data['timestamp'].max()}")
    print(f"   - 총 캔들 수: {len(price_data)}개")
    print(f"   - 가격 범위: ${price_data['close'].min():.2f} ~ ${price_data['close'].max():.2f}")
    
    # 3. CSV 데이터로 실제 지표 업데이트 및 전략 실행 (3분, 15분, 1시간봉 사용)
    signal_data = generate_signal_data_with_indicators(price_data, price_data_15m, price_data_1h, 
                                                     max_periods=min(1000, len(price_data)))
    
    if not signal_data:
        print("❌ 신호 데이터 생성 실패. 프로그램을 종료합니다.")
        return None, None, None
    
    print("=== 강화학습 에이전트 훈련 시작 ===")
    
    # 4. 에이전트 훈련 (에피소드 수 조정)
    agent, rewards = train_rl_agent(price_data, signal_data, episodes=200)
    
    print("\n=== 훈련 완료, 성능 평가 중 ===")
    
    # 5. 성능 평가
    eval_results = evaluate_agent(agent, price_data, signal_data, episodes=10)
    
    # 6. 성능 분석
    analyzer = BacktestAnalyzer()
    metrics = analyzer.calculate_performance_metrics(eval_results)
    report = analyzer.generate_report(eval_results, metrics)
    
    print(report)
    
    # 7. 모델 저장
    agent.save_model('ethusdc_crypto_rl_model.pth')
    print("\n모델이 'ethusdc_crypto_rl_model.pth'에 저장되었습니다.")
    
    return agent, eval_results, metrics


# 실시간 거래를 위한 클래스
class LiveTradingBot:
    """실시간 거래 봇"""
    
    def __init__(self, agent: CryptoRLAgent, exchange_api=None):
        self.agent = agent
        self.agent.epsilon = 0  # 실거래에서는 탐험 비활성화
        self.exchange_api = exchange_api
        self.current_position = 0.0
        self.last_action_time = datetime.now()
        
    def should_trade(self, current_signals: Dict) -> bool:
        """거래 조건 확인"""
        # 최소 시간 간격 체크 (3분)
        if (datetime.now() - self.last_action_time).seconds < 180:
            return False
        
        # 신호 품질 체크
        high_confidence_signals = 0
        for category in current_signals['decisions'].values():
            if category['meta']['synergy_meta']['confidence'] == 'HIGH':
                high_confidence_signals += 1
        
        return high_confidence_signals >= 1
    
    def get_trading_action(self, price_data: pd.DataFrame, current_signals: Dict) -> Dict:
        """거래 액션 결정"""
        if not self.should_trade(current_signals):
            return {'action': 'HOLD', 'reason': '거래 조건 미충족'}
        
        # 환경 설정
        env = TradingEnvironment(price_data, [current_signals])
        state = env._get_observation()
        
        # AI 액션 예측
        action = self.agent.act(state)
        
        return {
            'action': 'TRADE',
            'position_change': action[0],
            'leverage': action[1],
            'holding_minutes': action[2],
            'confidence': self._calculate_action_confidence(state, action),
            'timestamp': datetime.now()
        }
    
    def _calculate_action_confidence(self, state: np.ndarray, action: np.ndarray) -> float:
        """액션 신뢰도 계산"""
        # Q값들의 분산을 이용한 신뢰도 측정
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
        
        with torch.no_grad():
            pos_q, lev_q, hold_q = self.agent.q_network(state_tensor)
            
            # 각 차원별 Q값 분산
            pos_var = torch.var(pos_q).item()
            lev_var = torch.var(lev_q).item()
            hold_var = torch.var(hold_q).item()
            
            # 분산이 클수록 신뢰도 높음 (명확한 선택)
            confidence = (pos_var + lev_var + hold_var) / 3
            
            return min(confidence, 1.0)


if __name__ == "__main__":
    # 예시 실행
    main_example()