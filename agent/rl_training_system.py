"""
80차원 Signal 기반 강화학습 트레이딩 AI 훈련 시스템 - Part 1
- Signal의 모든 indicator와 raw score 활용
- 중복 계산 제거 및 정보 활용 극대화
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
import os
import warnings
from pathlib import Path

# PyTorch 호환성 설정
def setup_pytorch_compatibility():
    """PyTorch 버전 호환성 설정"""
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
        print("PyTorch 호환 설정 완료")
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
                        action: str, holding_time: int, signal_data: Dict = None,
                        trade_pnl: Optional[float] = None) -> float:
        """Signal 정보를 활용한 보상 계산"""
        reward = 0.0
        
        # 1. 포지션 보유 중 실시간 평가
        if abs(position) > 0.01:
            unrealized_pnl = self._calculate_unrealized_pnl(current_price, entry_price, position)
            
            if unrealized_pnl > 0:
                reward += min(unrealized_pnl * 10, 1.0)
            else:
                reward += max(unrealized_pnl * 15, -2.0)
            
            # 홀딩 시간 최적화
            if holding_time > 30:
                reward -= 0.1 * (holding_time - 30) / 30
        
        # 2. Signal 기반 추가 보상
        if signal_data:
            signal_reward = self._calculate_signal_reward(signal_data, position)
            reward += signal_reward
        
        # 3. 거래 완료시 승률 중심 평가
        if trade_pnl is not None:
            self.recent_trades.append(1 if trade_pnl > 0 else 0)
            current_win_rate = np.mean(self.recent_trades) if self.recent_trades else 0.5
            
            if trade_pnl > 0:
                reward += 5.0  # 승률 향상을 위한 큰 보상
                if current_win_rate > 0.6:
                    reward += 2.0  # 연속 승률 보너스
            else:
                reward -= 3.0
        
        return reward
    
    def _calculate_unrealized_pnl(self, current_price: float, entry_price: float, position: float) -> float:
        """미실현 손익 계산"""
        if entry_price <= 0:
            return 0.0
        
        price_change = (current_price - entry_price) / entry_price
        return position * price_change
    
    def _calculate_signal_reward(self, signal_data: Dict, position: float) -> float:
        """Signal 데이터 기반 추가 보상"""
        signal_reward = 0.0
        
        # 각 시간대별 신호와 포지션 일치도
        for timeframe in ['short_term', 'medium_term', 'long_term']:
            action = signal_data.get(f'{timeframe}_action', 'HOLD')
            net_score = float(signal_data.get(f'{timeframe}_net_score', 0.0))
            confidence = signal_data.get(f'{timeframe}_confidence', 'LOW')
            
            confidence_weight = 1.0 if confidence == 'HIGH' else (0.5 if confidence == 'MEDIUM' else 0.1)
            
            if action == 'LONG' and position > 0:
                signal_reward += abs(net_score) * confidence_weight * 0.5
            elif action == 'SHORT' and position < 0:
                signal_reward += abs(net_score) * confidence_weight * 0.5
            elif action == 'HOLD' and abs(position) < 0.1:
                signal_reward += 0.1 * confidence_weight
        
        return signal_reward

class DuelingDQN(nn.Module):
    """80차원 입력을 위한 Dueling DQN 네트워크"""
    
    def __init__(self, state_size: int, action_size: int = 3, hidden_size: int = 256):
        super().__init__()
        
        self.input_norm = nn.LayerNorm(state_size)
        
        # 공통 특성 추출 (80차원 입력 처리)
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_size)
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Advantage streams (각 액션 차원별)
        self.position_advantage = self._create_advantage_head(hidden_size, 21)  # -2.0~2.0
        self.leverage_advantage = self._create_advantage_head(hidden_size, 20)  # 1~20
        self.holding_advantage = self._create_advantage_head(hidden_size, 48)   # 30~1440분
        
        self.apply(self._init_weights)
    
    def _create_advantage_head(self, hidden_size: int, output_size: int) -> nn.Sequential:
        """Advantage head 생성"""
        return nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def _init_weights(self, module):
        """가중치 초기화"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        x = self.input_norm(x)
        features = self.feature_extractor(x)
        
        value = self.value_stream(features)
        
        pos_adv = self.position_advantage(features)
        lev_adv = self.leverage_advantage(features)
        hold_adv = self.holding_advantage(features)
        
        # Dueling 결합
        pos_q = value + pos_adv - pos_adv.mean(dim=1, keepdim=True)
        lev_q = value + lev_adv - lev_adv.mean(dim=1, keepdim=True)
        hold_q = value + hold_adv - hold_adv.mean(dim=1, keepdim=True)
        
        return pos_q, lev_q, hold_q

class TradingEnvironment(gym.Env):
    """80차원 Signal 기반 암호화폐 거래 강화학습 환경"""
    
    def __init__(self, price_data: pd.DataFrame, signal_data: List[Dict], 
                 initial_balance: float = 10000.0, max_position: float = 1.0):
        super().__init__()
        
        self.price_data = price_data
        self.signal_data = signal_data
        self.initial_balance = initial_balance
        self.max_position = max_position
        
        self.reward_calculator = RewardCalculator()
        
        # 액션/상태 스페이스 정의
        self.action_space = spaces.Box(
            low=np.array([-2.0, 1.0, 0.0]), 
            high=np.array([2.0, 20.0, 1440.0]), 
            dtype=np.float32
        )
        
        # 80차원 상태 공간
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(80,),  # 20(가격) + 25(기술점수) + 25(결정) + 10(포트폴리오)
            dtype=np.float32
        )
        
        self.reset()
    
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
        self.holding_time = 0
        self.in_position = False
        self.last_trade_pnl = None
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """환경 스텝 실행"""
        if self.current_step >= min(len(self.price_data), len(self.signal_data)) - 1:
            return self._get_observation(), 0.0, True, {}
        
        position_change = np.clip(action[0], -2.0, 2.0)
        leverage = np.clip(action[1], 1.0, 20.0)
        target_holding_minutes = np.clip(action[2], 1.0, 1440.0)
        
        current_price = self.price_data.iloc[self.current_step]['close']
        next_price = self.price_data.iloc[self.current_step + 1]['close']
        
        # 포지션 및 거래 처리
        trade_completed, old_position = self._process_position_change(
            position_change, leverage, current_price, target_holding_minutes
        )
        
        # Signal 데이터 가져오기
        current_signal = self.signal_data[self.current_step] if self.current_step < len(self.signal_data) else {}
        
        # 보상 계산 (Signal 정보 활용)
        reward = self.reward_calculator.calculate_reward(
            current_price=next_price,
            entry_price=self.entry_price,
            position=self.current_position,
            action='TRADE' if abs(position_change) > 0.1 else 'HOLD',
            holding_time=self.holding_time,
            signal_data=current_signal,
            trade_pnl=self.last_trade_pnl if trade_completed else None
        )
        
        # 다음 스텝으로 이동
        self.current_step += 1
        self.holding_time += 3
        
        # 홀딩 시간 초과시 강제 청산
        if self.in_position and self.holding_time >= target_holding_minutes:
            self._close_position(next_price)
        
        done = (self.current_step >= min(len(self.price_data), len(self.signal_data)) - 1 or 
                self.balance <= self.initial_balance * 0.1)
        
        info = self._create_info_dict()
        
        return self._get_observation(), reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """80차원 상태 관찰값 반환"""
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
        price_features = self._extract_price_indicators(current_signal, current_candle)
        
        # 2. Technical Scores (25차원)  
        technical_features = self._extract_technical_scores(current_signal)
        
        # 3. Decision Features (25차원)
        decision_features = self._extract_decision_features(current_signal)
        
        # 4. Portfolio Features (10차원)
        portfolio_features = self._get_portfolio_state()
        
        return np.concatenate([price_features, technical_features, decision_features, portfolio_features]).astype(np.float32)
    
    def _extract_price_indicators(self, signal_data: Dict, current_candle: Dict) -> np.ndarray:
        """Signal의 indicator들을 price feature로 활용 (20차원)"""
        features = []
        current_price = current_candle['close']
        
        # 1. 가격 대비 지표 위치
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
        
        # 3. 일별 기준점들
        prev_high = signal_data.get('indicator_prev_day_high', current_price)
        prev_low = signal_data.get('indicator_prev_day_low', current_price)
        or_high = signal_data.get('indicator_opening_range_high', current_price)
        or_low = signal_data.get('indicator_opening_range_low', current_price)
        
        prev_range = prev_high - prev_low
        prev_day_position = (current_price - prev_low) / prev_range if prev_range > 0 else 0.5
            
        or_range = or_high - or_low  
        or_position = (current_price - or_low) / or_range if or_range > 0 else 0.5
        
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
    
    def _extract_technical_scores(self, signals: Dict) -> np.ndarray:
        """각 전략의 raw score들 (25차원)"""
        features = []
        
        all_raw_scores = []
        for key, value in signals.items():
            if '_raw_' in key and '_score' in key and value is not None:
                try:
                    all_raw_scores.append(float(value))
                except:
                    all_raw_scores.append(0.0)
        
        if len(all_raw_scores) >= 25:
            sorted_scores = sorted(all_raw_scores, key=abs, reverse=True)
            features = sorted_scores[:25]
        else:
            features = all_raw_scores + [0.0] * (25 - len(all_raw_scores))
        
        return np.array(features, dtype=np.float32)
    
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
        for timeframe in ['short_term', 'medium_term', 'long_term']:
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
    
    
    def _process_position_change(self, position_change: float, leverage: float, 
                               current_price: float, target_holding_minutes: float) -> Tuple[bool, float]:
        """포지션 변경 처리"""
        old_position = self.current_position
        new_position = np.clip(self.current_position + position_change, -1.0, 1.0)
        trade_completed = False
        
        if abs(new_position - self.current_position) > 0.01:
            # 기존 포지션 청산
            if abs(self.current_position) > 0.01:
                trade_completed = True
                self.last_trade_pnl = self._calculate_trade_pnl(current_price, self.entry_price, old_position)
                self._close_position(current_price)
            
            # 새 포지션 진입
            if abs(new_position) > 0.01:
                self.current_position = new_position
                self.current_leverage = leverage
                self.entry_price = current_price
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
        pnl_usd = pnl * self.current_leverage * self.balance
        
        # 거래 수수료 차감
        fee = abs(pnl_usd) * 0.001
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
            'trade_completed': hasattr(self, 'last_trade_pnl') and self.last_trade_pnl is not None,
            'trade_pnl': self.last_trade_pnl if hasattr(self, 'last_trade_pnl') else None
        }

"""
80차원 Signal 기반 강화학습 트레이딩 AI 훈련 시스템 - Part 2
- RLAgent 클래스 및 훈련/평가 시스템
- 데이터 로더 및 유틸리티 함수들
"""

class RLAgent:
    """80차원 Signal 기반 강화학습 에이전트"""
    
    def __init__(self, state_size: int = 80, learning_rate: float = 5e-5, 
                 gamma: float = 0.995, epsilon: float = 0.9, epsilon_decay: float = 0.9995):
        
        self.state_size = state_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.05
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device} for {state_size}차원 모델")
        
        # 네트워크 초기화
        self.q_network = DuelingDQN(state_size, 3).to(self.device)
        self.target_network = DuelingDQN(state_size, 3).to(self.device)
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
            return self._get_random_action()
        
        return self._get_greedy_action(state)
    
    def _get_random_action(self) -> np.ndarray:
        """스마트한 랜덤 액션"""
        return np.array([
            np.random.uniform(-1.0, 1.0),
            np.random.uniform(1.0, 5.0),
            np.random.uniform(30.0, 180.0)
        ])
    
    def _get_greedy_action(self, state: np.ndarray) -> np.ndarray:
        """Q값 기반 탐욕적 액션 선택"""
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
        
        # 타겟 네트워크 업데이트
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.update_target_network()
    
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
        pos_loss = F.smooth_l1_loss(current_position_q, target_position_q)
        lev_loss = F.smooth_l1_loss(current_leverage_q, target_leverage_q)
        hold_loss = F.smooth_l1_loss(current_holding_q, target_holding_q)
        
        return pos_loss + lev_loss + hold_loss
    
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
            print(f"80차원 모델 저장 완료: {filepath}")
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
            
            print(f"✅ 80차원 모델 로드 성공! 엡실론: {self.epsilon:.3f}")
            return True
            
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            return False

    def load_model_with_compatibility(self, filepath: str) -> bool:
        """호환성을 고려한 80차원 모델 로드"""
        if not os.path.exists(filepath):
            print(f"모델 파일이 없습니다: {filepath}")
            return False
        
        try:
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
            
            # 상태 크기 확인
            model_state_size = checkpoint.get('state_size', 60)
            if model_state_size != 80:
                print(f"❌ 이 모델은 {model_state_size}차원입니다. 80차원 모델이 필요합니다.")
                return False
            
            # state_dict 키 이름 변환 (필요시)
            state_dict = checkpoint['q_network']
            converted_state_dict = {}
            
            for key, value in state_dict.items():
                # feature_extraction -> feature_extractor 변환
                new_key = key.replace('feature_extraction', 'feature_extractor')
                converted_state_dict[new_key] = value
            
            # 변환된 state_dict로 로드
            try:
                self.q_network.load_state_dict(converted_state_dict)
            except:
                self.q_network.load_state_dict(state_dict)  # 원본으로 시도
            
            # target_network도 동일하게 처리
            target_state_dict = checkpoint['target_network']
            converted_target_dict = {}
            for key, value in target_state_dict.items():
                new_key = key.replace('feature_extraction', 'feature_extractor')
                converted_target_dict[new_key] = value
            
            try:
                self.target_network.load_state_dict(converted_target_dict)
            except:
                self.target_network.load_state_dict(target_state_dict)
            
            # 나머지 파라미터들
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            
            print(f"✅ 80차원 호환성 모델 로드 성공!")
            return True
            
        except Exception as e:
            print(f"호환성 모델 로드 실패: {e}")
            return False

class DataLoader:
    """80차원 Signal 기반 데이터 로딩 클래스"""
    
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
        """80차원용 Signal 데이터 로드"""
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
        """Parquet을 Signal Dict 리스트로 변환 (80차원용)"""
        signal_data = []
        
        print("80차원용 Signal 데이터 변환 중...")
        
        for idx, row in signal_df.iterrows():
            # 각 행을 딕셔너리로 변환 (Flatten 형태 유지)
            signal_dict = {}
            
            for col, value in row.items():
                if pd.notna(value):
                    signal_dict[col] = value
                else:
                    signal_dict[col] = 0 if 'score' in col else ('HOLD' if 'action' in col else 'LOW' if 'confidence' in col else value)
            
            signal_data.append(signal_dict)
            
            if (idx + 1) % 5000 == 0:
                print(f"   변환 진행: {idx + 1:,}/{len(signal_df):,}")
        
        print(f"80차원용 Signal 데이터 변환 완료: {len(signal_data):,}개")
        return signal_data
    
    @staticmethod
    def generate_enhanced_signals(length: int) -> List[Dict]:
        """80차원용 향상된 기본 신호 데이터 생성"""
        print(f"80차원용 향상된 신호 데이터 생성 중: {length:,}개")
        
        signal_data = []
        for i in range(length):
            # 더 현실적인 신호 생성
            market_phase = i % 100  # 시장 사이클
            
            if market_phase < 30:  # 상승 구간
                short_action = 'LONG'
                short_score = 0.3 + (market_phase / 30) * 0.5
                short_conf = 'HIGH' if short_score > 0.6 else 'MEDIUM'
            elif market_phase < 70:  # 횡보 구간
                short_action = 'HOLD'
                short_score = 0.1
                short_conf = 'LOW'
            else:  # 하락 구간
                short_action = 'SHORT'
                short_score = 0.3 + ((market_phase - 70) / 30) * 0.5
                short_conf = 'HIGH' if short_score > 0.6 else 'MEDIUM'
            
            # 다양한 indicator 값들
            base_price = 2300 + np.sin(i * 0.01) * 100
            
            signal_dict = {
                # 시간대별 액션
                'short_term_action': short_action,
                'short_term_net_score': short_score if short_action != 'HOLD' else 0.0,
                'short_term_buy_score': short_score if short_action == 'LONG' else 0.0,
                'short_term_sell_score': short_score if short_action == 'SHORT' else 0.0,
                'short_term_confidence': short_conf,
                'short_term_leverage': min(int(short_score * 5) + 1, 5),
                'short_term_signals_used': 3,
                'short_term_max_holding_minutes': 60,
                
                'medium_term_action': 'HOLD',
                'medium_term_net_score': 0.0,
                'medium_term_buy_score': 0.0,
                'medium_term_sell_score': 0.0,
                'medium_term_confidence': 'LOW',
                'medium_term_leverage': 1,
                'medium_term_signals_used': 2,
                'medium_term_max_holding_minutes': 240,
                
                'long_term_action': 'HOLD',
                'long_term_net_score': 0.0,
                'long_term_buy_score': 0.0,
                'long_term_sell_score': 0.0,
                'long_term_confidence': 'LOW',
                'long_term_leverage': 1,
                'long_term_signals_used': 2,
                'long_term_max_holding_minutes': 1440,
                
                # Indicator들
                'indicator_vwap': base_price * (1 + np.random.normal(0, 0.01)),
                'indicator_atr': base_price * 0.02 * (1 + np.random.normal(0, 0.5)),
                'indicator_poc': base_price * (1 + np.random.normal(0, 0.01)),
                'indicator_hvn': base_price * (1 + np.random.normal(0, 0.01)),
                'indicator_lvn': base_price * (1 + np.random.normal(0, 0.01)),
                'indicator_vwap_std': base_price * 0.01,
                'indicator_prev_day_high': base_price * 1.02,
                'indicator_prev_day_low': base_price * 0.98,
                'indicator_opening_range_high': base_price * 1.005,
                'indicator_opening_range_low': base_price * 0.995,
                
                # Raw scores (다양한 전략들)
                'short_term_raw_vwap_pinball_score': np.random.uniform(-0.5, 0.5),
                'short_term_raw_zscore_mean_reversion_score': np.random.uniform(-0.5, 0.5),
                'short_term_raw_session_score': np.random.uniform(-0.5, 0.5),
                'short_term_raw_vol_spike_score': np.random.uniform(-0.3, 0.3),
                'short_term_raw_orderflow_cvd_score': np.random.uniform(-0.3, 0.3),
                'short_term_raw_liquidity_grab_score': np.random.uniform(-0.3, 0.3),
                
                'medium_term_raw_bollinger_squeeze_score': np.random.uniform(-0.3, 0.3),
                'medium_term_raw_support_resistance_score': np.random.uniform(-0.5, 0.5),
                'medium_term_raw_htf_trend_score': np.random.uniform(-0.3, 0.3),
                'medium_term_raw_ema_confluence_score': np.random.uniform(-0.5, 0.5),
                'medium_term_raw_multi_timeframe_score': np.random.uniform(-0.3, 0.3),
                
                'long_term_raw_vpvr_score': np.random.uniform(-0.5, 0.5),
                'long_term_raw_oi_delta_score': np.random.uniform(-0.3, 0.3),
                'long_term_raw_funding_rate_score': np.random.uniform(-0.3, 0.3),
                'long_term_raw_ichimoku_score': np.random.uniform(-0.5, 0.5),
                
                # 메타 정보
                'short_term_market_context': 'RANGING' if market_phase < 70 else 'TRENDING',
                'medium_term_market_context': 'NEUTRAL',
                'long_term_market_context': 'NEUTRAL',
                'long_term_institutional_bias': 'BULLISH' if market_phase < 40 else ('BEARISH' if market_phase > 60 else 'NEUTRAL'),
                'long_term_macro_trend_strength': 'MEDIUM',
                
                'timestamp': int(datetime.now().timestamp() * 1000) + i * 180000  # 3분 간격
            }
            
            signal_data.append(signal_dict)
        
        print("80차원용 향상된 신호 데이터 생성 완료")
        return signal_data

class PerformanceAnalyzer:
    """80차원 Signal 기반 성능 분석 클래스"""
    
    @staticmethod
    def evaluate_agent(agent: RLAgent, env: TradingEnvironment, num_episodes: int = 10) -> Tuple[List[Dict], Dict]:
        """80차원 에이전트 성능 평가"""
        print(f"80차원 에이전트 성능 평가 중 ({num_episodes} 에피소드)...")
        
        original_epsilon = agent.epsilon
        agent.epsilon = 0.0
        
        results = []
        all_trades = []
        
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            episode_trades = []
            episode_balance = env.initial_balance
            
            for step in range(500):
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
        """80차원 성능 리포트 출력"""
        print("\n" + "="*60)
        print(f"80차원 Signal 기반 성능 평가 결과")
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
        """성능 등급 산출"""
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
    def _get_recommendations(stats: Dict) -> List[str]:
        """성능 기반 개선 제안"""
        recommendations = []
        
        if stats['overall_win_rate'] < 0.55:
            recommendations.append("승률이 낮습니다. Signal 특성을 더 활용한 보상 함수 개선이 필요합니다.")
        
        if stats['avg_return'] < 0.02:
            recommendations.append("수익률이 낮습니다. 80차원 상태 공간의 장점을 더 활용하세요.")
        
        if stats['avg_max_drawdown'] > 0.2:
            recommendations.append("최대 낙폭이 큽니다. Signal 기반 리스크 관리를 강화하세요.")
        
        if stats['consistency'] < 0.5:
            recommendations.append("성과 일관성이 떨어집니다. 더 많은 훈련과 Signal 품질 개선이 필요합니다.")
        
        if stats['avg_trades_per_episode'] < 3:
            recommendations.append("거래 빈도가 낮습니다. Signal 감도를 조정해보세요.")
        
        if not recommendations:
            recommendations.append("80차원 Signal 기반 시스템이 잘 작동하고 있습니다!")
        
        return recommendations

class TrainingManager:
    """80차원 Signal 기반 훈련 관리 클래스"""
    
    @staticmethod
    def train_agent(agent: RLAgent, env: TradingEnvironment, 
                   episodes: int = 500, save_interval: int = 100) -> Tuple[RLAgent, List[float], List[float]]:
        """80차원 Signal 기반 에이전트 훈련"""
        print(f"80차원 Signal 기반 강화학습 훈련 시작 ({episodes} 에피소드)")
        print(f"상태 공간: {env.observation_space.shape[0]}차원")
        
        episode_rewards = []
        episode_win_rates = []
        best_win_rate = 0.0
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            episode_trades = []
            steps = 0
            
            while steps < 500:
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                
                agent.remember(state, action, reward, next_state, done)
                
                if info.get('trade_completed', False):
                    trade_pnl = info.get('trade_pnl', 0.0)
                    episode_trades.append(1 if trade_pnl > 0 else 0)
                
                state = next_state
                total_reward += reward
                steps += 1
                
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
                        f"ε: {agent.epsilon:.3f} | "
                        f"80D")
            
            # 베스트 모델 저장
            if episode % save_interval == 0 and episode > 0:
                current_avg_win_rate = np.mean(episode_win_rates[-100:]) if len(episode_win_rates) >= 100 else np.mean(episode_win_rates)
                
                if current_avg_win_rate > best_win_rate:
                    best_win_rate = current_avg_win_rate
                    agent.save_model(f'best_model_80d_ep{episode}_wr{current_avg_win_rate:.3f}.pth')
                    print(f"🎯 새로운 80차원 최고 성능! 승률: {current_avg_win_rate:.3f}")
            
            # 조기 종료 조건
            if episode > 200:
                recent_100_win_rate = np.mean(episode_win_rates[-100:])
                if recent_100_win_rate >= 0.65:
                    print(f"🏆 80차원 목표 달성! 승률 {recent_100_win_rate:.3f} 도달")
                    agent.save_model('agent/final_optimized_model_80d.pth')
                    break
        
        print(f"\n80차원 Signal 기반 훈련 완료!")
        print(f"   총 에피소드: {episode + 1}")
        print(f"   최고 승률: {best_win_rate:.3f}")
        print(f"   최종 승률: {np.mean(episode_win_rates[-50:]) if episode_win_rates else 0:.3f}")
        print(f"   상태 차원: 80차원 (Signal 기반)")
        
        return agent, episode_rewards, episode_win_rates

def main():
    """80차원 Signal 기반 메인 실행 함수"""
    print("80차원 Signal 기반 강화학습 트레이딩 시스템")
    print("=" * 80)
    
    try:
        # 1. 데이터 로딩
        print("\n1️⃣ 80차원용 데이터 로딩...")
        price_data = DataLoader.load_price_data()
        if price_data is None:
            print("가격 데이터 로드 실패")
            return
        
        signal_data = DataLoader.load_signal_data()
        if signal_data is None:
            signal_data = DataLoader.generate_enhanced_signals(min(len(price_data), 10000))
        
        # 데이터 길이 맞추기
        min_length = min(len(price_data), len(signal_data))
        price_data = price_data.iloc[:min_length].reset_index(drop=True)
        signal_data = signal_data[:min_length]
        
        print(f"최종 80차원용 데이터 준비 완료: {min_length:,}개")
        
        # 2. 환경 및 에이전트 생성
        print("\n2️⃣ 80차원 환경 및 에이전트 생성...")
        env = TradingEnvironment(price_data, signal_data)
        agent = RLAgent(env.observation_space.shape[0])  # 80차원
        
        print(f"상태 공간: {env.observation_space.shape[0]}차원")
        print("Signal의 모든 indicator와 raw score 활용")
        
        # 기존 80차원 모델 로드 시도
        model_files = ['agent/final_optimized_model_80d.pth', 'agent/best_model_80d.pth']
        model_loaded = False
        
        for model_file in model_files:
            if agent.load_model(model_file):
                model_loaded = True
                break
        
        if not model_loaded:
            print("새로운 80차원 모델로 시작합니다.")
        
        # 3. 현재 성능 평가
        print("\n3️⃣ 80차원 모델 현재 성능 평가...")
        results, stats = PerformanceAnalyzer.evaluate_agent(agent, env, num_episodes=5)
        PerformanceAnalyzer.print_performance_report(results, stats)
        
        # 4. 훈련 여부 결정
        if stats['overall_win_rate'] < 0.55 or not model_loaded:
            print(f"\n4️⃣ 80차원 Signal 기반 훈련 시작...")
            print(f"   현재 승률: {stats['overall_win_rate']:.3f}")
            print(f"   목표 승률: 0.65+")
            print(f"   Signal 특성 활용: 최대화")
            
            # 훈련 실행
            trained_agent, rewards, win_rates = TrainingManager.train_agent(agent, env, episodes=500)
            
            # 훈련 후 성능 재평가
            print("\n5️⃣ 80차원 모델 훈련 후 성능 평가...")
            final_results, final_stats = PerformanceAnalyzer.evaluate_agent(trained_agent, env, num_episodes=10)
            PerformanceAnalyzer.print_performance_report(final_results, final_stats)
            
            # 개선도 출력
            improvement = final_stats['overall_win_rate'] - stats['overall_win_rate']
            print(f"\n🚀 80차원 Signal 기반 성능 개선도:")
            print(f"   승률: {stats['overall_win_rate']:.3f} → {final_stats['overall_win_rate']:.3f} ({improvement:+.3f})")
            print(f"   평균 수익률: {stats['avg_return']:.3f} → {final_stats['avg_return']:.3f}")
            print(f"   Signal 활용도: 최대화됨")
            
            # 최종 모델 저장
            trained_agent.save_model('agent/final_optimized_model_80d.pth')
            
        else:
            print(f"80차원 모델 성능이 양호합니다 (승률: {stats['overall_win_rate']:.3f})")
            
            user_input = input("\n추가 80차원 훈련을 원하시나요? (y/n): ")
            if user_input.lower() == 'y':
                print("80차원 추가 훈련 시작...")
                TrainingManager.train_agent(agent, env, episodes=200)
    
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()