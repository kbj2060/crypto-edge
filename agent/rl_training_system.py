"""
61차원 RL Decision 기반 강화학습 트레이딩 AI 훈련 시스템 - Part 1
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
import gym
import os

from collections import deque, namedtuple
from gym import spaces
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
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
        """Signal 데이터 기반 추가 보상 - 새로운 RL 스키마 기반"""
        signal_reward = 0.0
        
        # 각 시간대별 신호와 포지션 일치도
        for timeframe in ['short_term', 'medium_term', 'long_term']:
            action_value = float(signal_data.get(f'{timeframe}_action_value', 0.0))
            net_score = float(signal_data.get(f'{timeframe}_net_score', 0.0))
            confidence_value = float(signal_data.get(f'{timeframe}_confidence_value', 0.0))
            
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
    """61차원 입력을 위한 Dueling DQN 네트워크"""
    
    def __init__(self, state_size: int, action_size: int = 3, hidden_size: int = 256):
        super().__init__()
        
        self.input_norm = nn.LayerNorm(state_size)
        
        # 공통 특성 추출 (61차원 입력 처리)
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
    """61차원 RL Decision 기반 암호화폐 거래 강화학습 환경"""
    
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
        
        # 61차원 상태 공간
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(61,),  # 20(가격) + 6(기술점수) + 26(결정) + 9(포트폴리오) = 61차원
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
        """61차원 상태 관찰값 반환"""
        if self.current_step >= min(len(self.price_data), len(self.signal_data)):
            return np.zeros(61, dtype=np.float32)
        
        # Signal과 현재 캔들 데이터
        current_signal = self.signal_data[self.current_step]
        current_candle = {
            'open': self.price_data.iloc[self.current_step]['open'],
            'high': self.price_data.iloc[self.current_step]['high'],
            'low': self.price_data.iloc[self.current_step]['low'],
            'close': self.price_data.iloc[self.current_step]['close'],
            'quote_volume': self.price_data.iloc[self.current_step]['quote_volume'],
        }
        
        # 1. Price Indicators (20차원)
        price_features = self._extract_price_indicators(current_signal, current_candle)
        # 2. Technical Scores (6차원)  
        technical_features = self._extract_technical_scores(current_signal)
        # 3. Decision Features (26차원)
        decision_features = self._extract_decision_features(current_signal)
        # 4. Portfolio Features (9차원)
        portfolio_features = self._get_portfolio_state()
        return np.concatenate([price_features, technical_features, decision_features, portfolio_features]).astype(np.float32)
    
    def _extract_price_indicators(self, signal_data: Dict, current_candle: Dict) -> np.ndarray:
        """Signal의 indicator들을 price feature로 활용 (20차원)"""
        current_price = current_candle['close']
        
        # 1. 가격 대비 지표 위치
        vwap = signal_data.get('indicator_vwap', current_price)
        poc = signal_data.get('indicator_poc', current_price)  
        hvn = signal_data.get('indicator_hvn', current_price)
        lvn = signal_data.get('indicator_lvn', current_price)
        
        # 2. 변동성 지표들
        atr = signal_data.get('indicator_atr', 0.0)
        vwap_std = signal_data.get('indicator_vwap_std', 0.0)
        
        # 3. 일별 기준점들
        prev_high = signal_data.get('indicator_prev_day_high', current_price)
        prev_low = signal_data.get('indicator_prev_day_low', current_price)
        or_high = signal_data.get('indicator_opening_range_high', current_price)
        or_low = signal_data.get('indicator_opening_range_low', current_price)
        
        prev_range = prev_high - prev_low
        prev_day_position = (current_price - prev_low) / prev_range if prev_range > 0 else 0.5
            
        or_range = or_high - or_low  
        or_position = (current_price - or_low) / or_range if or_range > 0 else 0.5
        
        # 4. 현재 캔들 정보
        high, low, close, open_price = current_candle['high'], current_candle['low'], current_candle['close'], current_candle['open']
        quote_volume = current_candle.get('quote_volume', 0)
        
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
    
    def _extract_technical_scores(self, signals: Dict) -> np.ndarray:
        """각 전략의 raw score들 (25차원) - 새로운 RL 스키마 기반"""
        # 새로운 RL 스키마에서 사용 가능한 점수들 수집
        score_fields = []
        
        # 각 시간대별 점수들
        # for timeframe in ['short_term', 'medium_term', 'long_term']:
        #     score_fields.extend([
        #         f'{timeframe}_net_score',
        #         f'{timeframe}_buy_score', 
        #         f'{timeframe}_sell_score',
        #         f'{timeframe}_confidence',
        #         f'{timeframe}_market_context'
        #     ])
        
        # Conflict 관련 점수들 (중복 제거 - Decision Features에서 처리)
        # score_fields.extend([
        #     'conflict_conflict_severity',
        #     'conflict_directional_consensus',
        #     'conflict_conflict_penalty',
        #     'conflict_consensus_bonus',
        #     'conflict_diversity_bonus'
        # ])
        
        # Indicator 관련 점수들
        indicator_fields = [
            'indicator_vwap', 'indicator_atr', 'indicator_poc', 
            'indicator_hvn', 'indicator_lvn', 'indicator_vwap_std'
        ]
        
        # 수집된 점수들 정규화
        all_scores = []
        for field in score_fields + indicator_fields:
            value = signals.get(field)
            try:
                score = float(value)
                # 정규화 (대부분 0~1 범위로 가정)
                if 'indicator_' in field:
                    # Indicator는 가격 대비 비율로 정규화
                    score = min(abs(score) / 1000.0, 1.0)  # 가격 대비 0.1% 단위
                all_scores.append(score)
            except:
                all_scores.append(0.0)
        
        # 6차원으로 맞추기 (Indicator만 사용)
        if len(all_scores) >= 6:
            return np.array(all_scores[:6], dtype=np.float32)
        else:
            return np.array(all_scores + [0.0] * (6 - len(all_scores)), dtype=np.float32)
    
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
61차원 RL Decision 기반 강화학습 트레이딩 AI 훈련 시스템 - Part 2
- RLAgent 클래스 및 훈련/평가 시스템
- 새로운 Decision 스키마 데이터 로더 및 유틸리티 함수들
"""

class RLAgent:
    """61차원 RL Decision 기반 강화학습 에이전트"""
    
    def __init__(self, state_size: int = 61, learning_rate: float = 5e-5, 
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
            print(f"61차원 모델 저장 완료: {filepath}")
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
            
            print(f"✅ 61차원 모델 로드 성공! 엡실론: {self.epsilon:.3f}")
            return True
            
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            return False

    def load_model_with_compatibility(self, filepath: str) -> bool:
        """호환성을 고려한 61차원 모델 로드"""
        if not os.path.exists(filepath):
            print(f"모델 파일이 없습니다: {filepath}")
            return False
        
        try:
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
            
            # 상태 크기 확인
            model_state_size = checkpoint.get('state_size', 60)
            if model_state_size != 61:
                print(f"❌ 이 모델은 {model_state_size}차원입니다. 61차원 모델이 필요합니다.")
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
            
            print(f"✅ 61차원 호환성 모델 로드 성공!")
            return True
            
        except Exception as e:
            print(f"호환성 모델 로드 실패: {e}")
            return False

class DataLoader:
    """61차원 RL Decision 기반 데이터 로딩 클래스"""
    
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
        """61차원용 RL Decision 데이터 로드"""
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
        """Parquet을 Signal Dict 리스트로 변환 (61차원용) - 새로운 RL 스키마"""
        signal_data = []
        
        print("61차원용 RL 스키마 Signal 데이터 변환 중...")
        
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
        
        print(f"61차원용 RL 스키마 Signal 데이터 변환 완료: {len(signal_data):,}개")
        return signal_data
    
    @staticmethod
    def generate_enhanced_signals(length: int) -> List[Dict]:
        """61차원용 향상된 기본 신호 데이터 생성"""
        print(f"61차원용 향상된 신호 데이터 생성 중: {length:,}개")
        
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
                # 시간대별 액션 (새로운 RL 스키마)
                'short_term_action_value': 1.0 if short_action == 'LONG' else (-1.0 if short_action == 'SHORT' else 0.0),
                'short_term_net_score': short_score if short_action != 'HOLD' else 0.0,
                'short_term_buy_score': short_score if short_action == 'LONG' else 0.0,
                'short_term_sell_score': short_score if short_action == 'SHORT' else 0.0,
                'short_term_confidence_value': 1.0 if short_conf == 'HIGH' else (0.5 if short_conf == 'MEDIUM' else 0.0),
                'short_term_market_context_value': 1.0 if market_phase < 70 else 0.0,
                'short_term_leverage': min(int(short_score * 5) + 1, 5),
                'short_term_signals_used': 3,
                'short_term_max_holding_minutes': 60,
                
                'medium_term_action_value': 0.0,
                'medium_term_net_score': 0.0,
                'medium_term_buy_score': 0.0,
                'medium_term_sell_score': 0.0,
                'medium_term_confidence_value': 0.0,
                'medium_term_market_context_value': 0.5,
                'medium_term_leverage': 1,
                'medium_term_signals_used': 2,
                'medium_term_max_holding_minutes': 240,
                
                'long_term_action_value': 0.0,
                'long_term_net_score': 0.0,
                'long_term_buy_score': 0.0,
                'long_term_sell_score': 0.0,
                'long_term_confidence_value': 0.0,
                'long_term_market_context_value': 0.5,
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
                
                # Conflict 정보 (새로운 RL 스키마)
                'conflict_conflict_severity': 0.0,
                'conflict_directional_consensus': 1.0 if short_action != 'HOLD' else 0.5,
                'conflict_conflict_penalty': 0.0,
                'conflict_consensus_bonus': 0.2 if short_action != 'HOLD' else 0.0,
                'conflict_diversity_bonus': 0.1,
                'conflict_active_categories': 1 if short_action != 'HOLD' else 0,
                'conflict_hold_ratio': 0.67 if short_action == 'HOLD' else 0.33,
                'conflict_max_leverage_used': min(int(short_score * 5) + 1, 5),
                'conflict_total_exposure': 1 if short_action != 'HOLD' else 0,
                'conflict_timeframe_diversity': 1,
                'conflict_long_count': 1 if short_action == 'LONG' else 0,
                'conflict_short_count': 1 if short_action == 'SHORT' else 0,
                'conflict_hold_count': 1 if short_action == 'HOLD' else 0,
                'conflict_risk_penalty': 0.0,
                
                # 메타 정보 (수치화)
                'long_term_institutional_bias': 1.0 if market_phase < 40 else (-1.0 if market_phase > 60 else 0.0),
                'long_term_macro_trend_strength': 0.5,
                
                'timestamp': int(datetime.now().timestamp() * 1000) + i * 180000  # 3분 간격
            }
            
            signal_data.append(signal_dict)
        
        print("61차원용 향상된 신호 데이터 생성 완료")
        return signal_data

class PerformanceAnalyzer:
    """61차원 RL Decision 기반 성능 분석 클래스"""
    
    @staticmethod
    def evaluate_agent(agent: RLAgent, env: TradingEnvironment, num_episodes: int = 10) -> Tuple[List[Dict], Dict]:
        """61차원 에이전트 성능 평가"""
        print(f"61차원 에이전트 성능 평가 중 ({num_episodes} 에피소드)...")
        
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
        """61차원 성능 리포트 출력"""
        print("\n" + "="*60)
        print(f"61차원 RL Decision 기반 성능 평가 결과")
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
            recommendations.append("수익률이 낮습니다. 61차원 상태 공간의 장점을 더 활용하세요.")
        
        if stats['avg_max_drawdown'] > 0.2:
            recommendations.append("최대 낙폭이 큽니다. Signal 기반 리스크 관리를 강화하세요.")
        
        if stats['consistency'] < 0.5:
            recommendations.append("성과 일관성이 떨어집니다. 더 많은 훈련과 Signal 품질 개선이 필요합니다.")
        
        if stats['avg_trades_per_episode'] < 3:
            recommendations.append("거래 빈도가 낮습니다. Signal 감도를 조정해보세요.")
        
        if not recommendations:
            recommendations.append("61차원 RL Decision 기반 시스템이 잘 작동하고 있습니다!")
        
        return recommendations

class TrainingManager:
    """61차원 RL Decision 기반 훈련 관리 클래스"""
    
    @staticmethod
    def train_agent(agent: RLAgent, train_env: TradingEnvironment, 
                   episodes: int = 200, save_interval: int = 100, 
                   test_env: TradingEnvironment = None) -> Tuple[RLAgent, List[float], List[float]]:
        """61차원 RL Decision 기반 에이전트 훈련 (테스트 환경 모니터링 포함)"""
        print(f"61차원 RL Decision 기반 강화학습 훈련 시작 ({episodes} 에피소드)")
        print(f"상태 공간: {train_env.observation_space.shape[0]}차원")
        if test_env:
            print(f"테스트 환경 모니터링: 활성화")
        
        episode_rewards = []
        episode_win_rates = []
        test_win_rates = []  # 테스트 데이터셋 승률 추적
        best_win_rate = 0.0
        best_test_win_rate = 0.0
        
        for episode in range(episodes):
            state = train_env.reset()
            total_reward = 0
            episode_trades = []
            steps = 0
            
            while steps < 500:
                action = agent.act(state)
                next_state, reward, done, info = train_env.step(action)
                
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
            
            # 테스트 데이터셋으로 성능 평가 (20 에피소드마다)
            if test_env and episode % 20 == 0 and episode > 0:
                print(f"\n📊 Episode {episode}: 테스트 데이터셋 성능 평가 중...")
                test_results, test_stats = PerformanceAnalyzer.evaluate_agent(agent, test_env, num_episodes=3)
                test_win_rate = test_stats['overall_win_rate']
                test_win_rates.append(test_win_rate)
                
                print(f"   테스트 승률: {test_win_rate:.3f} (이전 최고: {best_test_win_rate:.3f})")
                
                if test_win_rate > best_test_win_rate:
                    best_test_win_rate = test_win_rate
                    # 에피소드별 모델 저장
                    agent.save_model(f'best_test_model_ep{episode}_wr{test_win_rate:.3f}.pth')
                    # 최고 성능 모델 업데이트
                    agent.save_model('agent/best_test_performance_model_wr{:.3f}.pth'.format(test_win_rate))
                    print(f"🎯 새로운 테스트 데이터셋 최고 성능! 승률: {test_win_rate:.3f}")
                    print(f"   최고 성능 모델 업데이트: best_test_performance_model_wr{test_win_rate:.3f}.pth")
                print()  # 빈 줄 추가
            
            # 진행 상황 출력
            if episode % 10 == 0 or episode < 10:
                recent_rewards = episode_rewards[-50:] if len(episode_rewards) >= 50 else episode_rewards
                recent_win_rates = episode_win_rates[-50:] if len(episode_win_rates) >= 50 else episode_win_rates
                
                avg_reward = np.mean(recent_rewards)
                avg_win_rate = np.mean(recent_win_rates)
                
                # 테스트 성능도 함께 표시
                test_info = ""
                if test_win_rates:
                    recent_test_win_rate = np.mean(test_win_rates[-5:]) if len(test_win_rates) >= 5 else test_win_rates[-1]
                    test_info = f" | 테스트: {recent_test_win_rate:.3f}"
                
                print(f"Episode {episode:4d} | "
                        f"훈련승률: {avg_win_rate:.3f}{test_info} | "
                        f"리워드: {avg_reward:7.1f} | "
                        f"잔고: ${info['balance']:7.0f} | "
                        f"ε: {agent.epsilon:.3f} | "
                        f"81D")
            
            # 베스트 모델 저장 (훈련 데이터 기준)
            if episode % save_interval == 0 and episode > 0:
                current_avg_win_rate = np.mean(episode_win_rates[-100:]) if len(episode_win_rates) >= 100 else np.mean(episode_win_rates)
                
                if current_avg_win_rate > best_win_rate:
                    best_win_rate = current_avg_win_rate
                    agent.save_model(f'best_train_model_ep{episode}_wr{current_avg_win_rate:.3f}.pth')
                    print(f"🎯 새로운 훈련 데이터셋 최고 성능! 승률: {current_avg_win_rate:.3f}")
            
            # 조기 종료 조건 (테스트 데이터셋 기준)
            if episode > 200 and test_win_rates:
                recent_test_win_rate = np.mean(test_win_rates[-5:]) if len(test_win_rates) >= 5 else test_win_rates[-1]
                if recent_test_win_rate >= 0.65:
                    print(f"🏆 61차원 목표 달성! 테스트 데이터셋 승률 {recent_test_win_rate:.3f} 도달")
                    agent.save_model('agent/final_optimized_model_80d.pth')
                    break
        
        print(f"\n61차원 RL Decision 기반 훈련 완료!")
        print(f"   총 에피소드: {episode + 1}")
        print(f"   훈련 데이터 최고 승률: {best_win_rate:.3f}")
        print(f"   훈련 데이터 최종 승률: {np.mean(episode_win_rates[-50:]) if episode_win_rates else 0:.3f}")
        if test_win_rates:
            print(f"   테스트 데이터 최고 승률: {best_test_win_rate:.3f}")
            print(f"   테스트 데이터 최종 승률: {test_win_rates[-1]:.3f}")
        print(f"   상태 차원: 61차원 (RL Decision 기반)")
        
        # 테스트 데이터셋 최고 성능 모델 저장
        if test_win_rates and best_test_win_rate > 0:
            best_test_model_path = f'agent/best_test_performance_model_wr{best_test_win_rate:.3f}.pth'
            agent.save_model(best_test_model_path)
            print(f"✅ 테스트 데이터셋 최고 성능 모델 저장: {best_test_model_path}")
        
        return agent, episode_rewards, episode_win_rates

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
    """61차원 RL Decision 기반 메인 실행 함수"""
    print("61차원 RL Decision 기반 강화학습 트레이딩 시스템")
    print("=" * 80)
    
    try:
        # 1. 데이터 로딩
        print("\n1️⃣ 61차원용 데이터 로딩...")
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
        
        print(f"최종 61차원용 데이터 준비 완료: {min_length:,}개")
        
        # 2. 데이터 분할 (훈련 80%, 테스트 20%)
        print("\n2️⃣ 데이터 분할...")
        train_price, train_signal, test_price, test_signal = split_data(price_data, signal_data, 0.8, 0.2)
        
        # 3. 환경 및 에이전트 생성
        print("\n3️⃣ 61차원 환경 및 에이전트 생성...")
        train_env = TradingEnvironment(train_price, train_signal)
        test_env = TradingEnvironment(test_price, test_signal)
        agent = RLAgent(train_env.observation_space.shape[0])  # 61차원
        
        print(f"상태 공간: {train_env.observation_space.shape[0]}차원")
        print("Signal의 모든 indicator와 raw score 활용")
        
        # 기존 61차원 모델 로드 시도 (테스트 성능 우선)
        model_files = [
            'agent/best_test_performance_model_wr*.pth',  # 테스트 최고 성능 모델
            'agent/final_optimized_model_80d.pth',       # 최종 모델
            'agent/best_model_80d.pth'                   # 기존 모델
        ]
        model_loaded = False
        
        # 테스트 성능 모델 우선 로드
        import glob
        test_model_files = glob.glob('agent/best_test_performance_model_wr*.pth')
        if test_model_files:
            # 가장 높은 승률의 테스트 모델 선택
            best_test_model = max(test_model_files, key=lambda x: float(x.split('wr')[1].split('.pth')[0]))
            if agent.load_model(best_test_model):
                model_loaded = True
                print(f"✅ 테스트 데이터셋 최고 성능 모델 로드: {best_test_model}")
        
        # 테스트 모델이 없으면 다른 모델들 시도
        if not model_loaded:
            for model_file in ['agent/final_optimized_model_80d.pth', 'agent/best_model_80d.pth']:
                if agent.load_model(model_file):
                    model_loaded = True
                    break
        
        if not model_loaded:
            print("새로운 61차원 모델로 시작합니다.")
        
        # 4. 훈련 전 테스트 데이터셋 성능 평가 (베이스라인)
        print("\n4️⃣ 훈련 전 테스트 데이터셋 성능 평가...")
        baseline_results, baseline_stats = PerformanceAnalyzer.evaluate_agent(agent, test_env, num_episodes=5)
        print("=== 훈련 전 테스트 데이터셋 성능 ===")
        PerformanceAnalyzer.print_performance_report(baseline_results, baseline_stats)
        
        # 5. 훈련 데이터셋으로 훈련
        print(f"\n5️⃣ 훈련 데이터셋으로 61차원 RL Decision 기반 훈련 시작...")
        print(f"   훈련 데이터: {len(train_price):,}개")
        print(f"   테스트 데이터: {len(test_price):,}개")
        print(f"   목표 승률: 0.65+")
        print(f"   Signal 특성 활용: 최대화")
        
        # 훈련 실행 (테스트 환경 모니터링 포함)
        trained_agent, rewards, win_rates = TrainingManager.train_agent(agent, train_env, episodes=500, test_env=test_env)
        
        # 6. 훈련 후 테스트 데이터셋으로 성능 평가
        print("\n6️⃣ 훈련 후 테스트 데이터셋 성능 평가...")
        final_results, final_stats = PerformanceAnalyzer.evaluate_agent(trained_agent, test_env, num_episodes=10)
        print("=== 훈련 후 테스트 데이터셋 성능 ===")
        PerformanceAnalyzer.print_performance_report(final_results, final_stats)
        
        # 7. 성능 개선도 분석
        improvement = final_stats['overall_win_rate'] - baseline_stats['overall_win_rate']
        print(f"\n🚀 61차원 RL Decision 기반 성능 개선도 (테스트 데이터셋 기준):")
        print(f"   승률: {baseline_stats['overall_win_rate']:.3f} → {final_stats['overall_win_rate']:.3f} ({improvement:+.3f})")
        print(f"   평균 수익률: {baseline_stats['avg_return']:.3f} → {final_stats['avg_return']:.3f}")
        print(f"   Signal 활용도: 최대화됨")
        
        # 8. 최종 모델 저장
        trained_agent.save_model('agent/final_optimized_model_80d.pth')
        print(f"\n✅ 최종 모델이 저장되었습니다: agent/final_optimized_model_80d.pth")
        
        # 9. 추가 훈련 여부 확인
        if final_stats['overall_win_rate'] < 0.6:
            user_input = input("\n성능이 목표에 미달합니다. 추가 훈련을 원하시나요? (y/n): ")
            if user_input.lower() == 'y':
                print("61차원 추가 훈련 시작...")
                TrainingManager.train_agent(trained_agent, train_env, episodes=200, test_env=test_env)
                
                # 추가 훈련 후 재평가
                print("\n추가 훈련 후 테스트 데이터셋 성능 평가...")
                additional_results, additional_stats = PerformanceAnalyzer.evaluate_agent(trained_agent, test_env, num_episodes=10)
                print("=== 추가 훈련 후 테스트 데이터셋 성능 ===")
                PerformanceAnalyzer.print_performance_report(additional_results, additional_stats)
        else:
            print(f"\n🎉 목표 성능 달성! (테스트 데이터셋 승률: {final_stats['overall_win_rate']:.3f})")
    
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()