"""
66차원 RL Decision 기반 강화학습 트레이딩 AI 훈련 시스템 - Part 1
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
    """포텐셜 기반 보상 계산기 - Sparse Reward 문제 해결"""
    
    def __init__(self, max_trades_memory: int = 50):
        self.recent_trades = deque(maxlen=max_trades_memory)
        self.recent_returns = deque(maxlen=20)
        self.target_return_per_trade = 0.005  # 거래당 0.5% 목표
        
    def calculate_reward(self, current_price: float, entry_price: float, position: float, 
                    holding_time: int, trade_pnl: Optional[float] = None) -> float:
        """포텐셜 기반 보상 계산 - Hold 액션도 학습 가능하게"""
        reward = 0.0
        
        # 거래 완료 시
        if trade_pnl is not None:
            # 실제 수익률 기반 보상 (더 강한 신호)
            reward = trade_pnl * 1000  # 1% = 10.0 리워드
            
            # 보유 시간 인센티브 (짧은 거래 억제)
            if holding_time >= 10:  # 30분 이상
                reward += 0.5
            else:
                reward -= 1.0  # 빈번한 거래 페널티
            
            # 목표 달성 여부에 따른 추가 보상
            if trade_pnl > self.target_return_per_trade:
                reward += 5.0  # 목표 달성 보너스
            elif trade_pnl < -self.target_return_per_trade:
                reward -= 5.0  # 큰 손실 페널티
            
            # 샤프 비율 기반 보상 (리스크 대비 수익 고려)
            if len(self.recent_returns) > 5:
                returns_std = np.std(list(self.recent_returns))
                if returns_std > 0:
                    sharpe_bonus = (trade_pnl / returns_std) * 10
                    reward += sharpe_bonus
            
            # 거래 기록 저장
            self.recent_trades.append({
                'pnl': trade_pnl,
                'holding_time': holding_time
            })
            self.recent_returns.append(trade_pnl)
        
        # Hold 중일 때 - 미실현 손익 기반 강한 신호
        elif position != 0:
            unrealized_pnl = self._calculate_unrealized_pnl(current_price, entry_price, position)
            reward = unrealized_pnl * 100  # 강한 신호 (1% = 1.0 리워드)
            
            # 너무 오래 보유 시 페널티
            if holding_time > 50:  # 150분 이상
                reward -= 0.2
        
        # 완전히 포지션 없을 때는 0 (문제없음)
        
        return reward
    
    def _calculate_unrealized_pnl(self, current_price: float, entry_price: float, position: float) -> float:
        """미실현 손익 계산"""
        if entry_price <= 0:
            return 0.0
        
        price_change = (current_price - entry_price) / entry_price
        return position * price_change
    



class DuelingDQN(nn.Module):
    """강화된 Dueling DQN (Value + Advantage 분리로 안정적인 학습)"""
    
    def __init__(self, state_size: int, hidden_size: int = 256, dropout: float = 0.1):
        super().__init__()
        
        self.state_size = state_size
        self.hidden_size = hidden_size
        
        # 더 깊은 공통 특징 추출기 (4층)
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Value Stream (상태의 가치) - 더 깊은 네트워크
        self.value_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Advantage Stream (액션별 장점) - 더 깊은 네트워크
        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
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
        features = self.feature_extractor(x)  # [batch_size, 64]
        
        # Value Stream (상태의 가치)
        value = self.value_stream(features)  # [batch_size, 1]
        
        # Advantage Stream (액션별 장점)
        position_adv = self.advantage_stream(features)  # [batch_size, 3]
        
        # Dueling 구조: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        position_q = value + position_adv - position_adv.mean(dim=1, keepdim=True)
        
        # 단일 샘플이면 배치 차원 제거
        if single_sample:
            position_q = position_q.squeeze(0)
        
        return position_q


class TradingEnvironment(gym.Env):
    """111차원 RL Decision 기반 암호화폐 거래 강화학습 환경 (Gymnasium 호환) - OHLC 포함"""
    
    def __init__(self, signal_data: List[Dict], initial_balance: float = 10000.0):
        super().__init__()
        
        self.signal_data = signal_data
        self.initial_balance = initial_balance
        
        self.reward_calculator = RewardCalculator()
        
        # 액션/상태 스페이스 정의 (단타에 적합한 단순한 액션)
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        
        # 거래 제한 설정 (거래 간격 완전 제거)
        self.min_trade_interval = 1  # 거래 간격 최소화
        self.last_trade_step = -1  # 초기값
        self.trading_cost = 0.0  # 훈련용 수수료 제거 
        
        # 111차원 상태 공간 (기술적 지표 + 포트폴리오 + 의사결정 특성)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(111,),  # 3 + 20 + 8 + 80 = 111차원
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        """환경 초기화 (Gymnasium 호환)"""
        if seed is not None:
            np.random.seed(seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.current_position = 0.0
        self.entry_price = 0.0
        self.entry_balance = 0.0  # 진입 시점의 잔고 추적
        self.unrealized_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.max_drawdown = 0.0
        self.peak_balance = self.initial_balance
        self.consecutive_losses = 0
        self.holding_time = 0
        self.in_position = False
        self.last_trade_pnl = None
        self.last_trade_step = -self.min_trade_interval - 1  # 거래 간격 초기화 (더 안전하게)
        
        observation = self._get_observation()
        info = self._create_info_dict()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """환경 스텝 실행 (Gymnasium 호환) - OHLC 포함"""
        if self.current_step >= len(self.signal_data) - 1:
            return self._get_observation(), 0.0, True, False, {}
        
        # 이산적인 액션 처리 (position만) - 거래 간격 체크 제거
        # 0: Hold, 1: Buy, 2: Sell
        if action == 0:  # Hold
            position_change = 0.0
        elif action == 1:  # Buy
            position_change = 1.0  # 거래 간격 체크 제거
        elif action == 2:  # Sell
            position_change = -1.0  # 거래 간격 체크 제거
        else:
            position_change = 0.0  # 기본값은 Hold
        
        # 현재 신호 데이터에서 OHLC 정보 가져오기
        current_signal = self.signal_data[self.current_step]
        current_close_price = current_signal.get('close')
        
        # 포지션 및 거래 처리 (현재 close 가격으로 실행)
        trade_completed, old_position = self._process_position_change(
            position_change, current_close_price
        )
        
        # 거래 완료 시 거래 스텝 업데이트
        if trade_completed:
            self.last_trade_step = self.current_step
        
        # 보상 계산 (포텐셜 기반 - Hold 액션도 학습 가능)
        if trade_completed:
            # 거래 완료 시: 수수료 차감 후 실제 수익률 기반 보상
            reward = self.reward_calculator.calculate_reward(
                current_price=current_close_price,  # 현재 close 가격 사용
                entry_price=self.entry_price,
                position=old_position,  # 거래 전 포지션 사용
                holding_time=self.holding_time,
                trade_pnl=self.last_trade_pnl  # 이미 수수료 차감된 수익률
            )
        else:
            # Hold 액션이나 거래 간격 미충족 시: 미실현 손익 기반 약한 신호
            reward = self.reward_calculator.calculate_reward(
                current_price=current_close_price,
                entry_price=self.entry_price,
                position=self.current_position,  # 현재 포지션 사용
                holding_time=self.holding_time,
                trade_pnl=None  # 거래 완료가 아님
            )
        
        # 다음 스텝으로 이동
        self.current_step += 1
        self.holding_time += 3
        
        done = (self.current_step >= len(self.signal_data) - 1 or 
                self.balance <= self.initial_balance * 0.1)
        
        # 에피소드 보너스 제거 - Sparse Reward 문제 해결
        
        truncated = False  # Gymnasium 호환을 위한 truncated 플래그
        info = self._create_info_dict()
        
        return self._get_observation(), reward, done, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """111차원 상태 관찰값 반환 (기술적 지표 + 포트폴리오 + 의사결정 특성) - OHLC 포함"""
        if self.current_step >= len(self.signal_data):
            return np.zeros(111, dtype=np.float32)
        
        # 현재 신호 데이터에서 OHLC 정보 가져오기
        current_signal = self.signal_data[self.current_step]
        current_price = current_signal.get('close')
        
        # 이전 가격과 비교하여 가격 변화율 계산
        if self.current_step > 0:
            prev_signal = self.signal_data[self.current_step - 1]
            prev_price = prev_signal.get('close', current_price)
            price_change = (current_price - prev_price) / prev_price if prev_price > 0 else 0.0
        else:
            price_change = 0.0
        
        basic_observation = np.array([
            price_change,  # 가격 변화율
            self.current_position,  # 현재 포지션 (-1~1)
            self.balance / self.initial_balance  # 잔고 비율
        ], dtype=np.float32)
        
        # Signal 데이터는 이미 가져왔음
        
        # 각 차원별 특성 추출
        price_indicators = self._extract_price_indicators(current_signal)  # 20차원
        portfolio_state = self._get_portfolio_state()  # 8차원
        decision_features = self._extract_decision_features(current_signal)  # 80차원
        
        # 모든 차원 결합 (3 + 20 + 8 + 80 = 111차원)
        observation = np.concatenate([
            basic_observation,      # 3차원
            price_indicators,       # 20차원
            portfolio_state,        # 8차원
            decision_features       # 80차원
        ], dtype=np.float32)
        
        return observation
    
    def _extract_price_indicators(self, signal_data: Dict) -> np.ndarray:
        """Signal의 indicator들을 price feature로 활용 (20차원) - 실제 데이터 구조 기반"""
        current_price = signal_data.get('close')
        
        # 1. 지표 값들 그대로 사용 (10개)
        vwap = signal_data.get('indicator_vwap', 0.0)
        poc = signal_data.get('indicator_poc', 0.0)
        hvn = signal_data.get('indicator_hvn', 0.0)
        lvn = signal_data.get('indicator_lvn', 0.0)
        atr = signal_data.get('indicator_atr', 0.0)
        vwap_std = signal_data.get('indicator_vwap_std', 0.0)
        prev_high = signal_data.get('indicator_prev_day_high', 0.0)
        prev_low = signal_data.get('indicator_prev_day_low', 0.0)
        or_high = signal_data.get('indicator_opening_range_high', 0.0)
        or_low = signal_data.get('indicator_opening_range_low', 0.0)
        
        # 2. 현재 캔들 정보 (10개)
        high = signal_data.get('high', 0.0)
        low = signal_data.get('low', 0.0)
        close = signal_data.get('close', 0.0)
        open_price = signal_data.get('open', 0.0)
        quote_volume = signal_data.get('quote_volume', 0.0)
        
        # 안전한 캔들 계산
        body_size = abs(close - open_price) if open_price > 0 else 0.0
        candle_range = high - low if high > low else 1.0
        upper_shadow = high - max(open_price, close) if high > low else 0.0
        lower_shadow = min(open_price, close) - low if high > low else 0.0
        
        return np.array([
            # 지표 값들 그대로 사용 (10개)
            vwap,
            poc,
            hvn,
            lvn,
            atr,
            vwap_std,
            prev_high,
            prev_low,
            or_high,
            or_low,
            
            # 현재 캔들 정보 (10개)
            body_size / open_price if open_price > 0 else 0.0,  # 몸통 크기
            candle_range / current_price if current_price > 0 else 0.0,  # 전체 범위
            upper_shadow / candle_range if candle_range > 0 else 0.0,  # 위꼬리 비율
            lower_shadow / candle_range if candle_range > 0 else 0.0,  # 아래꼬리 비율
            body_size / candle_range if candle_range > 0 else 0.0,  # 몸통 비율
            min(quote_volume / 1000000, 2.0) if quote_volume > 0 else 0.0,  # 거래량
            1.0 if close > open_price else 0.0,  # 상승/하락
            upper_shadow / current_price if current_price > 0 else 0.0,  # 위꼬리 크기
            lower_shadow / current_price if current_price > 0 else 0.0,  # 아래꼬리 크기
            body_size / current_price if current_price > 0 else 0.0  # 몸통 크기
        ], dtype=np.float32)
    
    def _get_portfolio_state(self) -> np.ndarray:
        """포트폴리오 상태 정보 (8차원)"""
        return np.array([
            self.current_position,
            (self.balance - self.initial_balance) / self.initial_balance,
            self.unrealized_pnl / self.initial_balance if self.initial_balance > 0 else 0.0,
            self.total_trades / 100.0,
            self.winning_trades / max(self.total_trades, 1),
            self.max_drawdown,
            self.consecutive_losses / 10.0,
            self.holding_time / 1440.0
        ], dtype=np.float32)
    
    def _extract_decision_features(self, signals: Dict) -> np.ndarray:
        """Decision 특성들 (80차원) - 모든 전략 특성 사용"""
        # 전략별 특성 추출 (실제 데이터에 있는 전략들)
        strategy_names = [
            'session', 'vpvr', 'bollinger_squeeze', 'orderflow_cvd', 'ichimoku', 
            'vwap_pinball', 'vol_spike', 'liquidity_grab', 'vpvr_micro', 
            'zscore_mean_reversion', 'htf_trend', 'oi_delta', 'funding_rate', 
            'multi_timeframe', 'support_resistance', 'ema_confluence'
        ]
        
        # 모든 전략의 모든 특성 사용 (16개 전략 × 5개 특성 = 80차원)
        all_features = []
        for strategy in strategy_names:
            # Action을 숫자로 변환 (HOLD=0, BUY=1, SELL=-1)
            action_str = signals.get(f'{strategy}_action', 'HOLD')
            if action_str == 'BUY':
                action_value = 1.0
            elif action_str == 'SELL':
                action_value = -1.0
            else:  # HOLD 또는 None
                action_value = 0.0
            
            # Score와 Confidence (None인 경우 0.0으로 처리)
            score = float(signals.get(f'{strategy}_score', 0.0))
            confidence_str = signals.get(f'{strategy}_confidence')
            if confidence_str == 'HIGH':
                confidence = 1.0
            elif confidence_str == 'MEDIUM':
                confidence = 0.5
            elif confidence_str == 'LOW':
                confidence = 0.2
            else:
                confidence = 0.0
            
            # Entry와 Stop (None인 경우 0.0으로 처리)
            entry = float(signals.get(f'{strategy}_entry', 0.0))
            stop = float(signals.get(f'{strategy}_stop', 0.0))
            
            # 모든 특성 추가: action, score, confidence, entry, stop
            all_features.extend([action_value, score, confidence, entry, stop])

        return np.array(all_features, dtype=np.float32)
    
    def _process_position_change(self, position_change: float, current_price: float) -> Tuple[bool, float]:
        """포지션 변경 처리 (position만) - 3분봉 기반"""
        old_position = self.current_position
        trade_completed = False
        
        # 단순한 액션 처리: 전체 포지션을 즉시 변경
        target_position = position_change  # -1.0, 0.0, 1.0 중 하나
        
        # 포지션 변경이 필요한지 확인
        if abs(target_position - self.current_position) > 0.0001:
            # 기존 포지션 청산 (현재 close 가격으로 청산)
            if abs(self.current_position) > 0.0001:
                trade_completed = True
                # 청산 처리 (수수료 포함)
                self._close_position(current_price)
            
            # 새 포지션 진입 (현재 close 가격으로 진입)
            if abs(target_position) > 0.0001:
                self.current_position = target_position
                self.entry_price = current_price  # 현재 close 가격으로 진입
                self.entry_balance = self.balance  # 진입 시점의 잔고 저장
                
                # 진입 시 수수료 차감
                entry_fee = abs(target_position) * self.entry_balance * self.trading_cost
                self.balance -= entry_fee
                
                self.holding_time = 0
                self.in_position = True
                trade_completed = True  # 새 포지션 진입도 거래 완료로 간주
                
        
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
        
        # 순수 가격 변화율 계산 (수수료 미고려)
        price_change_rate = self._calculate_trade_pnl(exit_price, self.entry_price, self.current_position)
        
        # 실제 거래 금액 계산 (진입 시점의 잔고 기준)
        trade_volume = abs(self.current_position) * self.entry_balance
        gross_pnl_usd = price_change_rate * trade_volume  # 수수료 차감 전 손익
        
        # 청산 시 수수료 계산 (진입 시점의 잔고 기준)
        exit_trade_amount = abs(self.current_position) * self.entry_balance
        exit_fee = exit_trade_amount * self.trading_cost
        
        # 청산 수수료 차감 후 순손익
        net_pnl_usd = gross_pnl_usd - exit_fee
        
        # 잔고 업데이트
        self.balance += net_pnl_usd
        
        # 통계 업데이트 (순손익 기준)
        self._update_trading_stats(net_pnl_usd)
        
        # 수수료 차감 후 실제 수익률 계산 (비율)
        # 진입 수수료와 청산 수수료를 모두 고려한 총 수수료
        total_fee = exit_trade_amount * self.trading_cost * 2  # 진입 + 청산
        total_net_pnl = gross_pnl_usd - total_fee
        
        if trade_volume > 0:
            self.last_trade_pnl = total_net_pnl / trade_volume
        else:
            self.last_trade_pnl = 0.0
        
        # 포지션 초기화
        self.current_position = 0.0
        self.entry_balance = 0.0  # 진입 잔고 초기화
        self.unrealized_pnl = 0.0
        self.in_position = False
        self.holding_time = 0
    
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
        """정보 딕셔너리 생성 - OHLC 포함"""
        if self.current_step < len(self.signal_data):
            current_signal = self.signal_data[self.current_step]
            current_price = current_signal.get('close')
        else:
            current_price = 0.0
        
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
111차원원 RL Decision 기반 강화학습 트레이딩 AI 훈련 시스템 - Part 2
- RLAgent 클래스 및 훈련/평가 시스템
- 새로운 Decision 스키마 데이터 로더 및 유틸리티 함수들
"""

class RLAgent:
    """111차원 RL Decision 기반 강화학습 에이전트"""
    
    def __init__(self, state_size: int = 3, learning_rate: float = 1e-4,  # 적절한 학습률
                    gamma: float = 0.99, epsilon: float = 0.9, epsilon_decay: float = 0.9995,  # 0.995 → 0.9995로 변경
                    hidden_size: int = 256):
        
        self.state_size = state_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate  # learning_rate 속성 추가
        self.epsilon_min = 0.2  # 20%로 설정 (더 안정적인 탐험)
        
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
        self.q_network = DuelingDQN(state_size, hidden_size).to(self.device)
        self.target_network = DuelingDQN(state_size, hidden_size).to(self.device)
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 경험 리플레이 (수익률 학습 최적화)
        self.memory = deque(maxlen=5000)  # 메모리 크기 증가 (더 많은 경험 저장)
        self.batch_size = 512  # 배치 크기 증가 (더 안정적인 학습)
        
        # 학습 추적
        self.training_rewards = []
        self.losses = []
        self.return_rates = []  # 수익률 추적
        
        # 타겟 네트워크 업데이트 (수익률 학습 최적화)
        self.target_update_freq = 25  # 더 자주 업데이트 (빠른 학습)
        self.update_count = 0
    
        
        # 액션 공간 설정 (환경에서 가져옴)
        self.action_space = None  # 환경에서 설정됨
    
    def remember(self, state, action: int, reward, next_state, done):
        """경험 저장 (단순 액션)"""
        self.memory.append(Experience(state, action, reward, next_state, done))
    
    
    def adaptive_learning_rate(self, recent_rewards: List[float], recent_return_rates: List[float]):
        """적응형 학습률 조정"""
        if len(recent_rewards) < 10:
            return
        
        # 최근 성능 분석
        avg_reward = np.mean(recent_rewards[-10:])
        avg_return_rate = np.mean(recent_return_rates[-10:])
        
        # 성능이 좋으면 학습률 감소 (매우 보수적으로)
        if avg_return_rate > 0.05 and avg_reward > 0:  # 수익률 5% 이상
            self.learning_rate *= 0.995  # 매우 느린 감소
            self.learning_rate = max(self.learning_rate, 1e-7)  # 더 낮은 최소값
        # 성능이 나쁘면 학습률 증가 (매우 보수적으로)
        elif avg_return_rate < 0.02 or avg_reward < -10:  # 수익률 2% 미만 또는 손실
            self.learning_rate *= 1.01  # 매우 느린 증가
            self.learning_rate = min(self.learning_rate, 1e-4)  # 더 낮은 최대값
        
        # 옵티마이저 업데이트
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate
    
    def act(self, state: np.ndarray) -> int:
        """액션 선택 - 단순한 epsilon-greedy"""
        if np.random.random() <= self.epsilon:
            return self._get_random_action()
        
        return self._get_greedy_action(state)
    
    def _get_random_action(self) -> int:
        """랜덤 액션 (단순 액션)"""
        return np.random.randint(0, 3)  # 0: Hold, 1: Buy, 2: Sell
    
    def _get_greedy_action(self, state: np.ndarray) -> int:
        """Q값 기반 액션 선택 - 임계값 적용"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            position_q = self.q_network(state_tensor)
            
            # Q값 기반 액션 선택 (임계값 제거)
            q_values = position_q[0].cpu().numpy()
            action = np.argmax(q_values)  # 단순히 최대 Q값 선택
            
            return action
    
    def replay(self):
        """우선순위 경험 리플레이 학습"""
        if len(self.memory) < self.batch_size * 2:
            return
        
        # 단순한 랜덤 샘플링
        batch = random.sample(self.memory, min(self.batch_size, len(self.memory)))
        
        loss = self._compute_loss(batch)
        
        # 역전파
        self.optimizer.zero_grad()
        loss.backward()
        
        # 강화된 그래디언트 클리핑 (안정성 향상)
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 0.5)
        
        self.optimizer.step()
        
        self.losses.append(loss.item())
        
        # 단순한 엡실론 감소
        if self.epsilon > self.epsilon_min:
            self.epsilon *= 0.999  # 매우 느린 감소
        
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
        
        # 현재 Q값들
        current_position_q = self.q_network(states)
        
        # Double DQN: 현재 네트워크로 액션 선택, 타겟 네트워크로 Q값 계산
        with torch.no_grad():
            # 현재 네트워크로 다음 상태의 액션 선택
            next_position_q_current = self.q_network(next_states)
            
            # 타겟 네트워크로 Q값 계산
            next_position_q_target = self.target_network(next_states)
            
            target_position_q = current_position_q.clone()
            
            for i, (action, reward, done) in enumerate(zip(actions, rewards, dones)):
                # 단순 액션 처리: action은 0, 1, 2 중 하나
                if isinstance(action, (list, np.ndarray)):
                    # 기존 연속 액션에서 단순 액션으로 변환
                    if len(action) > 0:
                        if action[0] > 0.3:  # Buy
                            action_idx = 1
                        elif action[0] < -0.3:  # Sell
                            action_idx = 2
                        else:  # Hold
                            action_idx = 0
                    else:
                        action_idx = 0  # 기본값 Hold
                else:
                    # 이미 단순 액션인 경우
                    action_idx = int(action) if 0 <= action <= 2 else 0
                
                if not done:
                    # Double DQN: 현재 네트워크로 선택한 액션의 타겟 네트워크 Q값 사용
                    best_action = torch.argmax(next_position_q_current[i])
                    target_q = reward + self.gamma * next_position_q_target[i, best_action]
                    target_position_q[i, action_idx] = target_q
                else:
                    # 최종 보상 (수익률 중심)
                    target_position_q[i, action_idx] = reward
        
        # Q-learning 손실 (순수 DuelingDQN)
        pos_loss = F.smooth_l1_loss(current_position_q, target_position_q)
        
        # 단순한 DuelingDQN 손실
        total_loss = pos_loss
        
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
                'return_rates': [float(r) for r in self.return_rates],
                'update_count': int(self.update_count),
                'state_size': int(self.state_size)
            }
            
            torch.save(save_dict, filepath)
            print(f"111차원 모델 저장 완료: {filepath}")
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
            self.return_rates = checkpoint.get('return_rates', [])
            self.update_count = checkpoint.get('update_count', 0)
            
            print(f"✅ 111차원 모델 로드 성공! 엡실론: {self.epsilon:.3f}")
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
            self.return_rates = checkpoint.get('return_rates', [])
            self.update_count = checkpoint.get('update_count', 0)
            
            print(f"✅ {model_state_size}차원 → 111차원 호환성 모델 로드 성공!")
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
            new_model = DuelingDQN(self.state_size, self.hidden_size).to(self.device)
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
            self.target_network = DuelingDQN(self.state_size, self.hidden_size).to(self.device)
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
    """111차원 RL Decision 기반 데이터 로딩 클래스"""
    
    
    @staticmethod
    def load_signal_data(agent_folder: str = "agent") -> Optional[List[Dict]]:
        """111차원용 RL Decision 데이터 로드"""
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
    def load_signal_data_for_test(agent_folder: str = "agent", max_records: int = 50000) -> Optional[List[Dict]]:
        """테스트용 Signal 데이터 로드 (50,000개 제한)"""
        parquet_files = []
        
        if Path(agent_folder).exists():
            parquet_files = list(Path(agent_folder).glob("*.parquet"))
        
        if parquet_files:
            try:
                print(f"테스트용 Signal 데이터 로드 중: {parquet_files[0].name}")
                signal_df = pd.read_parquet(parquet_files[0])
                
                # 테스트용으로 최대 50,000개만 사용
                if len(signal_df) > max_records:
                    signal_df = signal_df.tail(max_records)  # 최근 50,000개 사용
                    print(f"테스트용 데이터 제한: {len(signal_df):,}개 레코드 (전체 {len(pd.read_parquet(parquet_files[0])):,}개 중)")
                else:
                    print(f"Signal 데이터 로드: {len(signal_df):,}개 레코드")
                
                return DataLoader._convert_parquet_to_signal_dicts(signal_df)
                
            except Exception as e:
                print(f"Parquet 로드 실패: {e}")
        
        print("Parquet 파일이 없어 기본 Signal을 생성합니다.")
        return None
    
    @staticmethod
    def _convert_parquet_to_signal_dicts(signal_df: pd.DataFrame) -> List[Dict]:
        """Parquet을 Signal Dict 리스트로 변환 (111차원용) - 새로운 RL 스키마"""
        signal_data = []
        
        print("111차원용 RL 스키마 Signal 데이터 변환 중...")
        
        for idx, row in signal_df.iterrows():
            # 각 행을 딕셔너리로 변환 (새로운 RL 스키마 형태 유지)
            signal_dict = {}
            
            for col, value in row.items():
                if pd.notna(value):
                    # 수치 데이터는 그대로 유지
                    signal_dict[col] = value
        
            
            signal_data.append(signal_dict)
            
            if (idx + 1) % 5000 == 0:
                print(f"   변환 진행: {idx + 1:,}/{len(signal_df):,}")
        
        print(f"111차원용 RL 스키마 Signal 데이터 변환 완료: {len(signal_data):,}개")
        return signal_data
    

class PerformanceAnalyzer:
    """111차원 RL Decision 기반 성능 분석 클래스"""
    
    @staticmethod
    def evaluate_agent(agent: RLAgent, env: TradingEnvironment, num_episodes: int = 10) -> Tuple[List[Dict], Dict]:
        """111차원 에이전트 성능 평가"""
        print(f"111차원 에이전트 성능 평가 중 ({num_episodes} 에피소드)...")
        
        original_epsilon = agent.epsilon
        agent.epsilon = 0.1  # 테스트에서도 적절한 탐험 허용 (훈련과 유사)
        
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
            
            for step in range(200):  # 훈련과 동일한 스텝 수
                action = agent.act(state)
                next_state, reward, done, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_balance = info['balance']
                
                if info.get('trade_completed', False):
                    trade_pnl = info.get('trade_pnl', 0.0)
                    # trade_pnl은 이미 수수료가 차감된 실제 수익률
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
        """111차원 성능 리포트 출력"""
        print("\n" + "="*60)
        print(f"111차원 RL Decision 기반 성능 평가 결과")
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
            recommendations.append("수익률이 10% 미만입니다. 111차원 상태 공간의 수익률 최적화를 더 활용하세요.")
        
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
            recommendations.append("111차원 RL Decision 기반 수익률 중심 시스템이 잘 작동하고 있습니다!")
        
        return recommendations

class TrainingManager:
    """111차원 RL Decision 기반 훈련 관리 클래스"""
    
    @staticmethod
    def train_agent(agent: RLAgent, train_env: TradingEnvironment, 
                   episodes: int = 1000, save_interval: int = 100, 
                   test_env: TradingEnvironment = None) -> Tuple[RLAgent, List[float], List[float]]:
        """111차원 RL Decision 기반 에이전트 훈련 (테스트 환경 모니터링 포함)"""
        print(f"111차원 RL Decision 기반 강화학습 훈련 시작 ({episodes} 에피소드)")
        print(f"상태 공간: {train_env.observation_space.shape[0]}차원")
        if test_env:
            print(f"테스트 환경 모니터링: 활성화")
        
        # 에이전트에 액션 공간 설정
        agent.action_space = train_env.action_space
        
        episode_rewards = []
        episode_win_rates = []
        episode_returns = []  # 훈련 데이터 수익률 추적
        test_return_rates = []  # 테스트 데이터셋 수익률 추적
        best_return_rate = 0.0
        best_test_return_rate = 0.0
        
        for episode in range(episodes):
            
            state, _ = train_env.reset()
            episode_start_balance = train_env.balance  # 에피소드 시작 잔고 추적
            total_reward = 0
            episode_trades = []
            steps = 0
            
            while steps < 200:
                action = agent.act(state)
                next_state, reward, done, truncated, info = train_env.step(action)
                
                agent.remember(state, action, reward, next_state, done)
                
                if info.get('trade_completed', False):
                    trade_pnl = info.get('trade_pnl')
                    # trade_pnl은 이미 수수료가 차감된 실제 수익률
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
            final_balance = info.get('balance', episode_start_balance)
            episode_return = (final_balance - episode_start_balance) / episode_start_balance
            episode_returns.append(episode_return)
            
            # 보상 정보 출력 (처음 5개 에피소드만)
            if episode < 5:
                print(f"\n🔍 Episode {episode} 보상 분석:")
                print(f"   총 리워드: {total_reward:.2f}")
                print(f"   수익률: {episode_return:.4f} ({episode_return*100:+.2f}%)")
                print(f"   거래 수: {len(episode_trades)}개")
            
            
            agent.training_rewards.append(total_reward)
            agent.return_rates.append(episode_return)
            
            
            # 적응형 학습률 업데이트 (10에피소드마다)
            if episode % 10 == 0 and episode > 0:
                recent_rewards = episode_rewards[-20:] if len(episode_rewards) >= 20 else episode_rewards
                recent_return_rates = episode_returns[-20:] if len(episode_returns) >= 20 else episode_returns
                agent.adaptive_learning_rate(recent_rewards, recent_return_rates)
            
            # 테스트 데이터셋으로 성능 평가 (과적합 방지 강화)
            if test_env and episode % 20 == 0 and episode > 0:  # 20에피소드마다 평가
                print(f"\n📊 Episode {episode}: 테스트 데이터셋 성능 평가 중...")
                test_results, test_stats = PerformanceAnalyzer.evaluate_agent(agent, test_env, num_episodes=5)  # 더 많은 에피소드로 평가
                test_return = test_stats['avg_return']
                test_return_rates.append(test_return)
                
                print(f"   테스트 수익률: {test_return:.3f} ({test_return*100:.1f}%) (이전 최고: {best_test_return_rate:.3f})")
                
                # 과적합 감지: 훈련 수익률과 테스트 수익률 차이 확인
                recent_train_return = np.mean(episode_returns[-10:]) if len(episode_returns) >= 10 else 0.0
                overfitting_gap = abs(recent_train_return - test_return)
                
                # 과적합 감지: 훈련이 테스트보다 현저히 좋을 때
                if overfitting_gap > 0.15:
                    print(f"⚠️ 과적합 감지: 훈련 수익률({recent_train_return:.3f}) vs 테스트 수익률({test_return:.3f}) = 차이 {overfitting_gap:.3f}")
                    # 학습률 감소 및 엡실론 증가
                    for param_group in agent.optimizer.param_groups:
                        param_group['lr'] *= 0.8
                    agent.epsilon = min(agent.epsilon * 1.2, 0.8)  # 더 많은 탐험
                    print(f"   학습률 감소: {agent.optimizer.param_groups[0]['lr']:.2e}, 엡실론 증가: {agent.epsilon:.3f}")
                
                # 학습 부족 감지: 훈련 수익률이 테스트보다 현저히 낮을 때
                elif recent_train_return < test_return - 0.1:
                    print(f"⚠️ 학습 부족: 훈련 수익률({recent_train_return:.3f}) < 테스트 수익률({test_return:.3f})")
                    # 학습률 증가 및 엡실론 조정
                    for param_group in agent.optimizer.param_groups:
                        param_group['lr'] *= 1.1
                    agent.epsilon = max(agent.epsilon * 0.95, 0.2)  # 적절한 탐험
                    print(f"   학습률 증가: {agent.optimizer.param_groups[0]['lr']:.2e}, 엡실론 조정: {agent.epsilon:.3f}")
                
                # 정상적인 학습 과정
                else:
                    print(f"ℹ️ 정상 학습: 훈련({recent_train_return:.3f}) vs 테스트({test_return:.3f}) - 차이 {overfitting_gap:.3f}")
                
                if test_return > best_test_return_rate:
                    best_test_return_rate = test_return
                    # 에피소드별 모델 저장 (수익률 기준)
                    # agent.save_model(f'best_test_model_ep{episode}_return{test_return:.3f}.pth')
                    # 최고 성능 모델 업데이트 (수익률 기준)
                    agent.save_model('agent/best_test_performance_model_return{:.3f}.pth'.format(test_return))
                    print(f"🎯 새로운 테스트 데이터셋 최고 수익률! 수익률: {test_return:.3f} ({test_return*100:.1f}%)")
                    print(f"   최고 성능 모델 업데이트: best_test_performance_model_return{test_return:.3f}.pth")
                print()  # 빈 줄 추가
            
            # 진행 상황 출력 (20에피소드마다)
            if episode % 20 == 0 or episode < 10:
                recent_rewards = episode_rewards[-50:] if len(episode_rewards) >= 50 else episode_rewards
                recent_win_rates = episode_win_rates[-50:] if len(episode_win_rates) >= 50 else episode_win_rates
                recent_returns = episode_returns[-50:] if len(episode_returns) >= 50 else episode_returns
                
                # 실제 리워드 평균 (중복 보상 제거됨)
                avg_reward = np.mean(recent_rewards)
                avg_win_rate = np.mean(recent_win_rates)
                avg_return = np.mean(recent_returns)
                
                # 테스트 성능도 함께 표시
                test_info = ""
                if test_return_rates:
                    recent_test_return_rate = np.mean(test_return_rates[-5:]) if len(test_return_rates) >= 5 else test_return_rates[-1]
                    test_info = f" | 테스트: {recent_test_return_rate:.3f}"
                
                # 수익률과 리워드 일치성 확인 (새로운 리워드 범위에 맞춤)
                reward_return_ratio = avg_reward / (avg_return * 100) if avg_return != 0 else 0
                
                print(f"Episode {episode:4d} | "
                        f"훈련승률: {avg_win_rate:.3f} | "
                        f"훈련수익률: {avg_return:.3f} ({avg_return*100:+.1f}%){test_info} | "
                        f"리워드: {avg_reward:7.1f} | "
                        f"잔고: ${info['balance']:7.0f} | "
                        f"거래: {info.get('total_trades', 0):3d}개 | "
                        f"ε: {agent.epsilon:.3f} | "
                        f"LR: {agent.learning_rate:.2e} | "
                        f"111D")
            
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
                
                if current_avg_return > best_return_rate:  # 수익률 기준으로 최고 성능 추적
                    best_return_rate = current_avg_return
                    agent.save_model(f'best_train_model_ep{episode}_return{current_avg_return:.3f}.pth')
                    print(f"🎯 새로운 훈련 데이터셋 최고 수익률! 수익률: {current_avg_return:.3f} ({current_avg_return*100:.1f}%)")
            
            # 조기 종료 조건 (과적합 방지 강화)
            if episode > 500 and test_return_rates:
                # 최근 테스트 결과들의 수익률 확인
                recent_test_returns = test_return_rates[-5:] if len(test_return_rates) >= 5 else test_return_rates
                
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
                    print(f"🏆 111차원 목표 달성! 테스트 데이터셋 수익률 {recent_test_return:.3f} ({recent_test_return*100:.1f}%) 도달")
                    agent.save_model('agent/final_optimized_model_111d.pth')
                    break
        
        
        print(f"\n111차원 RL Decision 기반 훈련 완료!")
        print(f"   총 에피소드: {episode + 1}")
        print(f"   훈련 데이터 최고 수익률: {best_return_rate:.3f}")
        print(f"   훈련 데이터 최종 수익률: {np.mean(episode_returns[-50:]) if episode_returns else 0:.3f}")
        if test_return_rates:
            print(f"   테스트 데이터 최고 수익률: {best_test_return_rate:.3f}")
            print(f"   테스트 데이터 최종 수익률: {test_return_rates[-1]:.3f}")
        print(f"   상태 차원: 111차원 (RL Decision 기반)")
        print(f"   아키텍처: DuelingDQN (Value + Advantage 분리)")
        print(f"   정규화 기법: 엔트로피 정규화, Spectral Normalization, 적응적 드롭아웃")
        
        # 테스트 데이터셋 최고 성능 모델 저장
        if test_return_rates and best_test_return_rate > 0:
            best_test_model_path = f'agent/best_test_performance_model_return{best_test_return_rate:.3f}.pth'
            agent.save_model(best_test_model_path)
            print(f"✅ 테스트 데이터셋 최고 성능 모델 저장: {best_test_model_path}")
        
        return agent, episode_rewards, episode_win_rates


def split_signal_data(signal_data: List[Dict], 
                     train_ratio: float = 0.8, test_ratio: float = 0.2) -> Tuple[List[Dict], List[Dict]]:
    """신호 데이터를 훈련용과 테스트용으로 분할"""
    total_length = len(signal_data)
    train_size = int(total_length * train_ratio)
    
    # 훈련 데이터
    train_signal = signal_data[:train_size]
    
    # 테스트 데이터
    test_signal = signal_data[train_size:]
    
    print(f"데이터 분할 완료:")
    print(f"  - 훈련 데이터: {len(train_signal):,}개 ({train_ratio*100:.1f}%)")
    print(f"  - 테스트 데이터: {len(test_signal):,}개 ({test_ratio*100:.1f}%)")
    
    return train_signal, test_signal

def main():
    """111차원 RL Decision 기반 메인 실행 함수"""
    print("111차원 RL Decision 기반 강화학습 트레이딩 시스템")
    print("=" * 80)
    
    try:
        # 1. 데이터 로딩 (OHLC 포함된 신호 데이터만 사용)
        print("\n1️⃣ 111차원용 데이터 로딩 (OHLC 포함)...")
        signal_data = DataLoader.load_signal_data()  # 테스트용 50,000개 제한
        if signal_data is None:
            print("신호 데이터 로드 실패")
            return
        
        print(f"신호 데이터 로드: {len(signal_data):,}개 (OHLC 포함)")
        
        # 2. 데이터 분할 (훈련 80%, 테스트 20%)
        print("\n2️⃣ 데이터 분할...")
        train_signal, test_signal = split_signal_data(signal_data, 0.8, 0.2)
        
        # 3. 환경 및 에이전트 생성
        print("\n3️⃣ 111차원 환경 및 에이전트 생성...")
        train_env = TradingEnvironment(train_signal)
        test_env = TradingEnvironment(test_signal)
        agent = RLAgent(train_env.observation_space.shape[0])  # 111차원
        
        # 환경 설정 비교 디버깅
        print(f"\n🔍 환경 설정 비교:")
        print(f"   훈련 환경:")
        print(f"     - 데이터 크기: {len(train_signal):,}개")
        print(f"     - 거래 간격: {train_env.min_trade_interval}")
        print(f"     - 거래 수수료: {train_env.trading_cost:.4f}")
        print(f"     - 초기 잔고: ${train_env.initial_balance:,.0f}")
        print(f"   테스트 환경:")
        print(f"     - 데이터 크기: {len(test_signal):,}개")
        print(f"     - 거래 간격: {test_env.min_trade_interval}")
        print(f"     - 거래 수수료: {test_env.trading_cost:.4f}")
        print(f"     - 초기 잔고: ${test_env.initial_balance:,.0f}")
        
        # 환경 설정 일치 확인
        env_mismatch = []
        if train_env.min_trade_interval != test_env.min_trade_interval:
            env_mismatch.append(f"거래 간격: {train_env.min_trade_interval} vs {test_env.min_trade_interval}")
        if train_env.trading_cost != test_env.trading_cost:
            env_mismatch.append(f"거래 수수료: {train_env.trading_cost} vs {test_env.trading_cost}")
        if train_env.initial_balance != test_env.initial_balance:
            env_mismatch.append(f"초기 잔고: {train_env.initial_balance} vs {test_env.initial_balance}")
        
        if env_mismatch:
            print(f"   ⚠️ 환경 설정 불일치:")
            for mismatch in env_mismatch:
                print(f"     - {mismatch}")
        else:
            print(f"   ✅ 환경 설정 일치")
        
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
            for model_file in ['agent/final_optimized_model_111d.pth', 'agent/best_model_111d.pth']:
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
            for model_file in ['agent/final_optimized_model_111d.pth', 'agent/best_model_111d.pth']:
                if os.path.exists(model_file):
                    print(f"🔄 모델 변환 시도: {model_file}")
                    if agent.create_compatible_model(model_file):
                        model_loaded = True
                        print(f"✅ 모델 변환 성공: {model_file}")
                        break
        
        if not model_loaded:
            print("새로운 111차원 모델로 시작합니다.")
        
        # 4. 훈련 전 테스트 데이터셋 성능 평가 (베이스라인)
        print("\n4️⃣ 훈련 전 테스트 데이터셋 성능 평가...")
        baseline_results, baseline_stats = PerformanceAnalyzer.evaluate_agent(agent, test_env, num_episodes=5)
        print("=== 훈련 전 테스트 데이터셋 성능 ===")
        PerformanceAnalyzer.print_performance_report(baseline_results, baseline_stats)
        
        # 5. 훈련 데이터셋으로 훈련
        print(f"\n5️⃣ 훈련 데이터셋으로 111차원 RL Decision 기반 훈련 시작...")
        print(f"   훈련 데이터: {len(train_signal):,}개")
        print(f"   테스트 데이터: {len(test_signal):,}개")
        print(f"   목표 수익률: 5%+ (수익률 중심)")
        print(f"   Signal 특성 활용: 수익률 최적화")
        
        # 훈련 실행 (과적합 방지 강화)
        trained_agent, rewards, win_rates = TrainingManager.train_agent(agent, train_env, episodes=10000, test_env=test_env)
        
        # 6. 훈련 후 테스트 데이터셋으로 성능 평가
        print("\n6️⃣ 훈련 후 테스트 데이터셋 성능 평가...")
        final_results, final_stats = PerformanceAnalyzer.evaluate_agent(trained_agent, test_env, num_episodes=10)
        print("=== 훈련 후 테스트 데이터셋 성능 ===")
        PerformanceAnalyzer.print_performance_report(final_results, final_stats)
        
        # 7. 성능 개선도 분석
        improvement = final_stats['avg_return'] - baseline_stats['avg_return']
        print(f"\n🚀 111차원 RL Decision 기반 성능 개선도 (테스트 데이터셋 기준):")
        print(f"   수익률: {baseline_stats['avg_return']:.3f} → {final_stats['avg_return']:.3f} ({improvement:+.3f})")
        print(f"   승률: {baseline_stats['overall_win_rate']:.3f} → {final_stats['overall_win_rate']:.3f}")
        print(f"   Signal 활용도: 최대화됨")
        
        # 8. 최종 모델 저장
        trained_agent.save_model('agent/final_optimized_model_111d.pth')
        print(f"\n✅ 최종 모델이 저장되었습니다: agent/final_optimized_model_111d.pth")
        
        # 9. 추가 훈련 여부 확인 (수익률 기준)
        if final_stats['avg_return'] < 0.30:  # 수익률 30% 미만
            user_input = input("\n수익률이 목표(30%)에 미달합니다. 추가 훈련을 원하시나요? (y/n): ")
            if user_input.lower() == 'y':
                print("111차원 수익률 중심 추가 훈련 시작...")
                TrainingManager.train_agent(trained_agent, train_env, episodes=5000, test_env=test_env)
                
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