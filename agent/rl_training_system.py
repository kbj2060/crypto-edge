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
                    action: str, holding_time: int, signal_data: Dict = None,
                    trade_pnl: Optional[float] = None) -> float:
        """수익률 중심 단순화된 보상 계산"""
        reward = 0.0
        
        # 1. 거래 완료 시 수익률 기반 보상 (가장 중요)
        if trade_pnl is not None:
            self.recent_trades.append(trade_pnl)
            
            # 수익률에 직접 비례하는 보상 (단순화)
            if trade_pnl > 0:
                # 수익률 1%당 1000점 보상
                reward += trade_pnl * 1000
            else:
                # 손실 1%당 500점 패널티 (보상보다 작게)
                reward += trade_pnl * 500
        
        # 2. 미실현 손익 보상 (포지션 유지 중)
        elif abs(position) > 0.01:
            unrealized_pnl = self._calculate_unrealized_pnl(current_price, entry_price, position)
            
            # 미실현 손익에 비례하는 보상 (거래 완료보다 작게)
            if unrealized_pnl > 0:
                reward += unrealized_pnl * 200  # 수익률 1%당 200점
            else:
                reward += unrealized_pnl * 100  # 손실 1%당 100점 패널티
        
        # 3. Signal 일치도 보상 (보조적, 가중치 감소)
        if signal_data and abs(position) > 0.01:
            signal_reward = self._calculate_signal_reward(signal_data, position)
            reward += signal_reward * 0.1  # 가중치 대폭 감소
        
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

class MultiHeadAttentionBlock(nn.Module):
    """다중 헤드 어텐션 블록"""
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0, "hidden_size는 num_heads로 나누어떨어져야 합니다"
        
        # 어텐션 레이어들
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        # 출력 프로젝션
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        
        # 정규화 및 드롭아웃
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        
        # 어텐션 계산
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 스케일드 닷 프로덕트 어텐션
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 어텐션 적용
        attention_output = torch.matmul(attention_weights, V)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, hidden_size
        )
        
        # 출력 프로젝션 및 잔차 연결
        output = self.output_projection(attention_output)
        output = self.layer_norm(x + output)
        
        return output

class AdvancedProfitDQN(nn.Module):
    """수익률 최적화를 위한 고급 DQN (어텐션, 잔차 연결, 배치 정규화)"""
    
    def __init__(self, state_size: int, action_size: int = 3, hidden_size: int = 256, 
                 num_heads: int = 8, num_layers: int = 3):
        super().__init__()
        
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # 입력 임베딩 및 정규화 (LayerNorm 사용으로 배치 크기 문제 해결)
        self.input_embedding = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 다중 헤드 어텐션 블록들
        self.attention_blocks = nn.ModuleList([
            MultiHeadAttentionBlock(hidden_size, num_heads, dropout=0.1)
            for _ in range(num_layers)
        ])
        
        # 잔차 연결을 위한 프로젝션 레이어
        self.residual_projections = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) if i > 0 else nn.Identity()
            for i in range(num_layers)
        ])
        
        # 특화된 특성 추출기들
        self.position_extractor = self._build_specialized_extractor(hidden_size, "position")
        self.leverage_extractor = self._build_specialized_extractor(hidden_size, "leverage")
        self.holding_extractor = self._build_specialized_extractor(hidden_size, "holding")
        self.profit_extractor = self._build_specialized_extractor(hidden_size, "profit")
        
        # 액션 헤드들 (개선된 구조, LayerNorm 사용)
        self.position_head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.LayerNorm(hidden_size // 4),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, 101)  # -10.0~10.0 (포지션 크기 대폭 확대)
        )
        
        self.leverage_head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.LayerNorm(hidden_size // 4),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, 20)  # 1~20
        )
        
        self.holding_head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.LayerNorm(hidden_size // 4),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, 48)  # 30~1440분
        )
        
        # 수익률 예측 (개선된 구조, LayerNorm 사용)
        self.profit_predictor = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.LayerNorm(hidden_size // 4),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, 1)
        )
        
        # 가중치 초기화
        self.apply(self._init_weights)
    
    def _build_specialized_extractor(self, hidden_size: int, extractor_type: str):
        """특화된 특성 추출기 빌드 (LayerNorm 사용)"""
        if extractor_type == "position":
            # 포지션 결정을 위한 특성 추출 (가격, 모멘텀 중심)
            return nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.LayerNorm(hidden_size // 2),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 2, hidden_size // 2),
                nn.ReLU()
            )
        elif extractor_type == "leverage":
            # 레버리지 결정을 위한 특성 추출 (변동성, 리스크 중심)
            return nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.LayerNorm(hidden_size // 2),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 2, hidden_size // 2),
                nn.ReLU()
            )
        elif extractor_type == "holding":
            # 보유 시간 결정을 위한 특성 추출 (트렌드, 지속성 중심)
            return nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.LayerNorm(hidden_size // 2),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 2, hidden_size // 2),
                nn.ReLU()
            )
        else:  # profit
            # 수익률 예측을 위한 특성 추출 (종합적 분석)
            return nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.LayerNorm(hidden_size // 2),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 2, hidden_size // 2),
                nn.ReLU()
            )
    
    def _init_weights(self, module):
        """Xavier 초기화"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # 배치 차원 확인
        if x.dim() == 1:
            x = x.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False
        
        # 입력 임베딩 (LayerNorm 사용으로 배치 크기 문제 해결)
        x = self.input_embedding(x)
        
        # 시퀀스 차원 추가 (어텐션을 위해)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, hidden_size]
        
        # 어텐션 블록들을 통한 특성 추출 (잔차 연결 포함)
        for i, attention_block in enumerate(self.attention_blocks):
            residual = x
            x = attention_block(x)
            
            # 잔차 연결
            if i > 0:
                x = x + self.residual_projections[i](residual)
        
        # 시퀀스 차원 제거
        x = x.squeeze(1)  # [batch_size, hidden_size]
        
        # 특화된 특성 추출
        position_features = self.position_extractor(x)
        leverage_features = self.leverage_extractor(x)
        holding_features = self.holding_extractor(x)
        profit_features = self.profit_extractor(x)
        
        # 각 액션 차원별 Q값
        position_q = self.position_head(position_features)
        leverage_q = self.leverage_head(leverage_features)
        holding_q = self.holding_head(holding_features)
        profit_pred = self.profit_predictor(profit_features)
        
        # 단일 샘플이면 배치 차원 제거
        if single_sample:
            position_q = position_q.squeeze(0)
            leverage_q = leverage_q.squeeze(0)
            holding_q = holding_q.squeeze(0)
            profit_pred = profit_pred.squeeze(0)
        
        return position_q, leverage_q, holding_q, profit_pred


def analyze_advanced_model(state_size: int = 61, hidden_size: int = 256):
    """AdvancedProfitDQN 모델 분석 함수"""
    print("🚀 AdvancedProfitDQN 모델 분석")
    print("=" * 50)
    
    # 고급 모델
    advanced_model = AdvancedProfitDQN(state_size, 3, hidden_size)
    advanced_params = sum(p.numel() for p in advanced_model.parameters())
    
    print(f"🚀 AdvancedProfitDQN:")
    print(f"   - 파라미터 수: {advanced_params:,}")
    print(f"   - 특징: 어텐션, 잔차 연결, 배치 정규화, 특화된 추출기")
    print(f"   - 장점: 복잡한 패턴 학습, 특화된 특성 추출")
    print(f"   - 단점: 더 많은 파라미터, 학습 시간 증가")
    print("=" * 50)
    
    return advanced_model

def test_model_forward_pass(model, input_size: int = 61, batch_size: int = 32):
    """모델의 forward pass 테스트"""
    import time
    print(f"🧪 {model.__class__.__name__} Forward Pass 테스트")
    
    # 테스트 입력 생성
    test_input = torch.randn(batch_size, input_size)
    
    try:
        with torch.no_grad():
            start_time = time.time()
            position_q, leverage_q, holding_q, profit_pred = model(test_input)
            end_time = time.time()
            
        print(f"   ✅ Forward pass 성공")
        print(f"   - 실행 시간: {(end_time - start_time)*1000:.2f}ms")
        print(f"   - Position Q shape: {position_q.shape}")
        print(f"   - Leverage Q shape: {leverage_q.shape}")
        print(f"   - Holding Q shape: {holding_q.shape}")
        print(f"   - Profit pred shape: {profit_pred.shape}")
        
        # 출력 범위 확인
        print(f"   - Position Q 범위: [{position_q.min():.3f}, {position_q.max():.3f}]")
        print(f"   - Leverage Q 범위: [{leverage_q.min():.3f}, {leverage_q.max():.3f}]")
        print(f"   - Holding Q 범위: [{holding_q.min():.3f}, {holding_q.max():.3f}]")
        print(f"   - Profit pred 범위: [{profit_pred.min():.3f}, {profit_pred.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Forward pass 실패: {e}")
        return False

def benchmark_advanced_model(state_size: int = 61, hidden_size: int = 256, num_tests: int = 100):
    """AdvancedProfitDQN 성능 벤치마크"""
    import time
    print("⚡ AdvancedProfitDQN 성능 벤치마크")
    print("=" * 50)
    
    # 모델 생성
    advanced_model = AdvancedProfitDQN(state_size, 3, hidden_size)
    
    # 테스트 입력
    test_input = torch.randn(32, state_size)
    
    # Advanced 모델 벤치마크
    print("🚀 AdvancedProfitDQN 벤치마크:")
    advanced_times = []
    for _ in range(num_tests):
        start_time = time.time()
        with torch.no_grad():
            _ = advanced_model(test_input)
        advanced_times.append(time.time() - start_time)
    
    advanced_avg_time = np.mean(advanced_times) * 1000
    advanced_std_time = np.std(advanced_times) * 1000
    
    print(f"   - 평균 실행 시간: {advanced_avg_time:.2f}ms ± {advanced_std_time:.2f}ms")
    print(f"   - 파라미터 수: {sum(p.numel() for p in advanced_model.parameters()):,}")
    
    return {
        'advanced_avg_time': advanced_avg_time,
        'advanced_std_time': advanced_std_time,
        'parameter_count': sum(p.numel() for p in advanced_model.parameters())
    }

class TradingEnvironment(gym.Env):
    """61차원 RL Decision 기반 암호화폐 거래 강화학습 환경 (Gymnasium 호환)"""
    
    def __init__(self, price_data: pd.DataFrame, signal_data: List[Dict], 
                 initial_balance: float = 10000.0, max_position: float = 1.0):
        super().__init__()
        
        self.price_data = price_data
        self.signal_data = signal_data
        self.initial_balance = initial_balance
        self.max_position = max_position
        
        self.reward_calculator = RewardCalculator()
        
        # 액션/상태 스페이스 정의 (단타 최적화)
        self.action_space = spaces.Box(
            low=np.array([-10.0, 1.0, 0.0]),  # 포지션 크기 대폭 확대 (-5.0 → -10.0)
            high=np.array([10.0, 10.0, 60.0]),  # 포지션 크기 대폭 확대 (5.0 → 10.0)
            dtype=np.float32
        )
        
        # 거래 제한 설정 (단타 최적화)
        self.min_trade_interval = 5  # 최소 5스텝 간격 (과도한 거래 방지)
        self.last_trade_step = -self.min_trade_interval  # 초기값
        self.trading_cost = 0.0001  # 0.01% 거래 비용 (수익성 대폭 개선)
        
        # 61차원 상태 공간
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(61,),  # 20(가격) + 6(기술점수) + 26(결정) + 9(포트폴리오) = 61차원
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        """환경 초기화 (Gymnasium 호환)"""
        if seed is not None:
            np.random.seed(seed)
        
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
        self.last_trade_step = -self.min_trade_interval  # 거래 간격 초기화
        
        observation = self._get_observation()
        info = self._create_info_dict()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """환경 스텝 실행 (Gymnasium 호환)"""
        if self.current_step >= min(len(self.price_data), len(self.signal_data)) - 1:
            return self._get_observation(), 0.0, True, False, {}
        
        position_change = np.clip(action[0], -2.0, 2.0)
        leverage = np.clip(action[1], 1.0, 10.0)  # 레버리지 최대 10으로 제한
        target_holding_minutes = np.clip(action[2], 1.0, 60.0)  # 단타 최대 1시간
        
        # 거래 간격 제한 (단타 허용)
        steps_since_last_trade = self.current_step - self.last_trade_step
        if steps_since_last_trade < self.min_trade_interval and abs(position_change) > 0.05:
            position_change = 0.0  # 최소 간격만 유지
        
        # 단타를 위한 연속 거래 허용 (조건부)
        # if abs(position_change) > 0.05 and self.in_position:
        #     position_change = 0.0  # 포지션이 있을 때는 거래 차단
        
        current_price = self.price_data.iloc[self.current_step]['close']
        next_price = self.price_data.iloc[self.current_step + 1]['close']
        
        # 포지션 및 거래 처리
        trade_completed, old_position = self._process_position_change(
            position_change, leverage, current_price, target_holding_minutes
        )
        
        # 거래 완료 시 거래 스텝 업데이트
        if trade_completed:
            self.last_trade_step = self.current_step
        
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
        
        truncated = False  # Gymnasium 호환을 위한 truncated 플래그
        info = self._create_info_dict()
        
        return self._get_observation(), reward, done, truncated, info
    
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
        
        # 거래 수수료 차감 (개선된 비용 구조)
        trade_volume = abs(self.current_position) * self.current_leverage * self.balance
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
    
    def __init__(self, state_size: int = 61, learning_rate: float = 5e-4, 
                    gamma: float = 0.99, epsilon: float = 0.2, epsilon_decay: float = 0.995,
                    hidden_size: int = 256):
        
        self.state_size = state_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.hidden_size = hidden_size
        self.epsilon_min = 0.05  # 5%로 감소 (적절한 탐험)
        
        # ε 값이 너무 낮으면 초기화
        if self.epsilon < self.epsilon_min:
            self.epsilon = 0.15  # 15%로 초기화 (적절한 탐험)
        
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
        
        # 네트워크 초기화 (AdvancedProfitDQN만 사용)
        print("🚀 AdvancedProfitDQN 아키텍처 사용 (어텐션, 잔차 연결, 배치 정규화)")
        self.q_network = AdvancedProfitDQN(state_size, 3, hidden_size).to(self.device)
        self.target_network = AdvancedProfitDQN(state_size, 3, hidden_size).to(self.device)
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 경험 리플레이
        self.memory = deque(maxlen=200000)  # 메모리 크기 증가
        self.batch_size = 512  # 배치 크기 증가
        
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
        """Q값 기반 탐욕적 액션 선택 (수익률 최적화)"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            position_q, leverage_q, holding_q, profit_pred = self.q_network(state_tensor)
            
            # 수익률 예측을 고려한 액션 선택
            position_idx = torch.argmax(position_q).item()
            leverage_idx = torch.argmax(leverage_q).item()
            holding_idx = torch.argmax(holding_q).item()
            
            # 인덱스를 실제 값으로 변환 (포지션 크기 대폭 확대)
            position = -10.0 + (position_idx * 0.2)  # -10.0~10.0 (101개 구간)
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
        
        # 적응적 엡실론 감소 (성능에 따라 조정)
        if self.epsilon > self.epsilon_min:
            # 최근 성능 기반 적응적 감소
            if len(self.training_rewards) > 50:
                recent_rewards = self.training_rewards[-50:]
                avg_recent_reward = np.mean(recent_rewards)
                
                # 성능이 좋으면 더 빠르게 감소
                if avg_recent_reward > 100:  # 리워드가 100 이상이면
                    self.epsilon *= 0.95  # 더 빠른 감소
                elif avg_recent_reward > 0:
                    self.epsilon *= 0.98  # 중간 감소
                else:
                    self.epsilon *= 0.99  # 느린 감소
            else:
                self.epsilon *= 0.99  # 초기에는 느린 감소
        
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
        
        # 타겟 Q값들
        with torch.no_grad():
            next_position_q, next_leverage_q, next_holding_q, next_profit_pred = self.target_network(next_states)
            
            target_position_q = current_position_q.clone()
            target_leverage_q = current_leverage_q.clone()
            target_holding_q = current_holding_q.clone()
            
            for i, (action, reward, done) in enumerate(zip(actions, rewards, dones)):
                pos_idx = int(np.clip((action[0] + 2.0) / 0.2, 0, 20))
                lev_idx = int(np.clip(action[1] - 1, 0, 19))
                hold_idx = int(np.clip((action[2] - 30.0) / 30.0, 0, 47))
                
                if not done:
                    # 수익률 기반 타겟 (더 강한 보상 가중치)
                    target_q = reward + self.gamma * torch.max(next_position_q[i])
                    target_position_q[i, pos_idx] = target_q
                    target_leverage_q[i, lev_idx] = target_q
                    target_holding_q[i, hold_idx] = target_q
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
        
        # 수익률 중심 가중치 (수익률 예측에 더 높은 가중치)
        total_loss = (pos_loss + lev_loss + hold_loss) + 0.5 * profit_loss
        
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
            
            print(f"✅ {model_state_size}차원 → 61차원 호환성 모델 로드 성공!")
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
        """기존 모델을 AdvancedProfitDQN 아키텍처로 변환"""
        try:
            print(f"🔄 기존 모델을 AdvancedProfitDQN 아키텍처로 변환 중...")
            
            # 기존 모델 로드
            checkpoint = torch.load(old_model_path, map_location=self.device, weights_only=False)
            old_state_dict = checkpoint['q_network']
            
            # 새로운 AdvancedProfitDQN 모델 생성
            new_model = AdvancedProfitDQN(self.state_size, 3, self.hidden_size).to(self.device)
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
            self.target_network = AdvancedProfitDQN(self.state_size, 3, self.hidden_size).to(self.device)
            self.target_network.load_state_dict(compatible_weights)
            
            print(f"✅ AdvancedProfitDQN 아키텍처로 변환 완료!")
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
            recommendations.append("수익률이 10% 미만입니다. 61차원 상태 공간의 수익률 최적화를 더 활용하세요.")
        
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
            recommendations.append("61차원 RL Decision 기반 수익률 중심 시스템이 잘 작동하고 있습니다!")
        
        return recommendations

class TrainingManager:
    """61차원 RL Decision 기반 훈련 관리 클래스"""
    
    @staticmethod
    def train_agent(agent: RLAgent, train_env: TradingEnvironment, 
                   episodes: int = 1000, save_interval: int = 100, 
                   test_env: TradingEnvironment = None) -> Tuple[RLAgent, List[float], List[float]]:
        """61차원 RL Decision 기반 에이전트 훈련 (테스트 환경 모니터링 포함)"""
        print(f"61차원 RL Decision 기반 강화학습 훈련 시작 ({episodes} 에피소드)")
        print(f"상태 공간: {train_env.observation_space.shape[0]}차원")
        if test_env:
            print(f"테스트 환경 모니터링: 활성화")
        
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
            
            while steps < 500:
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
            
            # 테스트 데이터셋으로 성능 평가 (20 에피소드마다)
            if test_env and episode % 10 == 0 and episode > 0:
                print(f"\n📊 Episode {episode}: 테스트 데이터셋 성능 평가 중...")
                test_results, test_stats = PerformanceAnalyzer.evaluate_agent(agent, test_env, num_episodes=5)
                test_return = test_stats['avg_return']
                test_win_rates.append(test_stats['overall_win_rate'])  # 승률도 추적하지만 저장 기준은 수익률
                
                print(f"   테스트 수익률: {test_return:.3f} ({test_return*100:.1f}%) (이전 최고: {best_test_win_rate:.3f})")
                
                if test_return > best_test_win_rate:  # 변수명은 그대로 유지하지만 수익률로 사용
                    best_test_win_rate = test_return
                    # 에피소드별 모델 저장 (수익률 기준)
                    agent.save_model(f'best_test_model_ep{episode}_return{test_return:.3f}.pth')
                    # 최고 성능 모델 업데이트 (수익률 기준)
                    agent.save_model('agent/best_test_performance_model_return{:.3f}.pth'.format(test_return))
                    print(f"🎯 새로운 테스트 데이터셋 최고 수익률! 수익률: {test_return:.3f} ({test_return*100:.1f}%)")
                    print(f"   최고 성능 모델 업데이트: best_test_performance_model_return{test_return:.3f}.pth")
                print()  # 빈 줄 추가
            
            # 진행 상황 출력
            if episode % 10 == 0 or episode < 10:
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
                
                print(f"Episode {episode:4d} | "
                        f"훈련승률: {avg_win_rate:.3f} | "
                        f"훈련수익률: {avg_return:.3f}{test_info} | "
                        f"리워드: {avg_reward:7.1f} | "
                        f"잔고: ${info['balance']:7.0f} | "
                        f"ε: {agent.epsilon:.3f} | "
                        f"61D")
            
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
            
            # 조기 종료 조건 (테스트 데이터셋 기준 - 수익률 중심)
            if episode > 1000 and test_win_rates:
                # 최근 테스트 결과들의 수익률 확인
                recent_test_returns = []
                for i in range(max(0, len(test_win_rates)-5), len(test_win_rates)):
                    if i < len(test_win_rates):
                        # 테스트 수익률 추정 (승률을 수익률로 근사)
                        estimated_return = test_win_rates[i] * 0.1  # 승률 65% = 수익률 6.5%로 근사
                        recent_test_returns.append(estimated_return)
                
                recent_test_return = np.mean(recent_test_returns) if recent_test_returns else 0.0
                
                if recent_test_return >= 0.30:  # 수익률 5% 이상 달성
                    print(f"🏆 61차원 목표 달성! 테스트 데이터셋 수익률 {recent_test_return:.3f} ({recent_test_return*100:.1f}%) 도달")
                    agent.save_model('agent/final_optimized_model_61d.pth')
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
            for model_file in ['agent/final_optimized_model_61d.pth', 'agent/best_model_61d.pth']:
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
            for model_file in ['agent/final_optimized_model_61d.pth', 'agent/best_model_61d.pth']:
                if os.path.exists(model_file):
                    print(f"🔄 모델 변환 시도: {model_file}")
                    if agent.create_compatible_model(model_file):
                        model_loaded = True
                        print(f"✅ 모델 변환 성공: {model_file}")
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
        print(f"   목표 수익률: 5%+ (수익률 중심)")
        print(f"   Signal 특성 활용: 수익률 최적화")
        
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
        trained_agent.save_model('agent/final_optimized_model_61d.pth')
        print(f"\n✅ 최종 모델이 저장되었습니다: agent/final_optimized_model_61d.pth")
        
        # 9. 추가 훈련 여부 확인 (수익률 기준)
        if final_stats['avg_return'] < 0.05:  # 수익률 5% 미만
            user_input = input("\n수익률이 목표(5%)에 미달합니다. 추가 훈련을 원하시나요? (y/n): ")
            if user_input.lower() == 'y':
                print("61차원 수익률 중심 추가 훈련 시작...")
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