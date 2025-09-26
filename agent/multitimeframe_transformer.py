"""
Multi-Timeframe Transformer 딥러닝 모델
- Decision 데이터를 입력으로 받아서 의사결정을 수행
- 다중 시간프레임 분석 (3m, 15m, 1h)
- Transformer 기반 어텐션 메커니즘
- 수익률 최적화 중심 설계
"""

# =============================================================================
# 하이퍼파라미터 환경변수 설정
# =============================================================================

# 모델 아키텍처 파라미터
MODEL_INPUT_SIZE = 58
MODEL_D_MODEL = 256
MODEL_NHEAD = 8
MODEL_NUM_LAYERS = 3
MODEL_DROPOUT = 0.15  # 드롭아웃 증가로 과적합 방지
MODEL_MAX_SEQ_LEN = 120

# 훈련 파라미터
TRAINING_BATCH_SIZE = 64
TRAINING_EPOCHS = 100
SEQUENCE_LENGTH = 10
TRAINING_VALIDATION_SPLIT = 0.2
TRAINING_LEARNING_RATE = 5e-4
TRAINING_WEIGHT_DECAY = 3e-5
TRAINING_PATIENCE = 5
EARLY_STOPPING_PATIENCE = 15  # 더 빠른 조기 종료

# 데이터 처리 파라미터
DATA_TEST_LIMIT = 10000
DATA_NORMALIZATION_ENABLED = True
NUM_BATCHES_PER_PRINT = 20
# 손실 함수 가중치 (균형 조정)
LOSS_PROFIT_WEIGHT = 2.0  # 수익률 가중치 감소
LOSS_ACTION_WEIGHT = 1.5  # 액션 가중치 감소
LOSS_OTHER_WEIGHT = 1.0   # 기타 가중치 유지

# 그래디언트 클리핑 (강화)
GRADIENT_CLIP_NORM = 0.5  # 더 강한 클리핑

# 모델 저장 경로
MODEL_SAVE_PATH = 'agent/best_multitimeframe_model.pth'
MODEL_SEQUENCE_SAVE_PATH = 'agent/best_multitimeframe_sequence_model.pth'
MODEL_FINAL_SAVE_PATH = 'agent/multitimeframe_transformer_trained.pth'
CHECKPOINT_SAVE_PATH = 'agent/training_checkpoint.pth'

# 데이터 파일 경로
DATA_FILE_PATH = 'agent/decisions_data.parquet'  # 최적화된 데이터 사용

# 테스트 파라미터
TEST_SAMPLES_COUNT = 10

# 고급 라벨링 파라미터
LABEL_PROFIT_THRESHOLD = 0.008  # 0.5% 기본 임계값
LABEL_VOLATILITY_FACTOR = 1.3  # 변동성 조정 계수
LABEL_TREND_FACTOR = 1.2       # 트렌드 강도 계수
LABEL_VOLUME_FACTOR = 1.2      # 거래량 계수
LABEL_MIN_CONFIDENCE = 0.3     # 최소 신뢰도
LABEL_MAX_CONFIDENCE = 0.8    # 최대 신뢰도
LABEL_LOOKAHEAD_STEPS = 5      # 미래 데이터 참조 스텝

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import os

from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# PyTorch 호환성 설정
def setup_pytorch_compatibility():
    """PyTorch 버전 호환성 설정"""
    try:
        safe_globals = [
            np.ndarray, np.dtype, np.float32, np.float64, np.int32, np.int64,
        ]
        
        try:
            import numpy._core.multiarray
            safe_globals.append(numpy._core.multiarray.scalar)
        except ImportError:
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

class DataNormalizer:
    """데이터 정규화 클래스"""
    
    def __init__(self):
        self.scalers = {}
        self.is_fitted = False
    
    def fit_transform(self, data_list: List[Dict]) -> List[Dict]:
        """데이터 정규화 적용"""
        if not data_list:
            return data_list
        
        # 특성별로 분류
        price_features = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
        indicator_features = [f'indicator_{col}' for col in [
            'vwap', 'atr', 'poc', 'hvn', 'lvn', 'vwap_std',
            'prev_day_high', 'prev_day_low', 'opening_range_high', 'opening_range_low'
        ]]
        score_features = [col for col in data_list[0].keys() if col.endswith('_score')]
        entry_stop_features = [col for col in data_list[0].keys() if col.endswith('_entry') or col.endswith('_stop')]
        
        # 데이터를 numpy 배열로 변환
        price_data = np.array([[float(data.get(f, 0.0)) for f in price_features] for data in data_list])
        indicator_data = np.array([[float(data.get(f, 0.0)) for f in indicator_features] for data in data_list])
        score_data = np.array([[float(data.get(f, 0.0)) for f in score_features] for data in data_list])
        entry_stop_data = np.array([[float(data.get(f, 0.0)) for f in entry_stop_features] for data in data_list])
        
        # 가격 데이터: Min-Max 정규화 (0-1 범위)
        self.scalers['price'] = MinMaxScaler()
        price_normalized = self.scalers['price'].fit_transform(price_data)
        
        # 지표 데이터: Standard 정규화
        self.scalers['indicator'] = StandardScaler()
        indicator_normalized = self.scalers['indicator'].fit_transform(indicator_data)
        
        # 점수 데이터: Min-Max 정규화 (0-1 범위)
        self.scalers['score'] = MinMaxScaler()
        score_normalized = self.scalers['score'].fit_transform(score_data)
        
        # Entry/Stop 데이터: Min-Max 정규화 (0-1 범위)
        self.scalers['entry_stop'] = MinMaxScaler()
        entry_stop_normalized = self.scalers['entry_stop'].fit_transform(entry_stop_data)
        
        # 정규화된 데이터를 원본 형태로 복원
        normalized_data = []
        for i, data in enumerate(data_list):
            normalized_item = data.copy()
            
            # 가격 데이터 정규화
            for j, feature in enumerate(price_features):
                normalized_item[feature] = price_normalized[i][j]
            
            # 지표 데이터 정규화
            for j, feature in enumerate(indicator_features):
                normalized_item[feature] = indicator_normalized[i][j]
            
            # 점수 데이터 정규화
            for j, feature in enumerate(score_features):
                normalized_item[feature] = score_normalized[i][j]
            
            # Entry/Stop 데이터 정규화
            for j, feature in enumerate(entry_stop_features):
                normalized_item[feature] = entry_stop_normalized[i][j]
            
            normalized_data.append(normalized_item)
        
        self.is_fitted = True
        return normalized_data
    
    def transform(self, data_list: List[Dict]) -> List[Dict]:
        """이미 피팅된 스케일러로 정규화 적용 (fit 없이)"""
        if not self.is_fitted:
            raise ValueError("스케일러가 피팅되지 않았습니다. fit_transform을 먼저 호출하세요.")
        
        if not data_list:
            return data_list
        
        # 특성별로 분류 (fit_transform과 동일)
        price_features = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
        indicator_features = [f'indicator_{col}' for col in [
            'vwap', 'atr', 'poc', 'hvn', 'lvn', 'vwap_std',
            'prev_day_high', 'prev_day_low', 'opening_range_high', 'opening_range_low'
        ]]
        score_features = [col for col in data_list[0].keys() if col.endswith('_score')]
        entry_stop_features = [col for col in data_list[0].keys() if col.endswith('_entry') or col.endswith('_stop')]
        
        # 데이터를 numpy 배열로 변환
        price_data = np.array([[float(data.get(f, 0.0)) for f in price_features] for data in data_list])
        indicator_data = np.array([[float(data.get(f, 0.0)) for f in indicator_features] for data in data_list])
        score_data = np.array([[float(data.get(f, 0.0)) for f in score_features] for data in data_list])
        entry_stop_data = np.array([[float(data.get(f, 0.0)) for f in entry_stop_features] for data in data_list])
        
        # 이미 피팅된 스케일러로 transform만 수행
        price_normalized = self.scalers['price'].transform(price_data)
        indicator_normalized = self.scalers['indicator'].transform(indicator_data)
        score_normalized = self.scalers['score'].transform(score_data)
        entry_stop_normalized = self.scalers['entry_stop'].transform(entry_stop_data)
        
        # 정규화된 데이터를 원본 형태로 복원
        normalized_data = []
        for i, data in enumerate(data_list):
            normalized_item = data.copy()
            
            # 가격 데이터 정규화
            for j, feature in enumerate(price_features):
                normalized_item[feature] = price_normalized[i][j]
            
            # 지표 데이터 정규화
            for j, feature in enumerate(indicator_features):
                normalized_item[feature] = indicator_normalized[i][j]
            
            # 점수 데이터 정규화
            for j, feature in enumerate(score_features):
                normalized_item[feature] = score_normalized[i][j]
            
            # Entry/Stop 데이터 정규화
            for j, feature in enumerate(entry_stop_features):
                normalized_item[feature] = entry_stop_normalized[i][j]
            
            normalized_data.append(normalized_item)
        
        return normalized_data

class PositionalEncoding(nn.Module):
    """위치 인코딩"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TemporalAttention(nn.Module):
    """개선된 시간적 패턴 어텐션 (Residual + Skip Connections)"""
    def __init__(self, d_model: int, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        
        # Multi-head attention for temporal patterns
        self.temporal_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        
        # Pre-norm architecture (더 안정적인 학습)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Enhanced Feed Forward Network with residual
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        # Skip connection projection (차원이 다를 때)
        self.skip_projection = nn.Linear(d_model, d_model) if d_model != d_model else nn.Identity()
        
        # Attention weights storage for visualization
        self.attention_weights = None
        
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len] (optional)
        """
        # Pre-norm + Self-attention
        norm_x = self.norm1(x)
        attn_out, attn_weights = self.temporal_attention(norm_x, norm_x, norm_x, attn_mask=mask)
        
        # Store attention weights for visualization
        self.attention_weights = attn_weights.detach().cpu()
        
        # Residual connection
        x = x + attn_out
        
        # Pre-norm + Feed Forward
        norm_x = self.norm2(x)
        ffn_out = self.ffn(norm_x)
        
        # Skip connection + Final residual
        skip_out = self.skip_projection(x)
        x = self.norm3(skip_out + ffn_out)
        
        return x

class MultiScaleFeatureExtractor(nn.Module):
    """간단한 다중 스케일 특성 추출기"""
    def __init__(self, d_model: int, scales: List[int] = [1, 2, 4], dropout: float = 0.1):
        super().__init__()
        self.scales = scales
        self.d_model = d_model
        
        # 단일 컨볼루션 (다중 스케일 대신)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 특성 융합
        self.fusion = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        # [batch_size, seq_len, d_model] -> [batch_size, d_model, seq_len]
        x_conv = x.transpose(1, 2)
        
        # 컨볼루션 적용
        conv_feat = self.conv(x_conv)  # [batch_size, d_model, seq_len]
        
        # [batch_size, d_model, seq_len] -> [batch_size, seq_len, d_model]
        conv_feat = conv_feat.transpose(1, 2)
        
        # 융합
        fused_features = self.fusion(conv_feat)
        
        return fused_features

class CrossTimeframeAttention(nn.Module):
    """개선된 Cross-Timeframe Attention (Multi-Scale + Residual)"""
    def __init__(self, feature_dim: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim  # d_model//2
        
        # Multi-head attention
        self.cross_attention = nn.MultiheadAttention(
            feature_dim, nhead, dropout=dropout, batch_first=True
        )
        
        # Pre-norm architecture
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        
        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Dropout(dropout)
        )
        
        # Skip connection
        self.skip_projection = nn.Identity()
    
    def forward(self, short_term, medium_term, long_term):
        # 모든 입력이 [batch_size, feature_dim]
        timeframes = torch.stack([short_term, medium_term, long_term], dim=1)
        # [batch_size, 3, feature_dim]
        
        # Pre-norm + Cross-attention
        norm_timeframes = self.norm1(timeframes)
        attn_out, _ = self.cross_attention(norm_timeframes, norm_timeframes, norm_timeframes)
        
        # Residual connection
        timeframes = timeframes + attn_out
        
        # Pre-norm + Feed Forward
        norm_timeframes = self.norm2(timeframes)
        ffn_out = self.ffn(norm_timeframes)
        
        # Final residual
        timeframes = timeframes + ffn_out
        
        return timeframes[:, 0, :], timeframes[:, 1, :], timeframes[:, 2, :]

class AdaptivePooling(nn.Module):
    """적응형 풀링 레이어"""
    def __init__(self, d_model: int, output_size: int = 1):
        super().__init__()
        self.d_model = d_model
        self.output_size = output_size
        
        # Attention-based pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
            nn.Softmax(dim=1)
        )
        
        # Adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool1d(output_size)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        # Attention weights 계산
        attention_weights = self.attention_pool(x)  # [batch_size, seq_len, 1]
        
        # Attention-weighted pooling
        attention_pooled = torch.sum(x * attention_weights, dim=1)  # [batch_size, d_model]
        
        # Adaptive pooling
        x_transposed = x.transpose(1, 2)  # [batch_size, d_model, seq_len]
        adaptive_pooled = self.adaptive_pool(x_transposed).squeeze(-1)  # [batch_size, d_model]
        
        # 결합
        combined = attention_pooled + adaptive_pooled
        
        return combined

class FinancialPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=0.1)
        
        # 모든 PE를 [max_len, d_model] 형태로 생성
        self.register_buffer('basic_pe', self._create_sinusoidal_pe(max_len, d_model))
        self.register_buffer('daily_pe', self._create_cyclical_pe(max_len, d_model, 24))
        self.register_buffer('weekly_pe', self._create_cyclical_pe(max_len, d_model, 168))
        
        # 학습 가능한 위치 임베딩
        self.learnable_pe = nn.Parameter(torch.randn(max_len, d_model) * 0.1)
    
    def _create_sinusoidal_pe(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # [max_len, d_model]
    
    def _create_cyclical_pe(self, max_len, d_model, period):
        pe = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = torch.sin(torch.tensor(2 * np.pi * pos / period, dtype=torch.float))
                if i + 1 < d_model:
                    pe[pos, i + 1] = torch.cos(torch.tensor(2 * np.pi * pos / period, dtype=torch.float))
        return pe  # [max_len, d_model] - 올바른 차원
    
    def forward(self, x):
        seq_len = x.size(1)
        
        # 올바른 슬라이싱
        combined_pe = (
            self.basic_pe[:seq_len] + 
            0.3 * self.daily_pe[:seq_len] + 
            0.2 * self.weekly_pe[:seq_len] + 
            0.1 * self.learnable_pe[:seq_len]
        )  # [seq_len, d_model]
        
        # 배치 차원 추가
        combined_pe = combined_pe.unsqueeze(0).expand(x.size(0), -1, -1)
        
        return self.dropout(x + combined_pe)

class MultiTimeframeTransformer(nn.Module):
    """Enhanced Multi-Timeframe Transformer 모델 with Advanced Attention Mechanisms"""
    
    def __init__(self, 
                    input_size: int = MODEL_INPUT_SIZE,
                    d_model: int = MODEL_D_MODEL,
                    nhead: int = MODEL_NHEAD,
                    num_layers: int = MODEL_NUM_LAYERS,
                    dropout: float = MODEL_DROPOUT,
                    max_seq_len: int = MODEL_MAX_SEQ_LEN,
                    use_enhanced_attention: bool = True):
        super().__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.use_enhanced_attention = use_enhanced_attention
        
        # 개선된 입력 임베딩
        self.input_embedding = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),  # ReLU 대신 GELU 사용
            nn.Dropout(dropout)
        )
        
        # 금융 특화 위치 인코딩
        self.pos_encoding = FinancialPositionalEncoding(d_model, max_seq_len)
        
        # Multi-Scale Feature Extractor
        if use_enhanced_attention:
            self.multi_scale_extractor = MultiScaleFeatureExtractor(d_model, scales=[1, 2, 4], dropout=dropout)
        
        # Temporal Attention (시간적 패턴 집중)
        if use_enhanced_attention:
            self.temporal_attention = TemporalAttention(d_model, nhead, dropout)
        
        # Transformer 인코더 (개선된 설정)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm architecture
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Adaptive Pooling
        if use_enhanced_attention:
            self.adaptive_pooling = AdaptivePooling(d_model, output_size=1)
        
        # Cross-Timeframe Attention (시간프레임 간 정보 교환)
        if use_enhanced_attention:
            self.cross_timeframe_attention = CrossTimeframeAttention(d_model//2, nhead//2, dropout)
        
        # 시간프레임별 특성 추출기 (개선됨)
        self.timeframe_extractors = nn.ModuleDict({
            'short_term': self._build_enhanced_timeframe_extractor(d_model, 'short'),
            'medium_term': self._build_enhanced_timeframe_extractor(d_model, 'medium'),
            'long_term': self._build_enhanced_timeframe_extractor(d_model, 'long')
        })
        
        # 개선된 의사결정 헤드들
        self.decision_heads = nn.ModuleDict({
            'action': self._build_decision_head(d_model, 3, dropout),  # BUY, SELL, HOLD
            'confidence': self._build_decision_head(d_model, 1, dropout, activation='sigmoid'),
            'risk': self._build_decision_head(d_model, 1, dropout)  # 리스크 예측 추가
        })
        
        # 수익률 예측기 (개선됨)
        self.profit_predictor = self._build_decision_head(d_model, 1, dropout)
        
        # Attention 가중치 저장용
        self.attention_weights = {}
        
        # 가중치 초기화
        self.apply(self._init_weights)
    
    def _build_enhanced_timeframe_extractor(self, d_model: int, timeframe: str):
        """압축된 시간프레임별 특성 추출기 (효율성 개선)"""
        if timeframe == 'short':
            # 단기: 압축된 구조로 빠른 처리
            return nn.Sequential(
                nn.Linear(d_model, d_model // 2),  # 차원 축소
                nn.GELU(),
                nn.LayerNorm(d_model // 2),
                nn.Dropout(0.05),
                nn.Linear(d_model // 2, d_model // 2)
            )
        elif timeframe == 'medium':
            # 중기: 균형잡힌 압축 구조
            return nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.LayerNorm(d_model // 2),
                nn.Dropout(0.1),
                nn.Linear(d_model // 2, d_model // 2)
            )
        else:  # long
            # 장기: 안정적인 압축 구조
            return nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.LayerNorm(d_model // 2),
                nn.Dropout(0.15),
                nn.Linear(d_model // 2, d_model // 2)
            )
    
    def _build_decision_head(self, d_model: int, output_size: int, dropout: float, activation: str = None):
        """개선된 의사결정 헤드 빌더"""
        layers = [
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.LayerNorm(d_model // 2),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_size)
        ]
        
        if activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif activation == 'softmax':
            layers.append(nn.Softmax(dim=-1))
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        
        return nn.Sequential(*layers)
    
    def _build_timeframe_extractor(self, d_model: int, timeframe: str):
        """기존 시간프레임별 특성 추출기 (하위 호환성)"""
        if timeframe == 'short':
            # 단기: 빠른 반응, 높은 민감도
            return nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.LayerNorm(d_model // 2),
                nn.Dropout(0.05),  # 낮은 드롭아웃
                nn.Linear(d_model // 2, d_model // 2)
            )
        elif timeframe == 'medium':
            # 중기: 균형잡힌 분석
            return nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.LayerNorm(d_model // 2),
                nn.Dropout(0.1),
                nn.Linear(d_model // 2, d_model // 2)
            )
        else:  # long
            # 장기: 안정적, 낮은 민감도
            return nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.LayerNorm(d_model // 2),
                nn.Dropout(0.15),  # 높은 드롭아웃
                nn.Linear(d_model // 2, d_model // 2)
            )
    
    def _init_weights(self, module):
        """Xavier 초기화"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x, mask=None):
        """
        Enhanced forward pass with advanced attention mechanisms
        Args:
            x: [batch_size, seq_len, input_size] 또는 [batch_size, input_size]
            mask: [batch_size, seq_len] (선택적)
        """
        # 배치 차원 확인
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, input_size]
            single_sample = True
        else:
            single_sample = False
        
        batch_size, seq_len, _ = x.shape
        
        # 입력 임베딩
        x = self.input_embedding(x)  # [batch_size, seq_len, d_model]
        
        # 금융 특화 위치 인코딩
        x = self.pos_encoding(x)  # [batch_size, seq_len, d_model]
        
        # Multi-Scale Feature Extraction
        if self.use_enhanced_attention:
            x = self.multi_scale_extractor(x)  # [batch_size, seq_len, d_model]
        
        # Temporal Attention (시간적 패턴 집중)
        if self.use_enhanced_attention:
            x = self.temporal_attention(x, mask)
            # Attention 가중치 저장
            if hasattr(self.temporal_attention, 'attention_weights'):
                self.attention_weights['temporal'] = self.temporal_attention.attention_weights
        
        # Transformer 인코더
        if mask is not None:
            # 마스크 적용 (패딩 토큰 등)
            x = self.transformer(x, src_key_padding_mask=mask)
        else:
            x = self.transformer(x)
        
        # Adaptive Pooling으로 최종 특성 추출
        if self.use_enhanced_attention:
            final_features = self.adaptive_pooling(x)  # [batch_size, d_model]
        else:
            # 기존 방식 (마지막 시퀀스 요소 사용)
            final_features = x[:, -1, :]  # [batch_size, d_model]
        
        # 시간프레임별 특성 추출
        timeframe_features = {}
        for timeframe, extractor in self.timeframe_extractors.items():
            timeframe_features[timeframe] = extractor(final_features)
        
        # Cross-Timeframe Attention (시간프레임 간 정보 교환)
        if self.use_enhanced_attention:
            enhanced_short, enhanced_medium, enhanced_long = self.cross_timeframe_attention(
                timeframe_features['short_term'],
                timeframe_features['medium_term'], 
                timeframe_features['long_term']
            )
            
            # Enhanced timeframe features 업데이트
            timeframe_features['short_term'] = enhanced_short
            timeframe_features['medium_term'] = enhanced_medium
            timeframe_features['long_term'] = enhanced_long
            
            # Cross-timeframe attention 가중치 저장
            if hasattr(self.cross_timeframe_attention, 'attention_weights'):
                self.attention_weights['cross_timeframe'] = self.cross_timeframe_attention.attention_weights
        
        # 의사결정 출력
        decisions = {}
        for head_name, head in self.decision_heads.items():
            decisions[head_name] = head(final_features)
        
        # 수익률 예측
        profit_pred = self.profit_predictor(final_features)
        
        # 단일 샘플이면 배치 차원 제거
        if single_sample:
            for key in decisions:
                decisions[key] = decisions[key].squeeze(0)
            profit_pred = profit_pred.squeeze(0)
            for key in timeframe_features:
                timeframe_features[key] = timeframe_features[key].squeeze(0)
        
        return decisions, profit_pred, timeframe_features
    
    def get_attention_weights(self):
        """Attention 가중치 반환 (시각화용)"""
        return self.attention_weights
    
    def clear_attention_weights(self):
        """Attention 가중치 초기화"""
        self.attention_weights = {}

class MultiTimeframeDecisionEngine:
    """Multi-Timeframe 의사결정 엔진"""
    
    def __init__(self, 
                    model_path: str = MODEL_SAVE_PATH,
                    input_size: int = MODEL_INPUT_SIZE,
                    d_model: int = MODEL_D_MODEL,
                    nhead: int = MODEL_NHEAD,
                    num_layers: int = MODEL_NUM_LAYERS,
                    device: str = 'auto',
                    use_enhanced_attention: bool = True):
        
        # 디바이스 설정
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Enhanced Multi-Timeframe Transformer 디바이스: {self.device}")
        print(f"Enhanced Attention 사용: {use_enhanced_attention}")
        
        # 모델 초기화
        self.model = MultiTimeframeTransformer(
            input_size=input_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            use_enhanced_attention=use_enhanced_attention
        ).to(self.device)
        
        # 개선된 옵티마이저 (더 나은 설정)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=TRAINING_LEARNING_RATE,
            weight_decay=TRAINING_WEIGHT_DECAY,
            betas=(0.9, 0.999),  # 기본값 유지
            eps=1e-8,
            amsgrad=False
        )
        
        # Warmup + Cosine Annealing 스케줄러
        from torch.optim.lr_scheduler import LinearLR, SequentialLR
        
        # Warmup 스케줄러 (처음 5 에포크)
        warmup_scheduler = LinearLR(
            self.optimizer, 
            start_factor=0.1, 
            end_factor=1.0, 
            total_iters=5
        )
        
        # Cosine Annealing 스케줄러
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # 10 에포크마다 리셋
            T_mult=2,  # 리셋 주기 2배씩 증가
            eta_min=1e-6  # 최소 학습률
        )
        
        # 순차적 스케줄러
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[5]  # 5 에포크 후 cosine으로 전환
        )
        
        # 모델 로드
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        # 의사결정 히스토리
        self.decision_history = []
        self.performance_history = []
        
        # 통계
        self.total_decisions = 0
        self.correct_decisions = 0
        self.total_profit = 0.0
    
    def _extract_features(self, decision_data: Dict) -> List[float]:
        """단일 Decision 데이터에서 58차원 특성 추출"""
        features = []
        
        # 1. 전략 Action (16개)
        strategy_actions = [
            'bollinger_squeeze_action', 'zscore_mean_reversion_action', 'vpvr_action', 
            'ema_confluence_action', 'ichimoku_action', 'htf_trend_action', 
            'funding_rate_action', 'orderflow_cvd_action', 'support_resistance_action', 
            'vwap_pinball_action', 'multi_timeframe_action', 'session_action', 
            'vol_spike_action', 'vpvr_micro_action', 'oi_delta_action', 'liquidity_grab_action'
        ]
        
        for action_col in strategy_actions:
            action = decision_data.get(action_col, 'HOLD')
            if isinstance(action, str):
                if action.upper() == 'BUY':
                    action_value = 1.0
                elif action.upper() == 'SELL':
                    action_value = -1.0
                else:
                    action_value = 0.0
            else:
                action_value = float(action) if action else 0.0
            features.append(action_value)
        
        # 2. 전략 Score (16개)
        strategy_scores = [
            'bollinger_squeeze_score', 'session_score', 'vpvr_score', 'oi_delta_score',
            'ema_confluence_score', 'multi_timeframe_score', 'vol_spike_score', 
            'vpvr_micro_score', 'htf_trend_score', 'liquidity_grab_score', 
            'ichimoku_score', 'funding_rate_score', 'support_resistance_score', 
            'zscore_mean_reversion_score', 'vwap_pinball_score', 'orderflow_cvd_score'
        ]
        
        for score_col in strategy_scores:
            features.append(float(decision_data.get(score_col, 0.0)))
        
        # 3. 전략 Confidence (1개 - vpvr_confidence만 있음)
        confidence = decision_data.get('vpvr_confidence', 0.0)
        if isinstance(confidence, str):
            confidence_value = 0.0
        else:
            confidence_value = float(confidence) if confidence else 0.0
        features.append(confidence_value)
        
        # 4. 전략 Entry (4개)
        strategy_entries = ['vpvr_micro_entry', 'vwap_pinball_entry', 'session_entry', 'vpvr_entry']
        for entry_col in strategy_entries:
            features.append(float(decision_data.get(entry_col, 0.0)))
        
        # 5. 전략 Stop (4개)
        strategy_stops = ['vpvr_stop', 'session_stop', 'vpvr_micro_stop', 'vwap_pinball_stop']
        for stop_col in strategy_stops:
            features.append(float(decision_data.get(stop_col, 0.0)))
        
        # 6. Indicator 정보 (10개)
        indicator_fields = [
            'indicator_prev_day_high', 'indicator_poc', 'indicator_hvn', 'indicator_vwap_std',
            'indicator_opening_range_high', 'indicator_lvn', 'indicator_opening_range_low',
            'indicator_vwap', 'indicator_prev_day_low', 'indicator_atr'
        ]
        
        for field in indicator_fields:
            features.append(float(decision_data.get(field, 0.0)))
        
        # 7. OHLC 데이터 (6개)
        features.extend([
            float(decision_data.get('open')),
            float(decision_data.get('high')),
            float(decision_data.get('low')),
            float(decision_data.get('close')),
            float(decision_data.get('volume')),
            float(decision_data.get('quote_volume'))
        ])
        
        # 8. Timestamp (1개)
        timestamp = decision_data.get('timestamp')
        if isinstance(timestamp, str):
            try:
                timestamp = pd.to_datetime(timestamp).timestamp()
            except:
                timestamp = 0.0
        else:
            timestamp = float(timestamp) if timestamp else 0.0
        
        features.append(timestamp)
        
        # 차원 검증
        assert len(features) == 58, f"Expected 58 features, got {len(features)}"
        
        return features
    
    def preprocess_sequence_data(self, decision_data: List[Dict], seq_len: int = SEQUENCE_LENGTH) -> torch.Tensor:
        """시퀀스 형태로 데이터 전처리"""
        sequences = []
        
        for i in range(len(decision_data) - seq_len + 1):
            sequence = []
            for j in range(seq_len):
                features = self._extract_features(decision_data[i + j])
                sequence.append(features)
            sequences.append(sequence)
        
        return torch.FloatTensor(sequences).to(self.device)
    
    def make_sequence_decision(self, decision_sequence: List[Dict], seq_len: int = SEQUENCE_LENGTH) -> Dict:
        """시퀀스 기반 의사결정 수행 (Transformer의 진정한 장점 활용)"""
        self.model.eval()
        
        with torch.no_grad():
            # 시퀀스 데이터 전처리
            if len(decision_sequence) < seq_len:
                # 시퀀스가 짧으면 패딩
                padded_sequence = decision_sequence + [decision_sequence[-1]] * (seq_len - len(decision_sequence))
                input_tensor = self.preprocess_sequence_data(padded_sequence, seq_len)
            else:
                # 마지막 seq_len개만 사용
                recent_sequence = decision_sequence[-seq_len:]
                input_tensor = self.preprocess_sequence_data(recent_sequence, seq_len)
            
            # 모델 추론 (시퀀스 형태)
            decisions, profit_pred, timeframe_features = self.model(input_tensor)
            
            # 의사결정 결과 생성 (단순화)
            result = {
                'action': self._interpret_action(decisions['action']),
                'confidence': float(decisions['confidence'].item()),
                'profit': float(profit_pred.item()),
                'sequence_length': seq_len,
                'timestamp': datetime.now().isoformat(),
                'model_version': 'MultiTimeframeTransformer_v2.0_Sequence'
            }
            
            # 히스토리에 저장
            self.decision_history.append({
                'input': decision_sequence[-1] if decision_sequence else {},  # 마지막 데이터만 저장
                'output': result,
                'timestamp': datetime.now()
            })
            
            self.total_decisions += 1
            
            return result
    
    def get_attention_visualization(self, decision_sequence: List[Dict], seq_len: int = SEQUENCE_LENGTH) -> Dict:
        """Attention 가중치 시각화를 위한 데이터 반환"""
        self.model.eval()
        
        with torch.no_grad():
            # 시퀀스 데이터 전처리
            if len(decision_sequence) < seq_len:
                padded_sequence = decision_sequence + [decision_sequence[-1]] * (seq_len - len(decision_sequence))
                input_tensor = self.preprocess_sequence_data(padded_sequence, seq_len)
            else:
                recent_sequence = decision_sequence[-seq_len:]
                input_tensor = self.preprocess_sequence_data(recent_sequence, seq_len)
            
            # 모델 추론
            decisions, profit_pred, timeframe_features = self.model(input_tensor)
            
            # Attention 가중치 가져오기
            attention_weights = self.model.get_attention_weights()
            
            # 시각화용 데이터 구성
            visualization_data = {
                'temporal_attention': attention_weights.get('temporal', None),
                'cross_timeframe_attention': attention_weights.get('cross_timeframe', None),
                'decisions': {
                    'action': decisions['action'].cpu().numpy(),
                    'confidence': decisions['confidence'].cpu().numpy(),
                    'risk': decisions.get('risk', torch.tensor([0.0])).cpu().numpy()
                },
                'profit_prediction': profit_pred.cpu().numpy(),
                'timeframe_features': {
                    k: v.cpu().numpy() for k, v in timeframe_features.items()
                }
            }
            
            return visualization_data
    
    def _interpret_action(self, action_logits: torch.Tensor) -> str:
        """액션 로짓을 해석 (확률 분포 확인 포함)"""
        # 소프트맥스 적용하여 확률로 변환
        probabilities = torch.softmax(action_logits, dim=-1)
        action_idx = torch.argmax(action_logits).item()
        actions = ['HOLD', 'BUY', 'SELL']
        
        # 디버깅: 액션 분포 출력
        
        return actions[action_idx]
    
    
    def train_on_sequence_batch(self, batch_sequences: List[List[Dict]], batch_labels: List[Dict], seq_len: int = SEQUENCE_LENGTH) -> float:
        """시퀀스 배치 학습 (Transformer의 진정한 장점 활용)"""
        self.model.train()
        
        # 배치 전처리
        inputs = []
        targets = []
        
        for sequence, labels in zip(batch_sequences, batch_labels):
            # ✅ 시퀀스를 직접 텐서로 변환 (preprocess_sequence_data 사용 안 함)
            if len(sequence) < seq_len:
                # 패딩
                padded_sequence = sequence + [sequence[-1]] * (seq_len - len(sequence))
            else:
                # 마지막 seq_len개만 사용
                padded_sequence = sequence[-seq_len:]
            
            # ✅ 직접 특성 추출
            sequence_features = []
            for data_point in padded_sequence:
                features = self._extract_features(data_point)
                sequence_features.append(features)
            
            # [seq_len, feature_dim] → [1, seq_len, feature_dim]
            input_tensor = torch.FloatTensor(sequence_features).unsqueeze(0).to(self.device)
            inputs.append(input_tensor)
            
            # 타겟 생성 (단순화)
            target = {
                'action': torch.tensor([labels.get('action', 0)], dtype=torch.long).to(self.device),
                'confidence': torch.tensor([labels.get('confidence', 0.5)], dtype=torch.float).to(self.device),
                'profit': torch.tensor([labels.get('profit', 0.0)], dtype=torch.float).to(self.device)
            }
            targets.append(target)
        
        # 배치 결합
        batch_input = torch.cat(inputs, dim=0)
        
        # 순전파
        decisions, profit_pred, _ = self.model(batch_input)
        
        # 손실 계산 (가중치 적용)
        total_loss = 0.0
        
        # 가중치 설정
        profit_loss_weight = LOSS_PROFIT_WEIGHT
        action_loss_weight = LOSS_ACTION_WEIGHT
        other_loss_weight = LOSS_OTHER_WEIGHT
        
        # 액션 분류 손실
        action_targets = torch.cat([t['action'] for t in targets])
        action_loss = F.cross_entropy(decisions['action'], action_targets)
        total_loss += action_loss * action_loss_weight
        
        # 회귀 손실들 (단순화)
        for key in ['confidence']:
            pred = decisions[key].squeeze(-1)  # 마지막 차원만 제거
            target = torch.cat([t[key] for t in targets])
            loss = F.mse_loss(pred, target)
            total_loss += loss * other_loss_weight
        
        # 수익률 예측 손실 (가장 중요)
        profit_targets = torch.cat([t['profit'] for t in targets])
        profit_loss = F.mse_loss(profit_pred.squeeze(-1), profit_targets)  # 마지막 차원만 제거
        total_loss += profit_loss * profit_loss_weight
        
        # 역전파
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRADIENT_CLIP_NORM)
        
        self.optimizer.step()
        
        return float(total_loss.item())
    
    def save_model(self, filepath: str) -> bool:
        """모델 저장"""
        try:
            os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
            
            save_dict = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'total_decisions': self.total_decisions,
                'correct_decisions': self.correct_decisions,
                'total_profit': self.total_profit,
                'decision_history': self.decision_history[-1000:],  # 최근 1000개만 저장
                'model_config': {
                    'input_size': self.model.input_size,
                    'd_model': self.model.d_model,
                    'nhead': self.model.nhead,
                    'num_layers': self.model.num_layers
                }
            }
            
            torch.save(save_dict, filepath)
            print(f"Multi-Timeframe Transformer 모델 저장 완료: {filepath}")
            return True
            
        except Exception as e:
            print(f"모델 저장 실패: {e}")
            return False
    
    def save_checkpoint(self, filepath: str, epoch: int, train_loss: float, val_loss: float, 
                       training_history: List[Dict], best_val_loss: float) -> bool:
        """학습 체크포인트 저장 (학습 이어하기용)"""
        try:
            os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
            
            checkpoint_dict = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'total_decisions': self.total_decisions,
                'correct_decisions': self.correct_decisions,
                'total_profit': self.total_profit,
                'decision_history': self.decision_history[-1000:],
                'model_config': {
                    'input_size': self.model.input_size,
                    'd_model': self.model.d_model,
                    'nhead': self.model.nhead,
                    'num_layers': self.model.num_layers
                },
                # 학습 상태 추가
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'training_history': training_history,
                'best_val_loss': best_val_loss,
                'timestamp': datetime.now().isoformat()
            }
            
            torch.save(checkpoint_dict, filepath)
            print(f"학습 체크포인트 저장 완료: {filepath}")
            print(f"  에포크: {epoch}, 검증 손실: {val_loss:.4f}")
            return True
            
        except Exception as e:
            print(f"체크포인트 저장 실패: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """모델 로드"""
        if not os.path.exists(filepath):
            print(f"모델 파일이 없습니다: {filepath}")
            return False
        
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            # 모델 상태 로드
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # 통계 로드
            self.total_decisions = checkpoint.get('total_decisions', 0)
            self.correct_decisions = checkpoint.get('correct_decisions', 0)
            self.total_profit = checkpoint.get('total_profit', 0.0)
            self.decision_history = checkpoint.get('decision_history', [])
            
            print(f"Multi-Timeframe Transformer 모델 로드 성공: {filepath}")
            print(f"   총 의사결정: {self.total_decisions}")
            print(f"   정확도: {self.correct_decisions/max(self.total_decisions, 1):.3f}")
            print(f"   총 수익: {self.total_profit:.3f}")
            
            return True
            
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            return False
    
    def load_checkpoint(self, filepath: str) -> Optional[Dict]:
        """학습 체크포인트 로드 (학습 이어하기용)"""
        if not os.path.exists(filepath):
            print(f"체크포인트 파일이 없습니다: {filepath}")
            return None
        
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            # 모델 상태 로드
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # 통계 로드
            self.total_decisions = checkpoint.get('total_decisions', 0)
            self.correct_decisions = checkpoint.get('correct_decisions', 0)
            self.total_profit = checkpoint.get('total_profit', 0.0)
            self.decision_history = checkpoint.get('decision_history', [])
            
            # 학습 상태 로드
            training_state = {
                'epoch': checkpoint.get('epoch', 0),
                'train_loss': checkpoint.get('train_loss', 0.0),
                'val_loss': checkpoint.get('val_loss', 0.0),
                'training_history': checkpoint.get('training_history', []),
                'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
                'timestamp': checkpoint.get('timestamp', '')
            }
            
            print(f"학습 체크포인트 로드 성공: {filepath}")
            print(f"   에포크: {training_state['epoch']}")
            print(f"   훈련 손실: {training_state['train_loss']:.4f}")
            print(f"   검증 손실: {training_state['val_loss']:.4f}")
            print(f"   최고 검증 손실: {training_state['best_val_loss']:.4f}")
            print(f"   저장 시간: {training_state['timestamp']}")
            
            return training_state
            
        except Exception as e:
            print(f"체크포인트 로드 실패: {e}")
            return None

class DecisionDataLoader:
    """Decision 데이터 로더"""
    
    @staticmethod
    def load_decision_data(file_path: str = 'agent/decisions_data.parquet') -> Optional[List[Dict]]:
        """Decision 데이터 로드"""
        if not os.path.exists(file_path):
            print(f"Decision 데이터 파일이 없습니다: {file_path}")
            return None
        
        try:
            print(f"Decision 데이터 로드 중: {file_path}")
            df = pd.read_parquet(file_path)
            print(f"Decision 데이터 로드 완료: {len(df):,}개 레코드")
            
            # DataFrame을 Dict 리스트로 변환 (메모리 최적화)
            decision_data = []
            for idx, row in df.iterrows():
                decision_dict = {}
                for col, value in row.items():
                    if pd.notna(value):
                        # 데이터 타입 최적화
                        if 'action' in col:
                            # 액션은 문자열이므로 그대로 유지
                            decision_dict[col] = str(value) if value else 'HOLD'
                        elif 'score' in col or 'value' in col:
                            decision_dict[col] = float(value) if value else 0.0
                        elif 'confidence' in col:
                            # confidence는 문자열이므로 그대로 유지
                            decision_dict[col] = str(value) if value else 'LOW'
                        elif 'count' in col or 'used' in col:
                            decision_dict[col] = int(value) if value else 0
                        elif 'timestamp' in col:
                            # timestamp는 문자열이므로 그대로 유지
                            decision_dict[col] = str(value) if value else ''
                        else:
                            decision_dict[col] = float(value) if value else 0.0
                    else:
                        # 기본값 설정 (최적화된 타입)
                        if 'action' in col:
                            decision_dict[col] = 'HOLD'
                        elif 'score' in col or 'value' in col:
                            decision_dict[col] = 0.0
                        elif 'confidence' in col:
                            decision_dict[col] = 'LOW'
                        elif 'count' in col or 'used' in col:
                            decision_dict[col] = 0
                        elif 'timestamp' in col:
                            decision_dict[col] = ''
                        else:
                            decision_dict[col] = 0.0
                
                decision_data.append(decision_dict)
                
                if (idx + 1) % 10000 == 0:
                    print(f"  변환 진행: {idx + 1:,}/{len(df):,}")
            
            print(f"Decision 데이터 변환 완료: {len(decision_data):,}개")
            return decision_data
            
        except Exception as e:
            print(f"Decision 데이터 로드 실패: {e}")
            return None
    
    @staticmethod
    def create_realistic_training_labels(decision_data: List[Dict], lookback_steps: int = SEQUENCE_LENGTH):
        """고급 라벨링 로직 - 시장 상황을 종합적으로 고려한 라벨 생성"""
        labels = []
        
        for i in range(lookback_steps, len(decision_data) - LABEL_LOOKAHEAD_STEPS):
            label = {
                'action': 0,        # int (0, 1, 2)
                'confidence': 0.5,  # float (0.0-1.0)
                'profit': 0.0       # float (수익률)
            }
            
            # 현재 데이터 추출
            current_data = decision_data[i]
            current_price = current_data.get('close')
            
            # 미래 데이터 추출
            future_prices = []
            future_volumes = []
            for j in range(1, LABEL_LOOKAHEAD_STEPS + 1):
                future_data = decision_data[i + j]
                future_prices.append(future_data.get('close'))
                future_volumes.append(future_data.get('volume'))
            
            if current_price > 0 and all(p > 0 for p in future_prices):
                # 1. 기본 수익률 계산
                future_returns = [(p - current_price) / current_price for p in future_prices]
                max_return = max(future_returns)
                min_return = min(future_returns)
                avg_return = np.mean(future_returns)
                
                # 2. 시장 상황 분석
                market_context = DecisionDataLoader._analyze_market_context(
                    decision_data, i, lookback_steps
                )
                
                # 3. 동적 임계값 계산
                dynamic_threshold = DecisionDataLoader._calculate_dynamic_threshold(
                    market_context, current_data
                )
                
                # 4. 고급 라벨 생성
                label = DecisionDataLoader._generate_advanced_label(
                    max_return, min_return, avg_return, 
                    market_context, dynamic_threshold, future_volumes
                )
            
            labels.append(label)
        
        # 액션 분포 통계 출력
        action_counts = {'HOLD': 0, 'BUY': 0, 'SELL': 0}
        for label in labels:
            action_counts[['HOLD', 'BUY', 'SELL'][label['action']]] += 1
        
        total_labels = len(labels)
        print(f"  📊 라벨 액션 분포:")
        print(f"    HOLD: {action_counts['HOLD']:,}개 ({action_counts['HOLD']/total_labels*100:.1f}%)")
        print(f"    BUY:  {action_counts['BUY']:,}개 ({action_counts['BUY']/total_labels*100:.1f}%)")
        print(f"    SELL: {action_counts['SELL']:,}개 ({action_counts['SELL']/total_labels*100:.1f}%)")
        
        return labels
    
    @staticmethod
    def _analyze_market_context(decision_data: List[Dict], current_idx: int, lookback_steps: int) -> Dict:
        """시장 상황 종합 분석"""
        context = {
            'volatility': 0.0,
            'trend_strength': 0.0,
            'volume_trend': 0.0,
            'price_momentum': 0.0,
            'market_regime': 'normal'  # normal, trending, volatile, consolidation
        }
        
        # 과거 데이터 추출
        start_idx = max(0, current_idx - lookback_steps)
        historical_data = decision_data[start_idx:current_idx]
        
        if len(historical_data) < 5:
            return context
        
        # 가격 데이터 추출
        prices = [d.get('close', 0) for d in historical_data if d.get('close', 0) > 0]
        volumes = [d.get('volume', 0) for d in historical_data if d.get('volume', 0) > 0]
        
        if len(prices) < 5:
            return context
        
        # 1. 변동성 계산 (ATR 기반)
        price_changes = [abs(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        context['volatility'] = np.mean(price_changes) if price_changes else 0.0
        
        # 2. 트렌드 강도 계산 (선형 회귀 기울기)
        x = np.arange(len(prices))
        y = np.array(prices)
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            context['trend_strength'] = abs(slope) / prices[-1] if prices[-1] > 0 else 0.0
        
        # 3. 거래량 트렌드
        if len(volumes) > 1:
            volume_changes = [volumes[i] - volumes[i-1] for i in range(1, len(volumes))]
            context['volume_trend'] = np.mean(volume_changes) / volumes[-1] if volumes[-1] > 0 else 0.0
        
        # 4. 가격 모멘텀 (최근 vs 과거)
        if len(prices) >= 10:
            recent_avg = np.mean(prices[-5:])
            past_avg = np.mean(prices[:5])
            context['price_momentum'] = (recent_avg - past_avg) / past_avg if past_avg > 0 else 0.0
        
        # 5. 시장 레짐 분류
        if context['volatility'] > 0.02:  # 2% 이상 변동성
            context['market_regime'] = 'volatile'
        elif abs(context['trend_strength']) > 0.001:  # 강한 트렌드
            context['market_regime'] = 'trending'
        elif context['volatility'] < 0.005:  # 낮은 변동성
            context['market_regime'] = 'consolidation'
        
        return context
    
    @staticmethod
    def _calculate_dynamic_threshold(market_context: Dict, current_data: Dict) -> float:
        """시장 상황에 따른 동적 임계값 계산"""
        base_threshold = LABEL_PROFIT_THRESHOLD
        
        # 변동성 조정
        volatility_factor = 1.0 + (market_context['volatility'] * LABEL_VOLATILITY_FACTOR)
        
        # 트렌드 조정
        trend_factor = 1.0 + (abs(market_context['trend_strength']) * LABEL_TREND_FACTOR)
        
        # 거래량 조정
        volume_factor = 1.0 + (abs(market_context['volume_trend']) * LABEL_VOLUME_FACTOR)
        
        # 시장 레짐별 조정
        regime_multiplier = {
            'volatile': 1.5,      # 변동성 시장: 임계값 높임
            'trending': 0.8,      # 트렌드 시장: 임계값 낮춤
            'consolidation': 1.2, # 횡보 시장: 임계값 약간 높임
            'normal': 1.0         # 정상 시장: 기본값
        }
        
        regime_factor = regime_multiplier.get(market_context['market_regime'], 1.0)
        
        # 최종 동적 임계값
        dynamic_threshold = base_threshold * volatility_factor * trend_factor * volume_factor * regime_factor
        
        # 최소/최대 제한
        return max(LABEL_PROFIT_THRESHOLD, min(0.05, dynamic_threshold))  # 0.5% ~ 5% 범위
    
    @staticmethod
    def _generate_advanced_label(max_return: float, min_return: float, avg_return: float,
                                market_context: Dict, dynamic_threshold: float, 
                                future_volumes: List[float]) -> Dict:
        """고급 라벨 생성 로직"""
        label = {
            'action': 0,                        # int (0, 1, 2)
            'confidence': LABEL_MIN_CONFIDENCE,  # float (0.0-1.0)
            'profit': 0.0                       # float (수익률)
        }
        
        # 거래량 가중치 계산
        volume_weight = 1.0
        if future_volumes:
            avg_volume = np.mean(future_volumes)
            if avg_volume > 0:
                # 거래량이 평균보다 높으면 신뢰도 증가
                volume_weight = min(1.5, avg_volume / (avg_volume * 0.8))
        
        # 1. BUY 조건 (상승 가능성)
        if max_return > dynamic_threshold:
            label['action'] = 1  # BUY
            
            # 수익률 기반 신뢰도
            return_confidence = min(0.8, 0.4 + max_return * 15)
            
            # 시장 상황 기반 신뢰도 조정
            market_confidence = 1.0
            if market_context['trend_strength'] > 0:  # 상승 트렌드
                market_confidence += 0.1
            if market_context['price_momentum'] > 0:  # 상승 모멘텀
                market_confidence += 0.1
            if market_context['market_regime'] == 'trending':  # 트렌드 시장
                market_confidence += 0.1
            
            # 최종 신뢰도
            label['confidence'] = min(LABEL_MAX_CONFIDENCE, 
                                    return_confidence * market_confidence * volume_weight)
            label['profit'] = max_return
            
        # 2. SELL 조건 (하락 가능성)
        elif min_return < -dynamic_threshold:
            label['action'] = 2  # SELL
            
            # 수익률 기반 신뢰도
            return_confidence = min(0.8, 0.4 + abs(min_return) * 15)
            
            # 시장 상황 기반 신뢰도 조정
            market_confidence = 1.0
            if market_context['trend_strength'] < 0:  # 하락 트렌드
                market_confidence += 0.1
            if market_context['price_momentum'] < 0:  # 하락 모멘텀
                market_confidence += 0.1
            if market_context['market_regime'] == 'trending':  # 트렌드 시장
                market_confidence += 0.1
            
            # 최종 신뢰도
            label['confidence'] = min(LABEL_MAX_CONFIDENCE, 
                                    return_confidence * market_confidence * volume_weight)
            label['profit'] = abs(min_return)
            
        # 3. HOLD 조건 (큰 움직임 없음)
        else:
            label['action'] = 0  # HOLD
            label['profit'] = 0.0
            
            # HOLD 신뢰도는 시장 안정성에 따라 결정
            stability_score = 1.0 - market_context['volatility'] * 10  # 변동성이 낮을수록 높은 신뢰도
            label['confidence'] = max(LABEL_MIN_CONFIDENCE, 
                                    min(0.7, stability_score * volume_weight))
        
        return label

class MultiTimeframeTrainer:
    """Multi-Timeframe Transformer 훈련 클래스"""
    
    def __init__(self, engine: MultiTimeframeDecisionEngine):
        self.engine = engine
        self.training_history = []
    
    def train_on_sequence_data(self, 
                                decision_data: List[Dict], 
                                seq_len: int = SEQUENCE_LENGTH,
                                batch_size: int = TRAINING_BATCH_SIZE,
                                epochs: int = TRAINING_EPOCHS,
                                validation_split: float = TRAINING_VALIDATION_SPLIT,
                                resume_from_checkpoint: bool = False) -> Dict:
        """시퀀스 데이터로 훈련 (Transformer의 진정한 장점 활용)"""
        print(f"Multi-Timeframe Transformer 시퀀스 훈련 시작")
        print(f"  데이터 크기: {len(decision_data):,}개")
        print(f"  시퀀스 길이: {seq_len}")
        print(f"  배치 크기: {batch_size}")
        print(f"  에포크: {epochs}")
        print(f"  검증 비율: {validation_split:.1%}")
        print(f"  체크포인트에서 이어하기: {resume_from_checkpoint}")
        
        # 체크포인트에서 이어하기
        start_epoch = 0
        best_val_loss = float('inf')
        training_history = []
        
        if resume_from_checkpoint and os.path.exists(CHECKPOINT_SAVE_PATH):
            print(f"\n🔄 체크포인트에서 학습 이어하기...")
            training_state = self.engine.load_checkpoint(CHECKPOINT_SAVE_PATH)
            if training_state:
                start_epoch = training_state['epoch'] + 1
                best_val_loss = training_state['best_val_loss']
                training_history = training_state['training_history']
                print(f"  이어할 에포크: {start_epoch}")
                print(f"  이전 최고 검증 손실: {best_val_loss:.4f}")
            else:
                print(f"  체크포인트 로드 실패, 처음부터 시작합니다.")
        
        # 데이터 분할
        split_idx = int(len(decision_data) * (1 - validation_split))
        train_data_raw = decision_data[:split_idx]
        val_data_raw = decision_data[split_idx:]
        
        # 정규화
        normalizer = DataNormalizer()
        train_data_raw = normalizer.fit_transform(train_data_raw)
        val_data_raw = normalizer.transform(val_data_raw)
        
        # 라벨 생성
        train_realistic_labels = DecisionDataLoader.create_realistic_training_labels(
            train_data_raw, lookback_steps=seq_len
        )
        val_realistic_labels = DecisionDataLoader.create_realistic_training_labels(
            val_data_raw, lookback_steps=seq_len
        )
        
        print(f"  훈련 라벨: {len(train_realistic_labels):,}개")
        print(f"  검증 라벨: {len(val_realistic_labels):,}개")
        
        # 시퀀스 생성 (수정됨)
        train_sequences = []
        train_sequence_labels = []
        
        for i in range(len(train_realistic_labels)):
            sequence = train_data_raw[i:i + seq_len]
            label = train_realistic_labels[i]
            
            train_sequences.append(sequence)
            train_sequence_labels.append(label)
        
        val_sequences = []
        val_sequence_labels = []
        
        for i in range(len(val_realistic_labels)):
            sequence = val_data_raw[i:i + seq_len]
            label = val_realistic_labels[i]
            
            val_sequences.append(sequence)
            val_sequence_labels.append(label)
        
        print(f"  훈련 시퀀스: {len(train_sequences):,}개")
        print(f"  검증 시퀀스: {len(val_sequences):,}개")
        
        # 훈련 루프
        patience = EARLY_STOPPING_PATIENCE
        patience_counter = 0
        
        for epoch in range(start_epoch, epochs):
            # 훈련
            train_loss = self._train_sequence_epoch(train_sequences, train_sequence_labels, seq_len, batch_size)
            
            # 검증
            val_loss = self._validate_sequence_epoch(val_sequences, val_sequence_labels, seq_len, batch_size)
            
            # 스케줄러 업데이트 (에포크 기반)
            self.engine.scheduler.step()
            
            # 히스토리 저장
            epoch_stats = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lr': self.engine.optimizer.param_groups[0]['lr']
            }
            training_history.append(epoch_stats)
            
            # 진행 상황 출력
            progress = (epoch - start_epoch + 1) / (epochs - start_epoch) * 100
            print(f"Epoch {epoch+1:3d}/{epochs} ({progress:5.1f}%) | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"LR: {self.engine.optimizer.param_groups[0]['lr']:.2e}")
            
            # 조기 종료
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 최고 모델 저장
                self.engine.save_model(MODEL_SEQUENCE_SAVE_PATH)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"조기 종료: {patience} 에포크 동안 개선 없음")
                    break
            
            # 체크포인트 저장 (매 에포크마다)
            self.engine.save_checkpoint(
                CHECKPOINT_SAVE_PATH, 
                epoch, 
                train_loss, 
                val_loss, 
                training_history, 
                best_val_loss
            )
        
        print(f"시퀀스 훈련 완료! 최고 검증 손실: {best_val_loss:.4f}")
        
        return {
            'best_val_loss': best_val_loss,
            'total_epochs': len(training_history),
            'training_history': training_history,
            'sequence_length': seq_len,
            'start_epoch': start_epoch,
            'completed_epochs': epochs - start_epoch
        }
    
    def _train_sequence_epoch(self, sequences: List[List[Dict]], labels: List[Dict], seq_len: int, batch_size: int) -> float:
        """시퀀스 에포크 훈련"""
        self.engine.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        # 데이터 셔플
        indices = list(range(len(sequences)))
        random.shuffle(indices)
        total_batches = (len(sequences) + batch_size - 1) // batch_size
        
        for i in range(0, len(sequences), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_sequences = [sequences[idx] for idx in batch_indices]
            batch_labels = [labels[idx] for idx in batch_indices]
            
            if num_batches % NUM_BATCHES_PER_PRINT == 0:
                print(f"    배치 {num_batches}/{total_batches} 처리 중... ({num_batches/total_batches*100:.1f}%)")
            
            loss = self.engine.train_on_sequence_batch(batch_sequences, batch_labels, seq_len)
            total_loss += loss
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _validate_sequence_epoch(self, sequences: List[List[Dict]], labels: List[Dict], seq_len: int, batch_size: int) -> float:
        """시퀀스 에포크 검증"""
        self.engine.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch_sequences = sequences[i:i + batch_size]
                batch_labels = labels[i:i + batch_size]
                
                # 검증용 손실 계산
                loss = self._compute_sequence_validation_loss(batch_sequences, batch_labels, seq_len)
                total_loss += loss
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _compute_sequence_validation_loss(self, batch_sequences: List[List[Dict]], batch_labels: List[Dict], seq_len: int) -> float:
        """시퀀스 검증 손실 계산"""
        # 배치 전처리
        inputs = []
        targets = []
        
        for sequence, labels in zip(batch_sequences, batch_labels):
            # 시퀀스 길이 조정
            if len(sequence) < seq_len:
                padded_sequence = sequence + [sequence[-1]] * (seq_len - len(sequence))
            else:
                padded_sequence = sequence[-seq_len:]
            
            # 직접 특성 추출
            sequence_features = []
            for data_point in padded_sequence:
                features = self.engine._extract_features(data_point)
                sequence_features.append(features)
            
            # [seq_len, feature_dim] → [1, seq_len, feature_dim]
            input_tensor = torch.FloatTensor(sequence_features).unsqueeze(0).to(self.engine.device)
            inputs.append(input_tensor)
            
            # 타겟 생성
            target = {
                'action': torch.tensor([labels.get('action', 0)], dtype=torch.long).to(self.engine.device),
                'confidence': torch.tensor([labels.get('confidence', 0.5)], dtype=torch.float).to(self.engine.device),
                'profit': torch.tensor([labels.get('profit', 0.0)], dtype=torch.float).to(self.engine.device)
            }
            targets.append(target)
        
        # 배치 결합
        batch_input = torch.cat(inputs, dim=0)
        
        # 순전파
        decisions, profit_pred, _ = self.engine.model(batch_input)
        
        # 손실 계산
        total_loss = 0.0
        
        # 액션 분류 손실
        action_targets = torch.cat([t['action'] for t in targets])
        action_loss = F.cross_entropy(decisions['action'], action_targets)
        total_loss += action_loss
        
        # 회귀 손실들 (단순화)
        for key in ['confidence']:
            pred = decisions[key].squeeze(-1)  # 마지막 차원만 제거
            target = torch.cat([t[key] for t in targets])
            loss = F.mse_loss(pred, target)
            total_loss += loss
        
        # 수익률 예측 손실
        profit_targets = torch.cat([t['profit'] for t in targets])
        profit_loss = F.mse_loss(profit_pred.squeeze(-1), profit_targets)  # 마지막 차원만 제거
        total_loss += profit_loss
        
        return float(total_loss.item())

def resume_training(checkpoint_path: str = CHECKPOINT_SAVE_PATH, 
                   additional_epochs: int = 50,
                   data_limit: int = DATA_TEST_LIMIT) -> Dict:
    """학습 이어하기 메인 함수"""
    print("🔄 Multi-Timeframe Transformer 학습 이어하기")
    print("=" * 60)
    
    # 체크포인트 파일 확인
    if not os.path.exists(checkpoint_path):
        print(f"❌ 체크포인트 파일이 없습니다: {checkpoint_path}")
        print("   처음부터 학습을 시작하려면 main() 함수를 사용하세요.")
        return {}
    
    # 모델 초기화 (Enhanced Attention 사용)
    engine = MultiTimeframeDecisionEngine(
        input_size=MODEL_INPUT_SIZE,
        d_model=MODEL_D_MODEL,
        nhead=MODEL_NHEAD,
        num_layers=MODEL_NUM_LAYERS,
        use_enhanced_attention=True
    )
    
    # 체크포인트 로드
    print(f"\n1️⃣ 체크포인트 로드 중: {checkpoint_path}")
    training_state = engine.load_checkpoint(checkpoint_path)
    
    if not training_state:
        print("❌ 체크포인트 로드 실패")
        return {}
    
    # 데이터 로드
    print(f"\n2️⃣ Decision 데이터 로드 중...")
    decision_data = DecisionDataLoader.load_decision_data(DATA_FILE_PATH)
    
    if not decision_data:
        print("❌ 데이터 로드 실패")
        return {}
    
    # 데이터 제한
    if len(decision_data) > data_limit:
        decision_data = decision_data[:data_limit]
        print(f"   🧪 테스트용으로 데이터를 {data_limit:,}개로 제한했습니다.")
    
    print(f"   데이터 크기: {len(decision_data):,}개")
    
    # 학습 이어하기
    print(f"\n3️⃣ 학습 이어하기 시작...")
    print(f"   이전 에포크: {training_state['epoch']}")
    print(f"   추가 에포크: {additional_epochs}")
    print(f"   총 목표 에포크: {training_state['epoch'] + additional_epochs}")
    
    trainer = MultiTimeframeTrainer(engine)
    training_results = trainer.train_on_sequence_data(
        decision_data=decision_data,
        seq_len=SEQUENCE_LENGTH,
        batch_size=TRAINING_BATCH_SIZE,
        epochs=training_state['epoch'] + additional_epochs,
        validation_split=TRAINING_VALIDATION_SPLIT,
        resume_from_checkpoint=True
    )
    
    # 결과 출력
    print(f"\n4️⃣ 학습 이어하기 완료!")
    print(f"   시작 에포크: {training_results['start_epoch']}")
    print(f"   완료된 에포크: {training_results['completed_epochs']}")
    print(f"   최고 검증 손실: {training_results['best_val_loss']:.4f}")
    
    # 최종 모델 저장
    engine.save_model(MODEL_FINAL_SAVE_PATH)
    print(f"\n✅ 최종 모델 저장 완료: {MODEL_FINAL_SAVE_PATH}")
    
    return training_results

def main():
    """메인 실행 함수"""
    print("Enhanced Multi-Timeframe Transformer 딥러닝 모델")
    print("=" * 60)
    
    # 모델 초기화 (Enhanced Attention 사용)
    engine = MultiTimeframeDecisionEngine(
        input_size=MODEL_INPUT_SIZE,
        d_model=MODEL_D_MODEL,
        nhead=MODEL_NHEAD,
        num_layers=MODEL_NUM_LAYERS,
        use_enhanced_attention=True
    )
    
    # 1. Decision 데이터 로드 (테스트용으로 제한)
    print("\n1️⃣ Decision 데이터 로드...")
    decision_data = DecisionDataLoader.load_decision_data(DATA_FILE_PATH)
    
    # 용으로 데이터 제한
    if len(decision_data) > DATA_TEST_LIMIT:
        decision_data = decision_data[:DATA_TEST_LIMIT]
        print(f"   🧪 테스트용으로 데이터를 {DATA_TEST_LIMIT:,}개로 제한했습니다.")
    
    # 2. 데이터 정규화는 훈련 함수 내부에서 수행 (Look-ahead Bias 방지)
    print("\n2️⃣ 데이터 정규화는 훈련 함수 내부에서 수행됩니다.")
    print("   🔥 Look-ahead Bias 방지를 위해 훈련/검증 분할 후 정규화 적용")
    # 정규화는 train_on_sequence_data 함수 내부에서 수행됨
    
    # 3. Decision 데이터에서 가격 정보 추출
    print("\n3️⃣ Decision 데이터에서 가격 정보 추출...")
    price_data = []
    for data in decision_data:
        price_row = {
            'open': data.get('open'),
            'high': data.get('high'),
            'low': data.get('low'),
            'close': data.get('close'),
            'volume': data.get('volume'),
            'quote_volume': data.get('quote_volume')
        }
        price_data.append(price_row)
    
    price_df = pd.DataFrame(price_data)
    print(f"가격 데이터 추출 완료: {len(price_df):,}개")

    # 4. 훈련 실행 (시퀀스 기반) - 라벨 생성은 함수 내부에서 처리
    print("\n4️⃣ Multi-Timeframe Transformer 시퀀스 훈련 시작...")
    print("   🔥 Look-ahead Bias 방지를 위해 현실적인 라벨 생성 방법 사용")
    print("   🔥 라벨 생성은 훈련 함수 내부에서 데이터 분할 후 수행")
    
    print(f"최종 데이터 길이: {len(decision_data):,}개")
    
    trainer = MultiTimeframeTrainer(engine)
    training_results = trainer.train_on_sequence_data(
        decision_data=decision_data,
        seq_len=SEQUENCE_LENGTH,
        batch_size=TRAINING_BATCH_SIZE,
        epochs=TRAINING_EPOCHS,
        validation_split=TRAINING_VALIDATION_SPLIT
    )
    
    # 6. 훈련 결과 출력
    print(f"\n6️⃣ 훈련 결과:")
    print(f"  최고 검증 손실: {training_results['best_val_loss']:.4f}")
    print(f"  총 에포크: {training_results['total_epochs']}")
    
    # 7. 테스트 (시퀀스 기반)
    print(f"\n7️⃣ 훈련된 시퀀스 모델 테스트...")
    
    # 시퀀스 데이터 테스트 (새로운 기능)
    if len(decision_data) >= SEQUENCE_LENGTH:
        print(f"\n시퀀스 데이터 테스트 ({TEST_SAMPLES_COUNT}개 샘플):")
        
        # 여러 샘플 테스트하여 액션 분포 확인
        test_action_counts = {'HOLD': 0, 'BUY': 0, 'SELL': 0}
        
        for i in range(min(TEST_SAMPLES_COUNT, len(decision_data) - SEQUENCE_LENGTH)):
            test_sequence = decision_data[i:i+SEQUENCE_LENGTH]
            sequence_decision = engine.make_sequence_decision(test_sequence, seq_len=SEQUENCE_LENGTH)
            
            action = sequence_decision['action']
            test_action_counts[action] += 1
            
            if i < 3:  # 처음 3개만 상세 출력
                print(f"  샘플 {i+1}:")
                print(f"    액션: {action}")
                print(f"    신뢰도: {sequence_decision['confidence']:.3f}")
                print(f"    수익률 예측: {sequence_decision['profit']:.3f}")
        
        # 액션 분포 통계
        total_tests = sum(test_action_counts.values())
        print(f"\n  📊 테스트 액션 분포 (총 {total_tests}개 샘플):")
        print(f"    HOLD: {test_action_counts['HOLD']}개 ({test_action_counts['HOLD']/total_tests*100:.1f}%)")
        print(f"    BUY:  {test_action_counts['BUY']}개 ({test_action_counts['BUY']/total_tests*100:.1f}%)")
        print(f"    SELL: {test_action_counts['SELL']}개 ({test_action_counts['SELL']/total_tests*100:.1f}%)")
    
    # 8. 최종 모델 저장
    engine.save_model(MODEL_FINAL_SAVE_PATH)
    print(f"\n✅ 훈련된 모델이 저장되었습니다: {MODEL_FINAL_SAVE_PATH}")

if __name__ == "__main__":
    import sys
    
    # 명령행 인수 확인
    if len(sys.argv) > 1 and sys.argv[1] == "resume":
        # 학습 이어하기
        additional_epochs = 50
        if len(sys.argv) > 2:
            try:
                additional_epochs = int(sys.argv[2])
            except ValueError:
                print("❌ 추가 에포크 수는 정수여야 합니다.")
                sys.exit(1)
        
        resume_training(additional_epochs=additional_epochs)
    else:
        # 처음부터 학습
        main()
