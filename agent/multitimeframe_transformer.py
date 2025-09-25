"""
Multi-Timeframe Transformer 딥러닝 모델
- Decision 데이터를 입력으로 받아서 의사결정을 수행
- 다중 시간프레임 분석 (3m, 15m, 1h)
- Transformer 기반 어텐션 메커니즘
- 수익률 최적화 중심 설계
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, accuracy_score

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

class MultiTimeframeTransformer(nn.Module):
    """Multi-Timeframe Transformer 모델"""
    
    def __init__(self, 
                 input_size: int = 58,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dropout: float = 0.1,
                 max_seq_len: int = 100):
        super().__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        # 입력 임베딩
        self.input_embedding = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 위치 인코딩
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer 인코더
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 시간프레임별 특성 추출기
        self.timeframe_extractors = nn.ModuleDict({
            'short_term': self._build_timeframe_extractor(d_model, 'short'),
            'medium_term': self._build_timeframe_extractor(d_model, 'medium'),
            'long_term': self._build_timeframe_extractor(d_model, 'long')
        })
        
        # 의사결정 헤드들 (단순화: action, confidence, profit만)
        self.decision_heads = nn.ModuleDict({
            'action': nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.LayerNorm(d_model // 2),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 3)  # BUY, SELL, HOLD
            ),
            'confidence': nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.LayerNorm(d_model // 2),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
                nn.Sigmoid()
            )
        })
        
        # 수익률 예측기
        self.profit_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.LayerNorm(d_model // 2),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        # 가중치 초기화
        self.apply(self._init_weights)
    
    def _build_timeframe_extractor(self, d_model: int, timeframe: str):
        """시간프레임별 특성 추출기"""
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
        
        # 위치 인코딩
        x = x.transpose(0, 1)  # [seq_len, batch_size, d_model]
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # [batch_size, seq_len, d_model]
        
        # Transformer 인코더
        if mask is not None:
            # 마스크 적용 (패딩 토큰 등)
            x = self.transformer(x, src_key_padding_mask=mask)
        else:
            x = self.transformer(x)
        
        # 최종 특성 (마지막 시퀀스 요소 사용)
        final_features = x[:, -1, :]  # [batch_size, d_model]
        
        # 시간프레임별 특성 추출
        timeframe_features = {}
        for timeframe, extractor in self.timeframe_extractors.items():
            timeframe_features[timeframe] = extractor(final_features)
        
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

class MultiTimeframeDecisionEngine:
    """Multi-Timeframe 의사결정 엔진"""
    
    def __init__(self, 
                 model_path: str = 'agent/best_multitimeframe_model.pth',
                 input_size: int = 58,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 device: str = 'auto'):
        
        # 디바이스 설정
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Multi-Timeframe Transformer 디바이스: {self.device}")
        
        # 모델 초기화
        self.model = MultiTimeframeTransformer(
            input_size=input_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers
        ).to(self.device)
        
        # 옵티마이저
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=5e-4,  # 더 높은 학습률
            weight_decay=1e-5
        )
        
        # 스케줄러
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=10,
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
            float(decision_data.get('open', 0.0)),
            float(decision_data.get('high', 0.0)),
            float(decision_data.get('low', 0.0)),
            float(decision_data.get('close', 0.0)),
            float(decision_data.get('volume', 0.0)),
            float(decision_data.get('quote_volume', 0.0))
        ])
        
        # 8. Timestamp (1개)
        timestamp = decision_data.get('timestamp', 0.0)
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
    
    def preprocess_decision_data(self, decision_data: Dict) -> torch.Tensor:
        """단일 Decision 데이터 전처리 (호환성 유지)"""
        features = self._extract_features(decision_data)
        return torch.FloatTensor(features).unsqueeze(0).to(self.device)
    
    def preprocess_sequence_data(self, decision_data: List[Dict], seq_len: int = 20) -> torch.Tensor:
        """시퀀스 형태로 데이터 전처리"""
        sequences = []
        
        for i in range(len(decision_data) - seq_len + 1):
            sequence = []
            for j in range(seq_len):
                features = self._extract_features(decision_data[i + j])
                sequence.append(features)
            sequences.append(sequence)
        
        return torch.FloatTensor(sequences).to(self.device)
    
    def make_decision(self, decision_data: Dict) -> Dict:
        """단일 데이터 의사결정 수행 (호환성 유지)"""
        self.model.eval()
        
        with torch.no_grad():
            # 데이터 전처리
            input_tensor = self.preprocess_decision_data(decision_data)
            
            # 모델 추론
            decisions, profit_pred, timeframe_features = self.model(input_tensor)
            
            # 의사결정 결과 생성 (단순화)
            result = {
                'action': self._interpret_action(decisions['action']),
                'confidence': float(decisions['confidence'].item()),
                'profit': float(profit_pred.item()),
                'timestamp': datetime.now().isoformat(),
                'model_version': 'MultiTimeframeTransformer_v1.0'
            }
            
            # 히스토리에 저장
            self.decision_history.append({
                'input': decision_data,
                'output': result,
                'timestamp': datetime.now()
            })
            
            self.total_decisions += 1
            
            return result
    
    def make_sequence_decision(self, decision_sequence: List[Dict], seq_len: int = 20) -> Dict:
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
    
    def _interpret_action(self, action_logits: torch.Tensor) -> str:
        """액션 로짓을 해석 (확률 분포 확인 포함)"""
        # 소프트맥스 적용하여 확률로 변환
        probabilities = torch.softmax(action_logits, dim=-1)
        action_idx = torch.argmax(action_logits).item()
        actions = ['HOLD', 'BUY', 'SELL']
        
        # 디버깅: 액션 분포 출력
        probs = probabilities.cpu().numpy()
        print(f"    액션 확률: HOLD={probs[0]:.3f}, BUY={probs[1]:.3f}, SELL={probs[2]:.3f}")
        
        return actions[action_idx]
    
    
    def train_on_batch(self, batch_data: List[Dict], batch_labels: List[Dict]) -> float:
        """단일 데이터 배치 학습 (호환성 유지)"""
        self.model.train()
        
        # 배치 전처리
        inputs = []
        targets = []
        
        for data, labels in zip(batch_data, batch_labels):
            input_tensor = self.preprocess_decision_data(data)
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
        profit_loss_weight = 3.0  # 수익률 예측에 가장 높은 가중치
        action_loss_weight = 2.0  # 액션 분류에 높은 가중치
        other_loss_weight = 1.0   # 기타 회귀에 기본 가중치
        
        # 액션 분류 손실
        action_targets = torch.cat([t['action'] for t in targets])
        action_loss = F.cross_entropy(decisions['action'], action_targets)
        total_loss += action_loss * action_loss_weight
        
        # 회귀 손실들 (단순화)
        for key in ['confidence']:
            pred = decisions[key].squeeze()
            target = torch.cat([t[key] for t in targets])
            loss = F.mse_loss(pred, target)
            total_loss += loss * other_loss_weight
        
        # 수익률 예측 손실 (가장 중요)
        profit_targets = torch.cat([t['profit'] for t in targets])
        profit_loss = F.mse_loss(profit_pred.squeeze(), profit_targets)
        total_loss += profit_loss * profit_loss_weight
        
        # 역전파
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        
        return float(total_loss.item())
    
    def train_on_sequence_batch(self, batch_sequences: List[List[Dict]], batch_labels: List[Dict], seq_len: int = 20) -> float:
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
        profit_loss_weight = 3.0  # 수익률 예측에 가장 높은 가중치
        action_loss_weight = 2.0  # 액션 분류에 높은 가중치
        other_loss_weight = 1.0   # 기타 회귀에 기본 가중치
        
        # 액션 분류 손실
        action_targets = torch.cat([t['action'] for t in targets])
        action_loss = F.cross_entropy(decisions['action'], action_targets)
        total_loss += action_loss * action_loss_weight
        
        # 회귀 손실들 (단순화)
        for key in ['confidence']:
            pred = decisions[key].squeeze()
            target = torch.cat([t[key] for t in targets])
            loss = F.mse_loss(pred, target)
            total_loss += loss * other_loss_weight
        
        # 수익률 예측 손실 (가장 중요)
        profit_targets = torch.cat([t['profit'] for t in targets])
        profit_loss = F.mse_loss(profit_pred.squeeze(), profit_targets)
        total_loss += profit_loss * profit_loss_weight
        
        # 역전파
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
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
    
    def get_performance_stats(self) -> Dict:
        """성능 통계 반환"""
        accuracy = self.correct_decisions / max(self.total_decisions, 1)
        avg_profit = self.total_profit / max(self.total_decisions, 1)
        
        return {
            'total_decisions': self.total_decisions,
            'correct_decisions': self.correct_decisions,
            'accuracy': accuracy,
            'total_profit': self.total_profit,
            'avg_profit_per_decision': avg_profit,
            'recent_decisions': len(self.decision_history)
        }

def create_sample_decision_data() -> Dict:
    """샘플 Decision 데이터 생성 (58차원)"""
    actions = ['HOLD', 'BUY', 'SELL']
    
    data = {}
    
    # 1. 전략 Action (16개)
    strategy_actions = [
        'bollinger_squeeze_action', 'zscore_mean_reversion_action', 'vpvr_action', 
        'ema_confluence_action', 'ichimoku_action', 'htf_trend_action', 
        'funding_rate_action', 'orderflow_cvd_action', 'support_resistance_action', 
        'vwap_pinball_action', 'multi_timeframe_action', 'session_action', 
        'vol_spike_action', 'vpvr_micro_action', 'oi_delta_action', 'liquidity_grab_action'
    ]
    
    for action_col in strategy_actions:
        data[action_col] = random.choice(actions)
    
    # 2. 전략 Score (16개)
    strategy_scores = [
        'bollinger_squeeze_score', 'session_score', 'vpvr_score', 'oi_delta_score',
        'ema_confluence_score', 'multi_timeframe_score', 'vol_spike_score', 
        'vpvr_micro_score', 'htf_trend_score', 'liquidity_grab_score', 
        'ichimoku_score', 'funding_rate_score', 'support_resistance_score', 
        'zscore_mean_reversion_score', 'vwap_pinball_score', 'orderflow_cvd_score'
    ]
    
    for score_col in strategy_scores:
        data[score_col] = random.uniform(0.0, 1.0)
    
    # 3. 전략 Confidence (1개)
    data['vpvr_confidence'] = random.uniform(0.0, 1.0)
    
    # 4. 전략 Entry (4개)
    strategy_entries = ['vpvr_micro_entry', 'vwap_pinball_entry', 'session_entry', 'vpvr_entry']
    for entry_col in strategy_entries:
        data[entry_col] = random.uniform(2000, 3000)
    
    # 5. 전략 Stop (4개)
    strategy_stops = ['vpvr_stop', 'session_stop', 'vpvr_micro_stop', 'vwap_pinball_stop']
    for stop_col in strategy_stops:
        data[stop_col] = random.uniform(2000, 3000)
    
    # 6. Indicator 정보 (10개)
    indicator_fields = [
        'indicator_prev_day_high', 'indicator_poc', 'indicator_hvn', 'indicator_vwap_std',
        'indicator_opening_range_high', 'indicator_lvn', 'indicator_opening_range_low',
        'indicator_vwap', 'indicator_prev_day_low', 'indicator_atr'
    ]
    
    for field in indicator_fields:
        if 'atr' in field or 'std' in field:
            data[field] = random.uniform(1, 50)
        else:
            data[field] = random.uniform(2000, 3000)
    
    # 7. OHLC 데이터 (6개)
    base_price = random.uniform(2000, 3000)
    data['open'] = base_price
    data['high'] = base_price + random.uniform(0, 50)
    data['low'] = base_price - random.uniform(0, 50)
    data['close'] = base_price + random.uniform(-20, 20)
    data['volume'] = random.uniform(1000, 10000)
    data['quote_volume'] = random.uniform(1000000, 10000000)
    
    # 8. Timestamp (1개)
    data['timestamp'] = int(datetime.now().timestamp() * 1000)
    
    return data

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
            
            # DataFrame을 Dict 리스트로 변환
            decision_data = []
            for idx, row in df.iterrows():
                decision_dict = {}
                for col, value in row.items():
                    if pd.notna(value):
                        decision_dict[col] = value
                    else:
                        # 기본값 설정
                        if 'score' in col or 'confidence' in col or 'value' in col:
                            decision_dict[col] = 0.0
                        elif 'count' in col or 'used' in col:
                            decision_dict[col] = 0
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
    def create_training_labels(decision_data: List[Dict], lookforward_steps: int = 5):
        """미래 N개 스텝의 평균 수익률로 라벨 생성 (기존 - Look-ahead Bias 있음)"""
        labels = []
        
        for i, data in enumerate(decision_data):
            label = {
                'action': 0,
                'confidence': 0.5,
                'profit': 0.0
            }
            
            # 미래 여러 스텝의 평균 수익률 계산
            if i < len(decision_data) - lookforward_steps:
                current_price = data.get('close', 0.0)
                future_prices = [
                    decision_data[i + j].get('close', 0.0) 
                    for j in range(1, lookforward_steps + 1)
                ]
                
                if current_price > 0 and all(p > 0 for p in future_prices):
                    # 평균 수익률
                    avg_price_change = np.mean([
                        (p - current_price) / current_price 
                        for p in future_prices
                    ])
                    
                    # 더 보수적인 임계값 (1% 이상/이하)
                    if avg_price_change > 0.01:
                        label['action'] = 1  # BUY
                        label['profit'] = avg_price_change
                        label['confidence'] = min(0.9, 0.5 + abs(avg_price_change) * 10)
                        
                    elif avg_price_change < -0.01:
                        label['action'] = 2  # SELL
                        label['profit'] = -avg_price_change
                        label['confidence'] = min(0.9, 0.5 + abs(avg_price_change) * 10)
            
            labels.append(label)
            
        return labels
    
    @staticmethod
    def create_realistic_training_labels(decision_data: List[Dict], lookback_steps: int = 20):
        """미래 데이터를 사용한 고품질 라벨 생성 (백테스트용)"""
        labels = []
        
        for i in range(lookback_steps, len(decision_data) - 5):  # 미래 5개 스텝 확보
            label = {
                'action': 0,
                'confidence': 0.5,
                'profit': 0.0
            }
            
            # 🔥 미래 데이터를 사용한 고품질 라벨 생성
            current_price = decision_data[i].get('close', 0.0)
            future_prices = [
                decision_data[i + j].get('close', 0.0) 
                for j in range(1, 6)  # 다음 5개 스텝
            ]
            
            if current_price > 0 and all(p > 0 for p in future_prices):
                # 🔥 미래 데이터를 사용한 정확한 라벨 생성
                # 미래 5개 스텝의 최고가, 최저가, 평균 수익률 계산
                future_returns = [(p - current_price) / current_price for p in future_prices]
                max_return = max(future_returns)
                min_return = min(future_returns)
                avg_return = np.mean(future_returns)
                
                # 거래량 분석 (현재 + 미래)
                current_volume = decision_data[i].get('volume')
                future_volumes = [decision_data[i + j].get('volume') for j in range(1, 6)]
                avg_volume = np.mean([current_volume] + future_volumes)
                
                # 🔥 미래 수익률 기반 라벨 생성 (훨씬 정확함)
                if max_return > 0.005:  # 0.5% 이상 상승 가능
                    label['action'] = 1  # BUY
                    label['profit'] = max_return  # 실제 미래 수익률
                    label['confidence'] = min(0.95, 0.6 + max_return * 20)  # 높은 신뢰도
                    
                elif min_return < -0.005:  # 0.5% 이상 하락 가능
                    label['action'] = 2  # SELL
                    label['profit'] = -min_return  # 실제 미래 수익률 (절댓값)
                    label['confidence'] = min(0.95, 0.6 + abs(min_return) * 20)  # 높은 신뢰도
                    
                else:
                    # Hold - 큰 움직임 없음
                    label['action'] = 0
                    label['profit'] = 0.0
                    label['confidence'] = 0.3
            
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

class MultiTimeframeTrainer:
    """Multi-Timeframe Transformer 훈련 클래스"""
    
    def __init__(self, engine: MultiTimeframeDecisionEngine):
        self.engine = engine
        self.training_history = []
    
    def train_on_sequence_data(self, 
                                decision_data: List[Dict], 
                                seq_len: int = 20,  # labels 파라미터 제거
                                batch_size: int = 32,
                                epochs: int = 10,
                                validation_split: float = 0.2) -> Dict:
        """시퀀스 데이터로 훈련 (Transformer의 진정한 장점 활용)"""
        print(f"Multi-Timeframe Transformer 시퀀스 훈련 시작")
        print(f"  데이터 크기: {len(decision_data):,}개")
        print(f"  시퀀스 길이: {seq_len}")
        print(f"  배치 크기: {batch_size}")
        print(f"  에포크: {epochs}")
        print(f"  검증 비율: {validation_split:.1%}")
        
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
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in range(epochs):
            # 훈련
            train_loss = self._train_sequence_epoch(train_sequences, train_sequence_labels, seq_len, batch_size)
            
            # 검증
            val_loss = self._validate_sequence_epoch(val_sequences, val_sequence_labels, seq_len, batch_size)
            
            # 스케줄러 업데이트
            self.engine.scheduler.step(val_loss)
            
            # 히스토리 저장
            epoch_stats = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lr': self.engine.optimizer.param_groups[0]['lr']
            }
            self.training_history.append(epoch_stats)
            
            # 진행 상황 출력
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"LR: {self.engine.optimizer.param_groups[0]['lr']:.2e}")
            
            # 조기 종료
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 최고 모델 저장
                self.engine.save_model('agent/best_multitimeframe_sequence_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"조기 종료: {patience} 에포크 동안 개선 없음")
                    break
        
        print(f"시퀀스 훈련 완료! 최고 검증 손실: {best_val_loss:.4f}")
        
        return {
            'best_val_loss': best_val_loss,
            'total_epochs': len(self.training_history),
            'training_history': self.training_history,
            'sequence_length': seq_len
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
            
            if i % 100 == 0:
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
                'position_size': torch.tensor([labels.get('position_size', 0.5)], dtype=torch.float).to(self.engine.device),
                'leverage': torch.tensor([labels.get('leverage', 0.5)], dtype=torch.float).to(self.engine.device),
                'holding_time': torch.tensor([labels.get('holding_time', 0.5)], dtype=torch.float).to(self.engine.device),
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
            pred = decisions[key].squeeze()
            target = torch.cat([t[key] for t in targets])
            loss = F.mse_loss(pred, target)
            total_loss += loss
        
        # 수익률 예측 손실
        profit_targets = torch.cat([t['profit'] for t in targets])
        profit_loss = F.mse_loss(profit_pred.squeeze(), profit_targets)
        total_loss += profit_loss
        
        return float(total_loss.item())
    
    def train_on_decision_data(self, 
                              decision_data: List[Dict], 
                              labels: List[Dict],
                              batch_size: int = 32,
                              epochs: int = 10,
                              validation_split: float = 0.2) -> Dict:
        """Decision 데이터로 훈련"""
        print(f"Multi-Timeframe Transformer 훈련 시작")
        print(f"  데이터 크기: {len(decision_data):,}개")
        print(f"  배치 크기: {batch_size}")
        print(f"  에포크: {epochs}")
        print(f"  검증 비율: {validation_split:.1%}")
        
        # 🔥 데이터 분할 먼저 (정규화 전에!)
        split_idx = int(len(decision_data) * (1 - validation_split))
        train_data = decision_data[:split_idx]
        train_labels = labels[:split_idx]
        val_data = decision_data[split_idx:]
        val_labels = labels[split_idx:]
        
        print(f"  훈련 데이터: {len(train_data):,}개")
        print(f"  검증 데이터: {len(val_data):,}개")
        
        # 🔥 훈련셋으로만 정규화 fit
        print("  훈련셋으로 정규화 fit 중...")
        normalizer = DataNormalizer()
        train_data = normalizer.fit_transform(train_data)
        
        # 🔥 검증셋은 transform만 (fit 없이!)
        print("  검증셋에 정규화 transform 적용 중...")
        val_data = normalizer.transform(val_data)
        
        # 훈련 루프
        best_val_loss = float('inf')
        patience = 20  # 더 관대한 조기 종료
        patience_counter = 0
        
        for epoch in range(epochs):
            # 훈련
            train_loss = self._train_epoch(train_data, train_labels, batch_size)
            
            # 검증
            val_loss = self._validate_epoch(val_data, val_labels, batch_size)
            
            # 스케줄러 업데이트
            self.engine.scheduler.step(val_loss)
            
            # 히스토리 저장
            epoch_stats = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lr': self.engine.optimizer.param_groups[0]['lr']
            }
            self.training_history.append(epoch_stats)
            
            # 진행 상황 출력
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"LR: {self.engine.optimizer.param_groups[0]['lr']:.2e}")
            
            # 조기 종료
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 최고 모델 저장
                self.engine.save_model('agent/best_multitimeframe_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"조기 종료: {patience} 에포크 동안 개선 없음")
                    break
        
        print(f"훈련 완료! 최고 검증 손실: {best_val_loss:.4f}")
        
        return {
            'best_val_loss': best_val_loss,
            'total_epochs': len(self.training_history),
            'training_history': self.training_history
        }
    
    def _train_epoch(self, data: List[Dict], labels: List[Dict], batch_size: int) -> float:
        """한 에포크 훈련"""
        self.engine.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        # 데이터 셔플
        indices = list(range(len(data)))
        random.shuffle(indices)
        
        for i in range(0, len(data), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_data = [data[idx] for idx in batch_indices]
            batch_labels = [labels[idx] for idx in batch_indices]
            
            loss = self.engine.train_on_batch(batch_data, batch_labels)
            total_loss += loss
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _validate_epoch(self, data: List[Dict], labels: List[Dict], batch_size: int) -> float:
        """한 에포크 검증"""
        self.engine.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(data), batch_size):
                batch_data = data[i:i + batch_size]
                batch_labels = labels[i:i + batch_size]
                
                # 검증용 손실 계산 (훈련하지 않음)
                loss = self._compute_validation_loss(batch_data, batch_labels)
                total_loss += loss
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _compute_validation_loss(self, batch_data: List[Dict], batch_labels: List[Dict]) -> float:
        """검증 손실 계산"""
        # 배치 전처리
        inputs = []
        targets = []
        
        for data, labels in zip(batch_data, batch_labels):
            input_tensor = self.engine.preprocess_decision_data(data)
            inputs.append(input_tensor)
            
            target = {
                'action': torch.tensor([labels.get('action', 0)], dtype=torch.long).to(self.engine.device),
                'confidence': torch.tensor([labels.get('confidence', 0.5)], dtype=torch.float).to(self.engine.device),
                'position_size': torch.tensor([labels.get('position_size', 0.5)], dtype=torch.float).to(self.engine.device),
                'leverage': torch.tensor([labels.get('leverage', 0.5)], dtype=torch.float).to(self.engine.device),
                'holding_time': torch.tensor([labels.get('holding_time', 0.5)], dtype=torch.float).to(self.engine.device),
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
            pred = decisions[key].squeeze()
            target = torch.cat([t[key] for t in targets])
            loss = F.mse_loss(pred, target)
            total_loss += loss
        
        # 수익률 예측 손실
        profit_targets = torch.cat([t['profit'] for t in targets])
        profit_loss = F.mse_loss(profit_pred.squeeze(), profit_targets)
        total_loss += profit_loss
        
        return float(total_loss.item())

def main():
    """메인 실행 함수"""
    print("Multi-Timeframe Transformer 딥러닝 모델")
    print("=" * 60)
    
    # 모델 초기화
    engine = MultiTimeframeDecisionEngine(
        input_size=58,
        d_model=256,
        nhead=8,
        num_layers=6
    )
    
    # 1. Decision 데이터 로드 (테스트용으로 제한)
    print("\n1️⃣ Decision 데이터 로드...")
    decision_data = DecisionDataLoader.load_decision_data('agent/decisions_data.parquet')
    
    # # 테스트용으로 데이터 제한 (처음 10,000개만 사용)
    # if len(decision_data) > 100000:
    #     decision_data = decision_data[:100000]
    #     print(f"   🧪 테스트용으로 데이터를 10,000개로 제한했습니다.")
    
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
        seq_len=30,  # 20 스텝 시퀀스
        batch_size=64,  # 시퀀스 데이터는 메모리 사용량이 크므로 배치 크기 조정
        epochs=100,
        validation_split=0.2
    )
    
    # 6. 훈련 결과 출력
    print(f"\n6️⃣ 훈련 결과:")
    print(f"  최고 검증 손실: {training_results['best_val_loss']:.4f}")
    print(f"  총 에포크: {training_results['total_epochs']}")
    
    # 7. 테스트 (시퀀스 기반)
    print(f"\n7️⃣ 훈련된 시퀀스 모델 테스트...")
    
    # 단일 데이터 테스트 (호환성)
    test_data = decision_data[0]
    decision = engine.make_decision(test_data)
    print("단일 데이터 테스트 결과:")
    print(f"  액션: {decision['action']}")
    print(f"  신뢰도: {decision['confidence']:.3f}")
    print(f"  수익률 예측: {decision['profit']:.3f}")
    
    # 시퀀스 데이터 테스트 (새로운 기능)
    if len(decision_data) >= 20:
        print("\n시퀀스 데이터 테스트 (10개 샘플):")
        
        # 여러 샘플 테스트하여 액션 분포 확인
        test_action_counts = {'HOLD': 0, 'BUY': 0, 'SELL': 0}
        
        for i in range(min(10, len(decision_data) - 20)):
            test_sequence = decision_data[i:i+20]  # 20개씩 슬라이딩 윈도우
            sequence_decision = engine.make_sequence_decision(test_sequence, seq_len=20)
            
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
    engine.save_model('agent/multitimeframe_transformer_trained.pth')
    print(f"\n✅ 훈련된 모델이 저장되었습니다: agent/multitimeframe_transformer_trained.pth")

if __name__ == "__main__":
    main()
