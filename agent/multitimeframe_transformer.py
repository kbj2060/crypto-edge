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
                 input_size: int = 61,
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
        
        # 의사결정 헤드들
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
            ),
            'position_size': nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.LayerNorm(d_model // 2),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
                nn.Sigmoid()
            ),
            'leverage': nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.LayerNorm(d_model // 2),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
                nn.Sigmoid()
            ),
            'holding_time': nn.Sequential(
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
                 model_path: str = None,
                 input_size: int = 61,
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
            lr=1e-4,
            weight_decay=1e-5
        )
        
        # 스케줄러
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=10,
            verbose=True
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
    
    def preprocess_decision_data(self, decision_data: Dict) -> torch.Tensor:
        """Decision 데이터 전처리"""
        # 61차원 특성 벡터 생성
        features = []
        
        # 각 시간프레임별 특성 추출
        for timeframe in ['short_term', 'medium_term', 'long_term']:
            # 기본 점수들
            features.extend([
                float(decision_data.get(f'{timeframe}_net_score', 0.0)),
                float(decision_data.get(f'{timeframe}_buy_score', 0.0)),
                float(decision_data.get(f'{timeframe}_sell_score', 0.0)),
                float(decision_data.get(f'{timeframe}_confidence', 0.0)),
                float(decision_data.get(f'{timeframe}_action_value', 0.0)),
                float(decision_data.get(f'{timeframe}_market_context', 0.0))
            ])
        
        # Conflict 정보
        features.extend([
            float(decision_data.get('conflict_conflict_severity', 0.0)),
            float(decision_data.get('conflict_directional_consensus', 0.0)),
            float(decision_data.get('conflict_conflict_penalty', 0.0))
        ])
        
        # Indicator 정보
        indicator_fields = [
            'indicator_vwap', 'indicator_atr', 'indicator_poc',
            'indicator_hvn', 'indicator_lvn', 'indicator_vwap_std',
            'indicator_prev_day_high', 'indicator_prev_day_low',
            'indicator_opening_range_high', 'indicator_opening_range_low'
        ]
        
        for field in indicator_fields:
            features.append(float(decision_data.get(field, 0.0)))
        
        # 캔들 데이터
        candle_data = decision_data.get('candle_data', {})
        features.extend([
            float(candle_data.get('open', 0.0)),
            float(candle_data.get('high', 0.0)),
            float(candle_data.get('low', 0.0)),
            float(candle_data.get('close', 0.0)),
            float(candle_data.get('volume', 0.0)),
            float(candle_data.get('quote_volume', 0.0))
        ])
        
        # 메타 정보
        timestamp = decision_data.get('timestamp', 0.0)
        if hasattr(timestamp, 'timestamp'):
            # Timestamp 객체인 경우
            timestamp = timestamp.timestamp()
        elif isinstance(timestamp, str):
            # 문자열인 경우 파싱
            try:
                timestamp = pd.to_datetime(timestamp).timestamp()
            except:
                timestamp = 0.0
        else:
            # 숫자인 경우 그대로 사용
            timestamp = float(timestamp) if timestamp else 0.0
        
        features.extend([
            timestamp,
            float(decision_data.get('signals_used', 0.0)),
            float(decision_data.get('strategies_count', 0.0))
        ])
        
        # 61차원으로 맞추기
        while len(features) < 61:
            features.append(0.0)
        
        features = features[:61]  # 정확히 61차원
        
        return torch.FloatTensor(features).unsqueeze(0).to(self.device)
    
    def make_decision(self, decision_data: Dict) -> Dict:
        """의사결정 수행"""
        self.model.eval()
        
        with torch.no_grad():
            # 데이터 전처리
            input_tensor = self.preprocess_decision_data(decision_data)
            
            # 모델 추론
            decisions, profit_pred, timeframe_features = self.model(input_tensor)
            
            # 의사결정 결과 생성
            result = {
                'action': self._interpret_action(decisions['action']),
                'confidence': float(decisions['confidence'].item()),
                'position_size': float(decisions['position_size'].item()),
                'leverage': self._interpret_leverage(decisions['leverage']),
                'holding_time': self._interpret_holding_time(decisions['holding_time']),
                'profit_prediction': float(profit_pred.item()),
                'timeframe_analysis': {
                    timeframe: {
                        'strength': float(features.mean().item()),
                        'trend': 'bullish' if features.mean().item() > 0 else 'bearish'
                    }
                    for timeframe, features in timeframe_features.items()
                },
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
    
    def _interpret_action(self, action_logits: torch.Tensor) -> str:
        """액션 로짓을 해석"""
        action_idx = torch.argmax(action_logits).item()
        actions = ['HOLD', 'BUY', 'SELL']
        return actions[action_idx]
    
    def _interpret_leverage(self, leverage_tensor: torch.Tensor) -> float:
        """레버리지 텐서를 해석 (1.0 ~ 20.0)"""
        leverage = float(leverage_tensor.item())
        return 1.0 + (leverage * 19.0)  # 0~1을 1~20으로 변환
    
    def _interpret_holding_time(self, holding_tensor: torch.Tensor) -> int:
        """보유시간 텐서를 해석 (30분 ~ 1440분)"""
        holding = float(holding_tensor.item())
        return int(30 + (holding * 1410))  # 0~1을 30~1440으로 변환
    
    def train_on_batch(self, batch_data: List[Dict], batch_labels: List[Dict]) -> float:
        """배치 학습"""
        self.model.train()
        
        # 배치 전처리
        inputs = []
        targets = []
        
        for data, labels in zip(batch_data, batch_labels):
            input_tensor = self.preprocess_decision_data(data)
            inputs.append(input_tensor)
            
            # 타겟 생성
            target = {
                'action': torch.tensor([labels.get('action', 0)], dtype=torch.long).to(self.device),
                'confidence': torch.tensor([labels.get('confidence', 0.5)], dtype=torch.float).to(self.device),
                'position_size': torch.tensor([labels.get('position_size', 0.5)], dtype=torch.float).to(self.device),
                'leverage': torch.tensor([labels.get('leverage', 0.5)], dtype=torch.float).to(self.device),
                'holding_time': torch.tensor([labels.get('holding_time', 0.5)], dtype=torch.float).to(self.device),
                'profit': torch.tensor([labels.get('profit', 0.0)], dtype=torch.float).to(self.device)
            }
            targets.append(target)
        
        # 배치 결합
        batch_input = torch.cat(inputs, dim=0)
        
        # 순전파
        decisions, profit_pred, _ = self.model(batch_input)
        
        # 손실 계산
        total_loss = 0.0
        
        # 액션 분류 손실
        action_targets = torch.cat([t['action'] for t in targets])
        action_loss = F.cross_entropy(decisions['action'], action_targets)
        total_loss += action_loss
        
        # 회귀 손실들
        for key in ['confidence', 'position_size', 'leverage', 'holding_time']:
            pred = decisions[key].squeeze()
            target = torch.cat([t[key] for t in targets])
            loss = F.mse_loss(pred, target)
            total_loss += loss
        
        # 수익률 예측 손실
        profit_targets = torch.cat([t['profit'] for t in targets])
        profit_loss = F.mse_loss(profit_pred.squeeze(), profit_targets)
        total_loss += profit_loss
        
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
    """샘플 Decision 데이터 생성"""
    return {
        'short_term_net_score': random.uniform(-1.0, 1.0),
        'short_term_buy_score': random.uniform(0.0, 1.0),
        'short_term_sell_score': random.uniform(0.0, 1.0),
        'short_term_confidence': random.uniform(0.0, 1.0),
        'short_term_action_value': random.uniform(-1.0, 1.0),
        'short_term_market_context': random.uniform(0.0, 1.0),
        
        'medium_term_net_score': random.uniform(-1.0, 1.0),
        'medium_term_buy_score': random.uniform(0.0, 1.0),
        'medium_term_sell_score': random.uniform(0.0, 1.0),
        'medium_term_confidence': random.uniform(0.0, 1.0),
        'medium_term_action_value': random.uniform(-1.0, 1.0),
        'medium_term_market_context': random.uniform(0.0, 1.0),
        
        'long_term_net_score': random.uniform(-1.0, 1.0),
        'long_term_buy_score': random.uniform(0.0, 1.0),
        'long_term_sell_score': random.uniform(0.0, 1.0),
        'long_term_confidence': random.uniform(0.0, 1.0),
        'long_term_action_value': random.uniform(-1.0, 1.0),
        'long_term_market_context': random.uniform(0.0, 1.0),
        
        'conflict_conflict_severity': random.uniform(0.0, 1.0),
        'conflict_directional_consensus': random.uniform(0.0, 1.0),
        'conflict_conflict_penalty': random.uniform(0.0, 1.0),
        
        'indicator_vwap': random.uniform(2000, 3000),
        'indicator_atr': random.uniform(10, 50),
        'indicator_poc': random.uniform(2000, 3000),
        'indicator_hvn': random.uniform(2000, 3000),
        'indicator_lvn': random.uniform(2000, 3000),
        'indicator_vwap_std': random.uniform(5, 20),
        'indicator_prev_day_high': random.uniform(2000, 3000),
        'indicator_prev_day_low': random.uniform(2000, 3000),
        'indicator_opening_range_high': random.uniform(2000, 3000),
        'indicator_opening_range_low': random.uniform(2000, 3000),
        
        'candle_data': {
            'open': random.uniform(2000, 3000),
            'high': random.uniform(2000, 3000),
            'low': random.uniform(2000, 3000),
            'close': random.uniform(2000, 3000),
            'volume': random.uniform(1000, 10000),
            'quote_volume': random.uniform(1000000, 10000000)
        },
        
        'timestamp': int(datetime.now().timestamp() * 1000),
        'signals_used': random.randint(5, 15),
        'strategies_count': random.randint(3, 8)
    }

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
    def create_training_labels(decision_data: List[Dict], price_data: pd.DataFrame = None) -> List[Dict]:
        """훈련용 라벨 생성 (수익률 기반)"""
        labels = []
        
        print("훈련용 라벨 생성 중...")
        
        for i, data in enumerate(decision_data):
            # 기본 라벨 (랜덤)
            label = {
                'action': random.randint(0, 2),  # 0: HOLD, 1: BUY, 2: SELL
                'confidence': random.uniform(0.3, 0.9),
                'position_size': random.uniform(0.1, 0.8),
                'leverage': random.uniform(0.2, 0.8),  # 1.4x ~ 16.2x
                'holding_time': random.uniform(0.1, 0.9),  # 30분 ~ 1296분
                'profit': random.uniform(-0.05, 0.05)  # -5% ~ +5%
            }
            
            # 가격 데이터가 있으면 실제 수익률 기반 라벨 생성
            if price_data is not None and i < len(price_data) - 1:
                current_price = price_data.iloc[i]['close']
                next_price = price_data.iloc[i + 1]['close']
                price_change = (next_price - current_price) / current_price
                
                # 가격 변화에 따른 액션 라벨
                if price_change > 0.01:  # 1% 이상 상승
                    label['action'] = 1  # BUY
                    label['confidence'] = min(0.9, 0.5 + abs(price_change) * 10)
                    label['profit'] = price_change
                elif price_change < -0.01:  # 1% 이상 하락
                    label['action'] = 2  # SELL
                    label['confidence'] = min(0.9, 0.5 + abs(price_change) * 10)
                    label['profit'] = -price_change  # 숏 포지션 수익
                else:  # 변동성 낮음
                    label['action'] = 0  # HOLD
                    label['confidence'] = 0.3
                    label['profit'] = 0.0
            
            labels.append(label)
            
            if (i + 1) % 10000 == 0:
                print(f"  라벨 생성 진행: {i + 1:,}/{len(decision_data):,}")
        
        print(f"훈련용 라벨 생성 완료: {len(labels):,}개")
        return labels

class MultiTimeframeTrainer:
    """Multi-Timeframe Transformer 훈련 클래스"""
    
    def __init__(self, engine: MultiTimeframeDecisionEngine):
        self.engine = engine
        self.training_history = []
    
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
        
        # 데이터 분할
        split_idx = int(len(decision_data) * (1 - validation_split))
        train_data = decision_data[:split_idx]
        train_labels = labels[:split_idx]
        val_data = decision_data[split_idx:]
        val_labels = labels[split_idx:]
        
        print(f"  훈련 데이터: {len(train_data):,}개")
        print(f"  검증 데이터: {len(val_data):,}개")
        
        # 훈련 루프
        best_val_loss = float('inf')
        patience = 5
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
        
        # 회귀 손실들
        for key in ['confidence', 'position_size', 'leverage', 'holding_time']:
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
        input_size=61,
        d_model=256,
        nhead=8,
        num_layers=6
    )
    
    # 1. Decision 데이터 로드
    print("\n1️⃣ Decision 데이터 로드...")
    decision_data = DecisionDataLoader.load_decision_data('agent/decisions_data.parquet')
    
    # 2. 가격 데이터 로드 (라벨 생성용)
    print("\n2️⃣ 가격 데이터 로드...")
    try:
        price_data = pd.read_csv('data/ETHUSDC_3m_historical_data.csv')
        price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
        price_data = price_data.set_index('timestamp')
        price_data = price_data[['open', 'high', 'low', 'close', 'volume', 'quote_volume']].reset_index()
        print(f"가격 데이터 로드 완료: {len(price_data):,}개")
    except Exception as e:
        print(f"가격 데이터 로드 실패: {e}")
        price_data = None
    
    # 4. decision_data와 price_data 길이 맞추기 (시간대 동기화)
    min_length = len(decision_data)
    # 전체 데이터 사용 (제한 없음)
    price_data = price_data.iloc[-min_length:].reset_index(drop=True)
    decision_data = decision_data[-min_length:]

    # 3. 훈련용 라벨 생성
    print("\n3️⃣ 훈련용 라벨 생성...")
    labels = DecisionDataLoader.create_training_labels(decision_data, price_data)
    
        
    print(f"최종 데이터 길이: {min_length:,}개 (decision_data와 price_data 동기화)")
    
    # 5. 전체 데이터 사용 (제한 없음)
    print(f"전체 데이터 사용: {len(decision_data):,}개")
    
    # 5. 훈련 실행
    print("\n4️⃣ Multi-Timeframe Transformer 훈련 시작...")
    trainer = MultiTimeframeTrainer(engine)
    training_results = trainer.train_on_decision_data(
        decision_data=decision_data,
        labels=labels,
        batch_size=64,
        epochs=20,
        validation_split=0.2
    )
    
    # 6. 훈련 결과 출력
    print(f"\n5️⃣ 훈련 결과:")
    print(f"  최고 검증 손실: {training_results['best_val_loss']:.4f}")
    print(f"  총 에포크: {training_results['total_epochs']}")
    
    # 7. 테스트
    print(f"\n6️⃣ 훈련된 모델 테스트...")
    test_data = decision_data[0]  # 첫 번째 데이터로 테스트
    decision = engine.make_decision(test_data)
    
    print("테스트 결과:")
    print(f"  액션: {decision['action']}")
    print(f"  신뢰도: {decision['confidence']:.3f}")
    print(f"  포지션 크기: {decision['position_size']:.3f}")
    print(f"  레버리지: {decision['leverage']:.1f}x")
    print(f"  보유시간: {decision['holding_time']}분")
    print(f"  수익률 예측: {decision['profit_prediction']:.3f}")
    
    # 8. 최종 모델 저장
    engine.save_model('agent/multitimeframe_transformer_trained.pth')
    print(f"\n✅ 훈련된 모델이 저장되었습니다: agent/multitimeframe_transformer_trained.pth")

if __name__ == "__main__":
    main()
