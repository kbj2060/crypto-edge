#!/usr/bin/env python3
"""
딥러닝 기반 메타 라벨링 신경망 엔진 (scikit-learn MLPClassifier 사용)

입력: 평면화된 데이터 (지표 값, 카테고리별 점수 등)
은닉 레이어: MLP (ReLU)
출력: Sigmoid로 0~1 사이의 성공 확률

Mac 호환: PyTorch 대신 scikit-learn의 MLPClassifier 사용
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


class MetaLabelingNNEngine:
    """딥러닝 기반 메타 라벨링 엔진 (scikit-learn MLPClassifier 사용)"""
    
    MODEL_PATH = "data/meta_labeling_nn_model.pkl"
    SCALER_PATH = "data/meta_labeling_nn_scaler.pkl"
    FEATURE_NAMES_PATH = "data/meta_labeling_nn_feature_names.pkl"
    
    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        max_iter: int = 500,
        confidence_threshold: float = 0.5
    ):
        """
        Args:
            hidden_layer_sizes: 은닉 레이어 크기 튜플 (예: (128, 64, 32))
            dropout: 드롭아웃 비율 (MLPClassifier는 alpha로 L2 정규화)
            learning_rate: 학습률 (MLPClassifier는 learning_rate_init)
            max_iter: 최대 반복 횟수
            confidence_threshold: 거래 실행 최소 신뢰도
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.confidence_threshold = confidence_threshold
        
        # scikit-learn MLPClassifier 사용
        self.model: Optional[MLPClassifier] = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.input_dim: Optional[int] = None
        self.is_trained = False
        
        print(f"🔧 MLPClassifier 사용 (은닉 레이어: {hidden_layer_sizes})")
    
    def extract_features(
        self,
        decision: Dict[str, Any],
        market_data: Optional[Dict[str, Any]] = None,
        indicators: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        평면화된 데이터에서 특성 추출 - 전략 score만 사용
        
        Args:
            decision: 거래 결정 딕셔너리
            market_data: 시장 데이터 (사용하지 않음)
            indicators: 지표 데이터 (사용하지 않음)
            
        Returns:
            특성 벡터 (16개 전략 score만)
        """
        features = []
        
        # 개별 전략의 score 값만 사용
        # 모든 전략 목록 (STRATEGY_CATEGORIES 기반)
        all_strategies = [
            # SHORT_TERM
            'vol_spike', 'orderflow_cvd', 'vpvr_micro', 
            'liquidity_grab', 'vwap_pinball', 'zscore_mean_reversion',
            # MEDIUM_TERM
            'multi_timeframe', 'htf_trend', 'bollinger_squeeze', 
            'support_resistance', 'ema_confluence',
            # LONG_TERM
            'oi_delta', 'vpvr', 'ichimoku', 'funding_rate'
        ]
        
        # 각 전략의 score 값만 추출 (평면화된 데이터에서)
        for strategy_name in all_strategies:
            strategy_score_key = f"{strategy_name}_score"
            
            # decision 딕셔너리에서 직접 찾기 (평면화된 형태)
            strategy_score = decision.get(strategy_score_key, 0.0)
            
            # score만 특성으로 추가
            features.append(float(strategy_score) if strategy_score is not None else 0.0)
        
        return np.array(features, dtype=np.float32)
    
    def train(
        self,
        decisions_df: pd.DataFrame,
        price_data: pd.DataFrame,
        test_size: float = 0.2,
        min_profit_threshold: float = 0.005,
        use_profit_based: bool = True
    ) -> Dict[str, Any]:
        """
        신경망 모델 학습
        
        Args:
            decisions_df: 결정 데이터프레임
            price_data: 가격 데이터프레임
            test_size: 테스트 데이터 비율
            min_profit_threshold: 최소 수익률 임계값
            use_profit_based: 실제 수익률 기반 사용 여부
            
        Returns:
            학습 결과 딕셔너리
        """
        from engines.meta_labeling_engine import MetaLabelingEngine
        
        # 기존 엔진을 사용하여 메타 라벨 생성
        temp_engine = MetaLabelingEngine()
        print("📊 메타 라벨 생성 중...")
        labeled_df = temp_engine.create_meta_labels(
            decisions_df, price_data,
            min_profit_threshold=min_profit_threshold,
            use_profit_based=use_profit_based
        )
        
        # 거래가 있는 결정만 필터링
        labeled_df = labeled_df[labeled_df['action'].isin(['LONG', 'SHORT'])]
        
        if len(labeled_df) < 100:
            return {"success": False, "message": f"학습 데이터 부족: {len(labeled_df)}개 (최소 100개 필요)"}
        
        # 특성 추출
        print("🔍 특성 추출 중...")
        X = []
        y = []
        
        for _, row in labeled_df.iterrows():
            try:
                decision_dict = row.to_dict()
                # indicators 추출 (평면화된 데이터에서)
                indicators = {}
                for key in decision_dict.keys():
                    if key.startswith('indicator_'):
                        indicator_name = key.replace('indicator_', '')
                        indicators[indicator_name] = decision_dict[key]
                
                features = self.extract_features(decision_dict, indicators=indicators)
                X.append(features)
                y.append(row['meta_label'])
            except Exception as e:
                print(f"⚠️ 특성 추출 실패 (건너뜀): {e}")
                continue
        
        if len(X) < 100:
            return {"success": False, "message": f"유효한 특성 부족: {len(X)}개"}
        
        X = np.array(X)
        y = np.array(y)
        
        # 특성 이름 저장 (전략 score만 사용)
        all_strategies = [
            'vol_spike', 'orderflow_cvd', 'vpvr_micro', 
            'liquidity_grab', 'vwap_pinball', 'zscore_mean_reversion',
            'multi_timeframe', 'htf_trend', 'bollinger_squeeze', 
            'support_resistance', 'ema_confluence',
            'oi_delta', 'vpvr', 'ichimoku', 'funding_rate'
        ]
        
        # 전략 score만 사용
        self.feature_names = [f'{strategy_name}_score' for strategy_name in all_strategies]
        
        self.input_dim = X.shape[1]
        print(f"📐 입력 차원: {self.input_dim}개 특성")
        
        # 클래스 분포 확인
        unique, counts = np.unique(y, return_counts=True)
        class_dist = dict(zip(unique, counts))
        print(f"📊 클래스 분포: {class_dist}")
        
        if len(class_dist) == 2:
            success_count = class_dist.get(1, 0)
            fail_count = class_dist.get(0, 0)
            total = success_count + fail_count
            success_rate = success_count / total if total > 0 else 0
            ratio = min(class_dist.values()) / max(class_dist.values())
            
            print(f"   ✅ 성공한 거래(1): {success_count:,}개 ({success_rate:.1%})")
            print(f"   ❌ 실패한 거래(0): {fail_count:,}개 ({(1-success_rate):.1%})")
            
            if ratio < 0.3:
                print(f"⚠️ 클래스 불균형 감지 (비율: {ratio:.2f})")
        
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # 스케일링
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # MLPClassifier 모델 초기화
        # alpha는 L2 정규화 (dropout 대신 사용)
        # learning_rate_init는 학습률
        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation='relu',
            solver='adam',
            alpha=self.dropout * 0.01,  # dropout을 alpha로 변환
            batch_size=min(200, len(X_train)),  # 배치 크기
            learning_rate_init=self.learning_rate,
            max_iter=self.max_iter,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            verbose=True
        )
        
        # 학습
        print(f"🎓 모델 학습 중... ({len(X_train)}개 샘플, 최대 {self.max_iter} 반복)")
        self.model.fit(X_train_scaled, y_train)
        
        # 최종 평가
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = np.mean(y_pred == y_test)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        precision = report.get('1', {}).get('precision', 0.0)
        recall = report.get('1', {}).get('recall', 0.0)
        
        print(f"✅ 모델 학습 완료!")
        print(f"   정확도: {accuracy:.3f}")
        print(f"   ROC-AUC: {roc_auc:.3f}")
        print(f"   Precision: {precision:.3f}")
        print(f"   Recall: {recall:.3f}")
        
        self.is_trained = True
        
        # 모델 저장
        self.save_model()
        
        return {
            "success": True,
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "precision": precision,
            "recall": recall,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "input_dim": self.input_dim
        }
    
    def predict(
        self,
        decision: Dict[str, Any],
        market_data: Optional[Dict[str, Any]] = None,
        indicators: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        거래 실행 여부 예측
        
        Args:
            decision: 거래 결정 딕셔너리
            market_data: 시장 데이터
            indicators: 지표 데이터
            
        Returns:
            예측 결과 딕셔너리
        """
        if not self.is_trained or self.model is None:
            return self._default_prediction(decision)
        
        try:
            # 특성 추출
            features = self.extract_features(decision, market_data, indicators)
            
            # feature 개수 확인 및 조정
            expected_features = self.scaler.n_features_in_ if hasattr(self.scaler, 'n_features_in_') else len(self.feature_names) if self.feature_names else 16
            actual_features = len(features)
            
            if actual_features != expected_features:
                print(f"⚠️ Feature 개수 불일치: 예상 {expected_features}개, 실제 {actual_features}개")
                # feature 개수를 맞춰주기 (부족하면 0으로 채우기, 많으면 자르기)
                if actual_features < expected_features:
                    # 부족한 feature를 0으로 채우기
                    features = np.pad(features, (0, expected_features - actual_features), 'constant', constant_values=0.0)
                    print(f"   → 부족한 {expected_features - actual_features}개 feature를 0으로 채웠습니다.")
                else:
                    # 많은 feature를 자르기
                    features = features[:expected_features]
                    print(f"   → 초과한 {actual_features - expected_features}개 feature를 제거했습니다.")
            
            features_scaled = self.scaler.transform([features])
            
            # 예측
            probability = self.model.predict_proba(features_scaled)[0][1]
            prediction = self.model.predict(features_scaled)[0]
            
            should_execute = probability >= (self.confidence_threshold * 0.9)
            
            return {
                "should_execute": should_execute,
                "prediction": int(prediction),
                "probability": float(probability),
                "confidence": "HIGH" if probability >= 0.7 else "MEDIUM" if probability >= 0.5 else "LOW"
            }
        except Exception as e:
            print(f"⚠️ 메타 라벨링 예측 실패: {e}")
            return self._default_prediction(decision)
    
    def _default_prediction(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """기본 예측 로직"""
        net_score = decision.get("net_score", 0.0)
        meta = decision.get("meta", {})
        synergy_meta = meta.get("synergy_meta", {})
        confidence = synergy_meta.get("confidence", "LOW")
        
        confidence_map = {"HIGH": 0.8, "MEDIUM": 0.5, "LOW": 0.2}
        confidence_value = confidence_map.get(confidence, 0.2)
        
        should_execute = (
            abs(net_score) > 0.2 and
            confidence_value >= 0.3
        )
        
        return {
            "should_execute": should_execute,
            "prediction": 1 if should_execute else 0,
            "probability": confidence_value * abs(net_score),
            "confidence": confidence,
            "note": "기본 휴리스틱 사용 (모델 미학습)"
        }
    
    def save_model(self):
        """모델 저장"""
        if self.model is None:
            return
        
        Path(self.MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
        
        # scikit-learn 모델은 pickle로 저장
        with open(self.MODEL_PATH, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'input_dim': self.input_dim,
                'hidden_layer_sizes': self.hidden_layer_sizes,
                'dropout': self.dropout,
                'is_trained': self.is_trained
            }, f)
        
        with open(self.SCALER_PATH, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(self.FEATURE_NAMES_PATH, 'wb') as f:
            pickle.dump(self.feature_names, f)
        
        print(f"💾 모델 저장 완료: {self.MODEL_PATH}")
    
    def load_model(self) -> bool:
        """모델 로드"""
        model_path = Path(self.MODEL_PATH)
        scaler_path = Path(self.SCALER_PATH)
        feature_names_path = Path(self.FEATURE_NAMES_PATH)
        
        # 모델 파일 존재 확인
        if not model_path.exists():
            # 이전 경로도 시도
            old_model_path = Path("engines/meta_labeling_nn_model.pkl")
            if old_model_path.exists():
                self.MODEL_PATH = str(old_model_path)
                model_path = old_model_path
            else:
                print(f"⚠️ 모델 파일을 찾을 수 없습니다. 시도한 경로: ['{self.MODEL_PATH}', 'engines/meta_labeling_nn_model.pkl']")
                return False
        
        # scaler 파일 존재 확인
        if not scaler_path.exists():
            old_scaler_path = Path("engines/meta_labeling_nn_scaler.pkl")
            if old_scaler_path.exists():
                self.SCALER_PATH = str(old_scaler_path)
                scaler_path = old_scaler_path
            else:
                print(f"⚠️ Scaler 파일을 찾을 수 없습니다. 시도한 경로: ['{self.SCALER_PATH}', 'engines/meta_labeling_nn_scaler.pkl']")
                return False
        
        # feature_names 파일 존재 확인
        if not feature_names_path.exists():
            old_feature_names_path = Path("engines/meta_labeling_nn_feature_names.pkl")
            if old_feature_names_path.exists():
                self.FEATURE_NAMES_PATH = str(old_feature_names_path)
                feature_names_path = old_feature_names_path
            else:
                print(f"⚠️ Feature names 파일을 찾을 수 없습니다. 시도한 경로: ['{self.FEATURE_NAMES_PATH}', 'engines/meta_labeling_nn_feature_names.pkl']")
                return False
        
        try:
            # 모델 파일 로드
            with open(self.MODEL_PATH, 'rb') as f:
                checkpoint = pickle.load(f)
            
            # checkpoint가 딕셔너리인지 확인
            if not isinstance(checkpoint, dict):
                print(f"❌ 모델 파일 형식 오류: 딕셔너리가 아닙니다.")
                return False
            
            # 필수 키 확인
            required_keys = ['model', 'input_dim', 'hidden_layer_sizes', 'dropout', 'is_trained']
            missing_keys = [key for key in required_keys if key not in checkpoint]
            if missing_keys:
                print(f"❌ 모델 파일에 필수 키가 없습니다: {missing_keys}")
                return False
            
            self.model = checkpoint.get('model')
            self.input_dim = checkpoint.get('input_dim')
            self.hidden_layer_sizes = checkpoint.get('hidden_layer_sizes')
            self.dropout = checkpoint.get('dropout')
            self.is_trained = checkpoint.get('is_trained')
            
            # scaler 로드 (별도 파일 또는 checkpoint에서)
            if 'scaler' in checkpoint:
                # 오래된 형식: checkpoint에 scaler 포함
                self.scaler = checkpoint['scaler']
            else:
                # 새로운 형식: 별도 파일에서 로드
                try:
                    with open(self.SCALER_PATH, 'rb') as f:
                        self.scaler = pickle.load(f)
                except FileNotFoundError:
                    print(f"⚠️ Scaler 파일을 찾을 수 없습니다: {self.SCALER_PATH}")
                    return False
            
            # feature_names 로드 (별도 파일 또는 checkpoint에서)
            if 'feature_names' in checkpoint:
                # 오래된 형식: checkpoint에 feature_names 포함
                self.feature_names = checkpoint['feature_names']
            else:
                # 새로운 형식: 별도 파일에서 로드
                try:
                    with open(self.FEATURE_NAMES_PATH, 'rb') as f:
                        self.feature_names = pickle.load(f)
                except FileNotFoundError:
                    print(f"⚠️ Feature names 파일을 찾을 수 없습니다: {self.FEATURE_NAMES_PATH}")
                    return False
            
            print(f"📂 모델 로드 완료: {self.MODEL_PATH}")
            return True
        except KeyError as e:
            print(f"❌ 모델 로드 실패 ({self.MODEL_PATH}): {e}")
            print("   모델 파일 형식이 올바르지 않습니다. 모델을 재학습하세요.")
            return False
        except FileNotFoundError as e:
            print(f"❌ 파일을 찾을 수 없습니다: {e}")
            return False
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
