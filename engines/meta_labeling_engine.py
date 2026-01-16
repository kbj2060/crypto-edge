#!/usr/bin/env python3
"""
Meta-Labeling Engine
마르코스 로페즈 데 프라도의 메타 라벨링 기법 구현

메타 라벨링은 2단계 접근법:
1. 1단계: 방향 예측 (기존 TradeDecisionEngine)
2. 2단계: 메타 라벨링 - 거래 실행 여부 결정 (이 모듈)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class MetaLabelingEngine:
    """
    메타 라벨링 엔진
    
    기존 모델의 방향 예측이 맞는지 여부를 예측하여
    거래 실행 여부를 결정합니다.
    """
    
    def __init__(
        self,
        model_type: str = "random_forest",
        min_samples_for_training: int = 100,
        confidence_threshold: float = 0.6,
        model_save_path: Optional[str] = None
    ):
        """
        Args:
            model_type: 모델 타입 ("random_forest", "gradient_boosting")
            min_samples_for_training: 학습에 필요한 최소 샘플 수
            confidence_threshold: 거래 실행을 위한 최소 신뢰도
            model_save_path: 모델 저장 경로
        """
        self.model_type = model_type
        self.min_samples_for_training = min_samples_for_training
        self.confidence_threshold = confidence_threshold
        self.model_save_path = model_save_path or "engines/meta_labeling_model.pkl"
        
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        
        # 모델 초기화
        self._init_model()
    
    def _init_model(self):
        """모델 초기화"""
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                class_weight='balanced'
            )
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def extract_features(self, decision: Dict[str, Any], market_data: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        결정에서 특성 추출
        
        Args:
            decision: 거래 결정 딕셔너리
            market_data: 시장 데이터 (선택적)
            
        Returns:
            특성 벡터
        """
        features = []
        
        # 1. 결정 관련 특성
        net_score = decision.get("net_score", 0.0)
        action = decision.get("action", "HOLD")
        
        # Action 인코딩
        action_encoded = {"LONG": 1, "SHORT": -1, "HOLD": 0}.get(action, 0)
        features.append(action_encoded)
        features.append(net_score)
        features.append(abs(net_score))  # 절대값
        
        # 2. 신뢰도 관련 특성
        meta = decision.get("meta", {})
        synergy_meta = meta.get("synergy_meta", {})
        
        confidence = synergy_meta.get("confidence", "LOW")
        confidence_map = {"HIGH": 0.8, "MEDIUM": 0.5, "LOW": 0.2}
        confidence_value = confidence_map.get(confidence, 0.2)
        features.append(confidence_value)
        
        # 3. 전략 사용 수
        strategies_used = decision.get("strategies_used", [])
        features.append(len(strategies_used))
        
        # 4. 시너지 메타 특성
        buy_score = synergy_meta.get("buy_score", 0.0)
        sell_score = synergy_meta.get("sell_score", 0.0)
        signals_used = synergy_meta.get("signals_used", 0)
        
        features.append(buy_score)
        features.append(sell_score)
        features.append(signals_used)
        features.append(abs(buy_score - sell_score))  # 점수 차이
        
        # 5. 포지션 크기 관련
        sizing = decision.get("sizing", {})
        risk_usd = sizing.get("risk_usd", 0.0)
        leverage = decision.get("leverage", 1)
        
        features.append(risk_usd)
        features.append(leverage)
        
        # 6. 카테고리 정보
        category = decision.get("category", "")
        category_map = {"SHORT_TERM": 0, "MEDIUM_TERM": 1, "LONG_TERM": 2}
        category_encoded = category_map.get(category, 0)
        features.append(category_encoded)
        
        # 7. 시장 데이터 특성 (있는 경우)
        if market_data:
            # ATR, 볼륨, 변동성 등 추가 가능
            atr = market_data.get("atr", 0.0)
            volume = market_data.get("volume", 0.0)
            volatility = market_data.get("volatility", 0.0)
            
            features.append(atr)
            features.append(volume)
            features.append(volatility)
        else:
            # 기본값
            features.extend([0.0, 0.0, 0.0])
        
        return np.array(features, dtype=np.float32)
    
    def create_meta_labels(
        self,
        decisions_df: pd.DataFrame,
        price_data: pd.DataFrame,
        lookforward_periods: int = 20
    ) -> pd.DataFrame:
        """
        과거 결정 데이터에서 메타 라벨 생성
        
        메타 라벨: 방향 예측이 맞았는지 여부 (1: 맞음, 0: 틀림)
        
        Args:
            decisions_df: 과거 결정 데이터프레임
            price_data: 가격 데이터프레임 (close 컬럼 필요)
            lookforward_periods: 미래 몇 기간을 보고 성공 여부 판단
            
        Returns:
            메타 라벨이 추가된 데이터프레임
        """
        df = decisions_df.copy()
        
        # timestamp를 인덱스로 설정
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            df = df.set_index('timestamp').sort_index()
        
        if 'close' not in price_data.columns:
            raise ValueError("price_data must have 'close' column")
        
        price_data = price_data.copy()
        if not isinstance(price_data.index, pd.DatetimeIndex):
            if 'timestamp' in price_data.columns:
                price_data['timestamp'] = pd.to_datetime(price_data['timestamp'], utc=True)
                price_data = price_data.set_index('timestamp')
        
        price_data = price_data.sort_index()
        
        # 메타 라벨 생성
        meta_labels = []
        
        for idx, row in df.iterrows():
            # 해당 시점의 가격 찾기
            try:
                current_price = price_data.loc[idx, 'close']
            except KeyError:
                # 가장 가까운 가격 찾기
                try:
                    nearest_idx = price_data.index.get_indexer([idx], method='nearest')[0]
                    current_price = price_data.iloc[nearest_idx]['close']
                except:
                    meta_labels.append(0)
                    continue
            
            # 미래 가격 찾기
            try:
                future_idx = price_data.index[price_data.index > idx][:lookforward_periods]
                if len(future_idx) < lookforward_periods:
                    meta_labels.append(0)
                    continue
                
                future_price = price_data.loc[future_idx[-1], 'close']
            except:
                meta_labels.append(0)
                continue
            
            # 방향 예측 확인
            action = row.get('action', 'HOLD')
            if action == 'HOLD':
                meta_labels.append(0)  # HOLD는 거래하지 않으므로 0
                continue
            
            # 실제 가격 변화
            price_change = (future_price - current_price) / current_price
            
            # 방향이 맞았는지 확인
            if action == 'LONG':
                # LONG 예측이 맞았는지 (가격 상승)
                is_correct = 1 if price_change > 0 else 0
            elif action == 'SHORT':
                # SHORT 예측이 맞았는지 (가격 하락)
                is_correct = 1 if price_change < 0 else 0
            else:
                is_correct = 0
            
            meta_labels.append(is_correct)
        
        df['meta_label'] = meta_labels
        return df.reset_index()
    
    def train(
        self,
        decisions_df: pd.DataFrame,
        price_data: pd.DataFrame,
        test_size: float = 0.2,
        retrain: bool = False
    ) -> Dict[str, Any]:
        """
        메타 라벨링 모델 학습
        
        Args:
            decisions_df: 결정 데이터프레임
            price_data: 가격 데이터프레임
            test_size: 테스트 데이터 비율
            retrain: 기존 모델이 있어도 재학습할지 여부
            
        Returns:
            학습 결과 딕셔너리
        """
        # 메타 라벨 생성
        print("📊 메타 라벨 생성 중...")
        labeled_df = self.create_meta_labels(decisions_df, price_data)
        
        # 거래가 있는 결정만 필터링 (HOLD 제외)
        labeled_df = labeled_df[labeled_df['action'].isin(['LONG', 'SHORT'])]
        
        if len(labeled_df) < self.min_samples_for_training:
            print(f"⚠️ 학습 데이터 부족: {len(labeled_df)} < {self.min_samples_for_training}")
            return {
                "success": False,
                "message": f"학습 데이터 부족: {len(labeled_df)}개"
            }
        
        # 특성 추출
        print("🔍 특성 추출 중...")
        X = []
        y = []
        
        for _, row in labeled_df.iterrows():
            try:
                # row를 딕셔너리로 변환
                if isinstance(row, pd.Series):
                    decision_dict = row.to_dict()
                else:
                    decision_dict = dict(row)
                
                features = self.extract_features(decision_dict)
                X.append(features)
                y.append(row['meta_label'])
            except Exception as e:
                print(f"⚠️ 특성 추출 실패 (건너뜀): {e}")
                continue
        
        if len(X) < self.min_samples_for_training:
            print(f"⚠️ 유효한 특성 부족: {len(X)} < {self.min_samples_for_training}")
            return {
                "success": False,
                "message": f"유효한 특성 부족: {len(X)}개"
            }
        
        X = np.array(X)
        y = np.array(y)
        
        # 특성 이름 저장
        self.feature_names = [
            'action_encoded', 'net_score', 'abs_net_score', 'confidence',
            'num_strategies', 'buy_score', 'sell_score', 'signals_used',
            'score_diff', 'risk_usd', 'leverage', 'category',
            'atr', 'volume', 'volatility'
        ]
        
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # 특성 스케일링
        try:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        except Exception as e:
            print(f"❌ 특성 스케일링 실패: {e}")
            # 스케일링 실패 시 원본 데이터 사용
            X_train_scaled = X_train
            X_test_scaled = X_test
        
        # 모델 학습
        print(f"🎓 모델 학습 중... ({len(X_train)}개 샘플)")
        self.model.fit(X_train_scaled, y_train)
        
        # 예측 및 평가
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # 정확도 계산
        accuracy = np.mean(y_pred == y_test)
        
        # ROC-AUC 계산
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except:
            roc_auc = 0.0
        
        # 분류 리포트
        report = classification_report(y_test, y_pred, output_dict=True)
        
        print(f"✅ 모델 학습 완료!")
        print(f"   정확도: {accuracy:.3f}")
        print(f"   ROC-AUC: {roc_auc:.3f}")
        print(f"   Precision: {report['1']['precision']:.3f}")
        print(f"   Recall: {report['1']['recall']:.3f}")
        
        self.is_trained = True
        
        # 모델 저장
        self.save_model()
        
        return {
            "success": True,
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "classification_report": report,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "feature_importance": dict(zip(
                self.feature_names,
                self.model.feature_importances_
            )) if hasattr(self.model, 'feature_importances_') else {}
        }
    
    def predict(self, decision: Dict[str, Any], market_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        거래 실행 여부 예측
        
        Args:
            decision: 거래 결정 딕셔너리
            market_data: 시장 데이터 (선택적)
            
        Returns:
            예측 결과 딕셔너리
        """
        if not self.is_trained:
            # 모델이 학습되지 않았으면 기본 로직 사용
            return self._default_prediction(decision)
        
        # 특성 추출
        try:
            features = self.extract_features(decision, market_data)
            features_scaled = self.scaler.transform([features])
            
            # 예측
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0][1]
        except Exception as e:
            # 예측 실패 시 기본 로직 사용
            print(f"⚠️ 메타 라벨링 예측 실패: {e}")
            return self._default_prediction(decision)
        
        # 거래 실행 여부 결정
        should_execute = (
            prediction == 1 and 
            probability >= self.confidence_threshold
        )
        
        return {
            "should_execute": should_execute,
            "prediction": int(prediction),
            "probability": float(probability),
            "confidence": "HIGH" if probability >= 0.7 else "MEDIUM" if probability >= 0.5 else "LOW"
        }
    
    def _default_prediction(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """모델이 학습되지 않았을 때 기본 예측 로직"""
        net_score = decision.get("net_score", 0.0)
        meta = decision.get("meta", {})
        synergy_meta = meta.get("synergy_meta", {})
        confidence = synergy_meta.get("confidence", "LOW")
        
        # 간단한 휴리스틱
        confidence_map = {"HIGH": 0.8, "MEDIUM": 0.5, "LOW": 0.2}
        confidence_value = confidence_map.get(confidence, 0.2)
        
        # 높은 점수와 신뢰도일 때만 실행
        should_execute = (
            abs(net_score) > 0.3 and
            confidence_value >= 0.5
        )
        
        return {
            "should_execute": should_execute,
            "prediction": 1 if should_execute else 0,
            "probability": confidence_value * abs(net_score),
            "confidence": confidence,
            "note": "기본 휴리스틱 사용 (모델 미학습)"
        }
    
    def save_model(self, path: Optional[str] = None):
        """모델 저장"""
        if self.model is None:
            return
        
        save_path = path or self.model_save_path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'model_type': self.model_type,
                'is_trained': self.is_trained
            }, f)
        
        print(f"💾 모델 저장 완료: {save_path}")
    
    def load_model(self, path: Optional[str] = None):
        """모델 로드"""
        load_path = path or self.model_save_path
        
        if not Path(load_path).exists():
            print(f"⚠️ 모델 파일 없음: {load_path}")
            return False
        
        try:
            with open(load_path, 'rb') as f:
                data = pickle.load(f)
            
            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_names = data.get('feature_names', [])
            self.model_type = data.get('model_type', self.model_type)
            self.is_trained = data.get('is_trained', False)
            
            print(f"📂 모델 로드 완료: {load_path}")
            return True
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            return False

