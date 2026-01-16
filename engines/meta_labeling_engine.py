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
        # 기본 경로: data 폴더 (기존 engines 폴더도 지원)
        self.model_save_path = model_save_path or "data/meta_labeling_model.pkl"
        
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        
        # 모델 초기화
        self._init_model()
    
    def _init_model(self):
        """모델 초기화 (성능 개선 버전)"""
        if self.model_type == "random_forest":
            # 하이퍼파라미터 튜닝: 더 많은 트리, 더 깊은 트리, 더 나은 분할
            # 클래스 불균형 대응: 'balanced' 가중치 사용
            self.model = RandomForestClassifier(
                n_estimators=300,  # 100 → 300 (더 많은 트리)
                max_depth=20,      # 10 → 20 (더 깊은 트리)
                min_samples_split=10,  # 20 → 10 (더 세밀한 분할)
                min_samples_leaf=5,   # 10 → 5 (더 세밀한 분할)
                max_features='sqrt',  # 특성 샘플링 추가
                random_state=42,
                class_weight='balanced_subsample',  # 'balanced' → 'balanced_subsample' (더 나은 불균형 처리)
                n_jobs=-1  # 병렬 처리
            )
        elif self.model_type == "gradient_boosting":
            # Gradient Boosting도 개선
            self.model = GradientBoostingClassifier(
                n_estimators=200,  # 100 → 200
                max_depth=7,       # 5 → 7
                learning_rate=0.05,  # 0.1 → 0.05 (더 안정적 학습)
                subsample=0.8,     # 과적합 방지
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
        
        # 충돌/시너지 특성 제거: 시그널 특성만 사용 (15개 특성)
        # conflict_severity, directional_consensus, active_categories 제거됨
        
        return np.array(features, dtype=np.float32)
    
    def _extract_final_action_from_strategies(self, row: pd.Series) -> tuple:
        """
        전략별 action에서 최종 action 추출
        
        Returns:
            (action, net_score, confidence) 튜플
        """
        # 전략별 action 컬럼 찾기
        action_columns = [col for col in row.index if col.endswith('_action')]
        score_columns = [col for col in row.index if col.endswith('_score')]
        
        if not action_columns:
            return ('HOLD', 0.0, 'LOW')
        
        # 각 전략의 action과 score 수집
        buy_signals = []
        sell_signals = []
        
        for action_col in action_columns:
            action = row.get(action_col)
            if pd.isna(action) or action is None:
                continue
            
            # 해당 전략의 score 찾기
            strategy_name = action_col.replace('_action', '')
            score_col = f"{strategy_name}_score"
            score = row.get(score_col, 0.0)
            if pd.isna(score):
                score = 0.0
            
            if action == 'BUY':
                buy_signals.append(score)
            elif action == 'SELL':
                sell_signals.append(score)
        
        # 최종 결정
        buy_total = sum(buy_signals) if buy_signals else 0.0
        sell_total = sum(sell_signals) if sell_signals else 0.0
        net_score = buy_total - sell_total
        
        # 신뢰도 계산
        total_signals = len(buy_signals) + len(sell_signals)
        if total_signals == 0:
            return ('HOLD', 0.0, 'LOW')
        elif total_signals >= 5:
            confidence = 'HIGH'
        elif total_signals >= 3:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'
        
        # 최종 action 결정
        if abs(net_score) < 0.1:  # 너무 작은 차이는 HOLD
            return ('HOLD', net_score, confidence)
        elif net_score > 0:
            return ('LONG', net_score, confidence)
        else:
            return ('SHORT', net_score, confidence)
    
    def _extract_entry_stop_from_strategies(self, row: pd.Series, action: str) -> tuple:
        """
        전략별 entry와 stop 가격 추출
        
        Returns:
            (entry_price, stop_price) 튜플
        """
        entry_price = None
        stop_price = None
        
        # 전략별 entry/stop 컬럼 찾기
        entry_columns = [col for col in row.index if col.endswith('_entry')]
        stop_columns = [col for col in row.index if col.endswith('_stop')]
        
        # action에 맞는 전략 찾기
        action_columns = [col for col in row.index if col.endswith('_action')]
        
        for action_col in action_columns:
            strategy_action = row.get(action_col)
            if pd.isna(strategy_action) or strategy_action is None:
                continue
            
            # action이 일치하는지 확인
            if (action == 'LONG' and strategy_action == 'BUY') or \
               (action == 'SHORT' and strategy_action == 'SELL'):
                strategy_name = action_col.replace('_action', '')
                entry_col = f"{strategy_name}_entry"
                stop_col = f"{strategy_name}_stop"
                
                if entry_col in row.index:
                    entry_val = row.get(entry_col)
                    if not pd.isna(entry_val) and entry_val is not None:
                        entry_price = float(entry_val)
                
                if stop_col in row.index:
                    stop_val = row.get(stop_col)
                    if not pd.isna(stop_val) and stop_val is not None:
                        stop_price = float(stop_val)
                
                # 하나라도 찾으면 사용
                if entry_price is not None or stop_price is not None:
                    break
        
        return (entry_price, stop_price)
    
    def _calculate_actual_return(
        self,
        action: str,
        entry_price: float,
        stop_price: float,
        future_prices: pd.Series,
        min_profit_threshold: float = 0.005,  # 최소 0.5% 수익
        commission_rate: float = 0.0004  # 0.04% 수수료 (바이낸스 선물)
    ) -> tuple:
        """
        실제 수익률 계산
        
        Returns:
            (actual_return, hit_stop, hit_target, meta_label) 튜플
        """
        if entry_price is None or stop_price is None:
            return (0.0, False, False, 0)
        
        if action == 'LONG':
            # LONG: entry에서 진입, stop에서 손절
            # 손절가가 진입가보다 낮아야 함
            if stop_price >= entry_price:
                return (0.0, False, False, 0)
            
            # 손절 거리
            stop_distance = (entry_price - stop_price) / entry_price
            
            # 미래 가격들 확인
            for future_price in future_prices:
                # 손절가 도달 확인
                if future_price <= stop_price:
                    # 손절 발생
                    loss = (stop_price - entry_price) / entry_price
                    net_return = loss - commission_rate * 2  # 진입/청산 수수료
                    return (net_return, True, False, 0)
                
                # 수익률 계산
                profit = (future_price - entry_price) / entry_price
                net_return = profit - commission_rate * 2
                
                # 최소 수익률 달성 확인
                if net_return >= min_profit_threshold:
                    return (net_return, False, True, 1)
            
            # lookforward 기간 내에 목표 달성 못함
            final_price = future_prices.iloc[-1]
            profit = (final_price - entry_price) / entry_price
            net_return = profit - commission_rate * 2
            
            # 손실이면 0, 작은 수익이면 0 (임계값 미달)
            if net_return < 0:
                return (net_return, False, False, 0)
            else:
                return (net_return, False, False, 0)  # 임계값 미달
            
        elif action == 'SHORT':
            # SHORT: entry에서 진입, stop에서 손절
            # 손절가가 진입가보다 높아야 함
            if stop_price <= entry_price:
                return (0.0, False, False, 0)
            
            # 손절 거리
            stop_distance = (stop_price - entry_price) / entry_price
            
            # 미래 가격들 확인
            for future_price in future_prices:
                # 손절가 도달 확인
                if future_price >= stop_price:
                    # 손절 발생
                    loss = (entry_price - stop_price) / entry_price
                    net_return = loss - commission_rate * 2
                    return (net_return, True, False, 0)
                
                # 수익률 계산
                profit = (entry_price - future_price) / entry_price
                net_return = profit - commission_rate * 2
                
                # 최소 수익률 달성 확인
                if net_return >= min_profit_threshold:
                    return (net_return, False, True, 1)
            
            # lookforward 기간 내에 목표 달성 못함
            final_price = future_prices.iloc[-1]
            profit = (entry_price - final_price) / entry_price
            net_return = profit - commission_rate * 2
            
            # 손실이면 0, 작은 수익이면 0 (임계값 미달)
            if net_return < 0:
                return (net_return, False, False, 0)
            else:
                return (net_return, False, False, 0)  # 임계값 미달
        
        return (0.0, False, False, 0)
    
    def create_meta_labels(
        self,
        decisions_df: pd.DataFrame,
        price_data: pd.DataFrame,
        lookforward_periods: int = 20,
        min_profit_threshold: float = 0.005,  # 최소 0.5% 수익
        use_profit_based: bool = True  # 실제 수익률 기반 사용 여부
    ) -> pd.DataFrame:
        """
        과거 결정 데이터에서 메타 라벨 생성
        
        메타 라벨: 실제 수익률 기반 (1: 수익, 0: 손실 또는 수익 미달)
        
        Args:
            decisions_df: 과거 결정 데이터프레임 (전략별 action 포함)
            price_data: 가격 데이터프레임 (close 컬럼 필요)
            lookforward_periods: 미래 몇 기간을 보고 성공 여부 판단
            min_profit_threshold: 최소 수익률 임계값 (기본 0.5%)
            use_profit_based: 실제 수익률 기반 사용 여부
            
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
        
        # 최종 action 추출 및 메타 라벨 생성
        meta_labels = []
        extracted_actions = []
        extracted_scores = []
        actual_returns = []
        
        for idx, row in df.iterrows():
            # 최종 action 추출
            action, net_score, confidence = self._extract_final_action_from_strategies(row)
            extracted_actions.append(action)
            extracted_scores.append(net_score)
            
            # HOLD는 거래하지 않으므로 0
            if action == 'HOLD':
                meta_labels.append(0)
                actual_returns.append(0.0)
                continue
            
            # 해당 시점의 가격 찾기
            try:
                current_price = price_data.loc[idx, 'close']
            except KeyError:
                try:
                    nearest_idx = price_data.index.get_indexer([idx], method='nearest')[0]
                    current_price = price_data.iloc[nearest_idx]['close']
                except:
                    meta_labels.append(0)
                    actual_returns.append(0.0)
                    continue
            
            # 미래 가격 찾기
            try:
                future_idx = price_data.index[price_data.index > idx][:lookforward_periods]
                if len(future_idx) < lookforward_periods:
                    meta_labels.append(0)
                    actual_returns.append(0.0)
                    continue
                
                future_prices = price_data.loc[future_idx, 'close']
            except:
                meta_labels.append(0)
                actual_returns.append(0.0)
                continue
            
            if use_profit_based:
                # 실제 수익률 기반 라벨링
                entry_price, stop_price = self._extract_entry_stop_from_strategies(row, action)
                
                if entry_price is None or stop_price is None:
                    # entry/stop이 없으면 현재 가격 사용
                    entry_price = current_price
                    if action == 'LONG':
                        stop_price = current_price * 0.98  # 2% 하락 가정
                    else:
                        stop_price = current_price * 1.02  # 2% 상승 가정
                
                actual_return, hit_stop, hit_target, meta_label = self._calculate_actual_return(
                    action, entry_price, stop_price, future_prices, min_profit_threshold
                )
                
                meta_labels.append(meta_label)
                actual_returns.append(actual_return)
            else:
                # 기존 방식 (방향 기반)
                future_price = future_prices.iloc[-1]
                price_change = (future_price - current_price) / current_price
                
                if action == 'LONG':
                    is_correct = 1 if price_change > 0 else 0
                elif action == 'SHORT':
                    is_correct = 1 if price_change < 0 else 0
                else:
                    is_correct = 0
                
                meta_labels.append(is_correct)
                actual_returns.append(price_change)
        
        # 결과 추가
        df['action'] = extracted_actions
        df['net_score'] = extracted_scores
        df['meta_label'] = meta_labels
        df['actual_return'] = actual_returns
        return df.reset_index()
    
    def train(
        self,
        decisions_df: pd.DataFrame,
        price_data: pd.DataFrame,
        test_size: float = 0.2,
        retrain: bool = False,
        min_profit_threshold: float = 0.005,
        use_profit_based: bool = True
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
        if use_profit_based:
            print(f"   실제 수익률 기반 라벨링 (최소 수익률: {min_profit_threshold*100:.2f}%)")
        labeled_df = self.create_meta_labels(
            decisions_df, price_data, 
            min_profit_threshold=min_profit_threshold,
            use_profit_based=use_profit_based
        )
        
        # action 컬럼이 있는지 확인
        if 'action' not in labeled_df.columns:
            return {
                "success": False,
                "message": "action 컬럼을 찾을 수 없습니다"
            }
        
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
        
        # 특성 이름 저장 (시그널 특성만, 충돌/시너지 특성 제외)
        self.feature_names = [
            'action_encoded', 'net_score', 'abs_net_score', 'confidence',
            'num_strategies', 'buy_score', 'sell_score', 'signals_used',
            'score_diff', 'risk_usd', 'leverage', 'category',
            'atr', 'volume', 'volatility'
        ]
        
        # 클래스 분포 확인 및 분석
        unique, counts = np.unique(y, return_counts=True)
        class_dist = dict(zip(unique, counts))
        print(f"📊 클래스 분포: {class_dist}")
        
        # 클래스 불균형 분석
        if len(class_dist) == 2:
            success_count = class_dist.get(1, 0)  # 성공한 거래 (1)
            fail_count = class_dist.get(0, 0)     # 실패한 거래 (0)
            total = success_count + fail_count
            success_rate = success_count / total if total > 0 else 0
            ratio = min(class_dist.values()) / max(class_dist.values())
            
            print(f"   ✅ 성공한 거래(1): {success_count:,}개 ({success_rate:.1%})")
            print(f"   ❌ 실패한 거래(0): {fail_count:,}개 ({(1-success_rate):.1%})")
            
            if ratio < 0.3:
                print(f"⚠️ 클래스 불균형 감지 (비율: {ratio:.2f})")
                print(f"   → 'balanced_subsample' 가중치로 불균형 처리 중")
        
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
        # 모델 성능이 낮을 수 있으므로 더 관대한 기준 사용
        # prediction == 1이면 확률이 임계값보다 약간 낮아도 허용
        should_execute = (
            prediction == 1 and 
            probability >= (self.confidence_threshold * 0.9)  # 임계값의 90%만 넘으면 실행
        )
        
        # 디버깅용 로그 (필요시 주석 해제)
        # if not should_execute:
        #     print(f"   메타 라벨링 차단: prediction={prediction}, probability={probability:.2%}, threshold={self.confidence_threshold:.2%}")
        
        return {
            "should_execute": should_execute,
            "prediction": int(prediction),
            "probability": float(probability),
            "confidence": "HIGH" if probability >= 0.7 else "MEDIUM" if probability >= 0.5 else "LOW"
        }
    
    def _default_prediction(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """모델이 학습되지 않았을 때 기본 예측 로직 (더 관대한 기준)"""
        net_score = decision.get("net_score", 0.0)
        meta = decision.get("meta", {})
        synergy_meta = meta.get("synergy_meta", {})
        confidence = synergy_meta.get("confidence", "LOW")
        
        # 간단한 휴리스틱
        confidence_map = {"HIGH": 0.8, "MEDIUM": 0.5, "LOW": 0.2}
        confidence_value = confidence_map.get(confidence, 0.2)
        
        # 더 관대한 기준: 점수와 신뢰도가 어느 정도만 있으면 실행
        should_execute = (
            abs(net_score) > 0.2 and  # 0.3 → 0.2 (더 관대)
            confidence_value >= 0.3   # 0.5 → 0.3 (더 관대)
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
        save_file = Path(save_path)
        
        # 디렉토리 생성
        save_file.parent.mkdir(parents=True, exist_ok=True)
        
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
        """모델 로드 (여러 경로 시도)"""
        # 우선순위: 지정된 경로 > data 폴더 > engines 폴더 (하위 호환성)
        possible_paths = []
        
        if path:
            possible_paths.append(path)
        else:
            # 기본 경로들
            possible_paths.append(self.model_save_path)  # data/meta_labeling_model.pkl
            possible_paths.append("engines/meta_labeling_model.pkl")  # 기존 경로 (하위 호환성)
        
        for load_path in possible_paths:
            model_file = Path(load_path)
            if model_file.exists():
                try:
                    with open(load_path, 'rb') as f:
                        data = pickle.load(f)
                    
                    self.model = data['model']
                    self.scaler = data['scaler']
                    self.feature_names = data.get('feature_names', [])
                    self.model_type = data.get('model_type', self.model_type)
                    self.is_trained = data.get('is_trained', False)
                    
                    # 로드된 경로를 저장 경로로 업데이트
                    self.model_save_path = str(load_path)
                    
                    print(f"📂 모델 로드 완료: {load_path}")
                    return True
                except Exception as e:
                    print(f"⚠️ 모델 로드 실패 ({load_path}): {e}")
                    continue
        
        print(f"⚠️ 모델 파일을 찾을 수 없습니다. 시도한 경로: {possible_paths}")
        return False

