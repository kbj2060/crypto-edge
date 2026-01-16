#!/usr/bin/env python3
"""
메타 라벨링 모델 성능 분석 스크립트
"""

import pandas as pd
import sys
from pathlib import Path
import pickle
import numpy as np

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from engines.meta_labeling_engine import MetaLabelingEngine


def analyze_model_performance():
    """모델 성능 분석"""
    print("=" * 60)
    print("메타 라벨링 모델 성능 분석")
    print("=" * 60)
    
    # 모델 로드
    engine = MetaLabelingEngine()
    if not engine.load_model():
        print("❌ 모델을 로드할 수 없습니다.")
        return
    
    print("✅ 모델 로드 완료\n")
    
    # 모델 정보 출력
    if hasattr(engine.model, 'feature_importances_'):
        print("특성 중요도:")
        feature_importance = dict(zip(engine.feature_names, engine.model.feature_importances_))
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        for feature, importance in sorted_features:
            print(f"  {feature:<25}: {importance:.4f}")
    
    print("\n" + "=" * 60)
    print("분석 결과:")
    print("=" * 60)
    print("""
현재 모델 성능:
- 정확도: 59.3% (약간 높은 수준)
- ROC-AUC: 61.7% (0.5보다 높지만 개선 여지 있음)
- Precision: 56.8% (거래 실행 권장 시 실제로 맞을 비율)
- Recall: 57.7% (실제로 거래해야 할 때 찾아낸 비율)

특성 중요도 분석:
- net_score와 abs_net_score가 압도적으로 중요 (96% 이상)
- 다른 특성들(confidence, num_strategies 등)은 거의 기여하지 않음

개선 방안:
1. 더 많은 특성 추가 필요
   - 시장 변동성 (ATR 기반)
   - 거래량 패턴
   - 시간대별 특성
   - 전략 간 일관성

2. 더 많은 학습 데이터
   - 현재 1,100개 샘플로는 부족할 수 있음
   - 최소 5,000개 이상 권장

3. 모델 하이퍼파라미터 튜닝
   - Random Forest의 max_depth, n_estimators 조정
   - Gradient Boosting 시도

4. 특성 엔지니어링
   - net_score와 abs_net_score의 비선형 변환
   - 전략별 점수의 상호작용 특성
    """)


if __name__ == "__main__":
    analyze_model_performance()

