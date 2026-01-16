#!/usr/bin/env python3
"""
메타 라벨링 모델 학습 결과 분석
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


def analyze_results():
    """학습 결과 분석"""
    print("=" * 60)
    print("메타 라벨링 모델 학습 결과 분석")
    print("=" * 60)
    
    # 모델 로드
    engine = MetaLabelingEngine()
    if not engine.load_model():
        print("❌ 모델을 로드할 수 없습니다.")
        return
    
    print("✅ 모델 로드 완료\n")
    
    # 모델 정보 출력
    if hasattr(engine.model, 'feature_importances_'):
        print("전체 특성 중요도:")
        feature_importance = dict(zip(engine.feature_names, engine.model.feature_importances_))
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        for feature, importance in sorted_features:
            bar = "█" * int(importance * 50)
            print(f"  {feature:<25}: {importance:.4f} {bar}")
    
    print("\n" + "=" * 60)
    print("결과 분석:")
    print("=" * 60)
    print("""
현재 모델 성능:
- 정확도: 51.0% (거의 랜덤 수준)
- ROC-AUC: 50.9% (거의 랜덤 수준)
- Precision: 49.1%
- Recall: 36.3%

주요 발견:
1. 데이터 증가 효과 없음
   - 이전 (1,100개 샘플): 정확도 59.3%, ROC-AUC 61.7%
   - 현재 (85,712개 샘플): 정확도 51.0%, ROC-AUC 50.9%
   - 더 많은 데이터로 학습했지만 성능이 오히려 떨어짐

2. 특성 중요도 문제
   - net_score와 abs_net_score가 98% 이상 차지
   - 다른 특성들이 거의 기여하지 않음
   - 모델이 단순히 점수만 보고 판단하고 있음

3. 가능한 원인
   - 메타 라벨 생성 로직의 문제
   - lookforward_periods (20기간)가 적절하지 않을 수 있음
   - 전략별 action 종합 로직이 부정확할 수 있음
   - 실제 거래 결과와 메타 라벨이 일치하지 않을 수 있음

개선 방안:
1. 메타 라벨 생성 로직 개선
   - lookforward_periods 조정 (10, 20, 30, 60 등 테스트)
   - 실제 수익률 기반 라벨링 (단순 방향이 아닌)
   - 손절/익절 기준 고려

2. 특성 엔지니어링 강화
   - 전략별 점수의 상호작용 특성
   - 시장 상황 특성 (변동성, 트렌드 등)
   - 시간대별 특성

3. 모델 개선
   - 더 복잡한 모델 시도 (Gradient Boosting)
   - 앙상블 모델
   - 하이퍼파라미터 튜닝

4. 데이터 품질 확인
   - 메타 라벨의 분포 확인
   - 실제 거래 결과와의 일치도 확인
    """)


if __name__ == "__main__":
    analyze_results()

