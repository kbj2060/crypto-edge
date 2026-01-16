#!/usr/bin/env python3
"""
Meta-Guided Consensus 아키텍처 테스트 스크립트
"""

import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from engines.trade_decision_engine import TradeDecisionEngine
from utils.display_utils import print_decision_interpretation


def test_meta_guided_consensus():
    """Meta-Guided Consensus 아키텍처 테스트"""
    print("=" * 80)
    print("Meta-Guided Consensus 아키텍처 테스트")
    print("=" * 80)
    
    # TradeDecisionEngine 초기화
    print("\n1️⃣ TradeDecisionEngine 초기화...")
    decision_engine = TradeDecisionEngine(use_meta_labeling=True)
    print("✅ 초기화 완료")
    
    # 메타 라벨링 엔진 확인
    if decision_engine.meta_labeling_engine:
        if decision_engine.meta_labeling_engine.is_trained:
            print(f"✅ 메타 라벨링 모델 로드됨 (임계값: {decision_engine.meta_labeling_engine.confidence_threshold:.1%})")
        else:
            print("⚠️ 메타 라벨링 모델이 학습되지 않음 (기본 휴리스틱 사용)")
    else:
        print("⚠️ 메타 라벨링 엔진이 없음")
    
    # 샘플 신호 생성
    print("\n2️⃣ 샘플 신호 생성...")
    sample_signals = {
        "VOL_SPIKE": {
            "action": "BUY",
            "score": 0.8,
            "entry": 3000.0,
            "stop": 2950.0
        },
        "ORDERFLOW_CVD": {
            "action": "BUY",
            "score": 0.7,
            "entry": 3000.0,
            "stop": 2950.0
        },
        "VPVR_MICRO": {
            "action": "BUY",
            "score": 0.6,
            "entry": 3000.0,
            "stop": 2950.0
        },
        "MULTI_TIMEFRAME": {
            "action": "BUY",
            "score": 0.65,
            "entry": 3000.0,
            "stop": 2950.0
        },
        "OI_DELTA": {
            "action": "SELL",
            "score": 0.5,
            "entry": 3000.0,
            "stop": 3050.0
        }
    }
    print(f"✅ {len(sample_signals)}개 신호 생성")
    
    # 거래 결정 생성
    print("\n3️⃣ 거래 결정 생성 (Meta-Guided Consensus)...")
    try:
        result = decision_engine.decide_trade_realtime(
            signals=sample_signals,
            account_balance=10000.0,
            base_risk_pct=0.005
        )
        
        print("✅ 결정 생성 완료")
        
        # 결과 구조 확인
        print("\n4️⃣ 결과 구조 확인...")
        final_decision = result.get("final_decision")
        category_decisions = result.get("category_decisions", {})
        conflicts = result.get("conflicts", {})
        
        if final_decision:
            print("✅ final_decision 존재")
            print(f"   - Action: {final_decision.get('action')}")
            print(f"   - Net Score: {final_decision.get('net_score')}")
            print(f"   - Confidence: {final_decision.get('confidence')}")
            print(f"   - Reason: {final_decision.get('reason')}")
            
            # 메타 라벨링 정보
            meta_labeling = final_decision.get("meta", {}).get("meta_labeling", {})
            if meta_labeling:
                print(f"   - 메타 라벨링 확률: {meta_labeling.get('probability', 0):.1%}")
                print(f"   - 실행 권장: {meta_labeling.get('should_execute', False)}")
        else:
            print("❌ final_decision이 없습니다!")
            return
        
        if category_decisions:
            print(f"✅ category_decisions 존재 ({len(category_decisions)}개 카테고리)")
            for cat_name, cat_decision in category_decisions.items():
                print(f"   - {cat_name}: {cat_decision.get('action')} ({cat_decision.get('net_score', 0):.2f})")
        
        if conflicts:
            print(f"✅ conflicts 정보 존재")
            print(f"   - 충돌 여부: {conflicts.get('has_conflicts', False)}")
            print(f"   - 충돌 심각도: {conflicts.get('conflict_severity', 0):.2f}")
        
        # 출력 테스트
        print("\n5️⃣ 결정 출력 테스트...")
        print_decision_interpretation(result)
        
        print("\n" + "=" * 80)
        print("✅ 테스트 완료!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_meta_guided_consensus()

