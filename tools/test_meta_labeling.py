#!/usr/bin/env python3
"""
메타 라벨링 모델 테스트 및 모니터링 스크립트
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from engines.meta_labeling_engine import MetaLabelingEngine
from engines.trade_decision_engine import TradeDecisionEngine


def test_meta_labeling():
    """메타 라벨링 모델 테스트"""
    print("=" * 60)
    print("메타 라벨링 모델 테스트")
    print("=" * 60)
    
    # 모델 로드 확인
    engine = MetaLabelingEngine()
    if not engine.load_model():
        print("❌ 모델을 로드할 수 없습니다.")
        print("   먼저 학습을 실행하세요: python tools/train_meta_labeling.py")
        return
    
    print("✅ 모델 로드 완료\n")
    
    # 샘플 결정 테스트
    sample_decisions = [
        {
            "action": "LONG",
            "net_score": 0.75,
            "meta": {
                "synergy_meta": {
                    "confidence": "HIGH",
                    "buy_score": 0.8,
                    "sell_score": 0.2,
                    "signals_used": 5
                }
            },
            "strategies_used": ["VOL_SPIKE", "ORDERFLOW_CVD", "VPVR_MICRO"],
            "sizing": {
                "risk_usd": 50.0,
                "entry_used": 3000.0,
                "stop_used": 2950.0
            },
            "leverage": 20,
            "category": "SHORT_TERM"
        },
        {
            "action": "SHORT",
            "net_score": -0.65,
            "meta": {
                "synergy_meta": {
                    "confidence": "MEDIUM",
                    "buy_score": 0.2,
                    "sell_score": 0.7,
                    "signals_used": 3
                }
            },
            "strategies_used": ["OI_DELTA", "VPVR"],
            "sizing": {
                "risk_usd": 30.0,
                "entry_used": 3000.0,
                "stop_used": 3050.0
            },
            "leverage": 10,
            "category": "MEDIUM_TERM"
        },
        {
            "action": "HOLD",
            "net_score": 0.1,
            "meta": {
                "synergy_meta": {
                    "confidence": "LOW",
                    "buy_score": 0.3,
                    "sell_score": 0.3,
                    "signals_used": 1
                }
            },
            "strategies_used": [],
            "sizing": {},
            "leverage": 1,
            "category": "SHORT_TERM"
        }
    ]
    
    print("샘플 결정 테스트:")
    print("-" * 60)
    
    for i, decision in enumerate(sample_decisions, 1):
        print(f"\n테스트 {i}: {decision['action']} (net_score: {decision['net_score']:.2f})")
        
        market_data = {
            "atr": 50.0,
            "volume": 1000000.0,
            "volatility": 0.02
        }
        
        result = engine.predict(decision, market_data)
        
        print(f"  예측 결과:")
        print(f"    - 거래 실행 권장: {result['should_execute']}")
        print(f"    - 예측: {result['prediction']} (1=실행, 0=보류)")
        print(f"    - 확률: {result['probability']:.3f}")
        print(f"    - 신뢰도: {result['confidence']}")
        
        if not result['should_execute']:
            print(f"    ⚠️ 메타 라벨링이 거래 실행을 권장하지 않음")
    
    print("\n" + "=" * 60)
    print("TradeDecisionEngine 통합 테스트")
    print("=" * 60)
    
    # TradeDecisionEngine 테스트
    decision_engine = TradeDecisionEngine(use_meta_labeling=True)
    
    # 샘플 신호 생성
    sample_signals = {
        "VOL_SPIKE": {
            "action": "BUY",
            "score": 0.7,
            "entry": 3000.0,
            "stop": 2950.0
        },
        "ORDERFLOW_CVD": {
            "action": "BUY",
            "score": 0.6,
            "entry": 3000.0,
            "stop": 2950.0
        }
    }
    
    print("\n샘플 신호로 결정 생성:")
    result = decision_engine.decide_trade_realtime(sample_signals)
    
    for category, decision in result["decisions"].items():
        print(f"\n{category}:")
        print(f"  Action: {decision['action']}")
        print(f"  Net Score: {decision['net_score']:.3f}")
        
        if "meta_labeling" in decision.get("meta", {}):
            ml_result = decision["meta"]["meta_labeling"]
            print(f"  메타 라벨링:")
            print(f"    - 실행 권장: {ml_result['should_execute']}")
            print(f"    - 확률: {ml_result['probability']:.3f}")
            print(f"    - 신뢰도: {ml_result['confidence']}")
            
            if not ml_result['should_execute']:
                print(f"    ⚠️ 원래 {decision.get('_original_action', decision['action'])}였지만 HOLD로 변경됨")


if __name__ == "__main__":
    test_meta_labeling()

