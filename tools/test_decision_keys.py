#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
decide_trade_realtime 함수의 키 개수 테스트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engines.trade_decision_engine import TradeDecisionEngine
from managers.data_manager import get_data_manager
from managers.strategy_executor import StrategyExecutor
from indicators.global_indicators import get_global_indicator_manager
import time

def test_decision_keys():
    """decide_trade_realtime 함수의 키 개수 테스트"""
    
    print("=== decide_trade_realtime 키 개수 테스트 ===")
    
    # 초기화
    data_manager = get_data_manager()
    strategy_executor = StrategyExecutor(data_manager)
    decision_engine = TradeDecisionEngine()
    global_manager = get_global_indicator_manager()
    
    # 테스트용 신호 생성
    test_signals = {
        "short_term": {
            "session_or": {"action": "BUY", "score": 0.7, "entry": 2300, "stop": 2280},
            "orderflow_cvd": {"action": "HOLD", "score": 0.0, "entry": None, "stop": None},
            "vwap_pinball": {"action": "SELL", "score": 0.6, "entry": 2295, "stop": 2305},
            "vol_spike": {"action": "HOLD", "score": 0.0, "entry": None, "stop": None},
            "liquidity_grab": {"action": "HOLD", "score": 0.0, "entry": None, "stop": None},
            "vpvr_micro": {"action": None, "score": None, "entry": None, "stop": None},
            "zscore_mean_reversion": {"action": "HOLD", "score": 0.0, "entry": None, "stop": None}
        },
        "medium_term": {
            "bollinger_squeeze": {"action": "BUY", "score": 0.3, "entry": 2290, "stop": 2270},
            "htf_trend": {"action": "HOLD", "score": 0.2, "entry": None, "stop": None},
            "multi_timeframe": {"action": "BUY", "score": 0.4, "entry": None, "stop": None},
            "support_resistance": {"action": "BUY", "score": 0.6, "entry": None, "stop": None},
            "ema_confluence": {"action": "SELL", "score": 0.5, "entry": None, "stop": None}
        },
        "long_term": {
            "vpvr": {"action": "SELL", "score": 0.8, "entry": 2295, "stop": 2310},
            "ichimoku": {"action": "SELL", "score": 1.0, "entry": None, "stop": None},
            "oi_delta": {"action": "HOLD", "score": 0.0, "entry": None, "stop": None},
            "funding_rate": {"action": "SELL", "score": 0.3, "entry": None, "stop": None}
        }
    }
    
    # 여러 번 테스트
    key_counts = []
    for i in range(5):
        print(f"\n--- 테스트 {i+1} ---")
        
        # 거래 결정
        result = decision_engine.decide_trade_realtime(test_signals)
        
        # 전체 키 개수
        total_keys = len(result)
        print(f"전체 키 개수: {total_keys}")
        
        # 각 카테고리별 키 개수
        decisions = result.get('decisions', {})
        for category, decision in decisions.items():
            decision_keys = len(decision)
            print(f"  {category}: {decision_keys}개")
            
            # 중첩된 키들도 확인
            if 'sizing' in decision:
                sizing_keys = len(decision['sizing'])
                print(f"    sizing: {sizing_keys}개")
            
            if 'meta' in decision:
                meta_keys = len(decision['meta'])
                print(f"    meta: {meta_keys}개")
                
                if 'synergy_meta' in decision['meta']:
                    synergy_keys = len(decision['meta']['synergy_meta'])
                    print(f"      synergy_meta: {synergy_keys}개")
        
        # conflicts 키 개수
        conflicts = result.get('conflicts', {})
        conflicts_keys = len(conflicts)
        print(f"  conflicts: {conflicts_keys}개")
        
        # meta 키 개수
        meta = result.get('meta', {})
        meta_keys = len(meta)
        print(f"  meta: {meta_keys}개")
        
        key_counts.append(total_keys)
        
        # 잠시 대기
        time.sleep(0.1)
    
    # 결과 분석
    print(f"\n=== 결과 분석 ===")
    print(f"키 개수 범위: {min(key_counts)} ~ {max(key_counts)}")
    print(f"키 개수 분포: {set(key_counts)}")
    
    if len(set(key_counts)) > 1:
        print("❌ 키 개수가 일관되지 않습니다!")
        return False
    else:
        print("✅ 키 개수가 일관됩니다!")
        return True

if __name__ == "__main__":
    test_decision_keys()
