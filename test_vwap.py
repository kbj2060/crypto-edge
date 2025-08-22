#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VWAP 지표 테스트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from indicators.global_indicators import (
    initialize_global_indicators,
    get_indicator,
    update_all_indicators_with_candle
)
from datetime import datetime, timezone

def test_vwap():
    """VWAP 지표 테스트"""
    print("🚀 VWAP 지표 테스트 시작...\n")
    
    # 1. 글로벌 지표 초기화
    print("📊 1. 글로벌 지표 초기화")
    initialize_global_indicators()
    vwap = get_indicator('vwap')
    print("✅ 초기화 완료\n")
    
    # 2. 초기 VWAP 상태 확인
    print("📊 2. 초기 VWAP 상태 확인")
    vwap_status = vwap.get_vwap_status()
    print(f"   📊 현재 VWAP: ${vwap_status.get('current_vwap', 0):.2f}")
    print(f"   📊 VWAP 표준편차: ${vwap_status.get('current_vwap_std', 0):.2f}")
    print(f"   📊 데이터 개수: {vwap_status.get('data_count', 0)}개")
    print(f"   📋 세션 상태: {vwap_status.get('session_status', 'UNKNOWN')}")
    print(f"   🎯 모드: {vwap_status.get('mode', 'unknown')}")
    
    if vwap_status.get('mode') == 'session':
        print(f"   📅 세션: {vwap_status.get('session_name', 'UNKNOWN')}")
        print(f"   ⏱️  세션 진행 시간: {vwap_status.get('elapsed_minutes', 0):.1f}분")
    else:
        print(f"   📊 세션 외 시간 VWAP 계산 중")
    print()
    
    # 3. 테스트 캔들로 업데이트 후 VWAP 변화 확인
    print("📊 3. 테스트 캔들로 업데이트 후 VWAP 변화 확인")
    test_candles = [
        {
            'timestamp': datetime.now(timezone.utc),
            'open': 4600.00,
            'high': 4615.50,
            'low': 4595.20,
            'close': 4610.30,
            'volume': 2000.0
        },
        {
            'timestamp': datetime.now(timezone.utc),
            'open': 4610.30,
            'high': 4625.80,
            'low': 4605.10,
            'close': 4620.50,
            'volume': 3000.0
        },
        {
            'timestamp': datetime.now(timezone.utc),
            'open': 4620.50,
            'high': 4635.20,
            'low': 4615.80,
            'close': 4630.10,
            'volume': 2500.0
        }
    ]
    
    for i, candle in enumerate(test_candles):
        print(f"   🔄 {i+1}. 캔들 업데이트: ${candle['close']:.2f}, 거래량: {candle['volume']:.0f}")
        update_all_indicators_with_candle(candle)
        
        # 업데이트 후 VWAP 확인
        vwap_status = vwap.get_vwap_status()
        print(f"      📊 VWAP: ${vwap_status.get('current_vwap', 0):.2f}")
        print(f"      📊 VWAP 표준편차: ${vwap_status.get('current_vwap_std', 0):.2f}")
        print(f"      📊 데이터 개수: {vwap_status.get('data_count', 0)}개")
        print()
    
    # 4. 최종 VWAP 분석
    print("📊 4. 최종 VWAP 분석")
    final_result = vwap.get_current_vwap()
    final_status = vwap.get_vwap_status()
    
    if final_result:
        vwap_value = final_result.get('vwap', 0)
        vwap_std = final_result.get('vwap_std', 0)
        total_volume = final_result.get('total_volume', 0)
        
        print(f"   📊 최종 VWAP: ${vwap_value:.2f}")
        print(f"   📊 최종 VWAP 표준편차: ${vwap_std:.2f}")
        print(f"   📊 총 거래량: {total_volume:,.2f}")
        
        # VWAP 분석
        print("\n📊 VWAP 분석:")
        print(f"   💡 VWAP ${vwap_value:.2f}는 거래량 가중 평균 가격입니다")
        print(f"   💡 VWAP 표준편차 ${vwap_std:.2f}는 가격 변동성을 나타냅니다")
        
        # VWAP 밴드
        if vwap_std > 0:
            vwap_upper_1 = vwap_value + vwap_std
            vwap_lower_1 = vwap_value - vwap_std
            vwap_upper_2 = vwap_value + 2 * vwap_std
            vwap_lower_2 = vwap_value - 2 * vwap_std
            
            print(f"   📈 VWAP +1σ: ${vwap_upper_1:.2f}")
            print(f"   📉 VWAP -1σ: ${vwap_lower_1:.2f}")
            print(f"   📈 VWAP +2σ: ${vwap_upper_2:.2f}")
            print(f"   📉 VWAP -2σ: ${vwap_lower_2:.2f}")
    
    print(f"\n   📊 최종 데이터 개수: {final_status.get('data_count', 0)}개")
    print(f"   📊 최종 모드: {final_status.get('mode', 'unknown')}")
    
    # 5. 글로벌 지표 상태 확인
    print("\n📊 5. 글로벌 지표 상태 확인")
    from indicators.global_indicators import get_indicators_status
    global_status = get_indicators_status()
    
    if 'indicators' in global_status and 'vwap' in global_status['indicators']:
        vwap_global = global_status['indicators']['vwap']
        print(f"   📊 글로벌 VWAP: ${vwap_global.get('current_vwap', 0):.2f}")
        print(f"   📊 글로벌 VWAP 표준편차: ${vwap_global.get('current_vwap_std', 0):.2f}")
        print(f"   📊 글로벌 VWAP 모드: {vwap_global.get('mode', 'unknown')}")
    
    print("\n🏁 VWAP 지표 테스트 완료!")

if __name__ == "__main__":
    test_vwap()
