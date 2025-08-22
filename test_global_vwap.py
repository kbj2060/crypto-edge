#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
글로벌 지표 VWAP 동작 확인 테스트
- 글로벌 지표 초기화 시 VWAP 동작 확인
- VWAP 데이터 로딩 및 업데이트 확인
- 다른 지표들과의 연동 확인
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from indicators.global_indicators import (
    initialize_global_indicators,
    get_indicator,
    update_all_indicators_with_candle,
    get_indicators_status
)
from datetime import datetime, timezone

def test_global_vwap():
    """글로벌 지표 VWAP 테스트"""
    print("🚀 글로벌 지표 VWAP 동작 확인 테스트 시작...\n")
    
    # 1. 글로벌 지표 초기화
    print("📊 1. 글로벌 지표 초기화")
    try:
        initialize_global_indicators()
        print("✅ 글로벌 지표 초기화 완료\n")
    except Exception as e:
        print(f"❌ 글로벌 지표 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 2. VWAP 지표 가져오기
    print("📊 2. VWAP 지표 가져오기")
    vwap = get_indicator('vwap')
    if vwap is None:
        print("❌ VWAP 지표를 가져올 수 없습니다")
        return
    
    print("✅ VWAP 지표 가져오기 성공")
    
    # 3. 초기 VWAP 상태 확인
    print("\n📊 3. 초기 VWAP 상태 확인")
    try:
        vwap_status = vwap.get_vwap_status()
        print(f"   📊 현재 VWAP: ${vwap_status.get('current_vwap', 0):.2f}")
        print(f"   📊 VWAP 표준편차: ${vwap_status.get('current_vwap_std', 0):.2f}")
        print(f"   📊 데이터 개수: {vwap_status.get('data_count', 0)}개")
        print(f"   📋 세션 상태: {vwap_status.get('session_status', 'UNKNOWN')}")
        print(f"   🎯 모드: {vwap_status.get('mode', 'unknown')}")
        
        if vwap_status.get('mode') == 'session':
            print(f"   📅 세션: {vwap_status.get('session_name', 'UNKNOWN')}")
            print(f"   ⏱️  세션 진행 시간: {vwap_status.get('elapsed_minutes', 0):.1f}분")
        elif vwap_status.get('mode') == 'outside_session':
            print(f"   📊 세션 외 시간 VWAP 계산 중")
        
        print()
    except Exception as e:
        print(f"❌ VWAP 상태 확인 실패: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. 모든 지표 상태 확인
    print("📊 4. 모든 지표 상태 확인")
    try:
        indicators_status = get_indicators_status()
        print(f"   📊 등록된 지표: {list(indicators_status.get('indicators', {}).keys())}")
        
        for name, status in indicators_status.get('indicators', {}).items():
            print(f"   📊 {name}: {status}")
        
        print()
    except Exception as e:
        print(f"❌ 지표 상태 확인 실패: {e}")
    
    # 5. 테스트 캔들로 업데이트
    print("📊 5. 테스트 캔들로 업데이트")
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
        try:
            update_all_indicators_with_candle(candle)
            
            # 업데이트 후 VWAP 확인
            vwap_status = vwap.get_vwap_status()
            print(f"      📊 VWAP: ${vwap_status.get('current_vwap', 0):.2f}")
            print(f"      📊 VWAP 표준편차: ${vwap_status.get('current_vwap_std', 0):.2f}")
            print(f"      📊 데이터 개수: {vwap_status.get('data_count', 0)}개")
            print()
        except Exception as e:
            print(f"      ❌ 업데이트 실패: {e}")
            import traceback
            traceback.print_exc()
    
    # 6. 최종 VWAP 분석
    print("📊 6. 최종 VWAP 분석")
    try:
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
            if vwap_std > 0:
                vwap_plus_1sigma = vwap_value + vwap_std
                vwap_minus_1sigma = vwap_value - vwap_std
                vwap_plus_2sigma = vwap_value + (2 * vwap_std)
                vwap_minus_2sigma = vwap_value - (2 * vwap_std)
                
                print(f"\n📊 VWAP 분석:")
                print(f"   💡 VWAP ${vwap_value:.2f}는 거래량 가중 평균 가격입니다")
                print(f"   💡 VWAP 표준편차 ${vwap_std:.2f}는 가격 변동성을 나타냅니다")
                print(f"   📈 VWAP +1σ: ${vwap_plus_1sigma:.2f}")
                print(f"   📉 VWAP -1σ: ${vwap_minus_1sigma:.2f}")
                print(f"   📈 VWAP +2σ: ${vwap_plus_2sigma:.2f}")
                print(f"   📉 VWAP -2σ: ${vwap_minus_2sigma:.2f}")
            
            print(f"\n   📊 최종 데이터 개수: {final_status.get('data_count', 0)}개")
            print(f"   📊 최종 모드: {final_status.get('mode', 'unknown')}")
        
    except Exception as e:
        print(f"❌ 최종 VWAP 분석 실패: {e}")
        import traceback
        traceback.print_exc()
    
    # 7. 글로벌 지표 상태 최종 확인
    print("\n📊 7. 글로벌 지표 상태 최종 확인")
    try:
        indicators_status = get_indicators_status()
        for name, status in indicators_status.items():
            if name == 'vwap':
                print(f"   📊 {name}: {status}")
                if isinstance(status, dict):
                    print(f"      📊 VWAP: ${status.get('current_vwap', 0):.2f}")
                    print(f"      📊 모드: {status.get('mode', 'unknown')}")
                    print(f"      📊 데이터 개수: {status.get('data_count', 0)}개")
        
    except Exception as e:
        print(f"❌ 글로벌 지표 상태 확인 실패: {e}")
    
    print("\n🏁 글로벌 지표 VWAP 동작 확인 테스트 완료!")

if __name__ == "__main__":
    test_global_vwap()
