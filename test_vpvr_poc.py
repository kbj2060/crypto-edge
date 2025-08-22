#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VPVR POC(Point of Control) 값 테스트
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

def test_vpvr_poc():
    """VPVR POC 값 테스트"""
    print("🚀 VPVR POC(Point of Control) 테스트 시작...\n")
    
    # 1. 글로벌 지표 초기화
    print("📊 1. 글로벌 지표 초기화")
    initialize_global_indicators()
    print("✅ 초기화 완료\n")
    
    # 2. VPVR 지표 가져오기
    print("📊 2. VPVR 지표 가져오기")
    vpvr = get_indicator('vpvr')
    if not vpvr:
        print("❌ VPVR 지표를 가져올 수 없습니다")
        return
    print("✅ VPVR 지표 획득\n")
    
    # 3. 초기 VPVR 상태 및 POC 확인
    print("📊 3. 초기 VPVR 상태 및 POC 확인")
    vpvr_status = vpvr.get_vpvr_status()
    print(f"   📈 활성 구간: {vpvr_status.get('active_bins', 0)}개")
    print(f"   📊 총 거래량: {vpvr_status.get('total_volume', 0):.2f}")
    print(f"   📋 세션 상태: {vpvr_status.get('session_status', 'UNKNOWN')}")
    
    # VPVR 결과 가져오기
    vpvr_result = vpvr.get_current_vpvr()
    if vpvr_result:
        print(f"   🎯 POC (Point of Control): ${vpvr_result.get('poc', 0):.2f}")
        print(f"   📈 HVN (High Volume Node): ${vpvr_result.get('hvn', 0):.2f}")
        print(f"   📉 LVN (Low Volume Node): ${vpvr_result.get('lvn', 0):.2f}")
    else:
        print("   ⚠️ VPVR 결과가 없습니다")
    print()
    
    # 4. 테스트 캔들로 업데이트 후 POC 변화 확인
    print("📊 4. 테스트 캔들로 업데이트 후 POC 변화 확인")
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
        
        # 업데이트 후 POC 확인
        vpvr_result = vpvr.get_current_vpvr()
        if vpvr_result:
            print(f"      🎯 POC: ${vpvr_result.get('poc', 0):.2f}")
            print(f"      📈 HVN: ${vpvr_result.get('hvn', 0):.2f}")
            print(f"      📉 LVN: ${vpvr_result.get('lvn', 0):.2f}")
        
        vpvr_status = vpvr.get_vpvr_status()
        print(f"      📊 활성 구간: {vpvr_status.get('active_bins', 0)}개")
        print(f"      📊 총 거래량: {vpvr_status.get('total_volume', 0):.2f}")
        print()
    
    # 5. 최종 VPVR 분석
    print("📊 5. 최종 VPVR 분석")
    final_result = vpvr.get_current_vpvr()
    final_status = vpvr.get_vpvr_status()
    
    if final_result:
        poc = final_result.get('poc', 0)
        hvn = final_result.get('hvn', 0)
        lvn = final_result.get('lvn', 0)
        
        print(f"   🎯 최종 POC: ${poc:.2f}")
        print(f"   📈 최종 HVN: ${hvn:.2f}")
        print(f"   📉 최종 LVN: ${lvn:.2f}")
        print()
        
        # POC 분석
        print("📊 POC 분석:")
        print(f"   💡 POC(Point of Control)는 가장 많은 거래량이 발생한 가격대입니다")
        print(f"   💡 현재 POC ${poc:.2f}는 주요 지지/저항 레벨로 활용할 수 있습니다")
        
        # HVN/LVN 분석
        if hvn != poc:
            print(f"   💡 HVN ${hvn:.2f}는 높은 거래량 구간으로 강한 지지/저항 레벨입니다")
        if lvn != poc:
            print(f"   💡 LVN ${lvn:.2f}는 낮은 거래량 구간으로 가격이 빠르게 움직일 수 있는 구간입니다")
    
    print(f"\n   📊 최종 활성 구간: {final_status.get('active_bins', 0)}개")
    print(f"   📊 최종 총 거래량: {final_status.get('total_volume', 0):.2f}")
    print(f"   📊 데이터 개수: {final_status.get('data_count', 0)}개")
    
    print("\n🏁 VPVR POC 테스트 완료!")

if __name__ == "__main__":
    test_vpvr_poc()
