#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
글로벌 지표 관리자 편의 함수들 테스트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from indicators.global_indicators import (
    get_global_indicator_manager,
    initialize_global_indicators,
    update_all_indicators_with_candle,
    get_indicator,
    get_indicators_status
)
from datetime import datetime, timezone

def test_convenience_functions():
    """편의 함수들 테스트"""
    print("🚀 글로벌 지표 관리자 편의 함수 테스트 시작...\n")
    
    # 1. 초기화 함수 테스트
    print("📊 1. initialize_global_indicators() 함수 테스트")
    manager = initialize_global_indicators()
    print(f"   ✅ 초기화 완료: {manager.is_initialized()}")
    print()
    
    # 2. get_indicator() 함수 테스트
    print("📊 2. get_indicator() 함수 테스트")
    vpvr = get_indicator('vpvr')
    atr = get_indicator('atr')
    daily_levels = get_indicator('daily_levels')
    
    print(f"   📈 VPVR: {type(vpvr).__name__ if vpvr else 'None'}")
    print(f"   📊 ATR: {type(atr).__name__ if atr else 'None'}")
    print(f"   📅 Daily Levels: {type(daily_levels).__name__ if daily_levels else 'None'}")
    print()
    
    # 3. get_indicators_status() 함수 테스트
    print("📊 3. get_indicators_status() 함수 테스트")
    status = get_indicators_status()
    print(f"   📋 전체 상태: {status['status']}")
    if 'indicators' in status:
        for name, indicator_status in status['indicators'].items():
            print(f"   📊 {name}: 상태 확인됨")
    print()
    
    # 4. update_all_indicators_with_candle() 함수 테스트
    print("📊 4. update_all_indicators_with_candle() 함수 테스트")
    test_candle = {
        'timestamp': datetime.now(timezone.utc),
        'open': 4600.00,
        'high': 4615.50,
        'low': 4595.20,
        'close': 4610.30,
        'volume': 1500.0
    }
    
    print("   🔄 테스트 캔들로 업데이트...")
    update_all_indicators_with_candle(test_candle)
    print()
    
    # 5. 업데이트 후 개별 지표 값 확인
    print("📊 5. 업데이트 후 개별 지표 값 확인")
    
    # VPVR 확인
    vpvr = get_indicator('vpvr')
    if vpvr:
        vpvr_status = vpvr.get_vpvr_status()
        print(f"   📈 VPVR 활성 구간: {vpvr_status.get('active_bins', 0)}개")
    
    # ATR 확인
    atr = get_indicator('atr')
    if atr:
        atr_value = atr.get_atr()
        print(f"   📊 ATR 값: {atr_value:.3f}")
        print(f"   📊 ATR 준비됨: {atr.is_ready()}")
    
    # Daily Levels 확인
    daily_levels = get_indicator('daily_levels')
    if daily_levels:
        levels = daily_levels.get_prev_day_high_low()
        print(f"   📅 어제 고가: ${levels.get('high', 0):.2f}")
        print(f"   📅 어제 저가: ${levels.get('low', 0):.2f}")
    
    print()
    
    # 6. 최종 상태 확인
    print("📊 6. 최종 상태 확인")
    final_status = get_indicators_status()
    print(f"   📋 상태: {final_status['status']}")
    print(f"   🕐 마지막 업데이트: {final_status.get('last_update', 'N/A')}")
    
    if 'indicators' in final_status:
        for name, indicator_status in final_status['indicators'].items():
            if name == 'vpvr':
                print(f"   📈 VPVR: {indicator_status.get('active_bins', 0)}개 구간, "
                      f"{indicator_status.get('total_volume', 0):.0f} 총 거래량")
            elif name == 'atr':
                print(f"   📊 ATR: {indicator_status.get('current_atr', 0):.3f}, "
                      f"준비됨: {indicator_status.get('is_ready', False)}")
            elif name == 'daily_levels':
                print(f"   📅 Daily Levels: 고가 ${indicator_status.get('prev_day_high', 0):.2f}, "
                      f"저가 ${indicator_status.get('prev_day_low', 0):.2f}")
    
    print("\n🏁 편의 함수 테스트 완료!")

if __name__ == "__main__":
    test_convenience_functions()
