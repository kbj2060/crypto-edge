#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
글로벌 지표 관리자 테스트
- 모든 지표들을 중앙에서 관리
- 새로운 3분봉 데이터로 전체 지표 자동 업데이트
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

def test_global_indicator_manager():
    """글로벌 지표 관리자 테스트"""
    print("🚀 글로벌 지표 관리자 테스트 시작...\n")
    
    # 1. 글로벌 지표 관리자 가져오기
    print("📊 1. 글로벌 지표 관리자 인스턴스 가져오기")
    manager = get_global_indicator_manager()
    print(f"   ✅ 관리자 인스턴스: {type(manager).__name__}")
    print(f"   📋 초기화 상태: {manager.is_initialized()}")
    print()
    
    # 2. 지표들 초기화
    print("📊 2. 모든 지표들 초기화")
    initialize_global_indicators()
    print(f"   📋 초기화 완료: {manager.is_initialized()}")
    print()
    
    # 3. 등록된 지표 목록 확인
    print("📊 3. 등록된 지표 목록 확인")
    indicators = manager.list_indicators()
    print(f"   📋 등록된 지표: {indicators}")
    print()
    
    # 4. 개별 지표 접근 테스트
    print("📊 4. 개별 지표 접근 테스트")
    for indicator_name in indicators:
        indicator = manager.get_indicator(indicator_name)
        print(f"   📊 {indicator_name}: {type(indicator).__name__}")
    print()
    
    # 5. 초기 상태 확인
    print("📊 5. 초기 지표 상태 확인")
    initial_status = get_indicators_status()
    print(f"   📋 상태: {initial_status['status']}")
    if 'indicators' in initial_status:
        for name, status in initial_status['indicators'].items():
            print(f"   📊 {name}: {status}")
    print()
    
    # 6. 테스트용 3분봉 데이터 생성 및 업데이트
    print("📊 6. 테스트 3분봉 데이터로 지표 업데이트")
    test_candles = [
        {
            'timestamp': datetime.now(timezone.utc),
            'open': 4600.00,
            'high': 4615.50,
            'low': 4595.20,
            'close': 4610.30,
            'volume': 1500.0
        },
        {
            'timestamp': datetime.now(timezone.utc),
            'open': 4610.30,
            'high': 4625.80,
            'low': 4605.10,
            'close': 4620.50,
            'volume': 1800.0
        },
        {
            'timestamp': datetime.now(timezone.utc),
            'open': 4620.50,
            'high': 4635.20,
            'low': 4615.80,
            'close': 4630.10,
            'volume': 2200.0
        }
    ]
    
    for i, candle in enumerate(test_candles):
        print(f"   🔄 {i+1}. 캔들 업데이트: ${candle['close']:.2f}")
        update_all_indicators_with_candle(candle)
        print()
    
    # 7. 업데이트 후 상태 확인
    print("📊 7. 업데이트 후 지표 상태 확인")
    final_status = get_indicators_status()
    print(f"   📋 상태: {final_status['status']}")
    if 'indicators' in final_status:
        for name, status in final_status['indicators'].items():
            print(f"   📊 {name}: {status}")
    print()
    
    # 8. 개별 지표 상세 정보 확인
    print("📊 8. 개별 지표 상세 정보 확인")
    
    # VPVR 상세 정보
    vpvr = get_indicator('vpvr')
    if vpvr:
        vpvr_status = vpvr.get_vpvr_status()
        print(f"   📈 VPVR 상세:")
        print(f"      활성 구간: {vpvr_status.get('active_bins', 0)}개")
        print(f"      총 거래량: {vpvr_status.get('total_volume', 0):.2f}")
        print(f"      데이터 수: {vpvr_status.get('data_count', 0)}개")
    
    # ATR 상세 정보
    atr = get_indicator('atr')
    if atr:
        print(f"   📊 ATR 상세:")
        print(f"      현재 ATR: {atr.get_atr():.3f}")
        print(f"      준비됨: {atr.is_ready()}")
        print(f"      안정됨: {len(atr.true_ranges) >= atr.length}")
        print(f"      캔들 수: {len(atr.candles)}개")
    
    # Daily Levels 상세 정보
    daily_levels = get_indicator('daily_levels')
    if daily_levels:
        daily_data = daily_levels.get_prev_day_high_low()
        print(f"   📅 Daily Levels 상세:")
        print(f"      로드됨: {daily_levels.is_loaded()}")
        print(f"      어제 고가: ${daily_data.get('high', 0):.2f}")
        print(f"      어제 저가: ${daily_data.get('low', 0):.2f}")
    
    print()
    
    # 9. 편의 함수 테스트
    print("📊 9. 편의 함수 테스트")
    print("   🔄 편의 함수로 지표 업데이트...")
    test_candle = {
        'timestamp': datetime.now(timezone.utc),
        'open': 4630.10,
        'high': 4640.50,
        'low': 4620.30,
        'close': 4625.80,
        'volume': 1900.0
    }
    update_all_indicators_with_candle(test_candle)
    print()
    
    print("🏁 글로벌 지표 관리자 테스트 완료!")

if __name__ == "__main__":
    test_global_indicator_manager()
