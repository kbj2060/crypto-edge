#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ATR 계산 문제 진단 및 디버깅
- ATR 지표의 현재 상태 확인
- 데이터 로딩 상태 점검
- 계산 과정 상세 분석
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from indicators.global_indicators import (
    initialize_global_indicators,
    get_indicator
)
from indicators.atr import ATR3M
import pandas as pd
import numpy as np

def debug_atr_calculation():
    """ATR 계산 문제 진단"""
    print("🔍 ATR 계산 문제 진단 시작...\n")
    
    # 1. 글로벌 지표 초기화
    print("📊 1. 글로벌 지표 초기화")
    try:
        initialize_global_indicators()
        print("✅ 글로벌 지표 초기화 완료\n")
    except Exception as e:
        print(f"❌ 글로벌 지표 초기화 실패: {e}")
        return
    
    # 2. ATR 지표 가져오기
    print("📊 2. ATR 지표 상태 확인")
    try:
        atr = get_indicator('atr')
        if not atr:
            print("❌ ATR 지표를 찾을 수 없습니다.")
            return
        
        print(f"✅ ATR 지표 가져오기 성공")
        print(f"   📊 클래스: {type(atr).__name__}")
        print(f"   📊 길이: {atr.length}")
        print(f"   📊 최대 캔들: {atr.max_candles}")
        print()
    except Exception as e:
        print(f"❌ ATR 지표 가져오기 실패: {e}")
        return
    
    # 3. ATR 내부 상태 상세 분석
    print("📊 3. ATR 내부 상태 상세 분석")
    try:
        print(f"🔍 ATR 객체 속성들:")
        for attr in dir(atr):
            if not attr.startswith('_'):
                try:
                    value = getattr(atr, attr)
                    if not callable(value):
                        print(f"   📊 {attr}: {value}")
                except Exception as e:
                    print(f"   ❌ {attr}: 접근 불가 ({e})")
        print()
        
        # 캔들 데이터 확인
        if hasattr(atr, 'candles'):
            candles = atr.candles
            print(f"📊 캔들 데이터 상태:")
            print(f"   📊 캔들 개수: {len(candles) if candles else 0}")
            if candles and len(candles) > 0:
                print(f"   📊 첫 번째 캔들: {candles[0] if len(candles) > 0 else 'None'}")
                print(f"   📊 마지막 캔들: {candles[-1] if len(candles) > 0 else 'None'}")
        else:
            print("❌ candles 속성이 없습니다.")
        print()
        
    except Exception as e:
        print(f"❌ ATR 내부 상태 분석 실패: {e}")
    
    # 4. ATR 계산 메서드 테스트
    print("📊 4. ATR 계산 메서드 테스트")
    try:
        # get_atr() 메서드 테스트
        atr_value = atr.get_atr()
        print(f"📊 get_atr() 결과: {atr_value}")
        
        # is_ready() 메서드 테스트
        if hasattr(atr, 'is_ready'):
            is_ready = atr.is_ready()
            print(f"✅ is_ready() 결과: {is_ready}")
        
        # is_mature() 메서드 테스트 (있는 경우)
        if hasattr(atr, 'is_mature'):
            is_mature = atr.is_mature()
            print(f"🎯 is_mature() 결과: {is_mature}")
        
        print()
        
    except Exception as e:
        print(f"❌ ATR 계산 메서드 테스트 실패: {e}")
    
    # 5. ATR 클래스 직접 테스트
    print("📊 5. ATR 클래스 직접 테스트")
    try:
        # 새로운 ATR 인스턴스 생성
        print("🔧 새로운 ATR 인스턴스 생성 테스트")
        test_atr = ATR3M(length=14, max_candles=100)
        print(f"✅ 테스트 ATR 생성: {test_atr}")
        print(f"   📊 길이: {test_atr.length}")
        print(f"   📊 최대 캔들: {test_atr.max_candles}")
        print(f"   📊 초기 ATR: {test_atr.get_atr()}")
        print(f"   📊 준비 상태: {test_atr.is_ready()}")
        print()
        
        # 테스트 캔들 데이터로 ATR 계산 테스트
        print("🔧 테스트 캔들로 ATR 계산 테스트")
        test_candles = [
            {'open': 4600, 'high': 4610, 'low': 4590, 'close': 4605, 'volume': 1000},
            {'open': 4605, 'high': 4620, 'low': 4600, 'close': 4615, 'volume': 1200},
            {'open': 4615, 'high': 4630, 'low': 4610, 'close': 4625, 'volume': 1100},
            {'open': 4625, 'high': 4640, 'low': 4620, 'close': 4635, 'volume': 1300},
            {'open': 4635, 'high': 4650, 'low': 4630, 'close': 4645, 'volume': 1400},
        ]
        
        for i, candle in enumerate(test_candles):
            test_atr.update_with_candle(candle)
            current_atr = test_atr.get_atr()
            print(f"   🔄 {i+1}번째 캔들 후 ATR: {current_atr:.3f}")
        
        print(f"   📊 최종 ATR: {test_atr.get_atr():.3f}")
        print(f"   ✅ 준비 상태: {test_atr.is_ready()}")
        print()
        
    except Exception as e:
        print(f"❌ ATR 클래스 직접 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
    
    # 6. 문제 해결 방안 제시
    print("📊 6. 문제 해결 방안")
    print("🔧 가능한 문제점들:")
    print("   1. 캔들 데이터가 제대로 로드되지 않음")
    print("   2. ATR 계산에 필요한 최소 캔들 수 부족")
    print("   3. 캔들 데이터 형식 문제")
    print("   4. ATR 클래스의 update_with_candle 메서드 문제")
    print()
    
    print("🔧 해결 방안:")
    print("   1. 글로벌 지표 업데이트 루프에서 ATR이 제대로 업데이트되는지 확인")
    print("   2. ATR에 테스트 캔들 데이터를 직접 주입하여 계산 테스트")
    print("   3. ATR 클래스의 내부 로직 점검")
    print()
    
    print("🏁 ATR 계산 문제 진단 완료!")

if __name__ == "__main__":
    debug_atr_calculation()
