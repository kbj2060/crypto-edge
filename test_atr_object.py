#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from indicators.atr import ATR3M
from datetime import datetime, timezone

def test_atr_object():
    """ATR3M 객체 테스트"""
    print("🚀 ATR3M 객체 테스트 시작...\n")
    
    # ATR 객체 생성
    atr = ATR3M(length=14)
    print("✅ ATR3M 인스턴스 생성 완료\n")
    
    # 1. 초기 상태 확인
    print("📊 1. 초기 ATR 상태")
    print(f"   현재 ATR: {atr.get_atr():.3f}")
    print(f"   준비됨: {atr.is_ready()}")
    print(f"   안정됨: {len(atr.true_ranges) >= atr.length}")
    print(f"   캔들 수: {len(atr.candles)}")
    print()
    
    # 2. 테스트용 캔들 데이터 생성
    print("📊 2. 테스트 캔들 데이터로 ATR 업데이트")
    test_candles = [
        {'timestamp': datetime.now(timezone.utc), 'open': 4600.00, 'high': 4615.50, 'low': 4595.20, 'close': 4610.30},
        {'timestamp': datetime.now(timezone.utc), 'open': 4610.30, 'high': 4625.80, 'low': 4605.10, 'close': 4620.50},
        {'timestamp': datetime.now(timezone.utc), 'open': 4620.50, 'high': 4635.20, 'low': 4615.80, 'close': 4630.10},
        {'timestamp': datetime.now(timezone.utc), 'open': 4630.10, 'high': 4640.50, 'low': 4620.30, 'close': 4625.80},
        {'timestamp': datetime.now(timezone.utc), 'open': 4625.80, 'high': 4635.60, 'low': 4610.20, 'close': 4615.40},
    ]
    
    for i, candle in enumerate(test_candles):
        print(f"   {i+1}. 캔들 추가: ${candle['close']:.2f}")
        atr.update_with_candle(candle)
    
    print()
    
    # 3. 업데이트 후 ATR 상태 확인
    print("📊 3. 업데이트 후 ATR 상태")
    print(f"   현재 ATR: {atr.get_atr():.3f}")
    print(f"   준비됨: {atr.is_ready()}")
    print(f"   안정됨: {len(atr.true_ranges) >= atr.length}")
    print(f"   캔들 수: {len(atr.candles)}")
    print(f"   TR 수: {len(atr.true_ranges)}")
    print()
    
    # 4. ATR 값 사용 예시
    print("📊 4. ATR 값 사용 예시")
    if atr.is_ready():
        atr_value = atr.get_atr()
        print(f"   ATR 값: {atr_value:.3f}")
        print(f"   ATR의 20%: {atr_value * 0.2:.3f}")
        print(f"   동적 bin 크기 계산에 사용 가능")
    else:
        print("   ATR이 아직 준비되지 않음")
    
    print("\n🏁 ATR3M 객체 테스트 완료!")

if __name__ == "__main__":
    test_atr_object()
