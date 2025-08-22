#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from indicators.daily_levels import DailyLevels

def test_simple_daily_levels():
    """간단한 DailyLevels 테스트"""
    print("🚀 간단한 DailyLevels 테스트 시작...\n")
    
    # DailyLevels 인스턴스 생성 (자동 로딩)
    print("📊 1. DailyLevels 인스턴스 생성 및 자동 데이터 로딩")
    daily_levels = DailyLevels('ETHUSDT', auto_load=True)
    print("✅ DailyLevels 인스턴스 생성 완료\n")
    
    # 2. 자동 로딩 결과 확인
    print("📊 2. 자동 로딩 결과 확인")
    print(f"   데이터 로드됨: {daily_levels.is_loaded()}")
    high, low = daily_levels.get_prev_day_high_low()
    print(f"   어제 고가: ${high:.2f}")
    print(f"   어제 저가: ${low:.2f}")
    print()
    
    # 3. 수동으로 다시 로드해보기
    print("📊 3. 수동으로 다시 로드해보기")
    success = daily_levels.fetch_prev_day_levels('ETHUSDT')
    
    if success:
        print("✅ 수동 데이터 로드 성공!")
    else:
        print("❌ 수동 데이터 로드 실패")
        return
    print()
    
    # 4. 최종 상태 확인
    print("📊 4. 최종 상태 확인")
    print(f"   데이터 로드됨: {daily_levels.is_loaded()}")
    high, low = daily_levels.get_prev_day_high_low()
    print(f"   어제 고가: ${high:.2f}")
    print(f"   어제 저가: ${low:.2f}")
    print()
    
    print("🏁 간단한 DailyLevels 테스트 완료!")

if __name__ == "__main__":
    test_simple_daily_levels()
