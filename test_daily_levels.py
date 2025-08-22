#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from indicators.daily_levels import DailyLevelsManager
from datetime import datetime, timezone

def test_daily_levels_manager():
    """DailyLevelsManager 기능 테스트"""
    print("🚀 DailyLevelsManager 테스트 시작...\n")
    
    # DailyLevelsManager 인스턴스 직접 생성
    daily_manager = DailyLevelsManager()
    print("✅ DailyLevelsManager 인스턴스 생성 완료\n")
    
    # 1. 어제 데이터 로드
    print("📊 1. 어제 데이터 로드")
    success = daily_manager.fetch_prev_day_data('ETHUSDT')
    
    if success:
        print("✅ 어제 데이터 로드 성공!")
    else:
        print("❌ 어제 데이터 로드 실패")
        return
    print()
    
    # 2. 어제 레벨 정보 확인
    print("📊 2. 어제 레벨 정보 확인")
    levels = daily_manager.get_prev_day_levels()
    
    if levels:
        print(f"   📈 고가: ${levels.get('prev_day_high', 0):.2f}")
        print(f"   📉 저가: ${levels.get('prev_day_low', 0):.2f}")
        print(f"   📊 종가: ${levels.get('prev_day_close', 0):.2f}")
        print(f"   📈 시가: ${levels.get('prev_day_open', 0):.2f}")
        print(f"   📊 거래량: {levels.get('prev_day_volume', 0):.2f}")
        print(f"   📊 캔들 개수: {levels.get('prev_day_candle_count', 0)}개")
    else:
        print("   ❌ 레벨 정보가 없습니다")
    print()
    
    # 3. 어제 데이터 사용 가능 여부 확인
    print("📊 3. 어제 데이터 사용 가능 여부 확인")
    is_available = daily_manager.is_prev_day_data_available()
    print(f"   데이터 사용 가능: {is_available}")
    
    if is_available:
        data_count = len(daily_manager.get_prev_day_data())
        print(f"   데이터 개수: {data_count}개")
    print()
    
    # 4. 고가/저가 직접 가져오기
    print("📊 4. 고가/저가 직접 가져오기")
    high, low = daily_manager.get_prev_day_high_low()
    print(f"   고가: ${high:.2f}")
    print(f"   저가: ${low:.2f}")
    print()
    
    # 5. 인스턴스 생성 테스트
    print("📊 5. 인스턴스 생성 테스트")
    daily_manager2 = DailyLevelsManager()
    print(f"   새로운 인스턴스 생성: {daily_manager is not daily_manager2}")
    print()
    
    print("🏁 DailyLevelsManager 테스트 완료!")

if __name__ == "__main__":
    test_daily_levels_manager()
