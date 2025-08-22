#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from indicators.daily_levels import DailyLevelsManager
from datetime import datetime, timezone, timedelta
import pandas as pd

def test_advanced_features():
    """DailyLevelsManager 고급 기능 테스트"""
    print("🚀 DailyLevelsManager 고급 기능 테스트 시작...\n")
    
    # DailyLevelsManager 인스턴스 생성
    daily_manager = DailyLevelsManager()
    print("✅ DailyLevelsManager 인스턴스 생성 완료\n")
    
    # 1. DataFrame 기반 일일 레벨 계산 테스트
    print("📊 1. DataFrame 기반 일일 레벨 계산 테스트")
    
    # 테스트용 데이터 생성 (어제 데이터 시뮬레이션)
    yesterday_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
    
    test_data = []
    for i in range(480):  # 3분봉 480개 (24시간)
        timestamp = yesterday_start + timedelta(minutes=i*3)
        test_data.append({
            'timestamp': int(timestamp.timestamp() * 1000),
            'open': 4200.0 + i * 0.1,
            'high': 4200.0 + i * 0.1 + 10.0,
            'low': 4200.0 + i * 0.1 - 5.0,
            'close': 4200.0 + i * 0.1 + 2.0,
            'volume': 100.0 + i * 0.5
        })
    
    df = pd.DataFrame(test_data)
    current_time = datetime.now(timezone.utc)
    
    # DataFrame 기반 일일 레벨 계산
    daily_levels = daily_manager.calculate_daily_levels_from_df(df, current_time)
    
    if daily_levels:
        print(f"   📈 고가: ${daily_levels.get('prev_day_high', 0):.2f}")
        print(f"   📉 저가: ${daily_levels.get('prev_day_low', 0):.2f}")
        print(f"   📈 시가: ${daily_levels.get('prev_day_open', 0):.2f}")
        print(f"   📊 종가: ${daily_levels.get('prev_day_close', 0):.2f}")
        print(f"   📊 거래량: {daily_levels.get('prev_day_volume', 0):.2f}")
        print(f"   📊 캔들 개수: {daily_levels.get('prev_day_candle_count', 0)}개")
    else:
        print("   ❌ 일일 레벨 계산 실패")
    print()
    
    # 2. 스윙 레벨 계산 테스트
    print("📊 2. 스윙 레벨 계산 테스트")
    
    # 더 복잡한 테스트 데이터 생성 (고점/저점 포함)
    swing_data = []
    for i in range(30):  # 30개 캔들
        if i in [5, 15, 25]:  # 고점
            high = 4300.0 + i * 2.0
            low = 4280.0 + i * 1.0
        elif i in [10, 20]:  # 저점
            high = 4270.0 + i * 1.0
            low = 4250.0 + i * 0.5
        else:  # 일반
            high = 4280.0 + i * 1.0
            low = 4260.0 + i * 0.5
        
        swing_data.append({
            'timestamp': int((datetime.now(timezone.utc) - timedelta(minutes=(30-i)*3)).timestamp() * 1000),
            'open': (high + low) / 2,
            'high': high,
            'low': low,
            'close': (high + low) / 2 + 5.0,
            'volume': 100.0 + i * 2.0
        })
    
    swing_df = pd.DataFrame(swing_data)
    swing_levels = daily_manager.calculate_swing_levels(swing_df, lookback=25)
    
    if swing_levels:
        print(f"   🔥 최근 스윙 고점: ${swing_levels.get('recent_swing_high', 0):.2f}")
        print(f"   ❄️  최근 스윙 저점: ${swing_levels.get('recent_swing_low', 0):.2f}")
    else:
        print("   ❌ 스윙 레벨 계산 실패")
    print()
    
    # 3. 모든 레벨 통합 계산 테스트
    print("📊 3. 모든 레벨 통합 계산 테스트")
    
    all_levels = daily_manager.calculate_all_levels(df, current_time)
    
    if all_levels:
        print("   📊 통합된 레벨 정보:")
        for key, value in all_levels.items():
            if 'volume' in key:
                print(f"      {key}: {value:.2f}")
            else:
                print(f"      {key}: ${value:.2f}")
    else:
        print("   ❌ 통합 레벨 계산 실패")
    print()
    
    # 4. 상태 요약 테스트
    print("📊 4. 상태 요약 테스트")
    
    # 먼저 어제 데이터 로드
    daily_manager.fetch_prev_day_data('ETHUSDT')
    
    summary = daily_manager.get_levels_summary()
    
    print("   📊 상태 요약:")
    print(f"      어제 데이터 보유: {summary.get('has_prev_day_data', False)}")
    print(f"      데이터 개수: {summary.get('data_count', 0)}개")
    print(f"      마지막 업데이트: {summary.get('last_update', 'None')}")
    
    if summary.get('prev_day_levels'):
        levels = summary['prev_day_levels']
        print(f"      어제 고가: ${levels.get('prev_day_high', 0):.2f}")
        print(f"      어제 저가: ${levels.get('prev_day_low', 0):.2f}")
    print()
    
    print("🏁 DailyLevelsManager 고급 기능 테스트 완료!")

if __name__ == "__main__":
    test_advanced_features()
