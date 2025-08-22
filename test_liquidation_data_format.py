#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
청산 전략 데이터 형식 테스트
- analyze_all_strategies 함수에 전달되는 데이터 형식 확인
- 웹소켓에서 생성되는 데이터 구조 검증
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from indicators.global_indicators import (
    initialize_global_indicators,
    get_global_indicator_manager
)
from indicators.opening_range import get_session_manager
from datetime import datetime, timezone
import pandas as pd

def test_liquidation_data_format():
    """청산 전략 데이터 형식 테스트"""
    print("🔍 청산 전략 데이터 형식 테스트 시작...\n")
    
    # 1. 글로벌 지표 초기화
    print("📊 1. 글로벌 지표 초기화")
    try:
        initialize_global_indicators()
        print("✅ 글로벌 지표 초기화 완료\n")
    except Exception as e:
        print(f"❌ 글로벌 지표 초기화 실패: {e}")
        return
    
    # 2. 웹소켓 시뮬레이션 데이터 생성
    print("📊 2. 웹소켓 시뮬레이션 데이터 생성")
    
    # 시뮬레이션용 kline 데이터 (웹소켓에서 받는 형식)
    simulation_kline = {
        'c': '4615.50',  # close price (현재 가격)
        'o': '4610.00',  # open price
        'h': '4620.00',  # high price
        'l': '4605.00',  # low price
        'v': '1500.00'   # volume
    }
    
    current_price = float(simulation_kline['c'])
    print(f"   📊 시뮬레이션 kline 데이터:")
    print(f"      💰 현재 가격 (close): ${current_price:.2f}")
    print(f"      📈 시가: ${float(simulation_kline['o']):.2f}")
    print(f"      📊 고가: ${float(simulation_kline['h']):.2f}")
    print(f"      📉 저가: ${float(simulation_kline['l']):.2f}")
    print(f"      📊 거래량: {float(simulation_kline['v']):.2f}")
    
    # 3. 글로벌 지표 데이터 수집 (웹소켓 코드와 동일)
    print("\n📊 3. 글로벌 지표 데이터 수집")
    
    try:
        global_manager = get_global_indicator_manager()
        
        # Daily Levels (Key Levels)
        daily_levels = global_manager.get_indicator('daily_levels')
        key_levels = {}
        if daily_levels and daily_levels.is_loaded():
            prev_day_data = daily_levels.get_prev_day_high_low()
            key_levels = {
                'prev_day_high': prev_day_data.get('high', 0),
                'prev_day_low': prev_day_data.get('low', 0)
            }
            print(f"   ✅ Key Levels 생성 완료: {key_levels}")
        
        # Opening Range
        opening_range = {}
        try:
            session_manager = get_session_manager()
            session_config = session_manager.get_indicator_mode_config()
            
            if session_config.get('use_session_mode'):
                opening_range = {
                    'session_name': session_config.get('session_name', 'UNKNOWN'),
                    'session_start': session_config.get('session_start_time'),
                    'elapsed_minutes': session_config.get('elapsed_minutes', 0),
                    'session_status': session_config.get('session_status', 'UNKNOWN')
                }
                print(f"   ✅ Opening Range 생성 완료: {opening_range}")
        except Exception as e:
            print(f"   ❌ Opening Range 생성 실패: {e}")
        
        # VWAP 및 VWAP 표준편차
        vwap_indicator = global_manager.get_indicator('vwap')
        vwap = 0.0
        vwap_std = 0.0
        if vwap_indicator:
            vwap_status = vwap_indicator.get_vwap_status()
            vwap = vwap_status.get('current_vwap', 0)
            vwap_std = vwap_status.get('current_vwap_std', 0)
            print(f"   ✅ VWAP 데이터 생성 완료: ${vwap:.2f}, STD: ${vwap_std:.2f}")
        
        # ATR
        atr_indicator = global_manager.get_indicator('atr')
        atr = 0.0
        if atr_indicator:
            atr = atr_indicator.get_atr()
            print(f"   ✅ ATR 데이터 생성 완료: {atr:.3f}")
        
    except Exception as e:
        print(f"   ❌ 글로벌 지표 데이터 수집 실패: {e}")
        return
    
    # 4. price_data DataFrame 생성 (웹소켓 코드와 동일)
    print("\n📊 4. price_data DataFrame 생성")
    
    try:
        # 웹소켓에서 생성하는 방식과 동일
        price_data = pd.DataFrame({
            'timestamp': [datetime.now(timezone.utc)],
            'open': [current_price],      # 현재 가격을 open으로 사용
            'high': [current_price],      # 현재 가격을 high로 사용
            'low': [current_price],       # 현재 가격을 low로 사용
            'close': [current_price],     # 현재 가격을 close로 사용
            'volume': [0.0]               # 웹소켓에서는 거래량 정보 없음
        })
        
        print(f"   ✅ price_data DataFrame 생성 완료:")
        print(f"      📊 데이터 타입: {type(price_data)}")
        print(f"      📊 행 개수: {len(price_data)}")
        print(f"      📊 열 개수: {len(price_data.columns)}")
        print(f"      📊 열 이름: {list(price_data.columns)}")
        print(f"      📊 데이터 형식:")
        print(price_data.to_string(index=False))
        
    except Exception as e:
        print(f"   ❌ price_data DataFrame 생성 실패: {e}")
        return
    
    # 5. analyze_all_strategies 함수 호출 시뮬레이션
    print("\n📊 5. analyze_all_strategies 함수 호출 시뮬레이션")
    
    print("🔍 analyze_all_strategies 함수에 전달될 매개변수:")
    print(f"   📊 price_data: {type(price_data)} (행: {len(price_data)}, 열: {len(price_data.columns)})")
    print(f"   📅 key_levels: {type(key_levels)} - {key_levels}")
    print(f"   🌅 opening_range: {type(opening_range)} - {opening_range}")
    print(f"   📊 vwap: {type(vwap)} - {vwap}")
    print(f"   📊 vwap_std: {type(vwap_std)} - {vwap_std}")
    print(f"   📊 atr: {type(atr)} - {atr}")
    
    # 6. 데이터 형식 검증
    print("\n📊 6. 데이터 형식 검증")
    
    validation_results = []
    
    # price_data 검증
    if isinstance(price_data, pd.DataFrame):
        if len(price_data) > 0 and all(col in price_data.columns for col in ['timestamp', 'open', 'high', 'low', 'close', 'volume']):
            validation_results.append(("price_data", "✅ DataFrame 형식 및 필수 열 검증 통과"))
        else:
            validation_results.append(("price_data", "❌ DataFrame 형식은 맞지만 필수 열 누락"))
    else:
        validation_results.append(("price_data", "❌ DataFrame 형식이 아님"))
    
    # key_levels 검증
    if isinstance(key_levels, dict) and 'prev_day_high' in key_levels and 'prev_day_low' in key_levels:
        validation_results.append(("key_levels", "✅ 딕셔너리 형식 및 필수 키 검증 통과"))
    else:
        validation_results.append(("key_levels", "❌ 딕셔너리 형식이 아니거나 필수 키 누락"))
    
    # opening_range 검증
    if isinstance(opening_range, dict) and 'session_name' in opening_range:
        validation_results.append(("opening_range", "✅ 딕셔너리 형식 및 필수 키 검증 통과"))
    else:
        validation_results.append(("opening_range", "❌ 딕셔너리 형식이 아니거나 필수 키 누락"))
    
    # vwap 검증
    if isinstance(vwap, (int, float)):
        validation_results.append(("vwap", "✅ 숫자 형식 검증 통과"))
    else:
        validation_results.append(("vwap", "❌ 숫자 형식이 아님"))
    
    # vwap_std 검증
    if isinstance(vwap_std, (int, float)):
        validation_results.append(("vwap_std", "✅ 숫자 형식 검증 통과"))
    else:
        validation_results.append(("vwap_std", "❌ 숫자 형식이 아님"))
    
    # atr 검증
    if isinstance(atr, (int, float)):
        validation_results.append(("atr", "✅ 숫자 형식 검증 통과"))
    else:
        validation_results.append(("atr", "❌ 숫자 형식이 아님"))
    
    # 검증 결과 출력
    for param_name, result in validation_results:
        print(f"   {result}")
    
    # 전체 검증 결과
    passed_count = sum(1 for _, result in validation_results if "✅" in result)
    total_count = len(validation_results)
    
    print(f"\n🎯 전체 데이터 형식 검증 결과: {passed_count}/{total_count}")
    
    if passed_count == total_count:
        print("   🟢 모든 데이터 형식이 analyze_all_strategies 함수 요구사항에 맞습니다!")
    else:
        print("   🔴 일부 데이터 형식에 문제가 있습니다.")
    
    print("\n🏁 청산 전략 데이터 형식 테스트 완료!")

if __name__ == "__main__":
    test_liquidation_data_format()
