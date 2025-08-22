#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
청산 전략용 글로벌 지표 데이터 테스트
- analyze_bucket_liquidations 함수에 필요한 모든 지표 데이터 확인
- 글로벌 지표 시스템과의 연동 테스트
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

def test_liquidation_indicators():
    """청산 전략용 지표 데이터 테스트"""
    print("🔍 청산 전략용 글로벌 지표 데이터 테스트 시작...\n")
    
    # 1. 글로벌 지표 초기화
    print("📊 1. 글로벌 지표 초기화")
    try:
        initialize_global_indicators()
        print("✅ 글로벌 지표 초기화 완료\n")
    except Exception as e:
        print(f"❌ 글로벌 지표 초기화 실패: {e}")
        return
    
    # 2. 글로벌 지표 관리자 가져오기
    print("📊 2. 글로벌 지표 관리자 확인")
    try:
        global_manager = get_global_indicator_manager()
        print("✅ 글로벌 지표 관리자 가져오기 성공\n")
    except Exception as e:
        print(f"❌ 글로벌 지표 관리자 가져오기 실패: {e}")
        return
    
    # 3. 각 지표별 데이터 확인
    print("📊 3. 청산 전략용 지표 데이터 확인")
    
    # 3.1 Daily Levels (Key Levels)
    print("🔍 3.1 Daily Levels (Key Levels) 확인")
    try:
        daily_levels = global_manager.get_indicator('daily_levels')
        if daily_levels and daily_levels.is_loaded():
            prev_day_data = daily_levels.get_prev_day_high_low()
            key_levels = {
                'prev_day_high': prev_day_data.get('high', 0),
                'prev_day_low': prev_day_data.get('low', 0)
            }
            print(f"   ✅ Daily Levels 로드 성공:")
            print(f"      📅 어제 고가: ${key_levels['prev_day_high']:.2f}")
            print(f"      📅 어제 저가: ${key_levels['prev_day_low']:.2f}")
        else:
            print("   ❌ Daily Levels 로드 실패 또는 데이터 없음")
            key_levels = {}
    except Exception as e:
        print(f"   ❌ Daily Levels 확인 오류: {e}")
        key_levels = {}
    
    # 3.2 Opening Range (세션 정보)
    print("\n🔍 3.2 Opening Range (세션 정보) 확인")
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
            print(f"   ✅ Opening Range 정보 로드 성공:")
            print(f"      🌅 세션 이름: {opening_range['session_name']}")
            print(f"      🕐 세션 시작: {opening_range['session_start']}")
            print(f"      ⏱️  경과 시간: {opening_range['elapsed_minutes']:.1f}분")
            print(f"      📋 세션 상태: {opening_range['session_status']}")
        else:
            print("   ⚠️ 세션 모드 비활성화")
            opening_range = {}
    except Exception as e:
        print(f"   ❌ Opening Range 확인 오류: {e}")
        opening_range = {}
    
    # 3.3 VWAP 및 VWAP 표준편차
    print("\n🔍 3.3 VWAP 및 VWAP 표준편차 확인")
    try:
        vwap_indicator = global_manager.get_indicator('vwap')
        vwap = 0.0
        vwap_std = 0.0
        
        if vwap_indicator:
            vwap_status = vwap_indicator.get_vwap_status()
            vwap = vwap_status.get('current_vwap', 0)
            vwap_std = vwap_status.get('current_vwap_std', 0)
            
            print(f"   ✅ VWAP 정보 로드 성공:")
            print(f"      📊 현재 VWAP: ${vwap:.2f}")
            print(f"      📊 VWAP 표준편차: ${vwap_std:.2f}")
            print(f"      📊 데이터 개수: {vwap_status.get('data_count', 0)}개")
            print(f"      🎯 모드: {vwap_status.get('mode', 'unknown')}")
        else:
            print("   ❌ VWAP 지표를 찾을 수 없음")
    except Exception as e:
        print(f"   ❌ VWAP 확인 오류: {e}")
    
    # 3.4 ATR
    print("\n🔍 3.4 ATR 확인")
    try:
        atr_indicator = global_manager.get_indicator('atr')
        atr = 0.0
        
        if atr_indicator:
            atr = atr_indicator.get_atr()
            atr_status = atr_indicator.get_status()
            
            print(f"   ✅ ATR 정보 로드 성공:")
            print(f"      📊 현재 ATR: {atr:.3f}")
            print(f"      ✅ 준비 상태: {atr_status.get('is_ready', False)}")
            print(f"      🎯 성숙 상태: {atr_status.get('is_mature', False)}")
            print(f"      📊 캔들 개수: {atr_status.get('candles_count', 0)}개")
            print(f"      📊 True Ranges: {atr_status.get('true_ranges_count', 0)}개")
        else:
            print("   ❌ ATR 지표를 찾을 수 없음")
    except Exception as e:
        print(f"   ❌ ATR 확인 오류: {e}")
    
    # 4. 청산 전략 함수 시뮬레이션
    print("\n📊 4. 청산 전략 함수 시뮬레이션")
    print("🔍 analyze_bucket_liquidations 함수에 전달될 지표 데이터:")
    
    # 시뮬레이션용 버킷 데이터
    simulation_bucket = [
        {'side': 'long', 'qty_usd': 1000, 'timestamp': datetime.now(timezone.utc)},
        {'side': 'short', 'qty_usd': 1500, 'timestamp': datetime.now(timezone.utc)},
        {'side': 'long', 'qty_usd': 800, 'timestamp': datetime.now(timezone.utc)}
    ]
    
    current_price = 4615.0  # 시뮬레이션 가격
    
    print(f"   📊 시뮬레이션 데이터:")
    print(f"      🎯 버킷 크기: {len(simulation_bucket)}개")
    print(f"      💰 현재 가격: ${current_price:.2f}")
    print(f"      📅 Key Levels: {key_levels}")
    print(f"      🌅 Opening Range: {opening_range}")
    print(f"      📊 VWAP: ${vwap:.2f}")
    print(f"      📊 VWAP STD: ${vwap_std:.2f}")
    print(f"      📊 ATR: {atr:.3f}")
    
    # 5. 지표 데이터 품질 평가
    print("\n📊 5. 지표 데이터 품질 평가")
    
    quality_score = 0
    total_indicators = 5
    
    # Key Levels 품질
    if key_levels and key_levels.get('prev_day_high', 0) > 0 and key_levels.get('prev_day_low', 0) > 0:
        print("   ✅ Key Levels: 품질 양호")
        quality_score += 1
    else:
        print("   ❌ Key Levels: 품질 불량")
    
    # Opening Range 품질
    if opening_range and opening_range.get('session_name') != 'UNKNOWN':
        print("   ✅ Opening Range: 품질 양호")
        quality_score += 1
    else:
        print("   ❌ Opening Range: 품질 불량")
    
    # VWAP 품질
    if vwap > 0:
        print("   ✅ VWAP: 품질 양호")
        quality_score += 1
    else:
        print("   ❌ VWAP: 품질 불량")
    
    # VWAP STD 품질
    if vwap_std >= 0:
        print("   ✅ VWAP STD: 품질 양호")
        quality_score += 1
    else:
        print("   ❌ VWAP STD: 품질 불량")
    
    # ATR 품질
    if atr > 0:
        print("   ✅ ATR: 품질 양호")
        quality_score += 1
    else:
        print("   ❌ ATR: 품질 불량")
    
    # 전체 품질 점수
    quality_percentage = (quality_score / total_indicators) * 100
    print(f"\n🎯 전체 지표 데이터 품질: {quality_score}/{total_indicators} ({quality_percentage:.1f}%)")
    
    if quality_percentage >= 80:
        print("   🟢 청산 전략 실행 가능: 모든 필수 지표가 준비됨")
    elif quality_percentage >= 60:
        print("   🟡 청산 전략 실행 가능: 일부 지표 부족")
    else:
        print("   🔴 청산 전략 실행 불가: 필수 지표 부족")
    
    print("\n🏁 청산 전략용 글로벌 지표 데이터 테스트 완료!")

if __name__ == "__main__":
    test_liquidation_indicators()
