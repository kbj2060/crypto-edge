#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
글로벌 지표 현재 상태 전체 확인
- 모든 등록된 지표의 현재 값들 확인
- 지표별 상세 정보 출력
- 전체 시스템 상태 요약
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from indicators.global_indicators import (
    initialize_global_indicators,
    get_indicators_status,
    get_indicator
)
from datetime import datetime, timezone, timedelta

def check_all_global_indicators():
    """글로벌 지표 전체 상태 확인"""
    print("🔍 글로벌 지표 전체 상태 확인 시작...\n")
    
    # 1. 글로벌 지표 초기화 (이미 초기화되어 있다면 건너뜀)
    print("📊 1. 글로벌 지표 초기화 확인")
    try:
        initialize_global_indicators()
        print("✅ 글로벌 지표 초기화 완료\n")
    except Exception as e:
        print(f"❌ 글로벌 지표 초기화 실패: {e}")
        return
    
    # 2. 전체 지표 상태 가져오기
    print("📊 2. 전체 지표 상태 수집")
    try:
        indicators_status = get_indicators_status()
        print("✅ 지표 상태 수집 완료\n")
    except Exception as e:
        print(f"❌ 지표 상태 수집 실패: {e}")
        return
    
    # 3. 시스템 전체 상태 요약
    print("📊 3. 시스템 전체 상태 요약")
    print(f"   🕐 수집 시간: {indicators_status.get('timestamp', 'N/A')}")
    print(f"   📊 시스템 상태: {indicators_status.get('system_status', 'N/A')}")
    print(f"   📈 등록된 지표: {len(indicators_status.get('indicators', {}))}개")
    print()
    
    # 4. 각 지표별 상세 정보
    print("📊 4. 각 지표별 상세 정보")
    indicators = indicators_status.get('indicators', {})
    
    for indicator_name, indicator_data in indicators.items():
        print(f"🔍 {indicator_name.upper()} 지표 상세 정보:")
        print(f"   📊 데이터: {indicator_data}")
        
        # 지표별 특화 정보 출력
        if indicator_name == 'vpvr':
            print(f"   📈 활성 가격 구간: {indicator_data.get('active_bins', 0)}개")
            print(f"   💰 총 거래량: {indicator_data.get('total_volume', 0):,.0f}")
            print(f"   📊 데이터 개수: {indicator_data.get('data_count', 0)}개")
            print(f"   📋 세션 상태: {indicator_data.get('session_status', 'N/A')}")
            
        elif indicator_name == 'atr':
            print(f"   📊 현재 ATR: {indicator_data.get('current_atr', 0):.3f}")
            print(f"   ✅ 준비 상태: {'준비됨' if indicator_data.get('is_ready', False) else '준비 안됨'}")
            print(f"   🎯 성숙 상태: {'성숙' if indicator_data.get('is_mature', False) else '미성숙'}")
            print(f"   📊 캔들 개수: {indicator_data.get('candles_count', 0)}개")
            
        elif indicator_name == 'daily_levels':
            print(f"   📅 어제 고가: ${indicator_data.get('prev_day_high', 0):.2f}")
            print(f"   📅 어제 저가: ${indicator_data.get('prev_day_low', 0):.2f}")
            print(f"   📊 로드 상태: {'로드됨' if indicator_data.get('is_loaded', False) else '로드 안됨'}")
            
        elif indicator_name == 'vwap':
            print(f"   📊 현재 VWAP: ${indicator_data.get('current_vwap', 0):.2f}")
            print(f"   📊 VWAP 표준편차: ${indicator_data.get('current_vwap_std', 0):.2f}")
            print(f"   📊 데이터 개수: {indicator_data.get('data_count', 0)}개")
            print(f"   🎯 모드: {indicator_data.get('mode', 'N/A')}")
        
        print()
    
    # 5. 개별 지표 객체에서 추가 정보 가져오기
    print("📊 5. 개별 지표 객체 상세 정보")
    
    # VWAP 추가 정보
    try:
        vwap = get_indicator('vwap')
        if vwap:
            print("🔍 VWAP 지표 추가 정보:")
            vwap_status = vwap.get_vwap_status()
            print(f"   📊 세션 이름: {vwap_status.get('session_name', 'N/A')}")
            print(f"   ⏱️  세션 진행 시간: {vwap_status.get('elapsed_minutes', 0):.1f}분")
            print(f"   📅 세션 시작: {vwap_status.get('session_start', 'N/A')}")
            print(f"   📊 마지막 업데이트: {vwap_status.get('last_update', 'N/A')}")
            
            # VWAP 결과 정보
            vwap_result = vwap.get_current_vwap()
            if vwap_result:
                print(f"   💰 총 거래량: {vwap_result.get('total_volume', 0):,.0f}")
                print(f"   📊 모드: {vwap_result.get('mode', 'N/A')}")
                if vwap_result.get('session'):
                    print(f"   📅 세션: {vwap_result.get('session', 'N/A')}")
                    print(f"   ⏱️  경과 시간: {vwap_result.get('elapsed_minutes', 0):.1f}분")
            print()
    except Exception as e:
        print(f"❌ VWAP 추가 정보 조회 실패: {e}")
    
    # VPVR 추가 정보
    try:
        vpvr = get_indicator('vpvr')
        if vpvr:
            print("🔍 VPVR 지표 추가 정보:")
            vpvr_status = vpvr.get_vpvr_status()
            print(f"   📊 활성 가격 구간: {vpvr_status.get('active_bins', 0)}개")
            print(f"   💰 총 거래량: {vpvr_status.get('total_volume', 0):,.0f}")
            print(f"   📊 처리된 캔들: {vpvr_status.get('data_count', 0)}개")
            print(f"   📋 세션 상태: {vpvr_status.get('session_status', 'N/A')}")
            print(f"   📅 세션 이름: {vpvr_status.get('session_name', 'N/A')}")
            print(f"   ⏱️  세션 진행 시간: {vpvr_status.get('elapsed_minutes', 0):.1f}분")
            print()
    except Exception as e:
        print(f"❌ VPVR 추가 정보 조회 실패: {e}")
    
    # ATR 추가 정보
    try:
        atr = get_indicator('atr')
        if atr:
            print("🔍 ATR 지표 추가 정보:")
            atr_value = atr.get_atr()
            print(f"   📊 현재 ATR: {atr_value:.3f}")
            print(f"   📊 길이: {atr.length}")
            print(f"   📊 최대 캔들: {atr.max_candles}")
            print(f"   ✅ 준비 상태: {atr.is_ready()}")
            print(f"   🎯 성숙 상태: {atr.is_mature()}")
            print()
    except Exception as e:
        print(f"❌ ATR 추가 정보 조회 실패: {e}")
    
    # Daily Levels 추가 정보
    try:
        daily_levels = get_indicator('daily_levels')
        if daily_levels:
            print("🔍 Daily Levels 지표 추가 정보:")
            print(f"   📅 어제 고가: ${daily_levels.prev_day_high:.2f}")
            print(f"   📅 어제 저가: ${daily_levels.prev_day_low:.2f}")
            print(f"   📊 로드 상태: {daily_levels.is_loaded()}")
            print()
    except Exception as e:
        print(f"❌ Daily Levels 추가 정보 조회 실패: {e}")
    
    # 6. 전체 시스템 요약
    print("📊 6. 전체 시스템 요약")
    print("=" * 60)
    
    # 현재 시간
    now = datetime.now(timezone.utc)
    print(f"🕐 현재 시간: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"🕐 한국 시간: {(now + timedelta(hours=9)).strftime('%Y-%m-%d %H:%M:%S KST')}")
    
    # 세션 정보
    if 'vwap' in indicators:
        vwap_data = indicators['vwap']
        if vwap_data.get('mode') == 'session':
            print(f"📅 현재 세션: 활성 (US/EU)")
            print(f"📊 VWAP: ${vwap_data.get('current_vwap', 0):.2f}")
        else:
            print(f"🌙 현재 세션: 비활성 (세션 외 시간)")
            print(f"📊 VWAP: ${vwap_data.get('current_vwap', 0):.2f}")
    
    # 거래량 정보
    if 'vpvr' in indicators:
        vpvr_data = indicators['vpvr']
        print(f"💰 총 거래량: {vpvr_data.get('total_volume', 0):,.0f}")
        print(f"📈 활성 가격 구간: {vpvr_data.get('active_bins', 0)}개")
    
    # ATR 정보
    if 'atr' in indicators:
        atr_data = indicators['atr']
        print(f"📊 ATR: {atr_data.get('current_atr', 0):.3f}")
        print(f"✅ ATR 준비: {'준비됨' if atr_data.get('is_ready', False) else '준비 안됨'}")
    
    # 어제 레벨 정보
    if 'daily_levels' in indicators:
        daily_data = indicators['daily_levels']
        print(f"📅 어제 고가: ${daily_data.get('prev_day_high', 0):.2f}")
        print(f"📅 어제 저가: ${daily_data.get('prev_day_low', 0):.2f}")
    
    print("=" * 60)
    print("\n🏁 글로벌 지표 전체 상태 확인 완료!")

if __name__ == "__main__":
    check_all_global_indicators()
