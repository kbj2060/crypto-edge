#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
세션 외 시간대 VWAP 테스트
- 세션 외 시간대 중간에 프로그램 시작 시나리오
- 이전 세션 종료 시점부터 현재까지 데이터 로딩 테스트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from indicators.vwap import SessionVWAP
from indicators.opening_range import get_session_manager
from datetime import datetime, timezone, timedelta
import time

def simulate_outside_session_time():
    """세션 외 시간대 시뮬레이션"""
    print("🌙 세션 외 시간대 시뮬레이션...")
    
    # 현재 시간이 17:37 UTC (US 세션 중)
    # 세션 외 시간대는 22:00-08:00 UTC
    # 이전 세션 종료 시점은 15:00 UTC (EU 세션 종료)
    
    current_time = datetime.now(timezone.utc)
    print(f"   🕐 현재 시간: {current_time.strftime('%H:%M UTC')}")
    
    # EU 세션 종료 시점 (15:00 UTC)
    eu_session_end = current_time.replace(hour=15, minute=0, second=0, microsecond=0)
    if eu_session_end > current_time:
        eu_session_end = eu_session_end - timedelta(days=1)
    
    print(f"   🌍 EU 세션 종료: {eu_session_end.strftime('%H:%M UTC')}")
    
    # 세션 외 시간대 시작 (22:00 UTC)
    outside_session_start = current_time.replace(hour=22, minute=0, second=0, microsecond=0)
    if outside_session_start > current_time:
        outside_session_start = outside_session_start - timedelta(days=1)
    
    print(f"   🌙 세션 외 시간 시작: {outside_session_start.strftime('%H:%M UTC')}")
    
    # 이전 세션 종료부터 현재까지의 시간
    time_since_session_end = current_time - eu_session_end
    hours_since_end = time_since_session_end.total_seconds() / 3600
    
    print(f"   ⏱️  EU 세션 종료 후 경과: {hours_since_end:.1f}시간")
    
    return eu_session_end, outside_session_start, hours_since_end

def test_vwap_outside_session_simulation():
    """세션 외 시간대 VWAP 시뮬레이션 테스트"""
    print("🌙 세션 외 시간대 VWAP 시뮬레이션 테스트 시작...\n")
    
    # 1. 현재 시간 및 세션 상태 확인
    print("📊 1. 현재 시간 및 세션 상태 확인")
    now = datetime.now(timezone.utc)
    print(f"   🕐 현재 시간: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"   🕐 현재 시간 (한국): {(now + timedelta(hours=9)).strftime('%Y-%m-%d %H:%M:%S KST')}")
    
    # 2. 세션 외 시간대 시뮬레이션
    print("\n📊 2. 세션 외 시간대 시뮬레이션")
    eu_session_end, outside_session_start, hours_since_end = simulate_outside_session_time()
    
    print(f"   📊 이전 세션 종료 시점: {eu_session_end.strftime('%H:%M UTC')}")
    print(f"   📊 세션 외 시간 시작: {outside_session_start.strftime('%H:%M UTC')}")
    print(f"   📊 세션 종료 후 경과: {hours_since_end:.1f}시간")
    
    # 3. 세션 매니저 상태 확인
    print("\n📊 3. 세션 매니저 상태 확인")
    session_manager = get_session_manager()
    session_manager.update_session_status()
    session_config = session_manager.get_indicator_mode_config()
    
    print(f"   📊 현재 세션: {session_config.get('session_name', 'UNKNOWN')}")
    print(f"   📊 세션 상태: {session_config.get('session_status', 'UNKNOWN')}")
    print(f"   📊 세션 모드: {'활성' if session_config['use_session_mode'] else '비활성'}")
    
    # 4. 세션 외 시간대 가정 및 테스트
    print("\n📊 4. 세션 외 시간대 가정 및 테스트")
    print("   🌙 현재 US 세션이지만 세션 외 시간대로 가정하여 테스트")
    print("   📊 EU 세션 종료(15:00 UTC)부터 현재까지 데이터 로딩 테스트")
    
    # 5. VWAP 인스턴스 생성 (세션 외 시간 가정)
    print("\n📊 5. 세션 외 시간대 VWAP 인스턴스 생성")
    try:
        vwap = SessionVWAP(symbol="ETHUSDT", auto_load=True)
        print("   ✅ VWAP 인스턴스 생성 완료")
        
        # 6. 초기 VWAP 상태 확인
        print("\n📊 6. 초기 VWAP 상태 확인")
        vwap_status = vwap.get_vwap_status()
        print(f"   📊 현재 VWAP: ${vwap_status.get('current_vwap', 0):.2f}")
        print(f"   📊 VWAP 표준편차: ${vwap_status.get('current_vwap_std', 0):.2f}")
        print(f"   📊 데이터 개수: {vwap_status.get('data_count', 0)}개")
        print(f"   📋 세션 상태: {vwap_status.get('session_status', 'UNKNOWN')}")
        print(f"   🎯 모드: {vwap_status.get('mode', 'unknown')}")
        
        if vwap_status.get('mode') == 'outside_session':
            print(f"   📊 세션 외 시간 VWAP 계산 중")
        else:
            print(f"   📅 세션: {vwap_status.get('session_name', 'UNKNOWN')}")
            print(f"   ⏱️  세션 진행 시간: {vwap_status.get('elapsed_minutes', 0):.1f}분")
        
        # 7. 세션 외 시간대 데이터 로딩 테스트
        print("\n📊 7. 세션 외 시간대 데이터 로딩 테스트")
        print("   📊 이전 세션 종료 시점부터 현재까지 데이터 확인")
        
        # VWAP의 내부 데이터 확인
        if hasattr(vwap, 'session_data') and vwap.session_data:
            print(f"   📊 로드된 데이터: {len(vwap.session_data)}개 캔들")
            
            # 첫 번째와 마지막 캔들 시간 확인
            first_candle = vwap.session_data[0]
            last_candle = vwap.session_data[-1]
            
            if 'timestamp' in first_candle:
                print(f"   📊 첫 번째 캔들: {first_candle['timestamp']}")
                print(f"   📊 마지막 캔들: {last_candle['timestamp']}")
            
            # 가격 범위 확인
            prices = [candle.get('close', 0) for candle in vwap.session_data]
            if prices:
                print(f"   📊 가격 범위: ${min(prices):.2f} ~ ${max(prices):.2f}")
                print(f"   📊 평균 가격: ${sum(prices)/len(prices):.2f}")
        
        # 8. 테스트 캔들로 업데이트
        print("\n📊 8. 테스트 캔들로 업데이트")
        test_candles = [
            {
                'timestamp': datetime.now(timezone.utc),
                'open': 4600.00,
                'high': 4615.50,
                'low': 4595.20,
                'close': 4610.30,
                'volume': 2000.0
            },
            {
                'timestamp': datetime.now(timezone.utc),
                'open': 4610.30,
                'high': 4625.80,
                'low': 4605.10,
                'close': 4620.50,
                'volume': 3000.0
            }
        ]
        
        for i, candle in enumerate(test_candles):
            print(f"   🔄 {i+1}. 캔들 업데이트: ${candle['close']:.2f}, 거래량: {candle['volume']:.0f}")
            vwap.update_with_candle(candle)
            
            # 업데이트 후 VWAP 확인
            vwap_status = vwap.get_vwap_status()
            print(f"      📊 VWAP: ${vwap_status.get('current_vwap', 0):.2f}")
            print(f"      📊 VWAP 표준편차: ${vwap_status.get('current_vwap_std', 0):.2f}")
            print(f"      📊 데이터 개수: {vwap_status.get('data_count', 0)}개")
            print()
        
        # 9. 최종 VWAP 분석
        print("📊 9. 최종 VWAP 분석")
        final_result = vwap.get_current_vwap()
        final_status = vwap.get_vwap_status()
        
        if final_result:
            vwap_value = final_result.get('vwap', 0)
            vwap_std = final_result.get('vwap_std', 0)
            total_volume = final_result.get('total_volume', 0)
            
            print(f"   📊 최종 VWAP: ${vwap_value:.2f}")
            print(f"   📊 최종 VWAP 표준편차: ${vwap_std:.2f}")
            print(f"   📊 총 거래량: {total_volume:,.2f}")
            
            # VWAP 분석
            if vwap_std > 0:
                vwap_plus_1sigma = vwap_value + vwap_std
                vwap_minus_1sigma = vwap_value - vwap_std
                vwap_plus_2sigma = vwap_value + (2 * vwap_std)
                vwap_minus_2sigma = vwap_value - (2 * vwap_std)
                
                print(f"\n📊 VWAP 분석:")
                print(f"   💡 VWAP ${vwap_value:.2f}는 거래량 가중 평균 가격입니다")
                print(f"   💡 VWAP 표준편차 ${vwap_std:.2f}는 가격 변동성을 나타냅니다")
                print(f"   📈 VWAP +1σ: ${vwap_plus_1sigma:.2f}")
                print(f"   📉 VWAP -1σ: ${vwap_minus_1sigma:.2f}")
                print(f"   📈 VWAP +2σ: ${vwap_plus_2sigma:.2f}")
                print(f"   📉 VWAP -2σ: ${vwap_minus_2sigma:.2f}")
            
            print(f"\n   📊 최종 데이터 개수: {final_status.get('data_count', 0)}개")
            print(f"   📊 최종 모드: {final_status.get('mode', 'unknown')}")
        
        print("\n🏁 세션 외 시간대 VWAP 시뮬레이션 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 테스트 오류: {e}")
        import traceback
        traceback.print_exc()

def test_vwap_outside_session():
    """세션 외 시간대 VWAP 테스트"""
    print("🌙 세션 외 시간대 VWAP 테스트 시작...\n")
    
    # 1. 현재 시간 및 세션 상태 확인
    print("📊 1. 현재 시간 및 세션 상태 확인")
    now = datetime.now(timezone.utc)
    print(f"   🕐 현재 시간: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"   🕐 현재 시간 (한국): {(now + timedelta(hours=9)).strftime('%Y-%m-%d %H:%M:%S KST')}")
    
    # 2. 세션 외 시간대 시뮬레이션
    print("\n📊 2. 세션 외 시간대 시뮬레이션")
    eu_session_end, outside_session_start, hours_since_end = simulate_outside_session_time()
    
    print(f"   📊 이전 세션 종료 시점: {eu_session_end.strftime('%H:%M UTC')}")
    print(f"   📊 세션 외 시간 시작: {outside_session_start.strftime('%H:%M UTC')}")
    print(f"   📊 세션 종료 후 경과: {hours_since_end:.1f}시간")
    
    # 3. 세션 매니저 상태 확인
    print("\n📊 3. 세션 매니저 상태 확인")
    session_manager = get_session_manager()
    session_manager.update_session_status()
    session_config = session_manager.get_indicator_mode_config()
    
    print(f"   📊 현재 세션: {session_config.get('session_name', 'UNKNOWN')}")
    print(f"   📊 세션 상태: {session_config.get('session_status', 'UNKNOWN')}")
    print(f"   📊 세션 모드: {'활성' if session_config['use_session_mode'] else '비활성'}")
    
    # 4. 세션 외 시간대 가정 및 테스트
    print("\n📊 4. 세션 외 시간대 가정 및 테스트")
    print("   🌙 현재 US 세션이지만 세션 외 시간대로 가정하여 테스트")
    print("   📊 EU 세션 종료(15:00 UTC)부터 현재까지 데이터 로딩 테스트")
    
    # 5. VWAP 인스턴스 생성 (세션 외 시간 가정)
    print("\n📊 5. 세션 외 시간대 VWAP 인스턴스 생성")
    try:
        vwap = SessionVWAP(symbol="ETHUSDT", auto_load=True)
        print("   ✅ VWAP 인스턴스 생성 완료")
        
        # 6. 초기 VWAP 상태 확인
        print("\n📊 6. 초기 VWAP 상태 확인")
        vwap_status = vwap.get_vwap_status()
        print(f"   📊 현재 VWAP: ${vwap_status.get('current_vwap', 0):.2f}")
        print(f"   📊 VWAP 표준편차: ${vwap_status.get('current_vwap_std', 0):.2f}")
        print(f"   📊 데이터 개수: {vwap_status.get('data_count', 0)}개")
        print(f"   📋 세션 상태: {vwap_status.get('session_status', 'UNKNOWN')}")
        print(f"   🎯 모드: {vwap_status.get('mode', 'unknown')}")
        
        if vwap_status.get('mode') == 'outside_session':
            print(f"   📊 세션 외 시간 VWAP 계산 중")
        else:
            print(f"   📅 세션: {vwap_status.get('session_name', 'UNKNOWN')}")
            print(f"   ⏱️  세션 진행 시간: {vwap_status.get('elapsed_minutes', 0):.1f}분")
        
        # 7. 세션 외 시간대 데이터 로딩 테스트
        print("\n📊 7. 세션 외 시간대 데이터 로딩 테스트")
        print("   📊 이전 세션 종료 시점부터 현재까지 데이터 확인")
        
        # VWAP의 내부 데이터 확인
        if hasattr(vwap, 'session_data') and vwap.session_data:
            print(f"   📊 로드된 데이터: {len(vwap.session_data)}개 캔들")
            
            # 첫 번째와 마지막 캔들 시간 확인
            first_candle = vwap.session_data[0]
            last_candle = vwap.session_data[-1]
            
            if 'timestamp' in first_candle:
                print(f"   📊 첫 번째 캔들: {first_candle['timestamp']}")
                print(f"   📊 마지막 캔들: {last_candle['timestamp']}")
            
            # 가격 범위 확인
            prices = [candle.get('close', 0) for candle in vwap.session_data]
            if prices:
                print(f"   📊 가격 범위: ${min(prices):.2f} ~ ${max(prices):.2f}")
                print(f"   📊 평균 가격: ${sum(prices)/len(prices):.2f}")
        
        # 8. 테스트 캔들로 업데이트
        print("\n📊 8. 테스트 캔들로 업데이트")
        test_candles = [
            {
                'timestamp': datetime.now(timezone.utc),
                'open': 4600.00,
                'high': 4615.50,
                'low': 4595.20,
                'close': 4610.30,
                'volume': 2000.0
            },
            {
                'timestamp': datetime.now(timezone.utc),
                'open': 4610.30,
                'high': 4625.80,
                'low': 4605.10,
                'close': 4620.50,
                'volume': 3000.0
            }
        ]
        
        for i, candle in enumerate(test_candles):
            print(f"   🔄 {i+1}. 캔들 업데이트: ${candle['close']:.2f}, 거래량: {candle['volume']:.0f}")
            vwap.update_with_candle(candle)
            
            # 업데이트 후 VWAP 확인
            vwap_status = vwap.get_vwap_status()
            print(f"      📊 VWAP: ${vwap_status.get('current_vwap', 0):.2f}")
            print(f"      📊 VWAP 표준편차: ${vwap_status.get('current_vwap_std', 0):.2f}")
            print(f"      📊 데이터 개수: {vwap_status.get('data_count', 0)}개")
            print()
        
        # 9. 최종 VWAP 분석
        print("📊 9. 최종 VWAP 분석")
        final_result = vwap.get_current_vwap()
        final_status = vwap.get_vwap_status()
        
        if final_result:
            vwap_value = final_result.get('vwap', 0)
            vwap_std = final_result.get('vwap_std', 0)
            total_volume = final_result.get('total_volume', 0)
            
            print(f"   📊 최종 VWAP: ${vwap_value:.2f}")
            print(f"   📊 최종 VWAP 표준편차: ${vwap_std:.2f}")
            print(f"   📊 총 거래량: {total_volume:,.2f}")
            
            # VWAP 분석
            if vwap_std > 0:
                vwap_plus_1sigma = vwap_value + vwap_std
                vwap_minus_1sigma = vwap_value - vwap_std
                vwap_plus_2sigma = vwap_value + (2 * vwap_std)
                vwap_minus_2sigma = vwap_value - (2 * vwap_std)
                
                print(f"\n📊 VWAP 분석:")
                print(f"   💡 VWAP ${vwap_value:.2f}는 거래량 가중 평균 가격입니다")
                print(f"   💡 VWAP 표준편차 ${vwap_std:.2f}는 가격 변동성을 나타냅니다")
                print(f"   📈 VWAP +1σ: ${vwap_plus_1sigma:.2f}")
                print(f"   📉 VWAP -1σ: ${vwap_minus_1sigma:.2f}")
                print(f"   📈 VWAP +2σ: ${vwap_plus_2sigma:.2f}")
                print(f"   📉 VWAP -2σ: ${vwap_minus_2sigma:.2f}")
            
            print(f"\n   📊 최종 데이터 개수: {final_status.get('data_count', 0)}개")
            print(f"   📊 최종 모드: {final_status.get('mode', 'unknown')}")
        
        print("\n🏁 세션 외 시간대 VWAP 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 테스트 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 시뮬레이션 테스트 실행
    test_vwap_outside_session_simulation()
    
    print("\n" + "="*80 + "\n")
    
    # 실제 테스트 실행
    test_vwap_outside_session()
