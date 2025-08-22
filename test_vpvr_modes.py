#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from indicators.vpvr import SessionVPVR
from indicators.opening_range import get_session_manager
from datetime import datetime, timezone, timedelta

def test_vpvr_modes():
    """VPVR 세션 모드와 lookback 모드 테스트"""
    print("🚀 VPVR 모드 테스트 시작...\n")
    
    # VPVR 객체 생성 (자동 데이터 로드 비활성화)
    vpvr = SessionVPVR(auto_load=False)
    print("✅ VPVR 인스턴스 생성 완료\n")
    
    # 1. 초기 상태 확인 (lookback 모드)
    print("📊 1. 초기 상태 확인 (Lookback 모드)")
    status = vpvr.get_vpvr_status()
    print(f"   모드: {status['mode']}")
    print(f"   세션 활성: {status['is_session_active']}")
    print(f"   데이터 개수: {status['data_count']}")
    print()
    
    # 2. Lookback 모드에서 캔들 데이터 추가
    print("📊 2. Lookback 모드에서 캔들 데이터 추가")
    test_candles = [
        {'timestamp': datetime.now(timezone.utc) - timedelta(minutes=15), 'open': 4200.0, 'high': 4210.0, 'low': 4195.0, 'close': 4205.0, 'volume': 100.0},
        {'timestamp': datetime.now(timezone.utc) - timedelta(minutes=12), 'open': 4205.0, 'high': 4215.0, 'low': 4200.0, 'close': 4210.0, 'volume': 150.0},
        {'timestamp': datetime.now(timezone.utc) - timedelta(minutes=9), 'open': 4210.0, 'high': 4220.0, 'low': 4205.0, 'close': 4215.0, 'volume': 200.0},
        {'timestamp': datetime.now(timezone.utc) - timedelta(minutes=6), 'open': 4215.0, 'high': 4225.0, 'low': 4210.0, 'close': 4220.0, 'volume': 180.0},
        {'timestamp': datetime.now(timezone.utc) - timedelta(minutes=3), 'open': 4220.0, 'high': 4230.0, 'low': 4215.0, 'close': 4225.0, 'volume': 120.0},
    ]
    
    for i, candle in enumerate(test_candles):
        print(f"   📊 캔들 {i+1} 추가: ${candle['close']:.2f}, 거래량: {candle['volume']:.2f}")
        vpvr.update_with_candle(candle)
        print()
    
    # 3. Lookback VPVR 결과 확인
    print("📊 3. Lookback VPVR 결과 확인")
    current_vpvr = vpvr.get_current_vpvr()
    if current_vpvr:
        print(f"   📍 POC: ${current_vpvr.get('poc', 0):.2f}")
        print(f"   🔥 HVN: ${current_vpvr.get('hvn', 0):.2f}")
        print(f"   ❄️  LVN: ${current_vpvr.get('lvn', 0):.2f}")
        print(f"   📊 모드: {current_vpvr.get('mode', 'unknown')}")
        print(f"   📈 총 거래량: {current_vpvr.get('total_volume', 0):.2f}")
    else:
        print("   ❌ VPVR 결과가 없습니다")
    print()
    
    # 4. 세션 시작 (세션 모드로 전환)
    print("📊 4. 세션 시작 (세션 모드로 전환)")
    
    # SessionManager 테스트를 위해 강제로 세션을 활성화
    # 실제 환경에서는 현재 시간에 따라 자동으로 결정됨
    session_manager = get_session_manager()
    
    # 세션 매니저의 세션 상태를 시뮬레이션 (테스트용)
    # 실제로는 현재 시간에 따라 자동으로 결정
    current_time = datetime.now(timezone.utc)
    session_info = session_manager.update_session_status(current_time)
    print(f"   현재 세션 상태: {session_info.get('status', 'UNKNOWN')}")
    
    # VPVR 세션 리셋 (SessionManager 기반)
    vpvr.reset_session()
    print()
    
    # 5. 세션 모드에서 캔들 데이터 추가
    print("📊 5. 세션 모드에서 캔들 데이터 추가")
    session_candles = [
        {'timestamp': datetime.now(timezone.utc) - timedelta(minutes=2), 'open': 4225.0, 'high': 4235.0, 'low': 4220.0, 'close': 4230.0, 'volume': 250.0},
        {'timestamp': datetime.now(timezone.utc) - timedelta(minutes=1), 'open': 4230.0, 'high': 4240.0, 'low': 4225.0, 'close': 4235.0, 'volume': 300.0},
        {'timestamp': datetime.now(timezone.utc), 'open': 4235.0, 'high': 4245.0, 'low': 4230.0, 'close': 4240.0, 'volume': 280.0},
    ]
    
    for i, candle in enumerate(session_candles):
        print(f"   📊 세션 캔들 {i+1} 추가: ${candle['close']:.2f}, 거래량: {candle['volume']:.2f}")
        vpvr.update_with_candle(candle)
        print()
    
    # 6. 세션 VPVR 결과 확인
    print("📊 6. 세션 VPVR 결과 확인")
    current_vpvr = vpvr.get_current_vpvr()
    if current_vpvr:
        print(f"   📍 POC: ${current_vpvr.get('poc', 0):.2f}")
        print(f"   🔥 HVN: ${current_vpvr.get('hvn', 0):.2f}")
        print(f"   ❄️  LVN: ${current_vpvr.get('lvn', 0):.2f}")
        print(f"   📊 모드: {current_vpvr.get('mode', 'unknown')}")
        print(f"   📈 총 거래량: {current_vpvr.get('total_volume', 0):.2f}")
        print(f"   🏷️  세션: {current_vpvr.get('session', 'unknown')}")
    else:
        print("   ❌ VPVR 결과가 없습니다")
    print()
    
    # 7. 세션 종료 (lookback 모드로 전환)
    print("📊 7. 세션 종료 (lookback 모드로 전환)")
    print("   세션은 SessionManager에서 자동으로 관리됩니다")
    print("   (실제 환경에서는 시간에 따라 자동 전환)")
    
    # 8. 최종 상태 확인
    print("📊 8. 최종 상태 확인")
    status = vpvr.get_vpvr_status()
    print(f"   모드: {status['mode']}")
    print(f"   세션 활성: {status['is_session_active']}")
    print(f"   데이터 개수: {status['data_count']}")
    print(f"   세션 히스토리: {len(vpvr.get_session_history())}개")
    print()
    
    print("🏁 VPVR 모드 테스트 완료!")

if __name__ == "__main__":
    test_vpvr_modes()
