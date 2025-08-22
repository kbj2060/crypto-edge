#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from indicators.opening_range import get_session_manager
from datetime import datetime, timezone, timedelta

def test_session_manager():
    """SessionManager 기능 테스트"""
    print("🚀 SessionManager 테스트 시작...\n")
    
    # SessionManager 인스턴스 가져오기
    session_manager = get_session_manager()
    print("✅ SessionManager 인스턴스 생성 완료\n")
    
    # 1. 현재 세션 상태 확인
    print("📊 1. 현재 세션 상태 확인")
    current_time = datetime.now(timezone.utc)
    session_status = session_manager.update_session_status(current_time)
    
    print(f"   현재 시간: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"   세션 활성: {session_status.get('is_active', False)}")
    print(f"   현재 세션: {session_status.get('current_session', 'None')}")
    print(f"   세션 상태: {session_status.get('status', 'UNKNOWN')}")
    print()
    
    # 2. indicator 모드 설정 정보 확인
    print("📊 2. Indicator 모드 설정 정보")
    mode_config = session_manager.get_indicator_mode_config()
    
    print(f"   세션 모드 사용: {mode_config.get('use_session_mode', False)}")
    print(f"   모드: {mode_config.get('mode', 'unknown')}")
    print(f"   세션 이름: {mode_config.get('session_name', 'None')}")
    print(f"   경과 시간: {mode_config.get('elapsed_minutes', 0):.1f}분")
    print()
    
    # 3. 다양한 시간대 테스트
    print("📊 3. 다양한 시간대 테스트")
    
    test_times = [
        # 유럽 세션 시간 (07:00 UTC)
        datetime.now(timezone.utc).replace(hour=8, minute=30, second=0, microsecond=0),
        # 미국 세션 시간 (13:30 UTC)  
        datetime.now(timezone.utc).replace(hour=15, minute=30, second=0, microsecond=0),
        # 세션 외 시간 (05:00 UTC)
        datetime.now(timezone.utc).replace(hour=5, minute=0, second=0, microsecond=0)
    ]
    
    for i, test_time in enumerate(test_times, 1):
        print(f"   테스트 시간 {i}: {test_time.strftime('%H:%M')} UTC")
        
        test_status = session_manager.update_session_status(test_time)
        test_config = session_manager.get_indicator_mode_config()
        
        print(f"      세션 활성: {test_status.get('is_active', False)}")
        print(f"      현재 세션: {test_status.get('current_session', 'None')}")
        print(f"      모드: {test_config.get('mode', 'unknown')}")
        print(f"      상태: {test_status.get('status', 'UNKNOWN')}")
        print()
    
    # 4. SessionManager 메서드 테스트
    print("📊 4. SessionManager 메서드 테스트")
    
    print(f"   is_session_active(): {session_manager.is_session_active()}")
    print(f"   get_current_session_name(): {session_manager.get_current_session_name()}")
    print(f"   should_use_session_mode(): {session_manager.should_use_session_mode()}")
    print(f"   get_session_elapsed_minutes(): {session_manager.get_session_elapsed_minutes():.1f}분")
    print()
    
    # 5. 세션 히스토리 확인
    print("📊 5. 세션 히스토리 확인")
    session_history = session_manager.get_session_history()
    
    if session_history:
        print(f"   세션 히스토리: {len(session_history)}개")
        for session_id, session_data in session_history.items():
            print(f"      {session_id}: {session_data.get('session_name', 'Unknown')}")
    else:
        print("   세션 히스토리가 없습니다")
    print()
    
    print("🏁 SessionManager 테스트 완료!")

if __name__ == "__main__":
    test_session_manager()
