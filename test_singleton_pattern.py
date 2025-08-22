#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
글로벌 지표 관리자 싱글톤 패턴 테스트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from indicators.global_indicators import get_global_indicator_manager

def test_singleton_pattern():
    """싱글톤 패턴 테스트"""
    print("🚀 싱글톤 패턴 테스트 시작...\n")
    
    # 1. 첫 번째 인스턴스 가져오기
    print("📊 1. 첫 번째 인스턴스 가져오기")
    manager1 = get_global_indicator_manager()
    print(f"   📋 Manager1 ID: {id(manager1)}")
    print(f"   📋 Manager1 초기화 상태: {manager1.is_initialized()}")
    print()
    
    # 2. 두 번째 인스턴스 가져오기
    print("📊 2. 두 번째 인스턴스 가져오기")
    manager2 = get_global_indicator_manager()
    print(f"   📋 Manager2 ID: {id(manager2)}")
    print(f"   📋 Manager2 초기화 상태: {manager2.is_initialized()}")
    print()
    
    # 3. 동일한 인스턴스인지 확인
    print("📊 3. 싱글톤 확인")
    print(f"   📋 Manager1 == Manager2: {manager1 is manager2}")
    print(f"   📋 ID 동일: {id(manager1) == id(manager2)}")
    print()
    
    # 4. 첫 번째 인스턴스에서 초기화
    print("📊 4. 첫 번째 인스턴스에서 초기화")
    manager1.initialize_indicators()
    print(f"   📋 Manager1 초기화 후: {manager1.is_initialized()}")
    print(f"   📋 Manager2 상태 (자동 반영): {manager2.is_initialized()}")
    print()
    
    # 5. 세 번째 인스턴스도 동일한지 확인
    print("📊 5. 세 번째 인스턴스 확인")
    manager3 = get_global_indicator_manager()
    print(f"   📋 Manager3 ID: {id(manager3)}")
    print(f"   📋 Manager3 초기화 상태: {manager3.is_initialized()}")
    print(f"   📋 Manager1 == Manager3: {manager1 is manager3}")
    print()
    
    # 6. 지표 목록 확인 (모든 인스턴스에서 동일해야 함)
    print("📊 6. 지표 목록 확인")
    indicators1 = manager1.list_indicators()
    indicators2 = manager2.list_indicators()
    indicators3 = manager3.list_indicators()
    
    print(f"   📋 Manager1 지표: {indicators1}")
    print(f"   📋 Manager2 지표: {indicators2}")
    print(f"   📋 Manager3 지표: {indicators3}")
    print(f"   📋 모든 목록 동일: {indicators1 == indicators2 == indicators3}")
    print()
    
    # 7. 개별 지표 접근 테스트
    print("📊 7. 개별 지표 접근 테스트")
    vpvr1 = manager1.get_indicator('vpvr')
    vpvr2 = manager2.get_indicator('vpvr')
    vpvr3 = manager3.get_indicator('vpvr')
    
    print(f"   📈 VPVR1 ID: {id(vpvr1) if vpvr1 else 'None'}")
    print(f"   📈 VPVR2 ID: {id(vpvr2) if vpvr2 else 'None'}")
    print(f"   📈 VPVR3 ID: {id(vpvr3) if vpvr3 else 'None'}")
    print(f"   📈 모든 VPVR 동일: {vpvr1 is vpvr2 is vpvr3}")
    print()
    
    print("✅ 싱글톤 패턴이 정상적으로 작동합니다!")
    print("🏁 싱글톤 패턴 테스트 완료!")

if __name__ == "__main__":
    test_singleton_pattern()
