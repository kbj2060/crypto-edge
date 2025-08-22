#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

# 현재 디렉토리와 프로젝트 루트를 패스에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append('.')

from indicators.vpvr import SessionVPVR
from indicators.daily_levels import DailyLevels

def test_vpvr():
    print('🚀 VPVR 클래스 테스트 시작...')
    
    try:
        # VPVR 인스턴스 생성
        vpvr = SessionVPVR()
        print('✅ VPVR 인스턴스 생성 성공')
        
        # 세션 상태 확인
        print('\n📊 세션 상태 확인...')
        try:
            session_status = vpvr.get_vpvr_status()
            print(f'✅ 세션 상태: {session_status}')
        except Exception as e:
            print(f'❌ 세션 상태 확인 오류: {e}')
        
        # 세션 히스토리 확인
        print('\n📊 세션 히스토리 확인...')
        try:
            session_history = vpvr.get_session_history()
            print(f'✅ 세션 히스토리: {session_history}')
        except Exception as e:
            print(f'❌ 세션 히스토리 확인 오류: {e}')
        
        # DailyLevels를 통한 어제 데이터 테스트
        print('\n📊 DailyLevels를 통한 어제 데이터 테스트...')
        try:
            daily_levels = DailyLevels()
            print('✅ DailyLevels 인스턴스 생성 성공')
            
            # 어제 high, low 가져오기
            levels = daily_levels.get_prev_day_high_low()
            print(f'✅ 어제 레벨: {levels}')
            
            # 개별 값 접근
            high = levels['high']
            low = levels['low']
            print(f'   📈 어제 고가: ${high:.2f}')
            print(f'   📉 어제 저가: ${low:.2f}')
            
        except Exception as e:
            print(f'❌ DailyLevels 테스트 오류: {e}')
            import traceback
            traceback.print_exc()
        
        # VPVR 모드 테스트
        print('\n📊 VPVR 모드 테스트...')
        try:
            # 현재 시간으로 세션 상태 업데이트
            from datetime import datetime, timezone
            current_time = datetime.now(timezone.utc)
            
            # VPVR 상태 확인
            vpvr_status = vpvr.get_vpvr_status()
            print(f'✅ VPVR 상태: {vpvr_status}')
            
        except Exception as e:
            print(f'❌ VPVR 모드 테스트 오류: {e}')
            
    except Exception as e:
        print(f'❌ 전체 테스트 실패: {e}')
        import traceback
        traceback.print_exc()

    print('\n🏁 테스트 완료!')

if __name__ == "__main__":
    test_vpvr()
