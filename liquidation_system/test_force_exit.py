#!/usr/bin/env python3
"""
강제종료 테스트 스크립트
Ctrl+C나 강제종료 신호가 제대로 처리되는지 테스트합니다.
"""

import asyncio
import signal
import sys
import time

async def test_force_exit():
    """강제종료 테스트"""
    print("🧪 강제종료 테스트 시작")
    print("💡 Ctrl+C를 눌러서 강제종료를 테스트하세요")
    print("⏰ 10초 후 자동 종료됩니다.")
    
    try:
        # 10초 대기 (중간에 Ctrl+C로 중단 가능)
        for i in range(10, 0, -1):
            print(f"⏳ {i}초 남음...")
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Ctrl+C 감지! 강제종료 테스트 성공!")
        return True
    
    print("✅ 자동 종료 완료")
    return False

def main():
    """메인 함수"""
    print("🚀 강제종료 테스트 프로그램")
    print("=" * 40)
    
    try:
        result = asyncio.run(test_force_exit())
        if result:
            print("🎉 강제종료가 정상적으로 처리되었습니다!")
        else:
            print("📝 자동 종료가 정상적으로 처리되었습니다.")
            
    except KeyboardInterrupt:
        print("\n⏹️  메인에서 Ctrl+C 감지!")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
    finally:
        print("👋 테스트 완료")

if __name__ == "__main__":
    main()
