#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ATR 연속 롤링 테스트
- ATR이 세션과 관계없이 연속적으로 업데이트되는지 확인
- 실시간 캔들 데이터로 ATR 계산 테스트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from indicators.global_indicators import (
    initialize_global_indicators,
    get_indicator,
    update_all_indicators_with_candle
)
from datetime import datetime, timezone
import time

def test_atr_rolling():
    """ATR 연속 롤링 테스트"""
    print("🔍 ATR 연속 롤링 테스트 시작...\n")
    
    # 1. 글로벌 지표 초기화
    print("📊 1. 글로벌 지표 초기화")
    try:
        initialize_global_indicators()
        print("✅ 글로벌 지표 초기화 완료\n")
    except Exception as e:
        print(f"❌ 글로벌 지표 초기화 실패: {e}")
        return
    
    # 2. ATR 지표 가져오기
    print("📊 2. ATR 지표 상태 확인")
    try:
        atr = get_indicator('atr')
        if not atr:
            print("❌ ATR 지표를 찾을 수 없습니다.")
            return
        
        print(f"✅ ATR 지표 가져오기 성공")
        print(f"   📊 클래스: {type(atr).__name__}")
        print(f"   📊 길이: {atr.length}")
        print(f"   📊 최대 캔들: {atr.max_candles}")
        
        # 초기 ATR 상태
        atr_status = atr.get_status()
        print(f"   📊 초기 ATR: {atr_status.get('current_atr', 0):.3f}")
        print(f"   ✅ 준비 상태: {atr_status.get('is_ready', False)}")
        print(f"   🎯 성숙 상태: {atr_status.get('is_mature', False)}")
        print(f"   📊 캔들 개수: {atr_status.get('candles_count', 0)}개")
        print()
        
    except Exception as e:
        print(f"❌ ATR 지표 가져오기 실패: {e}")
        return
    
    # 3. 연속 롤링 테스트 캔들 생성 및 업데이트
    print("📊 3. ATR 연속 롤링 테스트")
    
    # 테스트 캔들 데이터 (실제 시장과 유사한 가격 변동)
    test_candles = [
        {'timestamp': datetime.now(timezone.utc), 'open': 4610.0, 'high': 4615.0, 'low': 4605.0, 'close': 4612.0, 'volume': 1000},
        {'timestamp': datetime.now(timezone.utc), 'open': 4612.0, 'high': 4620.0, 'low': 4610.0, 'close': 4618.0, 'volume': 1200},
        {'timestamp': datetime.now(timezone.utc), 'open': 4618.0, 'high': 4625.0, 'low': 4615.0, 'close': 4622.0, 'volume': 1100},
        {'timestamp': datetime.now(timezone.utc), 'open': 4622.0, 'high': 4630.0, 'low': 4620.0, 'close': 4628.0, 'volume': 1300},
        {'timestamp': datetime.now(timezone.utc), 'open': 4628.0, 'high': 4635.0, 'low': 4625.0, 'close': 4632.0, 'volume': 1400},
        {'timestamp': datetime.now(timezone.utc), 'open': 4632.0, 'high': 4640.0, 'low': 4630.0, 'close': 4638.0, 'volume': 1500},
        {'timestamp': datetime.now(timezone.utc), 'open': 4638.0, 'high': 4645.0, 'low': 4635.0, 'close': 4642.0, 'volume': 1600},
        {'timestamp': datetime.now(timezone.utc), 'open': 4642.0, 'high': 4650.0, 'low': 4640.0, 'close': 4648.0, 'volume': 1700},
        {'timestamp': datetime.now(timezone.utc), 'open': 4648.0, 'high': 4655.0, 'low': 4645.0, 'close': 4652.0, 'volume': 1800},
        {'timestamp': datetime.now(timezone.utc), 'open': 4652.0, 'high': 4660.0, 'low': 4650.0, 'close': 4658.0, 'volume': 1900},
        {'timestamp': datetime.now(timezone.utc), 'open': 4658.0, 'high': 4665.0, 'low': 4655.0, 'close': 4662.0, 'volume': 2000},
        {'timestamp': datetime.now(timezone.utc), 'open': 4662.0, 'high': 4670.0, 'low': 4660.0, 'close': 4668.0, 'volume': 2100},
        {'timestamp': datetime.now(timezone.utc), 'open': 4668.0, 'high': 4675.0, 'low': 4665.0, 'close': 4672.0, 'volume': 2200},
        {'timestamp': datetime.now(timezone.utc), 'open': 4672.0, 'high': 4680.0, 'low': 4670.0, 'close': 4678.0, 'volume': 2300},
        {'timestamp': datetime.now(timezone.utc), 'open': 4678.0, 'high': 4685.0, 'low': 4675.0, 'close': 4682.0, 'volume': 2400},
        {'timestamp': datetime.now(timezone.utc), 'open': 4682.0, 'high': 4690.0, 'low': 4680.0, 'close': 4688.0, 'volume': 2500},
    ]
    
    print(f"🔄 {len(test_candles)}개 테스트 캔들로 ATR 연속 롤링 테스트 시작...")
    
    for i, candle in enumerate(test_candles):
        try:
            # 글로벌 지표 업데이트
            update_all_indicators_with_candle(candle)
            
            # ATR 상태 확인
            atr_status = atr.get_status()
            current_atr = atr_status.get('current_atr', 0)
            is_ready = atr_status.get('is_ready', False)
            is_mature = atr_status.get('is_mature', False)
            candles_count = atr_status.get('candles_count', 0)
            
            print(f"   🔄 {i+1:2d}번째 캔들 후:")
            print(f"      📊 ATR: {current_atr:.3f}")
            print(f"      ✅ 준비: {is_ready}")
            print(f"      🎯 성숙: {is_mature}")
            print(f"      📊 캔들: {candles_count}개")
            print(f"      💰 가격: ${candle['close']:.2f}")
            print()
            
            # 잠시 대기 (실시간 느낌)
            time.sleep(0.1)
            
        except Exception as e:
            print(f"❌ {i+1}번째 캔들 업데이트 실패: {e}")
    
    # 4. 최종 ATR 상태 분석
    print("📊 4. 최종 ATR 상태 분석")
    try:
        final_atr_status = atr.get_status()
        
        print(f"🎯 최종 ATR 분석 결과:")
        print(f"   📊 최종 ATR: {final_atr_status.get('current_atr', 0):.3f}")
        print(f"   ✅ 준비 상태: {final_atr_status.get('is_ready', False)}")
        print(f"   🎯 성숙 상태: {final_atr_status.get('is_mature', False)}")
        print(f"   📊 총 캔들: {final_atr_status.get('candles_count', 0)}개")
        print(f"   📊 True Ranges: {final_atr_status.get('true_ranges_count', 0)}개")
        print(f"   📊 길이: {final_atr_status.get('length', 0)}")
        print(f"   📊 최대 캔들: {final_atr_status.get('max_candles', 0)}개")
        
        if final_atr_status.get('last_update'):
            print(f"   🕐 마지막 업데이트: {final_atr_status.get('last_update')}")
        
        print()
        
        # ATR 계산 성공 여부
        if final_atr_status.get('is_ready', False):
            print("✅ ATR 연속 롤링 테스트 성공!")
            print("   📊 ATR이 세션과 관계없이 연속적으로 계산되고 있습니다.")
        else:
            print("❌ ATR 연속 롤링 테스트 실패!")
            print("   📊 ATR이 아직 준비되지 않았습니다.")
        
    except Exception as e:
        print(f"❌ 최종 ATR 상태 분석 실패: {e}")
    
    print("\n🏁 ATR 연속 롤링 테스트 완료!")

if __name__ == "__main__":
    test_atr_rolling()
