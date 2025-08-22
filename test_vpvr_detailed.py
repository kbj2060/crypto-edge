#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VPVR 상세 분석 테스트 - Volume Histogram 확인
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

def test_vpvr_detailed():
    """VPVR 상세 분석"""
    print("🚀 VPVR 상세 분석 테스트 시작...\n")
    
    # 1. 글로벌 지표 초기화
    print("📊 1. 글로벌 지표 초기화")
    initialize_global_indicators()
    vpvr = get_indicator('vpvr')
    print("✅ 초기화 완료\n")
    
    # 2. 초기 Volume Histogram 상위 10개 확인
    print("📊 2. 초기 Volume Histogram 상위 10개 확인")
    if hasattr(vpvr, 'volume_histogram') and vpvr.volume_histogram:
        # 거래량 기준으로 정렬
        sorted_volumes = sorted(vpvr.volume_histogram.items(), key=lambda x: x[1], reverse=True)
        
        print("   📈 상위 10개 거래량 구간:")
        for i, (bin_key, volume) in enumerate(sorted_volumes[:10]):
            if bin_key in vpvr.price_bins:
                price = vpvr.price_bins[bin_key]
                print(f"      {i+1:2d}. ${price:8.2f}: {volume:12,.2f} 거래량")
        
        print(f"\n   📊 총 구간 수: {len(vpvr.volume_histogram)}개")
        print(f"   📊 총 거래량: {sum(vpvr.volume_histogram.values()):,.2f}")
    else:
        print("   ⚠️ Volume Histogram이 없습니다")
    print()
    
    # 3. POC 계산 과정 확인
    print("📊 3. POC 계산 과정 확인")
    vpvr_result = vpvr.get_current_vpvr()
    if vpvr_result and hasattr(vpvr, 'volume_histogram'):
        poc_price = vpvr_result.get('poc', 0)
        
        # POC 가격 주변의 거래량 확인
        print(f"   🎯 현재 POC: ${poc_price:.2f}")
        
        # POC 주변 ±$10 범위의 거래량 확인
        poc_range_volumes = []
        for bin_key, volume in vpvr.volume_histogram.items():
            if bin_key in vpvr.price_bins:
                price = vpvr.price_bins[bin_key]
                if abs(price - poc_price) <= 10:  # ±$10 범위
                    poc_range_volumes.append((price, volume))
        
        # 가격 순으로 정렬
        poc_range_volumes.sort(key=lambda x: x[0])
        
        print(f"   📊 POC 주변 ±$10 범위의 거래량:")
        for price, volume in poc_range_volumes[:15]:  # 상위 15개만
            marker = "🎯" if abs(price - poc_price) < 1 else "  "
            print(f"      {marker} ${price:8.2f}: {volume:12,.2f}")
    print()
    
    # 4. 큰 거래량으로 테스트 캔들 추가
    print("📊 4. 큰 거래량 테스트 캔들로 POC 변화 확인")
    
    # 현재 가격대에 큰 거래량 추가
    large_volume_candles = [
        {
            'timestamp': datetime.now(timezone.utc),
            'open': 4620.00,
            'high': 4625.00,
            'low': 4615.00,
            'close': 4620.00,
            'volume': 10000.0  # 매우 큰 거래량
        },
        {
            'timestamp': datetime.now(timezone.utc),
            'open': 4620.00,
            'high': 4625.00,
            'low': 4615.00,
            'close': 4622.00,
            'volume': 15000.0  # 더 큰 거래량
        }
    ]
    
    for i, candle in enumerate(large_volume_candles):
        print(f"   🔄 {i+1}. 큰 거래량 캔들 추가: ${candle['close']:.2f}, 거래량: {candle['volume']:,.0f}")
        update_all_indicators_with_candle(candle)
        
        # 업데이트 후 POC 확인
        vpvr_result = vpvr.get_current_vpvr()
        if vpvr_result:
            new_poc = vpvr_result.get('poc', 0)
            print(f"      🎯 새로운 POC: ${new_poc:.2f}")
            
            # 상위 5개 거래량 구간 확인
            if hasattr(vpvr, 'volume_histogram'):
                sorted_volumes = sorted(vpvr.volume_histogram.items(), key=lambda x: x[1], reverse=True)
                print(f"      📈 상위 5개 거래량 구간:")
                for j, (bin_key, volume) in enumerate(sorted_volumes[:5]):
                    if bin_key in vpvr.price_bins:
                        price = vpvr.price_bins[bin_key]
                        marker = "🎯" if abs(price - new_poc) < 1 else "  "
                        print(f"         {marker} ${price:8.2f}: {volume:12,.2f}")
        print()
    
    # 5. 최종 VPVR 상태
    print("📊 5. 최종 VPVR 상태")
    final_result = vpvr.get_current_vpvr()
    final_status = vpvr.get_vpvr_status()
    
    if final_result:
        print(f"   🎯 최종 POC: ${final_result.get('poc', 0):.2f}")
        print(f"   📈 최종 HVN: ${final_result.get('hvn', 0):.2f}")
        print(f"   📉 최종 LVN: ${final_result.get('lvn', 0):.2f}")
    
    print(f"   📊 최종 활성 구간: {final_status.get('active_bins', 0)}개")
    print(f"   📊 최종 총 거래량: {final_status.get('total_volume', 0):,.2f}")
    
    # ATR 상태도 확인
    atr = get_indicator('atr')
    if atr:
        print(f"   📊 최종 ATR: {atr.get_atr():.3f}")
    
    print("\n🏁 VPVR 상세 분석 완료!")

if __name__ == "__main__":
    test_vpvr_detailed()
