#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VPVR 디버깅 테스트 - POC 계산 과정 상세 분석
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

def test_vpvr_debug():
    """VPVR 디버깅 테스트"""
    print("🚀 VPVR 디버깅 테스트 시작...\n")
    
    # 1. 글로벌 지표 초기화
    print("📊 1. 글로벌 지표 초기화")
    initialize_global_indicators()
    vpvr = get_indicator('vpvr')
    print("✅ 초기화 완료\n")
    
    # 2. 현재 Bin 크기 확인
    print("📊 2. 현재 Bin 크기 분석")
    if hasattr(vpvr, 'volume_histogram') and vpvr.volume_histogram:
        # 가격별 거래량 분포 확인
        price_volume_map = {}
        for bin_key, volume in vpvr.volume_histogram.items():
            if bin_key in vpvr.price_bins:
                price = vpvr.price_bins[bin_key]
                price_volume_map[price] = volume
        
        # 가격 순으로 정렬
        sorted_prices = sorted(price_volume_map.keys())
        
        print(f"   📊 총 가격 구간: {len(sorted_prices)}개")
        print(f"   📊 가격 범위: ${sorted_prices[0]:.2f} ~ ${sorted_prices[-1]:.2f}")
        
        # 연속된 가격 간격 확인
        if len(sorted_prices) > 1:
            gaps = []
            for i in range(1, len(sorted_prices)):
                gap = sorted_prices[i] - sorted_prices[i-1]
                gaps.append(gap)
            
            avg_gap = sum(gaps) / len(gaps)
            min_gap = min(gaps)
            max_gap = max(gaps)
            
            print(f"   📏 평균 가격 간격: ${avg_gap:.3f}")
            print(f"   📏 최소 가격 간격: ${min_gap:.3f}")
            print(f"   📏 최대 가격 간격: ${max_gap:.3f}")
            
            # Bin 크기 일관성 확인
            if abs(max_gap - min_gap) < 0.1:
                print(f"   ✅ Bin 크기가 일정함: ${avg_gap:.3f}")
            else:
                print(f"   ⚠️ Bin 크기가 불규칙함: ${min_gap:.3f} ~ ${max_gap:.3f}")
    
    print()
    
    # 3. 거래량 집중 구간 재분석
    print("📊 3. 거래량 집중 구간 재분석")
    if hasattr(vpvr, 'volume_histogram'):
        # 거래량 기준 상위 20개
        sorted_volumes = sorted(vpvr.volume_histogram.items(), key=lambda x: x[1], reverse=True)
        
        print("   📈 상위 20개 거래량 구간:")
        for i, (bin_key, volume) in enumerate(sorted_volumes[:20]):
            if bin_key in vpvr.price_bins:
                price = vpvr.price_bins[bin_key]
                percentage = (volume / sum(vpvr.volume_histogram.values())) * 100
                print(f"      {i+1:2d}. ${price:8.2f}: {volume:12,.2f} ({percentage:5.2f}%)")
        
        # 누적 거래량 확인
        print(f"\n   📊 누적 거래량 분석:")
        total_volume = sum(vpvr.volume_histogram.values())
        cumulative_volume = 0
        for i, (bin_key, volume) in enumerate(sorted_volumes):
            if bin_key in vpvr.price_bins:
                cumulative_volume += volume
                percentage = (cumulative_volume / total_volume) * 100
                price = vpvr.price_bins[bin_key]
                if i < 5 or percentage <= 50:  # 상위 5개 또는 50%까지
                    print(f"      누적 {i+1:2d}개: ${price:8.2f} → {percentage:5.1f}% ({cumulative_volume:12,.0f})")
                if percentage >= 50:
                    break
    
    print()
    
    # 4. POC 계산 로직 확인
    print("📊 4. POC 계산 로직 확인")
    vpvr_result = vpvr.get_current_vpvr()
    if vpvr_result:
        poc_price = vpvr_result.get('poc', 0)
        print(f"   🎯 현재 POC: ${poc_price:.2f}")
        
        # POC 주변 ±$20 범위의 거래량 확인
        poc_range_volumes = []
        for bin_key, volume in vpvr.volume_histogram.items():
            if bin_key in vpvr.price_bins:
                price = vpvr.price_bins[bin_key]
                if abs(price - poc_price) <= 20:  # ±$20 범위
                    poc_range_volumes.append((price, volume))
        
        # 가격 순으로 정렬
        poc_range_volumes.sort(key=lambda x: x[0])
        
        print(f"   📊 POC 주변 ±$20 범위의 거래량:")
        for price, volume in poc_range_volumes:
            marker = "🎯" if abs(price - poc_price) < 1 else "  "
            print(f"      {marker} ${price:8.2f}: {volume:12,.2f}")
    
    print()
    
    # 5. 사이트 결과와 비교
    print("📊 5. 사이트 결과와 비교")
    site_poc = 4643.0
    current_poc = vpvr_result.get('poc', 0) if vpvr_result else 0
    
    print(f"   🌐 사이트 POC: ${site_poc:.2f}")
    print(f"   🔧 VPVR POC: ${current_poc:.2f}")
    print(f"   📏 차이: ${abs(site_poc - current_poc):.2f} ({abs(site_poc - current_poc) / site_poc * 100:.2f}%)")
    
    # 사이트 POC 주변 거래량 확인
    if hasattr(vpvr, 'volume_histogram'):
        site_poc_volume = 0
        for bin_key, volume in vpvr.volume_histogram.items():
            if bin_key in vpvr.price_bins:
                price = vpvr.price_bins[bin_key]
                if abs(price - site_poc) < 5:  # ±$5 범위
                    site_poc_volume += volume
        
        print(f"   📊 사이트 POC ${site_poc:.2f} 주변 ±$5 거래량: {site_poc_volume:,.2f}")
        print(f"   📊 현재 POC ${current_poc:.2f} 주변 ±$5 거래량: {vpvr.volume_histogram.get(f'bin_{int(current_poc/2.128)}', 0):,.2f}")
    
    print("\n🏁 VPVR 디버깅 테스트 완료!")

if __name__ == "__main__":
    test_vpvr_debug()
