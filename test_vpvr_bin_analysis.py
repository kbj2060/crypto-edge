#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VPVR Bin 분포 상세 분석
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from indicators.global_indicators import (
    initialize_global_indicators,
    get_indicator
)

def test_vpvr_bin_analysis():
    """VPVR Bin 분포 상세 분석"""
    print("🚀 VPVR Bin 분포 상세 분석 시작...\n")
    
    # 1. 글로벌 지표 초기화
    print("📊 1. 글로벌 지표 초기화")
    initialize_global_indicators()
    vpvr = get_indicator('vpvr')
    print("✅ 초기화 완료\n")
    
    # 2. Bin 분포 상세 분석
    print("📊 2. Bin 분포 상세 분석")
    if hasattr(vpvr, 'volume_histogram') and vpvr.volume_histogram:
        # Bin 키별로 정렬
        bin_keys = sorted(vpvr.volume_histogram.keys(), key=lambda x: int(x.split('_')[1]))
        
        print(f"   📊 총 Bin 키 수: {len(bin_keys)}개")
        print(f"   📊 활성 Bin 수: {len([k for k, v in vpvr.volume_histogram.items() if v > 0])}개")
        
        # Bin 인덱스 범위 확인
        bin_indices = [int(key.split('_')[1]) for key in bin_keys]
        min_index = min(bin_indices)
        max_index = max(bin_indices)
        
        print(f"   📊 Bin 인덱스 범위: {min_index} ~ {max_index}")
        print(f"   📊 이론적 Bin 개수: {max_index - min_index + 1}개")
        
        # 연속된 Bin 확인
        expected_bins = set(range(min_index, max_index + 1))
        actual_bins = set(bin_indices)
        missing_bins = expected_bins - actual_bins
        
        print(f"   📊 누락된 Bin 개수: {len(missing_bins)}개")
        if missing_bins:
            print(f"   📊 누락된 Bin 인덱스: {sorted(list(missing_bins))[:10]}...")  # 처음 10개만
        
        # Bin별 상세 정보
        print(f"\n   📊 Bin별 상세 정보 (처음 20개):")
        for i, bin_key in enumerate(bin_keys[:20]):
            volume = vpvr.volume_histogram[bin_key]
            price = vpvr.price_bins.get(bin_key, "N/A")
            bin_index = int(bin_key.split('_')[1])
            print(f"      {i+1:2d}. {bin_key:8s}: 가격=${price:8.2f}, 거래량={volume:12,.2f}")
        
        # Bin 크기 계산 확인
        print(f"\n   📊 Bin 크기 계산 확인:")
        if len(bin_keys) >= 2:
            first_price = vpvr.price_bins.get(bin_keys[0], 0)
            second_price = vpvr.price_bins.get(bin_keys[1], 0)
            if first_price and second_price:
                actual_bin_size = abs(second_price - first_price)
                print(f"      첫 번째 Bin: {bin_keys[0]} = ${first_price:.2f}")
                print(f"      두 번째 Bin: {bin_keys[1]} = ${second_price:.2f}")
                print(f"      실제 Bin 간격: ${actual_bin_size:.2f}")
                print(f"      예상 Bin 크기: $10.00")
                print(f"      차이: ${abs(actual_bin_size - 10.0):.2f}")
    
    print()
    
    # 3. 가격 분포 히스토그램
    print("📊 3. 가격 분포 히스토그램")
    if hasattr(vpvr, 'volume_histogram'):
        # 가격별로 정렬
        price_volume_pairs = []
        for bin_key, volume in vpvr.volume_histogram.items():
            if bin_key in vpvr.price_bins:
                price = vpvr.price_bins[bin_key]
                price_volume_pairs.append((price, volume))
        
        price_volume_pairs.sort(key=lambda x: x[0])
        
        print(f"   📊 가격별 분포 (처음 15개):")
        for i, (price, volume) in enumerate(price_volume_pairs[:15]):
            if i > 0:
                gap = price - price_volume_pairs[i-1][0]
                print(f"      {i:2d}. ${price:8.2f}: {volume:12,.2f} (간격: ${gap:6.2f})")
            else:
                print(f"      {i:2d}. ${price:8.2f}: {volume:12,.2f}")
        
        # 간격 통계
        if len(price_volume_pairs) > 1:
            gaps = [price_volume_pairs[i][0] - price_volume_pairs[i-1][0] for i in range(1, len(price_volume_pairs))]
            avg_gap = sum(gaps) / len(gaps)
            min_gap = min(gaps)
            max_gap = max(gaps)
            
            print(f"\n   📊 가격 간격 통계:")
            print(f"      평균 간격: ${avg_gap:.2f}")
            print(f"      최소 간격: ${min_gap:.2f}")
            print(f"      최대 간격: ${max_gap:.2f}")
    
    print()
    
    # 4. 결론
    print("📊 4. 결론")
    print("   🔍 활성 가격 구간이 23개인 이유:")
    print("      1. Bin 크기: $10 (고정)")
    print("      2. 가격 범위: 약 $512")
    print("      3. 이론적 Bin: 약 51개")
    print("      4. 실제 활성 Bin: 23개")
    print("      → 64개 캔들이 23개 Bin에만 분포")
    print("      → 빈 Bin은 활성으로 계산되지 않음")
    
    print("\n🏁 VPVR Bin 분포 분석 완료!")

if __name__ == "__main__":
    test_vpvr_bin_analysis()
