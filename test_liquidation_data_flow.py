#!/usr/bin/env python3
"""
청산 데이터 수집과 분석 과정 테스트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from signals.advanced_liquidation_strategy import AdvancedLiquidationStrategy, AdvancedLiquidationConfig

def test_liquidation_data_flow():
    """청산 데이터 수집과 분석 과정 테스트"""
    print("🧪 청산 데이터 수집과 분석 과정 테스트 시작")
    print("=" * 60)
    
    # 전략 인스턴스 생성
    config = AdvancedLiquidationConfig()
    strategy = AdvancedLiquidationStrategy(config)
    
    print(f"📊 초기 상태:")
    print(f"   - 롱 버퍼 크기: {len(strategy.long_bins)}")
    print(f"   - 숏 버퍼 크기: {len(strategy.short_bins)}")
    print(f"   - 롱 μ: {strategy.mu_long:.2f}")
    print(f"   - 롱 σ: {strategy.sigma_long:.2f}")
    print(f"   - 숏 μ: {strategy.mu_short:.2f}")
    print(f"   - 숏 σ: {strategy.sigma_short:.2f}")
    print()
    
    # 1단계: 청산 이벤트 수집 시뮬레이션
    print("📥 1단계: 청산 이벤트 수집 시뮬레이션")
    print("-" * 40)
    
    # 60초간의 청산 이벤트 생성 (1초 간격)
    base_time = datetime.now(timezone.utc)
    
    for i in range(60):
        # 각 이벤트마다 고유한 타임스탬프 사용
        event_time = base_time - timedelta(seconds=59-i)
        event_timestamp = int(event_time.timestamp())
        
        # 롱 청산 이벤트 (일반적인 청산량)
        long_event = {
            'ts': event_timestamp,  # 'timestamp' 대신 'ts' 사용
            'side': 'long',
            'qty_usd': np.random.uniform(1000, 5000)  # $1K-$5K
        }
        
        # 숏 청산 이벤트 (일반적인 청산량)
        short_event = {
            'ts': event_timestamp,  # 'timestamp' 대신 'ts' 사용
            'side': 'short',
            'qty_usd': np.random.uniform(1000, 5000)  # $1K-$5K
        }
        
        # 이벤트 처리
        strategy.process_liquidation_event(long_event)
        strategy.process_liquidation_event(short_event)
        
        if i % 10 == 0:  # 10초마다 상태 출력
            print(f"   {i:2d}초: 롱 {len(strategy.long_bins)}개, 숏 {len(strategy.short_bins)}개")
            print(f"      타임스탬프: {event_timestamp}")
    
    print()
    print(f"📊 수집 후 상태:")
    print(f"   - 롱 버퍼 크기: {len(strategy.long_bins)}")
    print(f"   - 숏 버퍼 크기: {len(strategy.short_bins)}")
    print(f"   - 롱 μ: {strategy.mu_long:.2f}")
    print(f"   - 롱 σ: {strategy.sigma_long:.2f}")
    print(f"   - 숏 μ: {strategy.mu_short:.2f}")
    print(f"   - 숏 σ: {strategy.sigma_short:.2f}")
    
    # 버퍼 내용 일부 출력
    if strategy.long_bins:
        print(f"   - 롱 버퍼 샘플 (처음 5개): {list(strategy.long_bins)[:5]}")
    if strategy.short_bins:
        print(f"   - 숏 버퍼 샘플 (처음 5개): {list(strategy.short_bins)[:5]}")
    print()
    
    # 2단계: 스파이크 청산 이벤트 추가
    print("📈 2단계: 스파이크 청산 이벤트 추가")
    print("-" * 40)
    
    # 최근 10초에 스파이크 청산 추가
    for i in range(10):
        event_time = base_time - timedelta(seconds=9-i)
        event_timestamp = int(event_time.timestamp())
        
        # 큰 롱 청산 스파이크
        long_spike = {
            'ts': event_timestamp,
            'side': 'long',
            'qty_usd': np.random.uniform(20000, 50000)  # $20K-$50K (스파이크)
        }
        
        # 큰 숏 청산 스파이크
        short_spike = {
            'ts': event_timestamp,
            'side': 'short',
            'qty_usd': np.random.uniform(20000, 50000)  # $20K-$50K (스파이크)
        }
        
        strategy.process_liquidation_event(long_spike)
        strategy.process_liquidation_event(short_spike)
        
        print(f"   스파이크 {i+1}: 롱 ${long_spike['qty_usd']:,.0f}, 숏 ${short_spike['qty_usd']:,.0f}")
        print(f"      타임스탬프: {event_timestamp}")
    
    print()
    print(f"📊 스파이크 추가 후 상태:")
    print(f"   - 롱 버퍼 크기: {len(strategy.long_bins)}")
    print(f"   - 숏 버퍼 크기: {len(strategy.short_bins)}")
    print(f"   - 롱 μ: {strategy.mu_long:.2f}")
    print(f"   - 롱 σ: {strategy.sigma_long:.2f}")
    print(f"   - 숏 μ: {strategy.mu_short:.2f}")
    print(f"   - 숏 σ: {strategy.mu_short:.2f}")
    
    # 버퍼 내용 일부 출력
    if strategy.long_bins:
        print(f"   - 롱 버퍼 샘플 (처음 5개): {list(strategy.long_bins)[:5]}")
    if strategy.short_bins:
        print(f"   - 숏 버퍼 샘플 (처음 5개): {list(strategy.short_bins)[:5]}")
    print()
    
    # 3단계: 현재 청산 메트릭 계산
    print("🔍 3단계: 현재 청산 메트릭 계산")
    print("-" * 40)
    
    try:
        metrics = strategy.get_current_liquidation_metrics()
        print(f"   ✅ 메트릭 계산 성공:")
        print(f"   - 롱 30초 합계: {metrics['l_long_30s']:,.0f}")
        print(f"   - 숏 30초 합계: {metrics['l_short_30s']:,.0f}")
        print(f"   - 롱 Z-score: {metrics['z_long']:.2f}")
        print(f"   - 숏 Z-score: {metrics['z_short']:.2f}")
        print(f"   - LPI: {metrics['lpi']:.3f}")
        
        # 백그라운드 통계도 출력
        bg_stats = metrics.get('background_stats', {})
        if bg_stats:
            print(f"   - 롱 μ 스케일: {bg_stats.get('mu_long_scaled', 0):,.0f}")
            print(f"   - 롱 σ 스케일: {bg_stats.get('sigma_long_scaled', 0):,.0f}")
            print(f"   - 숏 μ 스케일: {bg_stats.get('mu_short_scaled', 0):,.0f}")
            print(f"   - 숏 σ 스케일: {bg_stats.get('sigma_short_scaled', 0):,.0f}")
            
    except Exception as e:
        print(f"   ❌ 메트릭 계산 실패: {e}")
    
    print()
    
    # 4단계: 워밍업 상태 확인
    print("🔥 4단계: 워밍업 상태 확인")
    print("-" * 40)
    
    try:
        warmup_status = strategy.get_warmup_status()
        print(f"   ✅ 워밍업 상태 확인 성공:")
        print(f"   - 기본 워밍업: {warmup_status['basic_warmup']}")
        print(f"   - 완전 워밍업: {warmup_status['full_warmup']}")
        print(f"   - μ/σ 안정성: {warmup_status['mu_stable']}")
        print(f"   - SETUP 가능: {warmup_status['can_setup']}")
        print(f"   - ENTRY 가능: {warmup_status['can_entry']}")
        print(f"   - 총 샘플: {warmup_status['total_samples']}")
        print(f"   - 롱 샘플: {warmup_status['long_samples']}")
        print(f"   - 숏 샘플: {warmup_status['short_samples']}")
    except Exception as e:
        print(f"   ❌ 워밍업 상태 확인 실패: {e}")
    
    print()
    
    # 5단계: 게이트 조건 확인
    print("🚪 5단계: 게이트 조건 확인")
    print("-" * 40)
    
    try:
        # 가짜 가격 데이터 생성 (DataFrame 형태)
        fake_price_data = pd.DataFrame({
            'open': [50000] * 5,
            'high': [51000] * 5,
            'low': [49000] * 5,
            'close': [50000] * 5,
            'volume': [1000] * 5
        })
        
        fake_atr = 500.0  # 1% ATR
        fake_current_price = 50000.0
        
        gate_result = strategy.check_gate_conditions(fake_price_data, fake_atr, fake_current_price)
        print(f"   ✅ 게이트 조건 확인 성공:")
        print(f"   - 게이트 통과: {gate_result['gate_passed']}")
        print(f"   - 기본 위생: {gate_result['basic_hygiene']}")
        print(f"   - 하드 블록: {gate_result['hard_blocked']}")
        if gate_result['hard_blocked']:
            print(f"   - 블록 이유: {gate_result['block_reason']}")
    except Exception as e:
        print(f"   ❌ 게이트 조건 확인 실패: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    
    # 6단계: 전략 분석 시도
    print("🎯 6단계: 전략 분석 시도")
    print("-" * 40)
    
    try:
        # 가짜 키 레벨 생성
        fake_key_levels = {
            'prev_day_high': 52000.0,
            'prev_day_low': 48000.0,
            'vwap': 50000.0,
            'vwap_std': 1000.0
        }
        
        # 전략 C (과열-소멸 페이드)가 트리거되도록 가격 데이터 설정
        # VWAP에서 2σ 이상 멀어진 상황 시뮬레이션
        fake_vwap = 50000.0
        fake_vwap_std = 1000.0
        fake_atr = 500.0  # 1% ATR
        fake_current_price = 50000.0
        
        # 과열된 가격 상황: VWAP(50000) + 2.5σ(2500) = 52500
        overheated_price = fake_vwap + (2.5 * fake_vwap_std)
        print(f"   📈 과열 가격 시뮬레이션: VWAP={fake_vwap:.0f}, σ={fake_vwap_std:.0f}")
        print(f"   📈 목표 가격: {overheated_price:.0f} (VWAP + 2.5σ)")
        
        fake_ohlcv = pd.DataFrame({
            'open': [50000] * 15 + [overheated_price] * 5,
            'high': [51000] * 15 + [overheated_price + 200] * 5,
            'low': [49000] * 15 + [overheated_price - 200] * 5,
            'close': [50000] * 15 + [overheated_price] * 5,
            'volume': [1000] * 20
        })
        
        # opening_range, vwap, vwap_std, atr 파라미터 추가
        fake_opening_range = {
            'high': 51000.0,
            'low': 49000.0,
            'mid': 50000.0
        }
        
        # 현재 가격을 과열된 가격으로 설정
        fake_current_price = overheated_price
        
        # 전략 분석 실행 전 조건 확인
        print(f"   📋 분석 전 조건 확인:")
        print(f"   - Z-score 임계값: z_spike={config.z_spike}, z_strong={config.z_strong}")
        print(f"   - 현재 Z-score: 롱={metrics['z_long']:.2f}, 숏={metrics['z_short']:.2f}")
        print(f"   - LPI 임계값: {config.lpi_bias}")
        print(f"   - 현재 LPI: {metrics['lpi']:.3f}")
        
        # 게이트 조건 재확인
        gate_check = strategy.check_gate_conditions(fake_ohlcv, fake_atr, fake_current_price)
        print(f"   - 게이트 통과: {gate_check['gate_passed']}")
        if not gate_check['gate_passed']:
            print(f"   - 게이트 실패 이유: {gate_check['block_reason']}")
        
        # 전략 분석 실행 (올바른 파라미터로)
        result = strategy.analyze_all_strategies(
            fake_ohlcv, 
            fake_key_levels, 
            fake_opening_range, 
            fake_vwap, 
            fake_vwap_std, 
            fake_atr
        )
        
        print(f"   ✅ 전략 분석 성공:")
        print(f"   - 결과 타입: {type(result)}")
        if isinstance(result, dict):
            print(f"   - 신호: {result.get('action', 'N/A')}")
            print(f"   - 등급: {result.get('tier', 'N/A')}")
            print(f"   - 점수: {result.get('total_score', 'N/A')}")
            print(f"   - 플레이북: {result.get('playbook', 'N/A')}")
            print(f"   - 이유: {result.get('reason', 'N/A')}")
        else:
            print(f"   - 결과: {result}")
            
    except Exception as e:
        print(f"   ❌ 전략 분석 실패: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("=" * 60)
    print("🧪 테스트 완료!")

if __name__ == "__main__":
    test_liquidation_data_flow()
