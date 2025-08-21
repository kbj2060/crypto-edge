#!/usr/bin/env python3
"""
통합 스마트 트레이더 고급 청산 전략 테스트
실시간 청산 데이터 처리 상태 확인
"""

import time
import datetime
from datetime import timezone
import numpy as np
from signals.advanced_liquidation_strategy import AdvancedLiquidationStrategy, AdvancedLiquidationConfig

def simulate_liquidation_events(strategy, duration_minutes=5):
    """청산 이벤트 시뮬레이션"""
    print(f"📥 {duration_minutes}분간 청산 이벤트 시뮬레이션 시작...")
    
    base_time = datetime.datetime.now(timezone.utc)
    events_processed = 0
    
    for minute in range(duration_minutes):
        for second in range(60):  # 1분 = 60초
            # 현재 시뮬레이션 시간
            sim_time = base_time + datetime.timedelta(minutes=minute, seconds=second)
            
            # 1초마다 1-3개의 청산 이벤트 생성
            num_events = np.random.randint(1, 4)
            
            for _ in range(num_events):
                # 롱/숏 랜덤 선택
                side = 'long' if np.random.random() > 0.5 else 'short'
                
                # 청산량 생성 (일반: $1K-$10K, 스파이크: $20K-$100K)
                if np.random.random() < 0.1:  # 10% 확률로 스파이크
                    qty_usd = np.random.uniform(20000, 100000)
                else:
                    qty_usd = np.random.uniform(1000, 10000)
                
                # 청산 이벤트 구성
                liquidation_event = {
                    'ts': int(sim_time.timestamp()),
                    'side': side,
                    'qty_usd': qty_usd
                }
                
                # 전략에 이벤트 전달
                strategy.process_liquidation_event(liquidation_event)
                events_processed += 1
            
            # 10초마다 상태 출력
            if second % 10 == 0:
                warmup_status = strategy.get_warmup_status()
                print(f"   {minute:02d}:{second:02d} - 이벤트: {events_processed}개, "
                      f"SETUP: {warmup_status['can_setup']}, ENTRY: {warmup_status['can_entry']}")
        
        # 1분마다 상세 상태 출력
        print(f"\n📊 {minute+1}분 완료 - 상세 상태:")
        _print_detailed_status(strategy)
        print()
    
    return events_processed

def _print_detailed_status(strategy):
    """상세 상태 출력"""
    # 워밍업 상태
    warmup_status = strategy.get_warmup_status()
    print(f"   🔥 워밍업 상태:")
    print(f"      - SETUP 가능: {warmup_status['can_setup']}")
    print(f"      - ENTRY 가능: {warmup_status['can_entry']}")
    print(f"      - 총 샘플: {warmup_status['total_samples']}")
    print(f"      - 롱 샘플: {warmup_status['long_samples']}")
    print(f"      - 숏 샘플: {warmup_status['short_samples']}")
    
    # 청산 메트릭
    try:
        metrics = strategy.get_current_liquidation_metrics()
        if metrics:
            print(f"   📈 청산 지표:")
            print(f"      - 롱 30초: {metrics['l_long_30s']:,.0f}")
            print(f"      - 숏 30초: {metrics['l_short_30s']:,.0f}")
            print(f"      - 롱 Z-score: {metrics['z_long']:.2f}")
            print(f"      - 숏 Z-score: {metrics['z_short']:.2f}")
            print(f"      - LPI: {metrics['lpi']:.3f}")
            print(f"      - 캐스케이드: {metrics['is_cascade']}")
            print(f"      - 쿨다운: {metrics['cooldown_active']}")
        else:
            print(f"   📈 청산 지표: 계산 불가")
    except Exception as e:
        print(f"   📈 청산 지표: 오류 - {e}")
    
    # 백그라운드 통계
    summary = strategy.get_strategy_summary()
    print(f"   📋 백그라운드 통계:")
    print(f"      - 롱 μ: {summary['background_stats']['mu_long']:,.0f}")
    print(f"      - 롱 σ: {summary['background_stats']['sigma_long']:,.0f}")
    print(f"      - 숏 μ: {summary['background_stats']['mu_short']:,.0f}")
    print(f"      - 숏 σ: {summary['background_stats']['sigma_short']:,.0f}")

def test_integrated_liquidation():
    """통합 청산 전략 테스트"""
    print("🧪 통합 스마트 트레이더 고급 청산 전략 테스트")
    print("=" * 70)
    
    # 1. 전략 인스턴스 생성
    config = AdvancedLiquidationConfig()
    strategy = AdvancedLiquidationStrategy(config)
    
    print(f"✅ 전략 인스턴스 생성 완료")
    print(f"📊 초기 상태:")
    print(f"   - 롱 버퍼 크기: {len(strategy.long_bins)}")
    print(f"   - 숏 버퍼 크기: {len(strategy.short_bins)}")
    print(f"   - 롱 μ: {strategy.mu_long:.2f}")
    print(f"   - 롱 σ: {strategy.sigma_long:.2f}")
    print(f"   - 숏 μ: {strategy.mu_short:.2f}")
    print(f"   - 숏 σ: {strategy.sigma_short:.2f}")
    print()
    
    # 2. 청산 이벤트 시뮬레이션 (5분간)
    events_processed = simulate_liquidation_events(strategy, duration_minutes=5)
    
    # 3. 최종 상태 확인
    print(f"\n🎯 최종 상태 확인:")
    print("=" * 50)
    _print_detailed_status(strategy)
    
    # 4. 전략 분석 시도
    print(f"\n🎯 전략 분석 시도:")
    print("-" * 30)
    
    try:
        # 가짜 가격 데이터 생성 (DataFrame 형태)
        fake_price_data = pd.DataFrame({
            'open': [50000] * 20,
            'high': [51000] * 20,
            'low': [49000] * 20,
            'close': [50000] * 20,
            'volume': [1000] * 20
        })
        
        # 가짜 키 레벨 및 지표
        fake_key_levels = {
            'prev_day_high': 52000.0,
            'prev_day_low': 48000.0,
            'vwap': 50000.0,
            'vwap_std': 1000.0
        }
        
        fake_opening_range = {
            'high': 51000.0,
            'low': 49000.0,
            'center': 50000.0,
            'range': 2000.0
        }
        
        fake_vwap = 50000.0
        fake_vwap_std = 1000.0
        fake_atr = 500.0
        
        # 전략 분석 실행
        result = strategy.analyze_all_strategies(
            fake_price_data, 
            fake_key_levels, 
            fake_opening_range, 
            fake_vwap, 
            fake_vwap_std, 
            fake_atr
        )
        
        if result:
            print(f"✅ 전략 분석 성공:")
            print(f"   - 신호: {result.get('action', 'N/A')}")
            print(f"   - 등급: {result.get('tier', 'N/A')}")
            print(f"   - 전략: {result.get('playbook', 'N/A')}")
            print(f"   - 점수: {result.get('total_score', 'N/A')}")
            print(f"   - 이유: {result.get('reason', 'N/A')}")
        else:
            print(f"❌ 전략 분석 실패: 신호 없음")
            
    except Exception as e:
        print(f"❌ 전략 분석 오류: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n📊 테스트 요약:")
    print(f"   - 처리된 청산 이벤트: {events_processed}개")
    print(f"   - 최종 롱 버퍼: {len(strategy.long_bins)}개")
    print(f"   - 최종 숏 버퍼: {len(strategy.short_bins)}개")
    print(f"   - 워밍업 완료: {strategy.get_warmup_status()['can_entry']}")
    
    print(f"\n✅ 테스트 완료!")

if __name__ == "__main__":
    test_integrated_liquidation()
