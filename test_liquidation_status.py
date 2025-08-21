#!/usr/bin/env python3
"""
고급 청산 전략 실시간 상태 확인 테스트
"""

import time
from datetime import datetime
from signals.advanced_liquidation_strategy import AdvancedLiquidationStrategy, AdvancedLiquidationConfig

def test_liquidation_status():
    """청산 전략 상태 확인"""
    print("🧪 고급 청산 전략 실시간 상태 확인 테스트")
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
    
    # 30초마다 상태 확인 (5회)
    for i in range(5):
        now = datetime.now()
        print(f"⏰ {now.strftime('%H:%M:%S')} - 상태 확인 {i+1}/5")
        print("-" * 40)
        
        # 워밍업 상태 확인
        warmup_status = strategy.get_warmup_status()
        print(f"🔥 워밍업 상태:")
        print(f"   - SETUP 가능: {warmup_status['can_setup']}")
        print(f"   - ENTRY 가능: {warmup_status['can_entry']}")
        print(f"   - 총 샘플: {warmup_status['total_samples']}")
        print(f"   - 롱 샘플: {warmup_status['long_samples']}")
        print(f"   - 숏 샘플: {warmup_status['short_samples']}")
        
        # 청산 메트릭 확인
        try:
            metrics = strategy.get_current_liquidation_metrics()
            if metrics:
                print(f"📈 청산 지표:")
                print(f"   - 롱 30초: {metrics['l_long_30s']:,.0f}")
                print(f"   - 숏 30초: {metrics['l_short_30s']:,.0f}")
                print(f"   - 롱 Z-score: {metrics['z_long']:.2f}")
                print(f"   - 숏 Z-score: {metrics['z_short']:.2f}")
                print(f"   - LPI: {metrics['lpi']:.3f}")
            else:
                print(f"📈 청산 지표: 계산 불가")
        except Exception as e:
            print(f"📈 청산 지표: 오류 - {e}")
        
        print()
        
        if i < 4:  # 마지막 반복이 아니면 대기
            print("⏳ 30초 대기 중...")
            time.sleep(30)
    
    print("✅ 테스트 완료!")

if __name__ == "__main__":
    test_liquidation_status()
