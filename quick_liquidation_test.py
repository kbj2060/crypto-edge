#!/usr/bin/env python3
"""
고급 청산 전략 빠른 상태 확인
"""

from signals.advanced_liquidation_strategy import AdvancedLiquidationStrategy, AdvancedLiquidationConfig
import numpy as np
from datetime import datetime, timezone

def quick_test():
    """빠른 테스트"""
    print("🧪 고급 청산 전략 빠른 테스트")
    print("=" * 50)
    
    # 전략 생성
    config = AdvancedLiquidationConfig()
    strategy = AdvancedLiquidationStrategy(config)
    
    print(f"📊 초기 상태:")
    print(f"   - 롱 버퍼: {len(strategy.long_bins)}개")
    print(f"   - 숏 버퍼: {len(strategy.short_bins)}개")
    print(f"   - 워밍업: SETUP={strategy.get_warmup_status()['can_setup']}, ENTRY={strategy.get_warmup_status()['can_entry']}")
    print()
    
    # 청산 이벤트 10개 추가
    print("📥 청산 이벤트 10개 추가...")
    now = datetime.now(timezone.utc)
    
    for i in range(10):
        side = 'long' if np.random.random() > 0.5 else 'short'
        qty_usd = np.random.uniform(1000, 50000)
        
        event = {
            'ts': int(now.timestamp()) + i,
            'side': side,
            'qty_usd': qty_usd
        }
        
        strategy.process_liquidation_event(event)
        print(f"   {i+1:2d}: {side} ${qty_usd:,.0f}")
    
    print()
    
    # 상태 확인
    warmup = strategy.get_warmup_status()
    print(f"📊 추가 후 상태:")
    print(f"   - 롱 버퍼: {warmup['long_samples']}개")
    print(f"   - 숏 버퍼: {warmup['short_samples']}개")
    print(f"   - 워밍업: SETUP={warmup['can_setup']}, ENTRY={warmup['can_entry']}")
    
    # 메트릭 확인
    try:
        metrics = strategy.get_current_liquidation_metrics()
        if metrics:
            print(f"   - 롱 Z: {metrics['z_long']:.2f}")
            print(f"   - 숏 Z: {metrics['z_short']:.2f}")
            print(f"   - LPI: {metrics['lpi']:.3f}")
    except Exception as e:
        print(f"   - 메트릭 오류: {e}")
    
    print(f"\n✅ 테스트 완료!")

if __name__ == "__main__":
    quick_test()
