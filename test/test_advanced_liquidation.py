#!/usr/bin/env python3
"""
고급 청산 전략 테스트 스크립트
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from signals.advanced_liquidation_strategy import AdvancedLiquidationStrategy, AdvancedLiquidationConfig

def create_test_data():
    """테스트용 가격 데이터 생성"""
    # 1분봉 데이터 (500개)
    dates = pd.date_range(start='2024-01-01', periods=500, freq='1min')
    
    # 가격 데이터 생성 (ETHUSDT 시뮬레이션)
    base_price = 3000.0
    np.random.seed(42)  # 재현 가능한 랜덤
    
    # 가격 변동성 추가
    price_changes = np.random.normal(0, 0.001, 500)  # 0.1% 변동성
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # 신호 조건을 만족하는 가격 패턴 추가
    # 마지막 10개 봉에서 하단 스윕 시뮬레이션
    for i in range(10):
        if i < 5:  # 하단 스윕
            prices[-(i+1)] = base_price * 0.98  # 2% 하락
        else:  # 리클레임
            prices[-(i+1)] = base_price * 0.99  # 1% 하락 (리클레임)
    
    # OHLC 데이터 생성
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        # 고가/저가/시가/종가 생성
        high = price * (1 + abs(np.random.normal(0, 0.0005)))
        low = price * (1 - abs(np.random.normal(0, 0.0005)))
        open_price = price * (1 + np.random.normal(0, 0.0002))
        close_price = price
        
        data.append({
            'timestamp': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': np.random.uniform(100, 1000)
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df

def create_test_liquidation_events():
    """테스트용 청산 이벤트 생성"""
    events = []
    current_time = datetime.now(timezone.utc)
    
    # 최근 5분 동안의 청산 이벤트
    for i in range(30):  # 30개 이벤트
        event_time = current_time - timedelta(seconds=i*10)  # 10초마다
        
        # 롱/숏 청산 랜덤 생성
        side = 'long' if np.random.random() > 0.5 else 'short'
        
        # 청산 스파이크 생성 (일부 이벤트는 매우 큰 청산량)
        if i < 5:  # 최근 5개 이벤트는 스파이크
            qty_usd = np.random.uniform(50000, 100000)  # $50K-$100K (스파이크)
        else:
            qty_usd = np.random.uniform(1000, 10000)  # $1K-$10K (정상)
        
        events.append({
            'ts': int(event_time.timestamp()),
            'side': side,
            'qty_usd': qty_usd,
            'symbol': 'ETHUSDT'
        })
    
    return events

def test_advanced_liquidation_strategy():
    """고급 청산 전략 테스트"""
    print("🧪 고급 청산 전략 테스트 시작...")
    
    try:
        # 1. 설정 생성
        config = AdvancedLiquidationConfig()
        print(f"✅ 설정 생성 완료: 워밍업 요구사항 - SETUP: {config.min_warmup_samples_setup}, ENTRY: {config.min_warmup_samples}")
        
        # 2. 전략 인스턴스 생성
        strategy = AdvancedLiquidationStrategy(config)
        print(f"✅ 전략 인스턴스 생성 완료")
        
        # 3. 테스트 데이터 생성
        df = create_test_data()
        liquidation_events = create_test_liquidation_events()
        print(f"✅ 테스트 데이터 생성 완료: 가격 {len(df)}개, 청산 {len(liquidation_events)}개")
        
        # 4. 청산 이벤트 처리
        print("\n📊 청산 이벤트 처리 중...")
        for event in liquidation_events:
            strategy.process_liquidation_event(event)
        
        # 5. 워밍업 상태 확인
        warmup_status = strategy.get_warmup_status()
        print(f"\n🔥 워밍업 상태:")
        print(f"   - 총 샘플: {warmup_status['total_samples']}")
        print(f"   - 롱 샘플: {warmup_status['long_samples']}")
        print(f"   - 숏 샘플: {warmup_status['short_samples']}")
        print(f"   - SETUP 가능: {warmup_status['can_setup']}")
        print(f"   - ENTRY 가능: {warmup_status['can_entry']}")
        
        # 6. 청산 지표 계산
        print(f"\n📈 청산 지표 계산...")
        metrics = strategy.get_current_liquidation_metrics()
        if metrics:
            print(f"   - 롱 30초: {metrics['l_long_30s']:.0f}")
            print(f"   - 숏 30초: {metrics['l_short_30s']:.0f}")
            print(f"   - Z 롱: {metrics['z_long']:.2f}")
            print(f"   - Z 숏: {metrics['z_short']:.2f}")
            print(f"   - LPI: {metrics['lpi']:.3f}")
            print(f"   - 캐스케이드: {metrics['is_cascade']}")
            print(f"   - 쿨다운: {metrics['cooldown_active']}")
        else:
            print("   ❌ 청산 지표 계산 실패")
            return
        
        # 7. 키 레벨 계산
        key_levels = {
            'prev_day_high': df['high'].max() * 1.01,
            'prev_day_low': df['low'].min() * 0.995,  # 현재 가격이 스윕할 수 있도록
            'vwap': df['close'].mean(),
            'vwap_std': df['close'].std()
        }
        
        opening_range = {
            'high': df['high'].max(),
            'low': df['low'].min()
        }
        
        # 8. ATR 계산
        atr = df['high'].rolling(14).max() - df['low'].rolling(14).min()
        atr = atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else df['close'].iloc[-1] * 0.02
        
        # 9. 전략 분석 실행
        print(f"\n🎯 전략 분석 실행...")
        signal = strategy.analyze_all_strategies(
            df, key_levels, opening_range, key_levels['vwap'], key_levels['vwap_std'], atr
        )
        
        if signal:
            if signal.get('action') == 'NEUTRAL':
                print(f"🔄 중립 신호: {signal.get('reason', '알 수 없음')}")
            else:
                print(f"✅ 신호 생성: {signal['action']} | {signal['playbook']} | {signal['tier']}")
                print(f"   - 점수: {signal.get('total_score', 0):.3f}")
                print(f"   - 진입가: {signal.get('entry_price', 0):.2f}")
                print(f"   - 스탑로스: {signal.get('stop_loss', 0):.2f}")
        else:
            print("❌ 신호 생성 실패")
        
        # 10. 전략 요약
        summary = strategy.get_strategy_summary()
        print(f"\n📋 전략 요약:")
        print(f"   - 세션 활성: {summary['session_active']}")
        print(f"   - 캐스케이드: {summary['cascade_detected']}")
        print(f"   - 쿨다운: {summary['cooldown_active']}")
        print(f"   - μ 롱: {summary['background_stats']['mu_long']:.0f}")
        print(f"   - σ 롱: {summary['background_stats']['sigma_long']:.0f}")
        print(f"   - μ 숏: {summary['background_stats']['mu_short']:.0f}")
        print(f"   - σ 숏: {summary['background_stats']['sigma_short']:.0f}")
        
        print(f"\n✅ 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_advanced_liquidation_strategy()
