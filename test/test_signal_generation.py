#!/usr/bin/env python3
"""
신호 생성 기능 집중 테스트
- 실제 신호가 생성되는 최소 조건 파악
- Gate/Score 조건 단계별 분석
- 신호 생성 확률 개선 방안 제시
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from signals.session_based_strategy import SessionBasedStrategy, SessionConfig

def create_minimal_signal_data():
    """신호 생성을 위한 최소한의 데이터 (스윕 포함)"""
    base_price = 4000
    session_start = datetime(2025, 1, 20, 8, 0, 0, tzinfo=pytz.UTC)
    
    # 스윕이 포함된 데이터 생성
    data = []
    for i in range(60):  # 1시간
        timestamp = session_start + timedelta(minutes=i)
        
        if i < 15:  # OR 구간
            price = base_price + i * 0.2  # 점진적 상승
        elif i < 25:  # 돌파 구간
            price = base_price + 15 + (i-15) * 0.8  # 더 빠른 상승
        elif i < 35:  # 스윕 구간 (전일 저가 하회)
            # 3995 이하로 확실히 스윕
            if i < 30:
                price = 3990 - (i-25) * 2  # 3990 → 3980
            else:
                price = 3980 + (i-30) * 2  # 3980 → 3990
        elif i == 59:  # 마지막 봉에서 스윕 발생 (현재 봉)
            price = 3990  # prev_day_low(3995) 아래로
        else:  # 리클레임 구간
            price = 3990 + (i-35) * 0.4  # 점진적 회복
        
        high = price + 2
        low = price - 2
        close = price + np.random.uniform(-1, 1)
        
        high = max(high, price, close)
        low = min(low, price, close)
        
        data.append({
            'open': price,
            'high': high,
            'low': low,
            'close': close,
            'volume': 10000
        })
    
    timestamps = [session_start + timedelta(minutes=i) for i in range(60)]
    df = pd.DataFrame(data, index=pd.DatetimeIndex(timestamps, tz=pytz.UTC))
    
    # 디버깅: 스윕 구간 가격 확인
    print(f"🔍 스윕 구간 디버깅:")
    print(f"  prev_day_low: 3995")
    print(f"  25분: {df.iloc[25]['low']:.2f}")
    print(f"  30분: {df.iloc[30]['low']:.2f}")
    print(f"  35분: {df.iloc[35]['low']:.2f}")
    print(f"  최저가: {df['low'].min():.2f}")
    
    return df

def test_gate_conditions_step_by_step():
    """Gate 조건을 단계별로 테스트"""
    print("🔒 Gate 조건 단계별 분석")
    print("-" * 50)
    
    config = SessionConfig()
    # 매우 관대한 설정
    config.min_sweep_depth_atr = 0.01
    config.max_slippage_gate = 0.1
    config.min_volume_ratio = 0.1
    
    strategy = SessionBasedStrategy(config)
    
    df = create_minimal_signal_data()
    key_levels = {
        'prev_day_high': 4030,
        'prev_day_low': 3995,  # 3970 → 3995로 수정 (실제 데이터 범위 내)
        'liquidation_data': {'long_volume': 50000, 'short_volume': 30000}
    }
    
    session_vwap = 4010
    or_info = {'high': 4020, 'low': 3990}
    atr = 15
    
    # 각 플레이북별 Gate 상세 분석
    for playbook in ['A', 'B', 'C']:
        print(f"\n📊 플레이북 {playbook} Gate 분석:")
        
        for side in ['LONG', 'SHORT']:
            gates_passed, gate_results = strategy.check_gates(
                df, session_vwap, or_info, atr, playbook, side, key_levels
            )
            
            print(f"  {side}:")
            print(f"    전체 통과: {gates_passed}")
            print(f"    방향 게이트: {gate_results.get('direction', False)}")
            print(f"    구조 게이트: {gate_results.get('structure', False)}")
            print(f"    슬리피지 게이트: {gate_results.get('slippage', False)}")
            print(f"    거래량 게이트: {gate_results.get('volume', False)}")
            
            if playbook == 'B':
                print(f"    스윕 ATR: {gate_results.get('sweep_atr', 0):.3f}")
                print(f"    스윕 깊이 계산: pdl={key_levels.get('prev_day_low', 'N/A')}, current_low={df['low'].iloc[-1]:.2f}")
                print(f"    구조 게이트: {gate_results.get('structure', False)}")
                print(f"    리클레임 확증: {gate_results.get('reclaim_confirmed', False)}")

def test_score_calculation_detailed():
    """Score 계산 상세 분석"""
    print("\n📊 Score 계산 상세 분석")
    print("-" * 50)
    
    config = SessionConfig()
    strategy = SessionBasedStrategy(config)
    
    df = create_minimal_signal_data()
    key_levels = {
        'prev_day_high': 4030,
        'prev_day_low': 3970,
        'liquidation_data': {
            'long_volume': 50000,
            'short_volume': 30000,
            'long_intensity': 1.5,
            'short_intensity': 0.8
        }
    }
    
    session_vwap = 4010
    or_info = {'high': 4020, 'low': 3990}
    atr = 15
    current_time = df.index[-1]
    
    # 통과하는 Gate 결과 시뮬레이션
    gate_results = {
        'direction': True,
        'structure': True,
        'slippage': True,
        'volume': True,
        'sweep_atr': 1.0,
        'slippage_value': 0.01,
        'volume_ratio': 1.5
    }
    
    for playbook in ['A', 'B', 'C']:
        print(f"\n📈 플레이북 {playbook} Score 분석:")
        
        for side in ['LONG', 'SHORT']:
            score = strategy.calculate_score(
                df, session_vwap, or_info, atr, playbook, side, 
                gate_results, current_time, key_levels
            )
            
            print(f"  {side}: {score:.3f}")
            
            # 임계값과 비교
            if score >= config.entry_thresh:
                tier = "ENTRY"
            elif score >= config.setup_thresh:
                tier = "SETUP"
            elif score >= config.headsup_thresh:
                tier = "HEADS_UP"
            else:
                tier = "NO_SIGNAL"
            
            print(f"    예상 티어: {tier}")

def test_relaxed_conditions():
    """매우 관대한 조건으로 신호 생성 테스트"""
    print("\n🎯 관대한 조건으로 신호 생성 테스트")
    print("-" * 50)
    
    # 매우 관대한 설정
    config = SessionConfig()
    config.entry_thresh = 0.30        # 매우 낮은 임계값
    config.setup_thresh = 0.20
    config.headsup_thresh = 0.10
    config.min_drive_return_R = 0.1   # 매우 낮은 진행거리
    config.min_sweep_depth_atr = 0.01 # 매우 낮은 스윕 깊이
    config.max_slippage_gate = 0.2    # 높은 슬리피지 허용
    config.min_volume_ratio = 0.1     # 낮은 거래량 비율
    
    strategy = SessionBasedStrategy(config)
    
    df = create_minimal_signal_data()
    key_levels = {
        'prev_day_high': 4030,
        'prev_day_low': 3995,  # 3970 → 3995로 수정 (실제 데이터 범위 내)
        'liquidation_data': {
            'long_volume': 50000,
            'short_volume': 30000,
            'long_intensity': 1.5,
            'short_intensity': 0.8
        }
    }
    current_time = df.index[-1]
    
    print(f"📊 데이터 정보:")
    print(f"  길이: {len(df)}분")
    print(f"  가격 범위: {df['low'].min():.2f} ~ {df['high'].max():.2f}")
    print(f"  OR 구간 (0-14): {df.iloc[:15]['low'].min():.2f} ~ {df.iloc[:15]['high'].max():.2f}")
    
    signal = strategy.analyze_session_strategy(df, key_levels, current_time)
    
    if signal:
        print(f"\n✅ 신호 생성 성공!")
        print(f"  플레이북: {signal['playbook']}")
        print(f"  방향: {signal['side']}")
        print(f"  등급: {signal['stage']}")
        print(f"  점수: {signal['score']:.3f}")
        print(f"  신뢰도: {signal['confidence']:.1%}")
        
        # Gate 결과 확인
        gate_results = signal.get('gate_results', {})
        if gate_results:
            print(f"\n  Gate 결과:")
            for key, value in gate_results.items():
                print(f"    {key}: {value}")
    else:
        print("❌ 관대한 조건에서도 신호 생성 실패")

def test_individual_playbooks():
    """각 플레이북별 개별 테스트"""
    print("\n🎮 개별 플레이북 테스트")
    print("-" * 50)
    
    config = SessionConfig()
    # 중간 정도 관대한 설정
    config.entry_thresh = 0.50
    config.setup_thresh = 0.35
    config.headsup_thresh = 0.25
    
    strategy = SessionBasedStrategy(config)
    
    # 플레이북 A용 데이터 (OR 돌파)
    print("\n📈 플레이북 A (OR 돌파) 테스트:")
    df_a = create_minimal_signal_data()
    
    # OR을 확실히 돌파하도록 데이터 조정
    or_high = df_a.iloc[:15]['high'].max()
    for i in range(20, 40):
        if df_a.iloc[i]['high'] <= or_high:
            df_a.iloc[i, df_a.columns.get_loc('high')] = or_high + 5
    
    key_levels = {'prev_day_high': 4050, 'prev_day_low': 3950}
    signal_a = strategy.analyze_session_strategy(df_a, key_levels, df_a.index[-1])
    
    if signal_a and signal_a.get('playbook') == 'A':
        print(f"  ✅ A 신호: {signal_a['side']} {signal_a['stage']} ({signal_a['score']:.3f})")
    else:
        print(f"  ❌ A 신호 없음")
    
    # 플레이북 B용 데이터 (스윕)
    print("\n🔄 플레이북 B (스윕) 테스트:")
    df_b = create_minimal_signal_data()
    
    # 전일 저가 스윕 시뮬레이션
    prev_day_low = 3980
    df_b.iloc[30:35, df_b.columns.get_loc('low')] = prev_day_low - 10  # 스윕
    df_b.iloc[35:, df_b.columns.get_loc('close')] = prev_day_low + 5   # 리클레임
    
    key_levels = {'prev_day_high': 4050, 'prev_day_low': prev_day_low}
    signal_b = strategy.analyze_session_strategy(df_b, key_levels, df_b.index[-1])
    
    if signal_b and signal_b.get('playbook') == 'B':
        print(f"  ✅ B 신호: {signal_b['side']} {signal_b['stage']} ({signal_b['score']:.3f})")
    else:
        print(f"  ❌ B 신호 없음")
    
    # 플레이북 C용 데이터 (VWAP 리버전)
    print("\n📊 플레이북 C (VWAP 리버전) 테스트:")
    df_c = create_minimal_signal_data()
    
    # VWAP 계산
    session_start = df_c.index[0]
    session_end = df_c.index[-1]
    vwap, std = strategy.calculate_session_vwap(df_c, session_start, session_end)
    
    # -2σ 아래로 가격 조정 후 재진입
    df_c.iloc[40:45, df_c.columns.get_loc('close')] = vwap - 2.1 * std  # -2σ 아래
    df_c.iloc[45:, df_c.columns.get_loc('close')] = vwap - 1.4 * std   # -1.5σ 안쪽
    
    key_levels = {'prev_day_high': 4050, 'prev_day_low': 3950}
    signal_c = strategy.analyze_session_strategy(df_c, key_levels, df_c.index[-1])
    
    if signal_c and signal_c.get('playbook') == 'C':
        print(f"  ✅ C 신호: {signal_c['side']} {signal_c['stage']} ({signal_c['score']:.3f})")
    else:
        print(f"  ❌ C 신호 없음")

def main():
    """메인 테스트 함수"""
    print("🚀 신호 생성 기능 집중 테스트")
    print("=" * 70)
    
    test_gate_conditions_step_by_step()
    test_score_calculation_detailed()
    test_relaxed_conditions()
    test_individual_playbooks()
    
    print("\n" + "=" * 70)
    print("📝 테스트 완료!")

if __name__ == "__main__":
    main()
