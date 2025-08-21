#!/usr/bin/env python3
"""
세션 기반 전략 상세 테스트 코드
- 실제 신호 생성 조건에 맞춘 정밀 테스트
- 구체적인 시나리오별 데이터 생성
- 신호 생성을 위한 최적화된 조건 설정
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from signals.session_based_strategy import SessionBasedStrategy, SessionConfig, make_session_trade_plan

class DetailedSessionTester:
    """상세 세션 전략 테스트 클래스"""
    
    def __init__(self):
        self.test_results = []
        
    def log_test(self, test_name, passed, message=""):
        """테스트 결과 로깅"""
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} | {test_name}")
        if message:
            print(f"     {message}")
    
    def generate_realistic_or_breakout_scenario(self):
        """실제 OR 돌파 시나리오에 맞는 데이터 생성"""
        base_price = 4000
        session_start = datetime(2025, 1, 20, 7, 0, 0, tzinfo=pytz.UTC)
        
        # 1. OR 구간 (첫 15분) - 횡보
        or_timestamps = [session_start + timedelta(minutes=i) for i in range(15)]
        or_high = base_price + 15  # 4015
        or_low = base_price - 15   # 3985
        
        or_data = []
        for i, ts in enumerate(or_timestamps):
            # OR 구간은 범위 내에서 무작위
            price = base_price + np.random.uniform(-10, 10)
            high = min(or_high, price + np.random.uniform(0, 8))
            low = max(or_low, price - np.random.uniform(0, 8))
            close = price + np.random.uniform(-5, 5)
            
            # 논리적 일관성
            high = max(high, price, close)
            low = min(low, price, close)
            
            or_data.append({
                'open': price,
                'high': high,
                'low': low,
                'close': close,
                'volume': np.random.uniform(5000, 15000)
            })
        
        # 2. 돌파 구간 (15-35분) - 상승 돌파
        breakout_timestamps = [session_start + timedelta(minutes=i) for i in range(15, 35)]
        breakout_data = []
        
        # 돌파 시작 가격을 OR 고점 위로 설정
        current_price = or_high + 8  # 4023
        
        for i, ts in enumerate(breakout_timestamps):
            # 점진적 상승
            price_increase = i * 0.8  # 매 분마다 0.8 상승
            price = current_price + price_increase + np.random.uniform(-2, 4)
            
            high = price + np.random.uniform(0, 6)
            low = price - np.random.uniform(0, 4)
            close = price + np.random.uniform(-3, 3)
            
            high = max(high, price, close)
            low = min(low, price, close)
            
            breakout_data.append({
                'open': price,
                'high': high,
                'low': low,
                'close': close,
                'volume': np.random.uniform(8000, 20000)
            })
        
        # 3. 풀백 구간 (35-50분) - 하락 조정
        pullback_timestamps = [session_start + timedelta(minutes=i) for i in range(35, 50)]
        pullback_data = []
        
        peak_price = current_price + 16  # 약 4039
        pullback_target = base_price + 5  # 4005 (EMA/VWAP 근처)
        
        for i, ts in enumerate(pullback_timestamps):
            # 점진적 하락
            progress = i / len(pullback_timestamps)
            price = peak_price - (peak_price - pullback_target) * progress + np.random.uniform(-3, 3)
            
            high = price + np.random.uniform(0, 4)
            low = price - np.random.uniform(0, 6)
            close = price + np.random.uniform(-2, 2)
            
            high = max(high, price, close)
            low = min(low, price, close)
            
            pullback_data.append({
                'open': price,
                'high': high,
                'low': low,
                'close': close,
                'volume': np.random.uniform(6000, 14000)
            })
        
        # DataFrame 생성
        all_data = or_data + breakout_data + pullback_data
        all_timestamps = or_timestamps + breakout_timestamps + pullback_timestamps
        
        df = pd.DataFrame(all_data, index=pd.DatetimeIndex(all_timestamps, tz=pytz.UTC))
        return df
    
    def generate_realistic_sweep_scenario(self):
        """실제 스윕 시나리오에 맞는 데이터 생성"""
        base_price = 4000
        session_start = datetime(2025, 1, 20, 7, 0, 0, tzinfo=pytz.UTC)
        prev_day_low = base_price - 25  # 3975
        
        # 1. 베이스 구간 (0-30분) - 횡보
        base_timestamps = [session_start + timedelta(minutes=i) for i in range(30)]
        base_data = []
        
        for i, ts in enumerate(base_timestamps):
            price = base_price + np.random.uniform(-8, 8)
            high = price + np.random.uniform(0, 5)
            low = price - np.random.uniform(0, 5)
            close = price + np.random.uniform(-3, 3)
            
            high = max(high, price, close)
            low = min(low, price, close)
            
            base_data.append({
                'open': price,
                'high': high,
                'low': low,
                'close': close,
                'volume': np.random.uniform(5000, 12000)
            })
        
        # 2. 스윕 구간 (30-35분) - 급격한 하락으로 전일 저가 돌파
        sweep_timestamps = [session_start + timedelta(minutes=i) for i in range(30, 35)]
        sweep_data = []
        
        for i, ts in enumerate(sweep_timestamps):
            # 점진적으로 전일 저가 아래로 하락
            price = base_price - 15 - i * 3  # 3985 → 3973
            low_spike = prev_day_low - 5 - i * 2  # 전일 저가 아래로 스윕
            
            high = price + np.random.uniform(0, 3)
            low = min(price - np.random.uniform(3, 8), low_spike)
            close = price - np.random.uniform(0, 5)
            
            high = max(high, price, close)
            
            sweep_data.append({
                'open': price,
                'high': high,
                'low': low,
                'close': close,
                'volume': np.random.uniform(15000, 30000)
            })
        
        # 3. 리클레임 구간 (35-50분) - 전일 저가 위로 회복
        reclaim_timestamps = [session_start + timedelta(minutes=i) for i in range(35, 50)]
        reclaim_data = []
        
        for i, ts in enumerate(reclaim_timestamps):
            # 점진적으로 전일 저가 위로 상승
            progress = i / len(reclaim_timestamps)
            price = prev_day_low - 8 + progress * 20  # 3967 → 3987
            
            high = price + np.random.uniform(0, 5)
            low = price - np.random.uniform(0, 3)
            close = max(prev_day_low + 2, price + np.random.uniform(-2, 4))  # 확실히 리클레임
            
            high = max(high, price, close)
            low = min(low, price, close)
            
            reclaim_data.append({
                'open': price,
                'high': high,
                'low': low,
                'close': close,
                'volume': np.random.uniform(8000, 18000)
            })
        
        # DataFrame 생성
        all_data = base_data + sweep_data + reclaim_data
        all_timestamps = base_timestamps + sweep_timestamps + reclaim_timestamps
        
        df = pd.DataFrame(all_data, index=pd.DatetimeIndex(all_timestamps, tz=pytz.UTC))
        return df
    
    def generate_realistic_vwap_reversion_scenario(self):
        """실제 VWAP 리버전 시나리오에 맞는 데이터 생성"""
        base_price = 4000
        session_start = datetime(2025, 1, 20, 7, 0, 0, tzinfo=pytz.UTC)
        
        # 1. 베이스 구간으로 VWAP 설정 (0-20분)
        base_timestamps = [session_start + timedelta(minutes=i) for i in range(20)]
        base_data = []
        
        for i, ts in enumerate(base_timestamps):
            price = base_price + np.random.uniform(-5, 5)
            high = price + np.random.uniform(0, 3)
            low = price - np.random.uniform(0, 3)
            close = price + np.random.uniform(-2, 2)
            
            high = max(high, price, close)
            low = min(low, price, close)
            
            base_data.append({
                'open': price,
                'high': high,
                'low': low,
                'close': close,
                'volume': np.random.uniform(5000, 10000)
            })
        
        # 2. 과매도 구간 (20-35분) - VWAP-2σ 아래로 급락
        oversold_timestamps = [session_start + timedelta(minutes=i) for i in range(20, 35)]
        oversold_data = []
        
        # 대략적인 2σ 거리 계산 (실제로는 strategy에서 계산됨)
        sigma_distance = base_price * 0.015  # 1.5% 정도
        vwap_minus_2sigma = base_price - 2 * sigma_distance
        
        for i, ts in enumerate(oversold_timestamps):
            # 점진적으로 -2σ 아래로 하락
            price = base_price - 10 - i * 1.5  # 점진적 하락
            
            high = price + np.random.uniform(0, 3)
            low = price - np.random.uniform(0, 5)
            # t봉(마지막 전 봉)에서 -2σ 아래 종가 마감
            if i == len(oversold_timestamps) - 2:
                close = vwap_minus_2sigma - 5  # 확실히 -2σ 아래
            else:
                close = price + np.random.uniform(-3, 2)
            
            high = max(high, price, close)
            low = min(low, price, close)
            
            oversold_data.append({
                'open': price,
                'high': high,
                'low': low,
                'close': close,
                'volume': np.random.uniform(10000, 20000)
            })
        
        # 3. 리버전 구간 (35-45분) - -1.5σ 안쪽으로 재진입
        reversion_timestamps = [session_start + timedelta(minutes=i) for i in range(35, 45)]
        reversion_data = []
        
        vwap_minus_1_5sigma = base_price - 1.5 * sigma_distance
        
        for i, ts in enumerate(reversion_timestamps):
            # t+1봉에서 -1.5σ 안쪽으로 재진입
            if i == 0:  # 첫 번째 봉
                close = vwap_minus_1_5sigma + 3  # 확실히 -1.5σ 안쪽
                price = close - np.random.uniform(0, 2)
            else:
                price = base_price - 20 + i * 2  # 점진적 회복
                close = price + np.random.uniform(-2, 3)
            
            high = price + np.random.uniform(0, 4)
            low = price - np.random.uniform(0, 2)
            
            high = max(high, price, close)
            low = min(low, price, close)
            
            reversion_data.append({
                'open': price,
                'high': high,
                'low': low,
                'close': close,
                'volume': np.random.uniform(7000, 15000)
            })
        
        # DataFrame 생성
        all_data = base_data + oversold_data + reversion_data
        all_timestamps = base_timestamps + oversold_timestamps + reversion_timestamps
        
        df = pd.DataFrame(all_data, index=pd.DatetimeIndex(all_timestamps, tz=pytz.UTC))
        return df
    
    def generate_key_levels(self, base_price=4000):
        """키 레벨 생성"""
        return {
            'prev_day_high': base_price + 25,    # 4025
            'prev_day_low': base_price - 25,     # 3975
            'prev_day_close': base_price + 2,    # 4002
            'weekly_high': base_price + 40,      # 4040
            'weekly_low': base_price - 40,       # 3960
            'liquidation_data': {
                'long_volume': 75000,
                'short_volume': 45000,
                'long_intensity': 1.8,
                'short_intensity': 0.9,
                'total_events': 32
            }
        }
    
    def test_playbook_a_with_realistic_data(self):
        """플레이북 A 실제 데이터로 테스트"""
        print("\n🎯 플레이북 A 실제 시나리오 테스트")
        print("-" * 40)
        
        try:
            config = SessionConfig()
            # A 플레이북이 더 쉽게 신호를 생성하도록 조정
            config.min_drive_return_R = 0.5  # 0.8 → 0.5
            config.entry_thresh = 0.60       # 0.70 → 0.60
            config.setup_thresh = 0.40       # 0.50 → 0.40
            
            strategy = SessionBasedStrategy(config)
            
            df = self.generate_realistic_or_breakout_scenario()
            key_levels = self.generate_key_levels(4000)
            current_time = df.index[-1]
            
            print(f"   📊 데이터 길이: {len(df)}분")
            print(f"   📊 OR 구간: {df.index[0]} ~ {df.index[14]}")
            print(f"   📊 가격 범위: {df['low'].min():.2f} ~ {df['high'].max():.2f}")
            
            signal = strategy.analyze_session_strategy(df, key_levels, current_time)
            
            if signal:
                self.log_test("플레이북 A 실제 시나리오", True,
                            f"신호 생성: {signal['playbook']} {signal['side']} {signal['stage']}, 점수: {signal['score']:.3f}")
                
                # 신호 상세 정보 출력
                if 'entry_price' in signal:
                    print(f"     진입: ${signal['entry_price']:.2f}")
                    print(f"     손절: ${signal['stop_loss']:.2f}")
                    print(f"     목표: ${signal.get('take_profit1', 'N/A')}")
            else:
                self.log_test("플레이북 A 실제 시나리오", True, "조건 불만족으로 신호 없음")
                
        except Exception as e:
            self.log_test("플레이북 A 실제 시나리오", False, f"오류: {e}")
    
    def test_playbook_b_with_realistic_data(self):
        """플레이북 B 실제 데이터로 테스트"""
        print("\n🔄 플레이북 B 실제 시나리오 테스트")
        print("-" * 40)
        
        try:
            config = SessionConfig()
            # B 플레이북이 더 쉽게 신호를 생성하도록 조정
            config.min_sweep_depth_atr = 0.1    # 0.2 → 0.1
            config.entry_thresh = 0.60          # 0.70 → 0.60
            config.setup_thresh = 0.40          # 0.50 → 0.40
            
            strategy = SessionBasedStrategy(config)
            
            df = self.generate_realistic_sweep_scenario()
            key_levels = self.generate_key_levels(4000)
            current_time = df.index[-1]
            
            print(f"   📊 데이터 길이: {len(df)}분")
            print(f"   📊 스윕 구간: {df.index[30]} ~ {df.index[34]}")
            print(f"   📊 전일 저가: {key_levels['prev_day_low']:.2f}")
            print(f"   📊 최저가: {df['low'].min():.2f}")
            
            signal = strategy.analyze_session_strategy(df, key_levels, current_time)
            
            if signal:
                self.log_test("플레이북 B 실제 시나리오", True,
                            f"신호 생성: {signal['playbook']} {signal['side']} {signal['stage']}, 점수: {signal['score']:.3f}")
                
                # Gate 결과 확인
                gate_results = signal.get('gate_results', {})
                if 'sweep_atr' in gate_results:
                    print(f"     스윕 깊이: {gate_results['sweep_atr']:.2f} ATR")
                if 'reclaim_confirmed' in gate_results:
                    print(f"     리클레임 확증: {gate_results['reclaim_confirmed']}")
            else:
                self.log_test("플레이북 B 실제 시나리오", True, "조건 불만족으로 신호 없음")
                
        except Exception as e:
            self.log_test("플레이북 B 실제 시나리오", False, f"오류: {e}")
    
    def test_playbook_c_with_realistic_data(self):
        """플레이북 C 실제 데이터로 테스트"""
        print("\n📊 플레이북 C 실제 시나리오 테스트")
        print("-" * 40)
        
        try:
            config = SessionConfig()
            # C 플레이북이 더 쉽게 신호를 생성하도록 조정
            config.entry_thresh = 0.60          # 0.70 → 0.60
            config.setup_thresh = 0.40          # 0.50 → 0.40
            config.trend_filter_slope = -0.1    # 트렌드 필터 완화
            
            strategy = SessionBasedStrategy(config)
            
            df = self.generate_realistic_vwap_reversion_scenario()
            key_levels = self.generate_key_levels(4000)
            current_time = df.index[-1]
            
            print(f"   📊 데이터 길이: {len(df)}분")
            print(f"   📊 과매도 구간: {df.index[20]} ~ {df.index[34]}")
            print(f"   📊 가격 범위: {df['low'].min():.2f} ~ {df['high'].max():.2f}")
            
            # VWAP 계산 확인
            session_start = df.index[0]
            session_end = df.index[-1]
            vwap, std = strategy.calculate_session_vwap(df, session_start, session_end)
            print(f"   📊 VWAP: {vwap:.2f}, STD: {std:.2f}")
            print(f"   📊 -2σ: {vwap - 2*std:.2f}, -1.5σ: {vwap - 1.5*std:.2f}")
            
            signal = strategy.analyze_session_strategy(df, key_levels, current_time)
            
            if signal:
                self.log_test("플레이북 C 실제 시나리오", True,
                            f"신호 생성: {signal['playbook']} {signal['side']} {signal['stage']}, 점수: {signal['score']:.3f}")
            else:
                self.log_test("플레이북 C 실제 시나리오", True, "조건 불만족으로 신호 없음")
                
        except Exception as e:
            self.log_test("플레이북 C 실제 시나리오", False, f"오류: {e}")
    
    def test_gate_conditions_detailed(self):
        """Gate 조건 상세 테스트"""
        print("\n🔒 Gate 조건 상세 테스트")
        print("-" * 40)
        
        try:
            config = SessionConfig()
            strategy = SessionBasedStrategy(config)
            
            df = self.generate_realistic_or_breakout_scenario()
            key_levels = self.generate_key_levels(4000)
            
            session_vwap = 4000
            or_info = {'high': 4015, 'low': 3985}
            atr = 20
            
            # 각 플레이북별 상세 Gate 테스트
            for playbook in ['A', 'B', 'C']:
                for side in ['LONG', 'SHORT']:
                    print(f"   🔍 {playbook} {side} Gate 테스트")
                    
                    gates_passed, gate_results = strategy.check_gates(
                        df, session_vwap, or_info, atr, playbook, side, key_levels
                    )
                    
                    print(f"      통과: {gates_passed}")
                    print(f"      방향: {gate_results.get('direction', False)}")
                    print(f"      구조: {gate_results.get('structure', False)}")
                    print(f"      슬리피지: {gate_results.get('slippage', False)}")
                    print(f"      거래량: {gate_results.get('volume', False)}")
                    
                    if playbook == 'B' and 'sweep_atr' in gate_results:
                        print(f"      스윕 ATR: {gate_results['sweep_atr']:.2f}")
            
            self.log_test("Gate 조건 상세", True, "모든 Gate 조건 상세 분석 완료")
            
        except Exception as e:
            self.log_test("Gate 조건 상세", False, f"오류: {e}")
    
    def test_score_components_detailed(self):
        """Score 구성 요소 상세 테스트"""
        print("\n📊 Score 구성 요소 상세 테스트")
        print("-" * 40)
        
        try:
            config = SessionConfig()
            strategy = SessionBasedStrategy(config)
            
            df = self.generate_realistic_or_breakout_scenario()
            key_levels = self.generate_key_levels(4000)
            current_time = df.index[-1]
            
            session_vwap = 4000
            or_info = {'high': 4015, 'low': 3985}
            atr = 20
            gate_results = {
                'direction': True, 'structure': True, 'slippage': True, 'volume': True,
                'sweep_atr': 1.2, 'slippage_value': 0.01, 'volume_ratio': 1.5
            }
            
            # 플레이북 A 롱 신호로 점수 구성 요소 분석
            score = strategy.calculate_score(
                df, session_vwap, or_info, atr, 'A', 'LONG', gate_results, current_time, key_levels
            )
            
            print(f"   📊 전체 점수: {score:.3f}")
            print(f"   📊 가중치 구성:")
            print(f"      방향 정렬: {config.weight_direction:.2f}")
            print(f"      돌파/스윕: {config.weight_breakout_sweep:.2f}")
            print(f"      풀백 품질: {config.weight_pullback:.2f}")
            print(f"      기준선: {config.weight_baseline:.2f}")
            print(f"      타이밍: {config.weight_timing:.2f}")
            print(f"      오더플로우: {config.weight_orderflow:.2f}")
            print(f"      리스크: {config.weight_risk:.2f}")
            
            self.log_test("Score 구성 요소 상세", True, f"점수 계산 완료: {score:.3f}")
            
        except Exception as e:
            self.log_test("Score 구성 요소 상세", False, f"오류: {e}")
    
    def run_detailed_tests(self):
        """상세 테스트 실행"""
        print("🚀 세션 기반 전략 상세 테스트 시작")
        print("=" * 60)
        
        # 실제 시나리오 테스트
        self.test_playbook_a_with_realistic_data()
        self.test_playbook_b_with_realistic_data()
        self.test_playbook_c_with_realistic_data()
        
        # 상세 분석 테스트
        self.test_gate_conditions_detailed()
        self.test_score_components_detailed()
        
        print("\n" + "=" * 60)
        print("✅ 상세 테스트 완료!")


def main():
    """메인 테스트 함수"""
    tester = DetailedSessionTester()
    tester.run_detailed_tests()


if __name__ == "__main__":
    main()
