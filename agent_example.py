
# 실제 사용 예시
from dataclasses import dataclass
import numpy as np
import pandas as pd

from collections import deque, namedtuple
import random
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple, Any, Optional

from config.integrated_config import IntegratedConfig
from data.data_manager import get_data_manager


def load_ethusdc_data():
    """ETHUSDC CSV 데이터 로드 - 3분, 15분, 1시간봉"""
    try:
        required_columns = [ 'open', 'high', 'low', 'close', 'volume', 'quote_volume']
        # 3분봉 데이터 로드
        df_3m = pd.read_csv('data/ETHUSDC_3m_historical_data.csv')
        df_3m['timestamp'] = pd.to_datetime(df_3m['timestamp'])
        df_3m = df_3m.set_index('timestamp')
        df_3m = df_3m[required_columns]

        df_15m = pd.read_csv('data/ETHUSDC_15m_historical_data.csv')
        df_15m['timestamp'] = pd.to_datetime(df_15m['timestamp'])
        df_15m = df_15m.set_index('timestamp')
        df_15m = df_15m[required_columns]

        # 3분봉에서 1시간봉 생성
        df_1h = pd.read_csv('data/ETHUSDC_1h_historical_data.csv')
        df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'])
        df_1h = df_1h.set_index('timestamp')
        df_1h = df_1h[required_columns]

        print(f"✅ ETHUSDC 3분봉 데이터 로드 완료: {len(df_3m)}개 캔들")
        print(f"✅ ETHUSDC 15분봉 데이터 생성 완료: {len(df_15m)}개 캔들")
        print(f"✅ ETHUSDC 1시간봉 데이터 생성 완료: {len(df_1h)}개 캔들")
        
        return df_3m, df_15m, df_1h

    except FileNotFoundError as e:
        print(f"❌ 데이터 파일을 찾을 수 없습니다: {e}")
        return None, None, None
    except Exception as e:
        print(f"❌ 데이터 로드 중 오류 발생: {e}")
        return None, None, None

def generate_signal_data_with_indicators(price_data: pd.DataFrame, price_data_15m: pd.DataFrame, 
                                        price_data_1h: pd.DataFrame, max_periods: int = 1000):
    """CSV 데이터로 실제 지표 업데이트 및 전략 실행 (3분, 15분, 1시간봉 사용)"""
    from data.strategy_executor import StrategyExecutor
    from engines.trade_decision_engine import TradeDecisionEngine
    from data.candle_creator import CandleCreator
    from data.data_manager import get_data_manager
    from indicators.global_indicators import get_global_indicator_manager
    from utils.time_manager import get_time_manager
    
    # 컴포넌트 초기화
    strategy_executor = StrategyExecutor()
    decision_engine = TradeDecisionEngine()
    global_manager = get_global_indicator_manager()
    time_manager = get_time_manager()
    data_manager = get_data_manager()

    signal_data = []
    
    print("🔄 CSV 데이터로 지표 업데이트 및 전략 실행 중...")
    print(f"   - 3분봉: {len(price_data)}개 캔들")
    print(f"   - 15분봉: {len(price_data_15m)}개 캔들")
    print(f"   - 1시간봉: {len(price_data_1h)}개 캔들")
    
    # 최근 데이터부터 처리 (최대 max_periods개)
    config = IntegratedConfig()
    start_idx = config.agent_start_idx

    data_manager.load_initial_data(symbol='ETHUSDC', df_3m=price_data[:start_idx], df_15m=price_data_15m[:start_idx], df_1h=price_data_1h[:start_idx])
    global_manager.initialize_indicators(target_time=price_data.iloc[start_idx].name)

    print(global_manager.get_indicator('atr'))
    for i in range(start_idx, len(price_data)):
        # 현재 캔들 데이터
        series_3m = price_data.iloc[i]
        
        # 글로벌 지표 업데이트
        global_manager.update_all_indicators(series_3m)
        
        # 전략 실행
        strategy_executor.execute_all_strategies()
        
        # 신호 수집
        signals = strategy_executor.get_signals()
        
        # 거래 결정
        decision = decision_engine.decide_trade_realtime(signals)
        
        signal_data.append(decision)
        
        if (i - start_idx) % 100 == 0:
            print(f"   진행률: {i - start_idx + 1}/{max_periods} ({((i - start_idx + 1) / max_periods) * 100:.1f}%)")
                
    
    print(f"✅ 신호 데이터 생성 완료: {len(signal_data)}개")
    return signal_data

def main_example():
    """강화학습 트레이딩 AI 사용 예시 - 실제 바이낸스 데이터 사용"""
    
    print("=== 강화학습 트레이딩 AI 훈련 시작 (실제 데이터) ===")
    
    # 1. 실제 ETHUSDC 데이터 로드 (3분, 15분, 1시간봉)
    price_data, price_data_15m, price_data_1h = load_ethusdc_data()
    
    # if price_data_3m is None:
    #     print("❌ 데이터 로드 실패. 프로그램을 종료합니다.")
    #     return None, None, None
    
    # 2. 가격 데이터 전처리 (3분봉을 메인으로 사용)
    # price_data = price_data_3m.reset_index()
    # price_data = price_data.rename(columns={'timestamp': 'timestamp'})
    
    # 필요한 컬럼만 선택
    
    print(f"📊 가격 데이터 정보:")
    print(f"   - 총 캔들 수: {len(price_data)}개")
    print(f"   - 가격 범위: ${price_data['close'].min():.2f} ~ ${price_data['close'].max():.2f}")
    
    # 3. CSV 데이터로 실제 지표 업데이트 및 전략 실행 (3분, 15분, 1시간봉 사용)
    signal_data = generate_signal_data_with_indicators(price_data, price_data_15m, price_data_1h, 
                                                      max_periods=min(1000, len(price_data)))
    
    if not signal_data:
        print("❌ 신호 데이터 생성 실패. 프로그램을 종료합니다.")
        return None, None, None
    
    print("=== 강화학습 에이전트 훈련 시작 ===")
    
    # 4. 에이전트 훈련 (에피소드 수 조정)
    # agent, rewards = train_rl_agent(price_data, signal_data, episodes=200)
    
    print("\n=== 훈련 완료, 성능 평가 중 ===")
    
    # 5. 성능 평가
    # eval_results = evaluate_agent(agent, price_data, signal_data, episodes=10)
    
    # 6. 성능 분석
    # analyzer = BacktestAnalyzer()
    # metrics = analyzer.calculate_performance_metrics(eval_results)
    # report = analyzer.generate_report(eval_results, metrics)
    
    # print(report)
    
    # 7. 모델 저장
    #agent.save_model('ethusdc_crypto_rl_model.pth')
    print("\n모델이 'ethusdc_crypto_rl_model.pth'에 저장되었습니다.")
    
    # return agent, eval_results, metrics



if __name__ == "__main__":
    # 예시 실행
    main_example()