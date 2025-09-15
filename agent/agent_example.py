
# 실제 사용 예시
import pandas as pd
from datetime import timedelta
import json
import os
import sys


# 상위 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from agent import BacktestAnalyzer, evaluate_agent, train_rl_agent


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

def save_signals_to_json(signal_data: list, filename: str = "agent/signals_data.json"):
    """신호 데이터를 JSON 파일로 저장 (중복 체크 후 추가)"""
    try:
        # 기존 데이터 로드
        existing_data = []
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = []
        
        # 기존 데이터의 timestamp들을 set으로 저장 (중복 체크용)
        existing_timestamps = set()
        for signal in existing_data:
            if isinstance(signal, dict) and 'timestamp' in signal:
                existing_timestamps.add(signal['timestamp'])
        
        # 새로운 신호 데이터를 직렬화 가능한 형태로 변환
        new_signals = []
        for signal in signal_data:
            if isinstance(signal, dict):
                # datetime 객체를 문자열로 변환
                serialized_signal = {}
                for key, value in signal.items():
                    if hasattr(value, 'isoformat'):  # datetime 객체인 경우
                        serialized_signal[key] = value.isoformat()
                    else:
                        serialized_signal[key] = value
                
                # 중복 체크 (timestamp 기준)
                if serialized_signal.get('timestamp') not in existing_timestamps:
                    new_signals.append(serialized_signal)
                    existing_timestamps.add(serialized_signal.get('timestamp'))
            else:
                new_signals.append(signal)
        
        # 기존 데이터에 새로운 신호 추가
        all_data = existing_data + new_signals
        
        # JSON 파일로 저장
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 신호 데이터 저장 완료: {filename} (기존: {len(existing_data)}개, 추가: {len(new_signals)}개, 총: {len(all_data)}개)")
        return True
        
    except Exception as e:
        print(f"❌ 신호 데이터 저장 오류: {e}")
        return False

def load_signals_from_json(filename: str = "signals_data.json"):
    """JSON 파일에서 신호 데이터 로드"""
    try:
        if not os.path.exists(filename):
            print(f"❌ 파일을 찾을 수 없습니다: {filename}")
            return None
            
        with open(filename, 'r', encoding='utf-8') as f:
            signal_data = json.load(f)
        
        print(f"✅ 신호 데이터 로드 완료: {filename} ({len(signal_data)}개 신호)")
        return signal_data
        
    except Exception as e:
        print(f"❌ 신호 데이터 로드 오류: {e}")
        return None

def generate_signal_data_with_indicators(price_data: pd.DataFrame, price_data_15m: pd.DataFrame, price_data_1h: pd.DataFrame):
    """CSV 데이터로 실제 지표 업데이트 및 전략 실행 (3분, 15분, 1시간봉 사용)"""
    from data.strategy_executor import StrategyExecutor
    from data.data_manager import get_data_manager
    from engines.trade_decision_engine import TradeDecisionEngine
    from indicators.global_indicators import get_global_indicator_manager
    from indicators.global_indicators import get_atr, get_daily_levels, get_opening_range, get_vpvr, get_vwap

    # 컴포넌트 초기화
    data_manager = get_data_manager()
    
    signal_data = []
    
    print("🔄 CSV 데이터로 지표 업데이트 및 전략 실행 중...")
    print(f"   - 3분봉: {len(price_data)}개 캔들")
    print(f"   - 15분봉: {len(price_data_15m)}개 캔들")
    print(f"   - 1시간봉: {len(price_data_1h)}개 캔들")
    
    # 최근 데이터부터 처리 (최대 max_periods개)
    target_datetime = price_data.iloc[0].name + timedelta(days=4)
    
    # 특정 날짜의 인덱스 위치 찾기
    start_idx = price_data.index.get_loc(target_datetime)
    print(f"✅ 기준 날짜 {target_datetime}의 인덱스 위치: {start_idx}")
    data_manager.load_initial_data(
        symbol='ETHUSDC', 
        df_3m=price_data[price_data.index < target_datetime], 
        df_15m=price_data_15m[price_data_15m.index < target_datetime], 
        df_1h=price_data_1h[price_data_1h.index < target_datetime]
        ) 
        
    target_time = price_data.index[start_idx]
    
    global_manager = get_global_indicator_manager(target_time)
    global_manager.initialize_indicators()

    strategy_executor = StrategyExecutor()
    decision_engine = TradeDecisionEngine()

    end_idx = len(price_data)
    batch_size = 100  # 100개씩 배치로 저장
    temp_signal_data = []  # 임시 저장용
    
    for i in range(start_idx, end_idx):
        # 현재 캔들 데이터
        series_3m = price_data.iloc[i]
        current_time = price_data.index[i]
        
        # 데이터 매니저에 캔들 데이터 업데이트
        data_manager.update_with_candle(series_3m)

        # 15분봉 마감 시간 체크 (15분 단위로 나누어떨어지는 시간)
        if current_time.minute % 15 == 0:
            # current_time과 같은 인덱스의 15분봉 데이터 가져오기
            series_15m = price_data_15m.loc[current_time]
            data_manager.update_with_candle_15m(series_15m)
        
        # 1시간봉 마감 시간 체크 (정시)
        if current_time.minute == 0:
            # current_time과 같은 인덱스의 1시간봉 데이터 가져오기
            series_1h = price_data_1h.loc[current_time]
            data_manager.update_with_candle_1h(series_1h)
                
        # 글로벌 지표 업데이트
        global_manager.update_all_indicators(series_3m)
        atr = get_atr()
        poc, hvn, lvn = get_vpvr()
        vwap, vwap_std = get_vwap()
        opening_range_high, opening_range_low = get_opening_range()
        prev_day_high, prev_day_low = get_daily_levels()
        indicators = {
            'atr': atr,
            'poc': poc,
            'hvn': hvn,
            'lvn': lvn,
            'vwap': vwap,
            'vwap_std': vwap_std,
            'opening_range_high': opening_range_high,
            'opening_range_low': opening_range_low,
            'prev_day_high': prev_day_high,
            'prev_day_low': prev_day_low,
        }
        # 전략 실행
        strategy_executor.execute_all_strategies()
        
        # 신호 수집
        signals = strategy_executor.get_signals()
        
        # 거래 결정
        decision = decision_engine.decide_trade_realtime(signals)
        decision.update({'timestamp': current_time, 'indicators': indicators})

        signal_data.append(decision)
        temp_signal_data.append(decision)
        
        # 100개마다 JSON 파일에 저장
        if len(temp_signal_data) >= batch_size:
            save_signals_to_json(temp_signal_data)
            temp_signal_data = []  # 임시 데이터 초기화
        
        if (i - start_idx) % 100 == 0:
            total_periods = end_idx - start_idx
            print(f"   진행률: {i - start_idx + 1}/{total_periods} ({((i - start_idx + 1) / total_periods) * 100:.1f}%)")
    
    # 남은 신호 데이터가 있으면 저장
    if temp_signal_data:
        save_signals_to_json(temp_signal_data)
                
    print(f"✅ 신호 데이터 생성 완료: {len(signal_data)}개")   
    
    return signal_data

def main_example():
    """강화학습 트레이딩 AI 사용 예시 - 실제 바이낸스 데이터 사용"""
    
    print("=== 강화학습 트레이딩 AI 훈련 시작 (실제 데이터) ===")
    
    # 1. 실제 ETHUSDC 데이터 로드 (3분, 15분, 1시간봉)
    price_data, price_data_15m, price_data_1h = load_ethusdc_data()
    
    print(f"📊 가격 데이터 정보:")
    print(f"   - 총 캔들 수: {len(price_data)}개")
    print(f"   - 가격 범위: ${price_data['close'].min():.2f} ~ ${price_data['close'].max():.2f}")
    
    # 3. CSV 데이터로 실제 지표 업데이트 및 전략 실행 (3분, 15분, 1시간봉 사용)
    signal_data = generate_signal_data_with_indicators(price_data, price_data_15m, price_data_1h)

    if not signal_data:
        print("❌ 신호 데이터 생성 실패. 프로그램을 종료합니다.")
        return None, None, None
    
    # 신호 데이터가 이미 JSON 파일로 저장되었음을 알림
    print(f"📁 신호 데이터가 JSON 파일로 저장되었습니다. 에이전트 훈련 시 사용할 수 있습니다.")
    
    print("=== 강화학습 에이전트 훈련 시작 ===")
    
    # 4. 에이전트 훈련 (에피소드 수 조정)
    # try:
    #     agent, rewards = train_rl_agent(price_data, signal_data, episodes=200)
        
    #     print("\n=== 훈련 완료, 성능 평가 중 ===")
        
    #     # 5. 성능 평가
    #     eval_results = evaluate_agent(agent, price_data, signal_data, episodes=10)
        
    #     # 6. 성능 분석
    #     analyzer = BacktestAnalyzer()
    #     metrics = analyzer.calculate_performance_metrics(eval_results)
    #     report = analyzer.generate_report(eval_results, metrics)
        
    #     print(report)
        
    #     # 7. 모델 저장
    #     agent.save_model('ethusdc_crypto_rl_model.pth')
    #     print("\n모델이 'ethusdc_crypto_rl_model.pth'에 저장되었습니다.")
        
    #     return agent, eval_results, metrics
        
    # except Exception as e:
    #     print(f"❌ 에이전트 훈련 중 오류 발생: {e}")
    #     import traceback
    #     traceback.print_exc()
    #     return None, None, None

if __name__ == "__main__":
    # 예시 실행
    main_example()
    
    # JSON 파일에서 에이전트 훈련 (선택사항)
    # train_agent_from_json("signals_data_20240115_100000.json")