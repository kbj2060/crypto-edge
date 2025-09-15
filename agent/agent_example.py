# 실제 사용 예시
import pandas as pd
from datetime import timedelta
import os
import sys
import pickle
from typing import Dict, Any, List, Optional

# 상위 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def flatten_decision_data(decision_data: Dict[str, Any]) -> Dict[str, Any]:
    """복잡한 중첩 구조를 평면화하여 Parquet 저장에 최적화"""
    flattened = {}
    
    # 기본 정보
    flattened['timestamp'] = decision_data.get('timestamp')
    
    # indicators 정보
    indicators = decision_data.get('indicators', {})
    for key, value in indicators.items():
        flattened[f'indicator_{key}'] = value
    
    # decisions 정보를 각 카테고리별로 평면화
    decisions = decision_data.get('decisions', {})
    
    for category_name, category_data in decisions.items():
        prefix = f"{category_name.lower()}_"
        
        # 기본 정보
        flattened[f'{prefix}action'] = category_data.get('action')
        flattened[f'{prefix}net_score'] = category_data.get('net_score')
        flattened[f'{prefix}leverage'] = category_data.get('leverage')
        flattened[f'{prefix}max_holding_minutes'] = category_data.get('max_holding_minutes')
        flattened[f'{prefix}reason'] = category_data.get('reason')
        
        # sizing 정보
        sizing = category_data.get('sizing', {})
        flattened[f'{prefix}qty'] = sizing.get('qty')
        flattened[f'{prefix}risk_usd'] = sizing.get('risk_usd')
        flattened[f'{prefix}entry_used'] = sizing.get('entry_used')
        flattened[f'{prefix}stop_used'] = sizing.get('stop_used')
        flattened[f'{prefix}risk_multiplier'] = sizing.get('risk_multiplier')
        
        # meta 정보
        meta = category_data.get('meta', {})
        flattened[f'{prefix}timeframe'] = meta.get('timeframe')
        
        # synergy_meta 정보
        synergy_meta = meta.get('synergy_meta', {})
        flattened[f'{prefix}confidence'] = synergy_meta.get('confidence')
        flattened[f'{prefix}market_context'] = synergy_meta.get('market_context')
        flattened[f'{prefix}buy_score'] = synergy_meta.get('buy_score')
        flattened[f'{prefix}sell_score'] = synergy_meta.get('sell_score')
        flattened[f'{prefix}signals_used'] = synergy_meta.get('signals_used')
        
        # 장기 전략 추가 정보
        if category_name == 'LONG_TERM':
            flattened[f'{prefix}institutional_bias'] = synergy_meta.get('institutional_bias')
            flattened[f'{prefix}macro_trend_strength'] = synergy_meta.get('macro_trend_strength')
        
        # raw 전략 데이터를 JSON 문자열로 저장 (필요시)
        raw_data = category_data.get('raw', {})
        # 주요 전략들만 개별 컬럼으로 저장
        for strategy_name, strategy_data in raw_data.items():
            if isinstance(strategy_data, dict):
                flattened[f'{prefix}raw_{strategy_name.lower()}_action'] = strategy_data.get('action')
                flattened[f'{prefix}raw_{strategy_name.lower()}_score'] = strategy_data.get('score')
                flattened[f'{prefix}raw_{strategy_name.lower()}_entry'] = strategy_data.get('entry')
                flattened[f'{prefix}raw_{strategy_name.lower()}_stop'] = strategy_data.get('stop')
    
    return flattened

def save_decisions_to_parquet(
    decision_data_list: List[Dict[str, Any]], 
    filename: str = "agent/decisions_data.parquet",
    append: bool = True
):
    """Decision 데이터를 Parquet 파일로 저장"""
    try:
        if not decision_data_list:
            print("저장할 데이터가 없습니다.")
            return False
        
        # 데이터 평면화
        flattened_data = [flatten_decision_data(decision) for decision in decision_data_list]
        new_df = pd.DataFrame(flattened_data)
        
        # timestamp를 datetime 타입으로 변환
        if 'timestamp' in new_df.columns:
            new_df['timestamp'] = pd.to_datetime(new_df['timestamp'])
        
        # 기존 파일이 있고 append 모드인 경우
        if append and os.path.exists(filename):
            try:
                existing_df = pd.read_parquet(filename)
                
                # 중복 제거 (timestamp 기준)
                if 'timestamp' in existing_df.columns and 'timestamp' in new_df.columns:
                    # 기존 데이터의 마지막 timestamp 이후 데이터만 추가
                    last_timestamp = existing_df['timestamp'].max()
                    new_df = new_df[new_df['timestamp'] > last_timestamp]
                
                if not new_df.empty:
                    # 컬럼 순서 맞추기
                    common_columns = list(set(existing_df.columns) & set(new_df.columns))
                    new_columns = [col for col in new_df.columns if col not in existing_df.columns]
                    
                    # 기존 DataFrame에 새 컬럼 추가 (NaN으로 채워짐)
                    for col in new_columns:
                        existing_df[col] = None
                    
                    # 새 DataFrame에 기존 컬럼 추가 (NaN으로 채워짐)
                    for col in existing_df.columns:
                        if col not in new_df.columns:
                            new_df[col] = None
                    
                    # 컬럼 순서 맞추기
                    all_columns = list(existing_df.columns)
                    new_df = new_df.reindex(columns=all_columns)
                    
                    # 데이터 합치기
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                else:
                    combined_df = existing_df
                    print("새로 추가할 데이터가 없습니다 (중복 제거됨)")
            except Exception as e:
                print(f"기존 파일 읽기 실패, 새 파일로 저장: {e}")
                combined_df = new_df
        else:
            combined_df = new_df
        
        # Parquet 파일로 저장 (압축 적용)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        combined_df.to_parquet(filename, compression='snappy', index=False)
        
        print(f"Decision 데이터 저장 완료: {filename} ({len(combined_df)}개 레코드)")
        print(f"파일 크기: {os.path.getsize(filename) / 1024 / 1024:.2f} MB")
        return True
        
    except Exception as e:
        print(f"Parquet 저장 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def load_decisions_from_parquet(filename: str = "agent/decisions_data.parquet") -> Optional[pd.DataFrame]:
    """Parquet 파일에서 Decision 데이터 로드"""
    try:
        if not os.path.exists(filename):
            print(f"파일을 찾을 수 없습니다: {filename}")
            return None
            
        df = pd.read_parquet(filename)
        print(f"Decision 데이터 로드 완료: {filename} ({len(df)}개 레코드)")
        return df
        
    except Exception as e:
        print(f"Parquet 로드 오류: {e}")
        return None

def save_progress_state(current_index: int, total_count: int, filename: str = "agent/progress_state.pkl"):
    """진행 상태 저장"""
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        state = {
            'current_index': current_index,
            'total_count': total_count,
            'timestamp': pd.Timestamp.now()
        }
        with open(filename, 'wb') as f:
            pickle.dump(state, f)
        print(f"진행 상태 저장: {current_index}/{total_count}")
    except Exception as e:
        print(f"진행 상태 저장 오류: {e}")

def load_progress_state(filename: str = "agent/progress_state.pkl") -> Optional[Dict[str, Any]]:
    """진행 상태 로드"""
    try:
        if not os.path.exists(filename):
            return None
        
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        
        print(f"진행 상태 복원: {state['current_index']}/{state['total_count']} "
              f"(저장 시간: {state['timestamp']})")
        return state
    except Exception as e:
        print(f"진행 상태 로드 오류: {e}")
        return None

def clear_progress_state(filename: str = "agent/progress_state.pkl"):
    """진행 상태 파일 삭제"""
    try:
        if os.path.exists(filename):
            os.remove(filename)
            print("진행 상태 파일 삭제 완료")
    except Exception as e:
        print(f"진행 상태 파일 삭제 오류: {e}")

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

        print(f"ETHUSDC 3분봉 데이터 로드 완료: {len(df_3m)}개 캔들")
        print(f"ETHUSDC 15분봉 데이터 생성 완료: {len(df_15m)}개 캔들")
        print(f"ETHUSDC 1시간봉 데이터 생성 완료: {len(df_1h)}개 캔들")
        
        return df_3m, df_15m, df_1h

    except FileNotFoundError as e:
        print(f"데이터 파일을 찾을 수 없습니다: {e}")
        return None, None, None
    except Exception as e:
        print(f"데이터 로드 중 오류 발생: {e}")
        return None, None, None

def generate_signal_data_with_indicators(
    price_data: pd.DataFrame, 
    price_data_15m: pd.DataFrame, 
    price_data_1h: pd.DataFrame,
    resume_from_progress: bool = True
):
    """CSV 데이터로 실제 지표 업데이트 및 전략 실행 (중단점 재시작 지원)"""
    from data.strategy_executor import StrategyExecutor
    from data.data_manager import get_data_manager
    from engines.trade_decision_engine import TradeDecisionEngine
    from indicators.global_indicators import get_global_indicator_manager
    from indicators.global_indicators import get_atr, get_daily_levels, get_opening_range, get_vpvr, get_vwap

    # 진행 상태 확인
    progress_state = None
    start_idx = None
    
    if resume_from_progress:
        progress_state = load_progress_state()
    
    # 컴포넌트 초기화
    data_manager = get_data_manager()
    
    print("CSV 데이터로 지표 업데이트 및 전략 실행 중...")
    print(f"   - 3분봉: {len(price_data)}개 캔들")
    print(f"   - 15분봉: {len(price_data_15m)}개 캔들")
    print(f"   - 1시간봉: {len(price_data_1h)}개 캔들")
    
    # 시작 위치 결정
    if progress_state:
        start_idx = progress_state['current_index']
        print(f"이전 진행 상태에서 재시작: {start_idx}번째 캔들부터")
    else:
        # 최근 데이터부터 처리 (최대 max_periods개)
        target_datetime = price_data.iloc[0].name + timedelta(days=4)
        start_idx = price_data.index.get_loc(target_datetime)
        print(f"기준 날짜 {target_datetime}의 인덱스 위치: {start_idx}")
    
    # 초기 데이터 로딩
    target_time = price_data.index[start_idx]
    data_manager.load_initial_data(
        symbol='ETHUSDC', 
        df_3m=price_data[price_data.index < target_time], 
        df_15m=price_data_15m[price_data_15m.index < target_time], 
        df_1h=price_data_1h[price_data_1h.index < target_time]
    ) 
    
    global_manager = get_global_indicator_manager(target_time)
    global_manager.initialize_indicators()

    strategy_executor = StrategyExecutor()
    decision_engine = TradeDecisionEngine()

    end_idx = len(price_data)
    batch_size = 500  # 500개씩 배치로 저장 (Parquet은 더 큰 배치가 효율적)
    temp_decision_data = []  # 임시 저장용
    
    try:
        for i in range(start_idx, end_idx):
            # 현재 캔들 데이터
            series_3m = price_data.iloc[i]
            current_time = price_data.index[i]
            
            # 데이터 매니저에 캔들 데이터 업데이트
            data_manager.update_with_candle(series_3m)

            # 15분봉 마감 시간 체크 (15분 단위로 나누어떨어지는 시간)
            if current_time.minute % 15 == 0:
                try:
                    series_15m = price_data_15m.loc[current_time]
                    data_manager.update_with_candle_15m(series_15m)
                except KeyError:
                    pass  # 해당 시간의 15분봉 데이터가 없으면 스킵
            
            # 1시간봉 마감 시간 체크 (정시)
            if current_time.minute == 0:
                try:
                    series_1h = price_data_1h.loc[current_time]
                    data_manager.update_with_candle_1h(series_1h)
                except KeyError:
                    pass  # 해당 시간의 1시간봉 데이터가 없으면 스킵
                    
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

            temp_decision_data.append(decision)
            
            # 배치 크기마다 Parquet 파일에 저장
            if len(temp_decision_data) >= batch_size:
                save_decisions_to_parquet(temp_decision_data)
                temp_decision_data = []  # 임시 데이터 초기화
            
            # 진행 상태 저장 (100개마다)
            if (i - start_idx) % 100 == 0:
                save_progress_state(i, end_idx)
                total_periods = end_idx - start_idx
                print(f"   진행률: {i - start_idx + 1}/{total_periods} ({((i - start_idx + 1) / total_periods) * 100:.1f}%)")
        
        # 남은 decision 데이터가 있으면 저장
        if temp_decision_data:
            save_decisions_to_parquet(temp_decision_data)
        
        # 완료 후 진행 상태 파일 삭제
        clear_progress_state()
        
        total_processed = end_idx - start_idx
        print(f"신호 데이터 생성 완료: {total_processed}개")
        
        return True
        
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")
        # 남은 데이터 저장
        if temp_decision_data:
            save_decisions_to_parquet(temp_decision_data)
            print("중단 전까지의 데이터를 저장했습니다.")
        
        # 진행 상태 저장
        save_progress_state(i, end_idx)
        print("다음에 '--resume' 옵션으로 재시작할 수 있습니다.")
        return False
        
    except Exception as e:
        print(f"처리 중 오류 발생: {e}")
        # 남은 데이터 저장
        if temp_decision_data:
            save_decisions_to_parquet(temp_decision_data)
        
        # 진행 상태 저장
        if 'i' in locals():
            save_progress_state(i, end_idx)
        
        import traceback
        traceback.print_exc()
        return False

def main_example():
    """강화학습 트레이딩 AI 사용 예시 - Parquet 저장 및 재시작 지원"""
    
    print("=== 강화학습 트레이딩 AI 훈련 시작 (Parquet 저장) ===")
    
    # 1. 실제 ETHUSDC 데이터 로드 (3분, 15분, 1시간봉)
    price_data, price_data_15m, price_data_1h = load_ethusdc_data()
    
    if price_data is None:
        print("데이터 로드 실패. 프로그램을 종료합니다.")
        return
    
    print(f"가격 데이터 정보:")
    print(f"   - 총 캔들 수: {len(price_data)}개")
    print(f"   - 가격 범위: ${price_data['close'].min():.2f} ~ ${price_data['close'].max():.2f}")
    
    # 2. CSV 데이터로 실제 지표 업데이트 및 전략 실행 (재시작 지원)
    success = generate_signal_data_with_indicators(price_data, price_data_15m, price_data_1h, resume_from_progress=True)

    if success:
        print("Decision 데이터가 Parquet 파일로 저장되었습니다.")
        
        # 저장된 데이터 확인
        df = load_decisions_from_parquet()
        if df is not None:
            print(f"저장된 데이터 요약:")
            print(f"   - 총 레코드 수: {len(df)}")
            print(f"   - 시간 범위: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
            print(f"   - 컬럼 수: {len(df.columns)}")
    else:
        print("데이터 생성이 완료되지 않았습니다. 나중에 재시작할 수 있습니다.")

if __name__ == "__main__":
    # 예시 실행
    main_example()