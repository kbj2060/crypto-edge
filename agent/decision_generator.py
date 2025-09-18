# 실제 사용 예시
import pandas as pd
from datetime import timedelta
import os
import sys
import pickle
from typing import Dict, Any, List, Optional

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from managers.strategy_executor import StrategyExecutor
from managers.data_manager import get_data_manager
from engines.trade_decision_engine import TradeDecisionEngine
from indicators.global_indicators import get_all_indicators, get_global_indicator_manager

def flatten_decision_data(decision_data: Dict[str, Any]) -> Dict[str, Any]:
    """복잡한 중첩 구조를 평면화하여 Parquet 저장에 최적화"""
    import numpy as np
    
    def safe_convert(value):
        """numpy 타입을 Python 기본 타입으로 안전하게 변환"""
        if isinstance(value, (np.integer, np.floating)):
            return float(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        return value
    
    flattened = {}
    
    # 기본 정보
    flattened['timestamp'] = decision_data.get('timestamp')
    
    # indicators 정보
    indicators = decision_data.get('indicators', {})
    for key, value in indicators.items():
        flattened[f'indicator_{key}'] = safe_convert(value)
    
    # decisions 정보를 각 카테고리별로 평면화
    decisions = decision_data.get('decisions', {})
    
    for category_name, category_data in decisions.items():
        prefix = f"{category_name.lower()}_"
        
        # RL 변환된 데이터에서 기본 정보 추출
        flattened[f'{prefix}action'] = safe_convert(category_data.get('action_value'))
        flattened[f'{prefix}net_score'] = safe_convert(category_data.get('net_score'))
        flattened[f'{prefix}leverage'] = safe_convert(category_data.get('leverage'))
        flattened[f'{prefix}max_holding_minutes'] = safe_convert(category_data.get('max_holding_minutes'))
        
        # 신뢰도 및 시장 상황
        flattened[f'{prefix}confidence'] = safe_convert(category_data.get('confidence_value'))
        flattened[f'{prefix}market_context'] = safe_convert(category_data.get('market_context_value'))
        
        # 점수 정보
        flattened[f'{prefix}buy_score'] = safe_convert(category_data.get('buy_score'))
        flattened[f'{prefix}sell_score'] = safe_convert(category_data.get('sell_score'))
        flattened[f'{prefix}signals_used'] = safe_convert(category_data.get('signals_used'))
        
        # 충돌 및 보너스 정보
        flattened[f'{prefix}conflicts_detected_count'] = safe_convert(category_data.get('conflicts_detected_count'))
        flattened[f'{prefix}bonuses_applied_count'] = safe_convert(category_data.get('bonuses_applied_count'))
        
        # 포지션 크기 정보
        flattened[f'{prefix}risk_multiplier'] = safe_convert(category_data.get('risk_multiplier'))
        flattened[f'{prefix}risk_usd'] = safe_convert(category_data.get('risk_usd'))
        
        # 전략 사용 정보
        flattened[f'{prefix}strategies_count'] = safe_convert(category_data.get('strategies_count'))
        
        # 카테고리별 특화 정보
        if category_name == 'short_term':
            flattened[f'{prefix}momentum_strength'] = safe_convert(category_data.get('momentum_strength'))
            flattened[f'{prefix}reversion_potential'] = safe_convert(category_data.get('reversion_potential'))
        elif category_name == 'medium_term':
            flattened[f'{prefix}trend_strength'] = safe_convert(category_data.get('trend_strength'))
            flattened[f'{prefix}consolidation_level'] = safe_convert(category_data.get('consolidation_level'))
        elif category_name == 'long_term':
            flattened[f'{prefix}institutional_bias'] = safe_convert(category_data.get('institutional_bias'))
            flattened[f'{prefix}macro_trend_strength'] = safe_convert(category_data.get('macro_trend_strength'))
    
    # conflicts 정보 (딕셔너리 형태로 처리)
    conflicts = decision_data.get('conflicts', {})
    for key, value in conflicts.items():
        flattened[f'conflict_{key}'] = safe_convert(value)

    return flattened

def safe_concat(existing_df, new_df):
    if existing_df is None or existing_df.empty:
        return new_df.copy() if not new_df.empty else pd.DataFrame()
    elif new_df.empty:
        return existing_df.copy()
    else:
        # 컬럼 일치 확인
        if list(existing_df.columns) != list(new_df.columns):
            # 공통 컬럼만 사용
            common_cols = list(set(existing_df.columns) & set(new_df.columns))
            if common_cols:
                existing_df = existing_df[common_cols]
                new_df = new_df[common_cols]
            else:
                return existing_df.copy()
        
        # FutureWarning 방지: 빈 DataFrame이나 모든 NA인 컬럼 처리
        if existing_df.empty or new_df.empty:
            return existing_df if not existing_df.empty else new_df
        
        # 모든 NA인 컬럼 제거
        existing_df_clean = existing_df.dropna(axis=1, how='all')
        new_df_clean = new_df.dropna(axis=1, how='all')
        
        # 공통 컬럼만 유지
        common_cols = list(set(existing_df_clean.columns) & set(new_df_clean.columns))
        if not common_cols:
            return existing_df_clean if not existing_df_clean.empty else new_df_clean
        
        existing_df_clean = existing_df_clean[common_cols]
        new_df_clean = new_df_clean[common_cols]
        
        return pd.concat([existing_df_clean, new_df_clean], ignore_index=True)

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
                    combined_df = safe_concat(existing_df, new_df)

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

def inspect_parquet_structure(filename: str = "agent/decisions_data.parquet") -> None:
    """Parquet 파일의 구조를 확인하는 함수"""
    try:
        if not os.path.exists(filename):
            print(f"파일을 찾을 수 없습니다: {filename}")
            return
            
        df = pd.read_parquet(filename)
        print(f"\n=== Parquet 파일 구조 분석 ===")
        print(f"파일: {filename}")
        print(f"총 레코드 수: {len(df)}")
        print(f"컬럼 수: {len(df.columns)}")
        print(f"메모리 사용량: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        print(f"\n컬럼 목록:")
        for i, col in enumerate(df.columns):
            non_null_count = df[col].count()
            print(f"  {i+1:2d}. {col:<30} (non-null: {non_null_count}/{len(df)})")
        
        print(f"\n첫 번째 레코드 샘플:")
        if len(df) > 0:
            sample_record = df.iloc[0].to_dict()
            for key, value in list(sample_record.items())[:10]:  # 처음 10개만 표시
                print(f"  {key}: {value}")
            if len(sample_record) > 10:
                print(f"  ... (총 {len(sample_record)}개 필드)")
        
        print(f"\n데이터 타입:")
        print(df.dtypes)
        
    except Exception as e:
        print(f"Parquet 구조 분석 오류: {e}")

def convert_parquet_to_signal_data(
    df: pd.DataFrame, 
    max_samples: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> List[Dict[str, Any]]:
    """DataFrame을 signal_data 리스트로 효율적으로 변환
    
    Args:
        df: 변환할 DataFrame (이미 딕셔너리 형태로 저장됨)
        max_samples: 최대 샘플 수 (None이면 전체 사용)
        start_date: 시작 날짜 (YYYY-MM-DD 형식)
        end_date: 종료 날짜 (YYYY-MM-DD 형식)
    """
    try:
        # 날짜 필터링
        if start_date or end_date:
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                if start_date:
                    df = df[df['timestamp'] >= start_date]
                if end_date:
                    df = df[df['timestamp'] <= end_date]
                print(f"날짜 필터링 후: {len(df)}개 레코드")
        
        # 샘플 수 제한
        if max_samples and len(df) > max_samples:
            # 최근 데이터부터 샘플링
            df = df.tail(max_samples)
            print(f"샘플링 후: {len(df)}개 레코드")
        
        # NaN 값을 None으로 변환하고 딕셔너리 리스트로 변환
        df_clean = df.where(pd.notnull(df), None)
        signal_data = df_clean.to_dict('records')
        
        print(f"Parquet 데이터를 signal_data로 변환 완료: {len(signal_data)}개 레코드")
        return signal_data
        
    except Exception as e:
        print(f"signal_data 변환 오류: {e}")
        return []

def load_signal_data_directly(
    filename: str = "agent/decisions_data.parquet",
    max_samples: Optional[int] = 5000,  # 기본값 5000개로 제한
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Parquet 파일에서 직접 signal_data를 로드하는 간단한 함수"""
    try:
        # parquet 파일 로드
        df = pd.read_parquet(filename)
        
        # 날짜 필터링
        if start_date or end_date:
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                if start_date:
                    df = df[df['timestamp'] >= start_date]
                if end_date:
                    df = df[df['timestamp'] <= end_date]
        
        # 샘플 수 제한
        if max_samples and len(df) > max_samples:
            df = df.tail(max_samples)
        
        # NaN을 None으로 변환하고 딕셔너리 리스트로 변환
        signal_data = df.where(pd.notnull(df), None).to_dict('records')
        
        print(f"Parquet에서 signal_data 직접 로드 완료: {len(signal_data)}개 레코드")
        return signal_data
        
    except Exception as e:
        print(f"signal_data 직접 로드 오류: {e}")
        return []

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
    batch_size = 100  # 500개씩 배치로 저장 (Parquet은 더 큰 배치가 효율적)
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
            indicators = get_all_indicators()
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

def check_existing_decision_data() -> bool:
    """기존 decision_data가 있는지 확인"""
    try:
        df = load_decisions_from_parquet()
        if df is not None and not df.empty:
            print(f"✅ 기존 Decision 데이터 발견: {len(df)}개 레코드")
            print(f"   - 시간 범위: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
            print(f"   - 컬럼 수: {len(df.columns)}")
            return True
        else:
            print("❌ 기존 Decision 데이터가 없습니다.")
            return False
    except Exception as e:
        print(f"❌ Decision 데이터 확인 중 오류: {e}")
        return False

def analyze_decision_data(df: pd.DataFrame) -> None:
    """Decision 데이터 분석 및 통계 출력"""
    print("\n=== Decision 데이터 분석 ===")
    
    if df.empty:
        print("분석할 데이터가 없습니다.")
        return
    
    print(f"총 레코드 수: {len(df)}")
    print(f"컬럼 수: {len(df.columns)}")
    print(f"시간 범위: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
    
    # 각 카테고리별 액션 분포 분석 (수치값을 문자열로 변환)
    categories = ['short_term', 'medium_term', 'long_term']
    action_mapping = {1.0: 'LONG', -1.0: 'SHORT', 0.0: 'HOLD'}
    
    for category in categories:
        action_col = f'{category}_action'
        if action_col in df.columns:
            # 수치값을 문자열로 변환
            action_values = df[action_col].map(action_mapping).fillna('UNKNOWN')
            action_counts = action_values.value_counts()
            print(f"\n{category.upper()} 액션 분포:")
            for action, count in action_counts.items():
                percentage = (count / len(df)) * 100
                print(f"  {action}: {count}개 ({percentage:.1f}%)")
    
    # 신뢰도 분포 분석 (수치값을 문자열로 변환)
    confidence_mapping = {0.8: 'HIGH', 0.5: 'MEDIUM', 0.2: 'LOW'}
    
    for category in categories:
        confidence_col = f'{category}_confidence'
        if confidence_col in df.columns:
            # 수치값을 문자열로 변환
            conf_values = df[confidence_col].map(confidence_mapping).fillna('UNKNOWN')
            confidence_counts = conf_values.value_counts()
            print(f"\n{category.upper()} 신뢰도 분포:")
            for conf, count in confidence_counts.items():
                percentage = (count / len(df)) * 100
                print(f"  {conf}: {count}개 ({percentage:.1f}%)")
    
    # 지표 데이터 완성도 분석
    indicator_cols = [col for col in df.columns if col.startswith('indicator_')]
    print(f"\n지표 데이터 완성도:")
    for col in indicator_cols:
        non_null_count = df[col].count()
        percentage = (non_null_count / len(df)) * 100
        print(f"  {col}: {non_null_count}/{len(df)} ({percentage:.1f}%)")
    
    # 카테고리별 특화 지표 분석
    print(f"\n카테고리별 특화 지표:")
    for category in categories:
        prefix = f'{category}_'
        special_cols = [col for col in df.columns if col.startswith(prefix) and 
                       col.endswith(('_strength', '_potential', '_level', '_bias'))]
        if special_cols:
            print(f"  {category.upper()}:")
            for col in special_cols:
                non_null_count = df[col].count()
                percentage = (non_null_count / len(df)) * 100
                print(f"    {col}: {non_null_count}/{len(df)} ({percentage:.1f}%)")
    
    # 충돌 정보 분석
    conflict_cols = [col for col in df.columns if col.startswith('conflict_')]
    if conflict_cols:
        print(f"\n충돌 정보:")
        for col in conflict_cols:
            if col in ['conflict_conflicts_count', 'conflict_conflicts_details']:
                continue  # 이전 방식의 컬럼은 스킵
            non_null_count = df[col].count()
            if non_null_count > 0:
                if df[col].dtype in ['float64', 'int64']:
                    stats = df[col].describe()
                    print(f"  {col}: 평균={stats['mean']:.3f}, 최대={stats['max']:.3f}, 최소={stats['min']:.3f}")
                else:
                    print(f"  {col}: {non_null_count}개 값")
    
    # 기존 방식의 충돌 정보 (하위 호환성)
    if 'conflicts_count' in df.columns:
        conflict_stats = df['conflicts_count'].describe()
        print(f"\n기존 충돌 정보:")
        print(f"  평균 충돌 수: {conflict_stats['mean']:.2f}")
        print(f"  최대 충돌 수: {conflict_stats['max']:.0f}")
        print(f"  충돌이 있는 레코드: {(df['conflicts_count'] > 0).sum()}개")

def main_example():
    """Decision 데이터 생성 전용 - Parquet 저장 및 재시작 지원"""
    print("\n📊 Decision 데이터를 생성합니다...")
    
    # 1. 기존 데이터 확인
    if check_existing_decision_data():
        response = input("기존 데이터가 있습니다. 새로 생성하시겠습니까? (y/N): ")
        if response.lower() != 'y':
            print("기존 데이터를 사용합니다.")
            df = load_decisions_from_parquet()
            if df is not None:
                analyze_decision_data(df)
            return
    
    # 2. 실제 ETHUSDC 데이터 로드 (3분, 15분, 1시간봉)
    price_data, price_data_15m, price_data_1h = load_ethusdc_data()
    
    if price_data is None:
        print("데이터 로드 실패. 프로그램을 종료합니다.")
        return
    
    print(f"가격 데이터 정보:")
    print(f"   - 총 캔들 수: {len(price_data)}개")
    print(f"   - 가격 범위: ${price_data['close'].min():.2f} ~ ${price_data['close'].max():.2f}")
    
    # 3. CSV 데이터로 실제 지표 업데이트 및 전략 실행 (재시작 지원)
    success = generate_signal_data_with_indicators(price_data, price_data_15m, price_data_1h, resume_from_progress=True)

    if success:
        print("\n✅ Decision 데이터가 Parquet 파일로 저장되었습니다.")
        
        # 저장된 데이터 확인 및 분석
        df = load_decisions_from_parquet()
        if df is not None:
            print(f"\n저장된 데이터 요약:")
            print(f"   - 총 레코드 수: {len(df)}")
            print(f"   - 시간 범위: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
            print(f"   - 컬럼 수: {len(df.columns)}")
            
            # 상세 분석
            analyze_decision_data(df)
            
            # Parquet 구조 확인
            inspect_parquet_structure()
            
            print(f"\n🎯 Decision 데이터 생성 완료!")
            print(f"   - 파일 위치: agent/decisions_data.parquet")
            print(f"   - 이 데이터를 강화학습이나 다른 분석에 사용할 수 있습니다.")
        
    else:
        print("❌ 데이터 생성이 완료되지 않았습니다. 나중에 재시작할 수 있습니다.")
        print("   - 진행 상태가 저장되어 있어서 다음에 이어서 실행할 수 있습니다.")

if __name__ == "__main__":
    # 예시 실행
    main_example()