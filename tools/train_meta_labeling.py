#!/usr/bin/env python3
"""
메타 라벨링 모델 학습 스크립트

과거 거래 결정 데이터와 가격 데이터를 사용하여
메타 라벨링 모델을 학습합니다.
"""

import pandas as pd
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional
import time

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from engines.meta_labeling_engine import MetaLabelingEngine
from agent.decision_generator import load_decisions_from_parquet
from managers.binance_dataloader import BinanceDataLoader


def load_price_data_from_csv(csv_path: str = "data/ETHUSDT_3m_20240913_20250913.csv") -> pd.DataFrame:
    """CSV 파일에서 가격 데이터 로드"""
    print(f"📊 CSV 파일에서 가격 데이터 로드 중... ({csv_path})")
    
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        
        # timestamp 처리
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            df = df.set_index('timestamp')
        elif not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("CSV 파일에 timestamp 컬럼이 없습니다")
        
        # 필요한 컬럼만 선택
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        available_columns = [col for col in required_columns if col in df.columns]
        df = df[available_columns]
        
        # 정렬
        df = df.sort_index()
        
        print(f"✅ CSV 가격 데이터 로드 완료: {len(df)}개 캔들")
        print(f"   시간 범위: {df.index.min()} ~ {df.index.max()}")
        return df
    except Exception as e:
        raise ValueError(f"CSV 파일 로드 실패: {e}")


def load_price_data_from_api_batch(
    symbol: str = "ETHUSDT", 
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    batch_size: int = 1500
) -> pd.DataFrame:
    """API에서 가격 데이터를 배치로 로드 (1년치 데이터)"""
    print(f"📊 API에서 가격 데이터 배치 로드 중... (심볼: {symbol})")
    print(f"   시간 범위: {start_time} ~ {end_time}")
    
    dataloader = BinanceDataLoader()
    all_dataframes = []
    
    current_start = start_time
    batch_count = 0
    
    while current_start < end_time:
        batch_count += 1
        print(f"   배치 {batch_count} 로드 중... ({current_start.strftime('%Y-%m-%d %H:%M')})")
        
        try:
            # 각 배치의 end_time 계산 (batch_size만큼의 캔들)
            # 3분봉이므로 batch_size * 3분 = batch_size * 3분
            batch_end = min(
                current_start + timedelta(minutes=batch_size * 3),
                end_time
            )
            
            df = dataloader.fetch_data(
                interval="3m",
                symbol=symbol,
                limit=batch_size,
                start_time=current_start,
                end_time=batch_end
            )
            
            if df is None or df.empty:
                print(f"   ⚠️ 배치 {batch_count} 데이터 없음, 다음 배치로...")
                current_start = batch_end
                continue
            
            # 인덱스 처리
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                    df = df.set_index('timestamp')
            
            all_dataframes.append(df)
            
            # 다음 배치 시작 시간 (마지막 캔들 시간 + 3분)
            if len(df) > 0:
                current_start = df.index[-1] + timedelta(minutes=3)
            else:
                current_start = batch_end
            
            # API 제한을 피하기 위한 대기
            time.sleep(0.1)
            
        except Exception as e:
            print(f"   ⚠️ 배치 {batch_count} 로드 실패: {e}")
            # 실패해도 다음 배치 시도
            if current_start < end_time:
                current_start = current_start + timedelta(days=1)
            else:
                break
    
    if not all_dataframes:
        raise ValueError(f"API에서 가격 데이터를 가져올 수 없습니다: {symbol}")
    
    # 모든 데이터프레임 합치기
    combined_df = pd.concat(all_dataframes)
    combined_df = combined_df.sort_index()
    
    # 중복 제거
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
    
    print(f"✅ API 가격 데이터 로드 완료: {len(combined_df)}개 캔들 (총 {batch_count}개 배치)")
    return combined_df


def load_price_data(
    decisions_df: Optional[pd.DataFrame] = None,
    symbol: str = "ETHUSDT",
    years_back: int = 1
) -> pd.DataFrame:
    """가격 데이터 로드 (CSV 우선, API 폴백)"""
    # 1. CSV 파일 시도
    csv_paths = [
        "data/ETHUSDT_3m_20240913_20250913.csv",
        "data/ETHUSDT_3m_20230913_20240913.csv"
    ]
    
    for csv_path in csv_paths:
        csv_file = Path(csv_path)
        if csv_file.exists():
            try:
                df = load_price_data_from_csv(csv_path)
                
                # 결정 데이터의 시간 범위에 맞춰 필터링
                if decisions_df is not None and not decisions_df.empty:
                    if 'timestamp' in decisions_df.columns:
                        decisions_df = decisions_df.set_index('timestamp')
                    
                    min_time = decisions_df.index.min()
                    max_time = decisions_df.index.max()
                    
                    # 시간 범위 확장 (앞뒤로 여유 시간)
                    time_buffer = pd.Timedelta(hours=24)
                    min_time = min_time - time_buffer
                    max_time = max_time + time_buffer
                    
                    df = df[(df.index >= min_time) & (df.index <= max_time)]
                    print(f"   결정 데이터 시간 범위에 맞춰 필터링: {len(df)}개 캔들")
                elif years_back > 0:
                    # 년수로 필터링
                    now = datetime.now(timezone.utc)
                    cutoff_time = now - timedelta(days=years_back * 365)
                    df = df[df.index >= cutoff_time]
                    print(f"   {years_back}년치 데이터 필터링: {len(df)}개 캔들")
                
                return df
            except Exception as e:
                print(f"⚠️ CSV 로드 실패 ({csv_path}): {e}")
                continue
    
    # 2. API 시도
    print("⚠️ CSV 파일을 찾을 수 없어 API에서 로드 시도...")
    
    # 시간 범위 결정
    now = datetime.now(timezone.utc)
    end_time = now
    start_time = now - timedelta(days=years_back * 365)
    
    if decisions_df is not None and not decisions_df.empty:
        if 'timestamp' in decisions_df.columns:
            decisions_df = decisions_df.set_index('timestamp')
        
        # 결정 데이터의 시간 범위와 교집합
        decision_min = decisions_df.index.min().to_pydatetime()
        decision_max = decisions_df.index.max().to_pydatetime()
        
        start_time = max(start_time, decision_min)
        end_time = min(end_time, decision_max)
    
    print(f"   API에서 {years_back}년치 데이터 요청: {start_time} ~ {end_time}")
    return load_price_data_from_api_batch(symbol, start_time, end_time)


def load_decision_data(
    filename: str = "agent/decisions_data.parquet",
    years_back: int = 1
) -> pd.DataFrame:
    """결정 데이터 로드 (지정된 년수만큼)"""
    print(f"📊 결정 데이터 로드 중... ({filename})")
    
    df = load_decisions_from_parquet(filename)
    
    if df is None or df.empty:
        raise ValueError(f"결정 데이터를 로드할 수 없습니다: {filename}")
    
    # timestamp 처리 (문자열로 저장된 경우 처리)
    if 'timestamp' in df.columns:
        # 문자열인 경우 파싱
        if df['timestamp'].dtype == 'object':
            # "YYYY-MM-DD HH:MM:SS UTC" 형식 처리
            df['timestamp'] = df['timestamp'].str.replace(' UTC', '', regex=False)
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        
        # 유효하지 않은 timestamp 제거
        df = df.dropna(subset=['timestamp'])
        df = df.set_index('timestamp')
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("결정 데이터에 timestamp가 없습니다")
    
    # 시간 범위 필터링 (지금으로부터 N년 전까지)
    if years_back > 0:
        now = datetime.now(timezone.utc)
        cutoff_time = now - timedelta(days=years_back * 365)
        
        original_count = len(df)
        # 현재 시간 이전의 데이터만 사용 (미래 데이터 제외)
        df = df[(df.index >= cutoff_time) & (df.index <= now)]
        
        print(f"   전체 데이터: {original_count}개")
        print(f"   필터링 후: {len(df)}개 (지금으로부터 {years_back}년, {cutoff_time.strftime('%Y-%m-%d')} ~ {now.strftime('%Y-%m-%d')})")
    
    print(f"✅ 결정 데이터 로드 완료: {len(df)}개 레코드")
    if len(df) > 0:
        print(f"   시간 범위: {df.index.min()} ~ {df.index.max()}")
    
    return df.reset_index()


def prepare_data(decisions_df: pd.DataFrame, price_df: pd.DataFrame) -> tuple:
    """데이터 준비 및 정렬"""
    print("🔧 데이터 준비 중...")
    
    # timestamp 기준으로 정렬
    if 'timestamp' in decisions_df.columns:
        decisions_df = decisions_df.set_index('timestamp')
    
    decisions_df = decisions_df.sort_index()
    price_df = price_df.sort_index()
    
    # 시간 범위 맞추기
    min_time = max(decisions_df.index.min(), price_df.index.min())
    max_time = min(decisions_df.index.max(), price_df.index.max())
    
    decisions_df = decisions_df[
        (decisions_df.index >= min_time) & 
        (decisions_df.index <= max_time)
    ]
    price_df = price_df[
        (price_df.index >= min_time) & 
        (price_df.index <= max_time)
    ]
    
    print(f"✅ 데이터 준비 완료:")
    print(f"   결정 데이터: {len(decisions_df)}개")
    print(f"   가격 데이터: {len(price_df)}개")
    print(f"   시간 범위: {min_time} ~ {max_time}")
    
    return decisions_df.reset_index(), price_df


def main():
    """메인 함수"""
    print("=" * 60)
    print("메타 라벨링 모델 학습 시작 (1년치 데이터)")
    print("=" * 60)
    
    years_back = 1  # 지금으로부터 1년
    
    # 1. 데이터 로드
    try:
        decisions_df = load_decision_data("agent/decisions_data.parquet", years_back=years_back)
        price_df = load_price_data(decisions_df, "ETHUSDT", years_back=years_back)
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 2. 데이터 준비
    try:
        decisions_df, price_df = prepare_data(decisions_df, price_df)
    except Exception as e:
        print(f"❌ 데이터 준비 실패: {e}")
        return
    
    # 3. 메타 라벨링 엔진 초기화 (성능 개선 버전)
    print("\n🤖 메타 라벨링 엔진 초기화...")
    engine = MetaLabelingEngine(
        model_type="random_forest",  # 또는 "gradient_boosting" 시도 가능
        min_samples_for_training=100,
        confidence_threshold=0.7  # 0.6 → 0.7 (더 보수적)
    )
    
    # 4. 모델 학습
    print("\n🎓 모델 학습 시작...")
    try:
        result = engine.train(
            decisions_df=decisions_df,
            price_data=price_df,
            test_size=0.2,
            retrain=True,
            min_profit_threshold=0.005,  # 최소 0.5% 수익
            use_profit_based=True  # 실제 수익률 기반 라벨링
        )
        
        if result["success"]:
            print("\n" + "=" * 60)
            print("✅ 모델 학습 완료!")
            print("=" * 60)
            print(f"정확도: {result['accuracy']:.3f}")
            print(f"ROC-AUC: {result['roc_auc']:.3f}")
            print(f"학습 샘플: {result['train_samples']}개")
            print(f"테스트 샘플: {result['test_samples']}개")
            
            if result.get('feature_importance'):
                print("\n특성 중요도 (상위 5개):")
                sorted_features = sorted(
                    result['feature_importance'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                for feature, importance in sorted_features:
                    print(f"  {feature}: {importance:.3f}")
        else:
            print(f"\n❌ 모델 학습 실패: {result.get('message', '알 수 없는 오류')}")
    
    except Exception as e:
        print(f"\n❌ 모델 학습 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


