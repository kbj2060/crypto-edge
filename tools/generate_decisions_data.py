#!/usr/bin/env python3
"""
decisions_data.parquet 파일 생성 스크립트
"""

import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent.decision_generator import (
    generate_signal_data_with_indicators,
    load_decisions_from_parquet,
    analyze_decision_data,
    inspect_parquet_structure,
    check_existing_decision_data,
    clear_progress_state
)
from managers.binance_dataloader import BinanceDataLoader
from datetime import datetime, timezone, timedelta
import pandas as pd
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from collections import defaultdict


def fetch_single_batch(
    interval: str,
    batch_start: datetime,
    batch_end: datetime,
    batch_num: int,
    print_lock: Lock
):
    """단일 배치 데이터를 가져오는 함수 (병렬 처리용)"""
    try:
        # 각 스레드마다 독립적인 dataloader 인스턴스 생성 (스레드 안전)
        dataloader = BinanceDataLoader()
        batch_size = 1500
        
        df = dataloader.fetch_data(
            interval=interval,
            symbol="ETHUSDT",
            limit=batch_size,
            start_time=batch_start,
            end_time=batch_end
        )
        
        if df is None or df.empty:
            return batch_num, None
        
        # 인덱스 처리
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                df = df.set_index('timestamp')
        
        if batch_num % 20 == 0:
            with print_lock:
                print(f"      [{interval}] 배치 {batch_num} 완료 ({len(df)}개 캔들)")
        
        return batch_num, df
        
    except Exception as e:
        with print_lock:
            print(f"      ⚠️ [{interval}] 배치 {batch_num} 오류: {e}")
        return batch_num, None


def load_interval_data_parallel(
    interval: str,
    start_time: datetime,
    end_time: datetime,
    print_lock: Lock,
    max_workers: int = 10
):
    """단일 간격 데이터를 병렬로 로드하는 함수 (고속 병렬 처리)"""
    with print_lock:
        print(f"\n   {interval} 데이터 로드 시작 (병렬 처리: 최대 {max_workers}개 동시 요청)...")
    
    batch_size = 1500
    interval_minutes = {'3m': 3, '15m': 15, '1h': 60}[interval]
    
    # 모든 배치 시간 범위를 미리 계산
    batch_ranges = []
    current_start = start_time
    batch_num = 0
    
    while current_start < end_time:
        batch_num += 1
        batch_end = min(
            current_start + timedelta(minutes=batch_size * interval_minutes),
            end_time
        )
        batch_ranges.append((batch_num, current_start, batch_end))
        current_start = batch_end
    
    with print_lock:
        print(f"      [{interval}] 총 {len(batch_ranges)}개 배치를 병렬로 처리합니다...")
    
    # 배치를 병렬로 가져오기 (더 많은 동시 요청)
    batch_results = {}
    completed_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 모든 배치에 대해 병렬 작업 제출
        future_to_batch = {
            executor.submit(
                fetch_single_batch,
                interval,
                batch_start,
                batch_end,
                batch_num,
                print_lock
            ): batch_num
            for batch_num, batch_start, batch_end in batch_ranges
        }
        
        # 완료된 작업을 실시간으로 수집
        for future in as_completed(future_to_batch):
            batch_num = future_to_batch[future]
            try:
                result_batch_num, df = future.result()
                if df is not None:
                    batch_results[result_batch_num] = df
                completed_count += 1
                
                # 진행 상황 출력 (10%마다)
                if completed_count % max(1, len(batch_ranges) // 10) == 0:
                    progress = (completed_count / len(batch_ranges)) * 100
                    with print_lock:
                        print(f"      [{interval}] 진행률: {progress:.1f}% ({completed_count}/{len(batch_ranges)})")
                        
            except Exception as e:
                with print_lock:
                    print(f"      ⚠️ [{interval}] 배치 {batch_num} 처리 중 오류: {e}")
    
    # 배치 번호 순서대로 정렬하여 합치기
    if batch_results:
        sorted_batches = sorted(batch_results.items())
        all_dataframes = [df for _, df in sorted_batches if df is not None]
        
        if all_dataframes:
            combined_df = pd.concat(all_dataframes, ignore_index=False)
            combined_df = combined_df.sort_index()
            combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
            with print_lock:
                print(f"   ✅ {interval} 데이터 로드 완료: {len(combined_df)}개 캔들")
            return combined_df
    
    with print_lock:
        print(f"   ⚠️ {interval} 데이터 로드 실패: 데이터 없음")
    return None


def get_cache_filepath(interval: str, months_back: int) -> str:
    """캐시 파일 경로 생성"""
    os.makedirs("data", exist_ok=True)
    end_date = datetime.now(timezone.utc).strftime("%Y%m%d")
    start_date = (datetime.now(timezone.utc) - timedelta(days=months_back * 30)).strftime("%Y%m%d")
    return f"data/ETHUSDT_{interval}_{start_date}_{end_date}.parquet"


def is_cache_valid(cache_path: str, max_age_hours: int = 24) -> bool:
    """캐시 파일이 유효한지 확인 (기본 24시간)"""
    if not os.path.exists(cache_path):
        return False
    
    # 파일 수정 시간 확인
    file_time = datetime.fromtimestamp(os.path.getmtime(cache_path), tz=timezone.utc)
    age_hours = (datetime.now(timezone.utc) - file_time).total_seconds() / 3600
    
    return age_hours < max_age_hours


def load_from_cache(interval: str, months_back: int) -> pd.DataFrame:
    """캐시 파일에서 데이터 로드"""
    cache_path = get_cache_filepath(interval, months_back)
    
    if not os.path.exists(cache_path):
        return None
    
    try:
        df = pd.read_parquet(cache_path)
        # 인덱스가 timestamp인지 확인
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                df = df.set_index('timestamp')
        print(f"   ✅ [{interval}] 캐시에서 로드: {len(df)}개 캔들")
        return df
    except Exception as e:
        print(f"   ⚠️ [{interval}] 캐시 로드 실패: {e}")
        return None


def save_to_cache(df: pd.DataFrame, interval: str, months_back: int):
    """데이터를 캐시 파일로 저장"""
    if df is None or df.empty:
        return
    
    cache_path = get_cache_filepath(interval, months_back)
    try:
        # timestamp를 인덱스로 설정
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                df = df.set_index('timestamp')
        
        df.to_parquet(cache_path, compression='snappy')
        file_size = os.path.getsize(cache_path) / 1024 / 1024
        print(f"   💾 [{interval}] 캐시 저장 완료: {cache_path} ({file_size:.2f}MB)")
    except Exception as e:
        print(f"   ⚠️ [{interval}] 캐시 저장 실패: {e}")


def load_ethusdt_data_from_api(months_back: int = 3, use_cache: bool = True, max_cache_age_hours: int = 24):
    """API에서 ETHUSDT 데이터 로드 (3분, 15분, 1시간봉) - 캐시 지원"""
    intervals = ['3m', '15m', '1h']
    results = {}
    
    # 1. 캐시에서 로드 시도
    if use_cache:
        print("📂 캐시 파일 확인 중...")
        for interval in intervals:
            cache_path = get_cache_filepath(interval, months_back)
            if is_cache_valid(cache_path, max_cache_age_hours):
                df = load_from_cache(interval, months_back)
                if df is not None:
                    results[interval] = df
        
        # 모든 간격의 캐시가 있으면 반환
        if len(results) == len(intervals):
            print("✅ 모든 데이터를 캐시에서 로드했습니다.")
            return results.get('3m'), results.get('15m'), results.get('1h')
        
        # 일부만 캐시에 있으면 출력
        if results:
            print(f"⚠️  캐시에서 {len(results)}/{len(intervals)}개 간격만 로드했습니다. 나머지는 API에서 가져옵니다.")
    
    # 2. API에서 로드 (캐시에 없는 간격만)
    print(f"\n📥 API에서 가격 데이터 로드 중... (최근 {months_back}개월)")
    print("   🚀 고속 병렬 처리 모드 활성화")
    
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=months_back * 30)
    
    # 스레드 안전을 위한 Lock
    print_lock = Lock()
    
    # 캐시에 없는 간격만 API에서 로드
    intervals_to_fetch = [iv for iv in intervals if iv not in results]
    
    if intervals_to_fetch:
        # 간격별 병렬 처리 (최대 3개 간격 동시 처리)
        with ThreadPoolExecutor(max_workers=3) as interval_executor:
            # 모든 간격에 대해 병렬 작업 제출
            future_to_interval = {
                interval_executor.submit(
                    load_interval_data_parallel,
                    interval,
                    start_time,
                    end_time,
                    print_lock,
                    max_workers=10  # 각 간격 내에서 최대 10개 배치 동시 처리
                ): interval
                for interval in intervals_to_fetch
            }
            
            # 완료된 작업 처리
            for future in as_completed(future_to_interval):
                interval = future_to_interval[future]
                try:
                    df = future.result()
                    results[interval] = df
                    
                    # 캐시에 저장
                    if df is not None and use_cache:
                        save_to_cache(df, interval, months_back)
                        
                except Exception as e:
                    with print_lock:
                        print(f"   ❌ {interval} 데이터 로드 중 오류: {e}")
                    results[interval] = None
    
    return results.get('3m'), results.get('15m'), results.get('1h')


def main():
    """메인 함수"""
    print("=" * 60)
    print("📊 Decision 데이터 생성 시작")
    print("=" * 60)
    
    # 1. 기존 데이터 확인
    if check_existing_decision_data():
        print("\n⚠️  기존 데이터가 있습니다.")
        print("기존 파일을 삭제하고 새로 생성합니다...")
        import os
        if os.path.exists("agent/decisions_data.parquet"):
            os.remove("agent/decisions_data.parquet")
            print("✅ 기존 파일 삭제 완료")
        # 진행 상태도 삭제
        clear_progress_state()
    
    # 2. 실제 ETHUSDT 데이터 로드 (3분, 15분, 1시간봉)
    print("\n📥 가격 데이터 로드 중...")
    
    # 먼저 기존 CSV 파일 시도 (하위 호환성)
    try:
        from agent.decision_generator import load_ethusdt_data
        price_data, price_data_15m, price_data_1h = load_ethusdt_data()
        if price_data is not None:
            print("✅ 기존 CSV 파일에서 로드했습니다.")
    except:
        price_data, price_data_15m, price_data_1h = None, None, None
    
    # CSV가 없으면 캐시/API에서 로드 (캐시 우선)
    if price_data is None:
        price_data, price_data_15m, price_data_1h = load_ethusdt_data_from_api(
            months_back=1,  # 1달 데이터
            use_cache=True,  # 캐시 사용
            max_cache_age_hours=24  # 24시간 이내 캐시는 유효
        )
    
    if price_data is None:
        print("❌ 데이터 로드 실패")
        return
    
    print(f"\n✅ 가격 데이터 로드 완료:")
    print(f"   - 3분봉: {len(price_data)}개 캔들")
    print(f"   - 15분봉: {len(price_data_15m)}개 캔들")
    print(f"   - 1시간봉: {len(price_data_1h)}개 캔들")
    print(f"   - 가격 범위: ${price_data['close'].min():.2f} ~ ${price_data['close'].max():.2f}")
    
    # 3. CSV 데이터로 실제 지표 업데이트 및 전략 실행
    print("\n🔄 Decision 데이터 생성 중...")
    print("   (이 작업은 시간이 걸릴 수 있습니다)")
    
    success = generate_signal_data_with_indicators(
        price_data, 
        price_data_15m, 
        price_data_1h, 
        resume_from_progress=False  # 처음부터 시작
    )
    
    if success:
        print("\n" + "=" * 60)
        print("✅ Decision 데이터 생성 완료!")
        print("=" * 60)
        
        # 저장된 데이터 확인 및 분석
        df = load_decisions_from_parquet()
        if df is not None:
            print(f"\n📊 저장된 데이터 요약:")
            print(f"   - 총 레코드 수: {len(df)}개")
            print(f"   - 시간 범위: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
            print(f"   - 컬럼 수: {len(df.columns)}")
            
            # 상세 분석
            analyze_decision_data(df)
            
            # Parquet 구조 확인
            inspect_parquet_structure()
            
            print(f"\n🎯 파일 위치: agent/decisions_data.parquet")
            print(f"   이 데이터를 메타 라벨링 학습에 사용할 수 있습니다.")
    else:
        print("\n❌ 데이터 생성이 완료되지 않았습니다.")
        print("   진행 상태가 저장되어 있어서 다음에 이어서 실행할 수 있습니다.")


if __name__ == "__main__":
    main()

