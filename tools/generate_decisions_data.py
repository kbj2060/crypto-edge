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


def load_ethusdt_data_from_api(months_back: int = 3):
    """API에서 ETHUSDT 데이터 로드 (3분, 15분, 1시간봉)"""
    print(f"📥 API에서 가격 데이터 로드 중... (최근 {months_back}개월)")
    
    dataloader = BinanceDataLoader()
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=months_back * 30)
    
    all_dataframes = {}
    
    for interval in ['3m', '15m', '1h']:
        print(f"\n   {interval} 데이터 로드 중...")
        all_dataframes[interval] = []
        
        current_start = start_time
        batch_count = 0
        
        # 배치 크기 계산 (API 제한: 최대 1500개)
        batch_size = 1500
        # 간격별 분 단위
        interval_minutes = {'3m': 3, '15m': 15, '1h': 60}[interval]
        
        while current_start < end_time:
            batch_count += 1
            if batch_count % 10 == 0:
                print(f"      배치 {batch_count}... ({current_start.strftime('%Y-%m-%d %H:%M')})")
            
            # 배치 종료 시간 계산
            batch_end = min(
                current_start + timedelta(minutes=batch_size * interval_minutes),
                end_time
            )
            
            try:
                df = dataloader.fetch_data(
                    interval=interval,
                    symbol="ETHUSDT",
                    limit=batch_size,
                    start_time=current_start,
                    end_time=batch_end
                )
                
                if df is None or df.empty:
                    current_start = batch_end
                    continue
                
                # 인덱스 처리
                if not isinstance(df.index, pd.DatetimeIndex):
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                        df = df.set_index('timestamp')
                
                all_dataframes[interval].append(df)
                
                # 다음 배치 시작 시간
                if len(df) > 0:
                    current_start = df.index[-1] + timedelta(minutes=interval_minutes)
                else:
                    current_start = batch_end
                
                # API 제한 방지
                time.sleep(0.1)
                
            except Exception as e:
                print(f"      ⚠️ 배치 {batch_count} 오류: {e}")
                current_start = batch_end
                continue
        
        # 데이터 합치기
        if all_dataframes[interval]:
            combined_df = pd.concat(all_dataframes[interval], ignore_index=False)
            combined_df = combined_df.sort_index()
            combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
            all_dataframes[interval] = combined_df
            print(f"   ✅ {interval} 데이터 로드 완료: {len(combined_df)}개 캔들")
        else:
            all_dataframes[interval] = None
    
    return all_dataframes.get('3m'), all_dataframes.get('15m'), all_dataframes.get('1h')


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
    
    # 먼저 CSV 파일 시도
    try:
        from agent.decision_generator import load_ethusdt_data
        price_data, price_data_15m, price_data_1h = load_ethusdt_data()
    except:
        price_data, price_data_15m, price_data_1h = None, None, None
    
    # CSV가 없으면 API에서 로드
    if price_data is None:
        print("⚠️ CSV 파일이 없어 API에서 데이터를 가져옵니다...")
        price_data, price_data_15m, price_data_1h = load_ethusdt_data_from_api(months_back=3)
    
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

