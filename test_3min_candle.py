#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BinanceDataLoader 3분봉 1개 데이터 테스트 파일
- 3분봉 1개만 가져오는 테스트
- 다양한 시간 범위로 테스트
- 데이터 구조 및 내용 검증
"""

import sys
import os
from datetime import datetime, timezone, timedelta

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.binance_dataloader import BinanceDataLoader


def test_single_3min_candle():
    """3분봉 1개 데이터 가져오기 테스트"""
    print("🚀 BinanceDataLoader 3분봉 1개 데이터 테스트 시작...")
    print("=" * 60)
    
    # BinanceDataLoader 초기화
    loader = BinanceDataLoader()
    
    # 현재 시간 기준으로 테스트
    current_time = datetime.now(timezone.utc)
    print(f"⏰ 현재 시간: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print()
    
    # 테스트 1: 현재 진행 중인 3분봉 데이터
    print("📊 테스트 1: 현재 진행 중인 3분봉 데이터")
    print("-" * 40)
    
    # 현재 3분봉의 시작 시간 계산
    current_minute = current_time.minute
    minutes_since_3min = current_minute % 3
    candle_start = current_time.replace(
        minute=current_minute - minutes_since_3min,
        second=0, 
        microsecond=0
    )
    
    # 3분봉 종료 시간
    candle_end = candle_start + timedelta(minutes=3)
    
    print(f"🔍 3분봉 시간 범위: {candle_start.strftime('%H:%M:%S')} ~ {candle_end.strftime('%H:%M:%S')} UTC")
    print(f"📅 날짜: {candle_start.strftime('%Y-%m-%d')}")
    
    # 3분봉 데이터 요청
    df_3m = loader.fetch_data(
        interval=3,
        symbol="ETHUSDT",
        start_time=candle_start,
        end_time=current_time
    )
    
    if df_3m is not None and not df_3m.empty:
        print(f"✅ 데이터 로드 성공: {len(df_3m)}개 캔들")
        
        # 첫 번째 캔들 데이터 출력
        first_candle = df_3m.iloc[0]
        print(f"📊 첫 번째 캔들:")
        print(f"   시간: {first_candle.name}")
        print(f"   Open: ${float(first_candle['open']):.2f}")
        print(f"   High: ${float(first_candle['high']):.2f}")
        print(f"   Low:  ${float(first_candle['low']):.2f}")
        print(f"   Close: ${float(first_candle['close']):.2f}")
        print(f"   Volume: {float(first_candle['volume']):.2f} ETH")
        print(f"   Quote Volume: ${float(first_candle['quote_volume']):.2f}")
        
        # 전체 DataFrame 정보
        print(f"\n📋 DataFrame 정보:")
        print(f"   Shape: {df_3m.shape}")
        print(f"   Columns: {list(df_3m.columns)}")
        print(f"   Index: {df_3m.index.name}")
        print(f"   Data Types: {df_3m.dtypes.to_dict()}")
        
    else:
        print("❌ 데이터 로드 실패")
    
    print("\n" + "=" * 60)
    
    # 테스트 2: 과거 특정 3분봉 데이터
    print("📊 테스트 2: 과거 특정 3분봉 데이터")
    print("-" * 40)
    
    # 1시간 전 3분봉 데이터
    past_time = current_time - timedelta(hours=1)
    past_minute = past_time.minute
    past_minutes_since_3min = past_minute % 3
    past_candle_start = past_time.replace(
        minute=past_minute - past_minutes_since_3min,
        second=0, 
        microsecond=0
    )
    past_candle_end = past_candle_start + timedelta(minutes=3)
    
    print(f"🔍 과거 3분봉 시간 범위: {past_candle_start.strftime('%H:%M:%S')} ~ {past_candle_end.strftime('%H:%M:%S')} UTC")
    print(f"📅 날짜: {past_candle_start.strftime('%Y-%m-%d')}")
    
    # 과거 3분봉 데이터 요청
    df_past_3m = loader.fetch_data(
        interval=3,
        symbol="ETHUSDT",
        start_time=past_candle_start,
        end_time=past_candle_end
    )
    
    if df_past_3m is not None and not df_past_3m.empty:
        print(f"✅ 과거 데이터 로드 성공: {len(df_past_3m)}개 캔들")
        
        # 과거 캔들 데이터 출력
        past_candle = df_past_3m.iloc[0]
        print(f"📊 과거 캔들:")
        print(f"   시간: {past_candle.name}")
        print(f"   Open: ${float(past_candle['open']):.2f}")
        print(f"   High: ${float(past_candle['high']):.2f}")
        print(f"   Low:  ${float(past_candle['low']):.2f}")
        print(f"   Close: ${float(past_candle['close']):.2f}")
        print(f"   Volume: {float(past_candle['volume']):.2f} ETH")
        print(f"   Quote Volume: ${float(past_candle['quote_volume']):.2f}")
        
    else:
        print("❌ 과거 데이터 로드 실패")
    
    print("\n" + "=" * 60)
    
    # 테스트 3: 정확한 3분봉 마감 시점 데이터
    print("📊 테스트 3: 정확한 3분봉 마감 시점 데이터")
    print("-" * 40)
    
    # 가장 최근에 완료된 3분봉 찾기
    if current_minute % 3 == 0:
        # 정확히 3분봉 시작 시점이면 이전 3분봉 사용
        completed_candle_start = current_time - timedelta(minutes=3)
        completed_candle_start = completed_candle_start.replace(second=0, microsecond=0)
    else:
        # 진행 중인 3분봉의 이전 3분봉 사용
        completed_candle_start = current_time.replace(
            minute=current_minute - minutes_since_3min - 3,
            second=0, 
            microsecond=0
        )
    
    completed_candle_end = completed_candle_start + timedelta(minutes=3)
    
    print(f"🔍 완료된 3분봉 시간 범위: {completed_candle_start.strftime('%H:%M:%S')} ~ {completed_candle_end.strftime('%H:%M:%S')} UTC")
    print(f"📅 날짜: {completed_candle_start.strftime('%Y-%m-%d')}")
    
    # 완료된 3분봉 데이터 요청
    df_completed_3m = loader.fetch_data(
        interval=3,
        symbol="ETHUSDT",
        start_time=completed_candle_start,
        end_time=completed_candle_end
    )
    
    if df_completed_3m is not None and not df_completed_3m.empty:
        print(f"✅ 완료된 3분봉 데이터 로드 성공: {len(df_completed_3m)}개 캔들")
        
        # 완료된 캔들 데이터 출력
        completed_candle = df_completed_3m.iloc[0]
        print(f"📊 완료된 캔들:")
        print(f"   시간: {completed_candle.name}")
        print(f"   Open: ${float(completed_candle['open']):.2f}")
        print(f"   High: ${float(completed_candle['high']):.2f}")
        print(f"   Low:  ${float(completed_candle['low']):.2f}")
        print(f"   Close: ${float(completed_candle['close']):.2f}")
        print(f"   Volume: {float(completed_candle['volume']):.2f} ETH")
        print(f"   Quote Volume: ${float(completed_candle['quote_volume']):.2f}")
        
    else:
        print("❌ 완료된 3분봉 데이터 로드 실패")
    
    print("\n" + "=" * 60)
    
    # 테스트 5: 마지막으로 완성된 3분봉 데이터
    print("📊 테스트 5: 마지막으로 완성된 3분봉 데이터")
    print("-" * 40)
    
    # 가장 최근에 완료된 3분봉 찾기 (현재 진행 중이 아닌 완료된 것)
    # 현재 시간이 3분봉의 어느 시점인지 정확히 계산
    current_minute = current_time.minute
    current_second = current_time.second
    
    # 현재 진행 중인 3분봉의 시작 시간
    current_candle_start = current_time.replace(
        minute=current_minute - (current_minute % 3),
        second=0, 
        microsecond=0
    )
    
    # 마지막 완성된 3분봉은 현재 진행 중인 3분봉의 이전 3분봉
    last_completed_start = current_candle_start - timedelta(minutes=3)
    last_completed_end = last_completed_start + timedelta(minutes=3)
    
    print(f"🔍 현재 진행 중인 3분봉: {current_candle_start.strftime('%H:%M:%S')} ~ {(current_candle_start + timedelta(minutes=3)).strftime('%H:%M:%S')} UTC")
    print(f"🔍 마지막 완성된 3분봉 시간 범위: {last_completed_start.strftime('%H:%M:%S')} ~ {last_completed_end.strftime('%H:%M:%S')} UTC")
    print(f"📅 날짜: {last_completed_start.strftime('%Y-%m-%d')}")
    print(f"⏰ 현재 시간: {current_time.strftime('%H:%M:%S')} UTC")
    
    # 마지막 완성된 3분봉 데이터 요청
    df_last_completed = loader.fetch_data(
        interval=3,
        symbol="ETHUSDT",
        start_time=last_completed_start,
        end_time=last_completed_end
    )
    
    if df_last_completed is not None and not df_last_completed.empty:
        print(f"✅ 마지막 완성된 3분봉 데이터 로드 성공: {len(df_last_completed)}개 캔들")
        
        # 마지막 완성된 캔들 데이터 출력
        last_completed_candle = df_last_completed.iloc[0]
        print(f"📊 마지막 완성된 캔들:")
        print(f"   시간: {last_completed_candle.name}")
        print(f"   Open: ${float(last_completed_candle['open']):.2f}")
        print(f"   High: ${float(last_completed_candle['high']):.2f}")
        print(f"   Low:  ${float(last_completed_candle['low']):.2f}")
        print(f"   Close: ${float(last_completed_candle['close']):.2f}")
        print(f"   Volume: {float(last_completed_candle['volume']):.2f} ETH")
        print(f"   Quote Volume: ${float(last_completed_candle['quote_volume']):.2f}")
        
        # 완성 여부 확인
        current_time_utc = current_time.replace(tzinfo=timezone.utc)
        candle_close_time = last_completed_candle.name
        
        if current_time_utc > candle_close_time:
            print(f"✅ 완성 확인: 현재 시간({current_time_utc.strftime('%H:%M:%S')}) > 캔들 종료 시간({candle_close_time.strftime('%H:%M:%S')})")
            print(f"   🎯 이 3분봉은 완전히 완성되었습니다!")
            
            # 다음 3분봉 시작까지 남은 시간 계산
            next_candle_start = current_candle_start + timedelta(minutes=3)
            time_until_next = next_candle_start - current_time
            minutes_until_next = time_until_next.total_seconds() / 60
            
            print(f"   ⏰ 다음 3분봉 시작까지: {minutes_until_next:.1f}분 남음")
            
        else:
            print(f"⚠️ 주의: 이 3분봉은 아직 완성되지 않았습니다")
            
    else:
        print("❌ 마지막 완성된 3분봉 데이터 로드 실패")
    
    print("\n" + "=" * 60)
    
    # 테스트 4: 데이터 품질 검증
    print("📊 테스트 4: 데이터 품질 검증")
    print("-" * 40)
    
    if df_3m is not None and not df_3m.empty:
        # 데이터 품질 검증
        print("🔍 데이터 품질 검증 결과:")
        
        # 1. 시간 순서 검증
        is_sorted = df_3m.index.is_monotonic_increasing
        print(f"   ✅ 시간 순서 정렬: {is_sorted}")
        
        # 2. OHLC 관계 검증
        first_candle = df_3m.iloc[0]
        o, h, l, c = float(first_candle['open']), float(first_candle['high']), float(first_candle['low']), float(first_candle['close'])
        
        high_is_highest = h >= max(o, c)
        low_is_lowest = l <= min(o, c)
        
        print(f"   ✅ High가 최고값: {high_is_highest} (High: {h:.2f}, Open: {o:.2f}, Close: {c:.2f})")
        print(f"   ✅ Low가 최저값: {low_is_lowest} (Low: {l:.2f}, Open: {o:.2f}, Close: {c:.2f})")
        
        # 3. 거래량 검증
        volume = float(first_candle['volume'])
        quote_volume = float(first_candle['quote_volume'])
        
        print(f"   ✅ 거래량 양수: {volume > 0}")
        print(f"   ✅ USDT 거래량 양수: {quote_volume > 0}")
        
        # 4. 가격 변화율 계산
        price_change = ((c - o) / o) * 100
        print(f"   📈 가격 변화율: {price_change:.2f}%")
        
        # 5. 변동성 계산
        volatility = ((h - l) / o) * 100
        print(f"   📊 변동성: {volatility:.2f}%")
        
    else:
        print("❌ 데이터가 없어서 품질 검증을 수행할 수 없습니다")
    
    print("\n" + "=" * 60)
    print("🏁 3분봉 1개 데이터 테스트 완료!")


def test_error_handling():
    """에러 처리 테스트"""
    print("\n🚨 에러 처리 테스트")
    print("=" * 60)
    
    loader = BinanceDataLoader()
    
    # 잘못된 심볼 테스트
    print("📊 잘못된 심볼 테스트:")
    df_invalid = loader.fetch_data(
        interval=3,
        symbol="INVALID",
        start_time=datetime.now(timezone.utc) - timedelta(minutes=10),
        end_time=datetime.now(timezone.utc)
    )
    
    if df_invalid is None:
        print("✅ 잘못된 심볼에 대한 에러 처리 정상")
    else:
        print("❌ 잘못된 심볼에 대한 에러 처리 실패")
    
    # 잘못된 시간 범위 테스트
    print("\n📊 잘못된 시간 범위 테스트:")
    future_time = datetime.now(timezone.utc) + timedelta(hours=1)
    df_future = loader.fetch_data(
        interval=3,
        symbol="ETHUSDT",
        start_time=future_time,
        end_time=future_time + timedelta(minutes=3)
    )
    
    if df_future is None or df_future.empty:
        print("✅ 미래 시간에 대한 에러 처리 정상")
    else:
        print("❌ 미래 시간에 대한 에러 처리 실패")


if __name__ == "__main__":
    try:
        # 메인 테스트 실행
        test_single_3min_candle()
        
        # 에러 처리 테스트 실행
        test_error_handling()
        
    except Exception as e:
        print(f"❌ 테스트 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
