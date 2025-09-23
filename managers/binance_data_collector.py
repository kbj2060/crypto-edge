#!/usr/bin/env python3
"""
바이낸스 API를 사용하여 3분봉과 5분봉 데이터를 수집하고 CSV로 저장하는 스크립트
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta, timezone
import os
from typing import List
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BinanceDataCollector:
    def __init__(self):
        self.base_url = "https://fapi.binance.com/fapi/v1/klines"  # 선물 API 엔드포인트
        self.session = requests.Session()
        
    def get_klines(self, symbol: str, interval: str, start_time: int = None, end_time: int = None, limit: int = 1000) -> List[List]:
        """
        바이낸스 선물 API에서 Kline 데이터를 가져옵니다.
        
        Args:
            symbol: 거래쌍 (예: 'ETHUSDT')
            interval: 시간 간격 ('3m', '5m' 등)
            start_time: 시작 시간 (밀리초)
            end_time: 종료 시간 (밀리초)
            limit: 최대 개수 (최대 1000)
        """
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
            
        try:
            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API 요청 실패: {e}")
            return []
    
    def get_historical_data(self, symbol: str, interval: str, start_time, end_time) -> pd.DataFrame:
        """
        지정된 기간 동안의 히스토리컬 데이터를 가져옵니다.
        
        Args:
            symbol: 거래쌍
            interval: 시간 간격
            start_time: 시작 시간 (datetime 객체)
            end_time: 종료 시간 (datetime 객체)
        """
        logger.info(f"{symbol} {interval} 데이터 수집 시작 ({start_time} ~ {end_time})")
        
        # datetime을 밀리초 타임스탬프로 변환 (UTC 기준)
        start_timestamp = int(start_time.timestamp() * 1000)
        end_timestamp = int(end_time.timestamp() * 1000)
        
        logger.info(f"시작 타임스탬프: {start_timestamp} ({datetime.fromtimestamp(start_timestamp/1000)})")
        logger.info(f"종료 타임스탬프: {end_timestamp} ({datetime.fromtimestamp(end_timestamp/1000)})")
        
        all_data = []
        current_end_time = end_timestamp
        
        while current_end_time > start_timestamp:
            # API 제한을 고려하여 1000개씩 가져오기
            # end_time만 지정하여 최신 데이터부터 역순으로 가져오기
            klines = self.get_klines(symbol, interval, end_time=current_end_time, limit=1000)
            
            if not klines:
                logger.warning(f"데이터를 가져올 수 없습니다: {symbol} {interval}")
                break
                
            # 시작 시간보다 오래된 데이터는 제외
            filtered_klines = [k for k in klines if k[0] >= start_timestamp]
            all_data.extend(filtered_klines)
            
            # 다음 배치를 위한 시간 설정 (가장 오래된 데이터의 시간 - 1ms)
            current_end_time = klines[0][0] - 1
            
            # API 제한을 피하기 위한 대기
            time.sleep(0.1)
            
            logger.info(f"수집된 데이터: {len(all_data)}개")
            
            # 시작 시간에 도달했으면 중단
            if klines[0][0] <= start_timestamp:
                break
        
        # DataFrame으로 변환
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # 데이터 타입 변환 (UTC 시간)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', utc=True)
        
        # 한국 시간으로 변환 (선택사항)
        # df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Seoul')
        # df['close_time'] = df['close_time'].dt.tz_convert('Asia/Seoul')
        
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 
                          'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
        
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 시간순 정렬
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 지정된 시간 범위 내의 데이터만 필터링
        df = df[(df['timestamp'] >= pd.to_datetime(start_time, utc=True)) & 
                (df['timestamp'] < pd.to_datetime(end_time, utc=True))]
        
        logger.info(f"{symbol} {interval} 데이터 수집 완료: {len(df)}개")
        return df
    
    def save_to_csv(self, df: pd.DataFrame, symbol: str, interval: str, start_time, end_time, output_dir: str = "data"):
        """DataFrame을 CSV 파일로 저장합니다."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        filename = f"{symbol}_{interval}_{start_time.strftime('%Y%m%d')}_{end_time.strftime('%Y%m%d')}.csv"
        filepath = os.path.join(output_dir, filename)
        
        df.to_csv(filepath, index=False)
        logger.info(f"데이터 저장 완료: {filepath}")
        return filepath

def main():
    """메인 실행 함수"""
    collector = BinanceDataCollector()
    
    # 수집할 거래쌍과 시간 간격 (선물 데이터)
    symbols = ['ETHUSDT']
    intervals = ['1h', '3m', '15m']
    
    logger.info("바이낸스 선물 데이터 수집 시작")
    logger.info(f"거래쌍: {symbols}")
    logger.info(f"시간 간격: {intervals}")
    logger.info(f"수집 기간: 2023년 9월 13일 00시 00분 ~ 2024년 9월 13일 00시 00분")
    
    for symbol in symbols:
        for interval in intervals:
            try:
                logger.info(f"\n{'='*50}")
                logger.info(f"수집 중: {symbol} {interval}")
                logger.info(f"{'='*50}")
                
                # UTC 시간으로 정확히 설정 (한국 시간 - 9시간)
                start_time = datetime(2024, 9, 13, 0, 0, 0, tzinfo=timezone.utc)  # 2023년 9월 13일 00시 00분 UTC
                end_time = datetime(2025, 9, 13, 0, 0, 0, tzinfo=timezone.utc)    # 2024년 9월 13일 00시 00분 UTC
                # 데이터 수집
                df = collector.get_historical_data(symbol, interval, start_time, end_time)
                
                if not df.empty:
                    # CSV 저장
                    collector.save_to_csv(df, symbol, interval, start_time, end_time)
                    
                    # 데이터 요약 정보 출력
                    logger.info(f"수집된 데이터 요약:")
                    logger.info(f"  - 총 캔들 수: {len(df)}개")
                    logger.info(f"  - 시작 시간: {df['timestamp'].min()}")
                    logger.info(f"  - 종료 시간: {df['timestamp'].max()}")
                    logger.info(f"  - 가격 범위: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
                    logger.info(f"  - 평균 거래량: {df['volume'].mean():.2f}")
                else:
                    logger.warning(f"{symbol} {interval} 데이터가 비어있습니다.")
                
                # API 제한을 피하기 위한 대기
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"{symbol} {interval} 수집 중 오류 발생: {e}")
                continue
    
    logger.info("\n데이터 수집 완료!")

if __name__ == "__main__":
    main()
