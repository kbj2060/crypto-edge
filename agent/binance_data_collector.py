#!/usr/bin/env python3
"""
바이낸스 API를 사용하여 3분봉과 5분봉 데이터를 수집하고 CSV로 저장하는 스크립트
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import os
from typing import List, Dict, Any
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BinanceDataCollector:
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        self.session = requests.Session()
        
    def get_klines(self, symbol: str, interval: str, start_time: int = None, end_time: int = None, limit: int = 1000) -> List[List]:
        """
        바이낸스 API에서 Kline 데이터를 가져옵니다.
        
        Args:
            symbol: 거래쌍 (예: 'BTCUSDT')
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
            response = self.session.get(f"{self.base_url}/klines", params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API 요청 실패: {e}")
            return []
    
    def get_historical_data(self, symbol: str, interval: str, days_back: int = 365) -> pd.DataFrame:
        """
        지정된 기간 동안의 히스토리컬 데이터를 가져옵니다.
        
        Args:
            symbol: 거래쌍
            interval: 시간 간격
            days_back: 몇 일 전까지의 데이터를 가져올지
        """
        logger.info(f"{symbol} {interval} 데이터 수집 시작 (최근 {days_back}일)")
        
        # 종료 시간 (현재 시간)
        end_time = int(datetime.now().timestamp() * 1000)
        
        # 시작 시간 (days_back일 전의 0시 0분)
        start_time = int((datetime.now() - timedelta(days=days_back)).replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
        
        all_data = []
        current_end_time = end_time
        
        while current_end_time > start_time:
            # API 제한을 고려하여 1000개씩 가져오기
            klines = self.get_klines(symbol, interval, end_time=current_end_time, limit=1000)
            
            if not klines:
                logger.warning(f"데이터를 가져올 수 없습니다: {symbol} {interval}")
                break
                
            all_data.extend(klines)
            
            # 다음 배치를 위한 시간 설정 (가장 오래된 데이터의 시간 - 1ms)
            current_end_time = klines[0][0] - 1
            
            # API 제한을 피하기 위한 대기
            time.sleep(0.1)
            
            logger.info(f"수집된 데이터: {len(all_data)}개")
        
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
        
        logger.info(f"{symbol} {interval} 데이터 수집 완료: {len(df)}개")
        return df
    
    def save_to_csv(self, df: pd.DataFrame, symbol: str, interval: str, output_dir: str = "data"):
        """DataFrame을 CSV 파일로 저장합니다."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        filename = f"{symbol}_{interval}_historical_data.csv"
        filepath = os.path.join(output_dir, filename)
        
        df.to_csv(filepath, index=False)
        logger.info(f"데이터 저장 완료: {filepath}")
        return filepath

def main():
    """메인 실행 함수"""
    collector = BinanceDataCollector()
    
    # 수집할 거래쌍과 시간 간격
    symbols = ['ETHUSDC']
    intervals = [ '1h', '3m', '15m']
    
    # 수집할 기간 (일)
    days_back = 365  # 1년치 데이터
    
    logger.info("바이낸스 데이터 수집 시작")
    logger.info(f"거래쌍: {symbols}")
    logger.info(f"시간 간격: {intervals}")
    logger.info(f"수집 기간: 최근 {days_back}일")
    
    for symbol in symbols:
        for interval in intervals:
            try:
                logger.info(f"\n{'='*50}")
                logger.info(f"수집 중: {symbol} {interval}")
                logger.info(f"{'='*50}")
                
                # 데이터 수집
                df = collector.get_historical_data(symbol, interval, days_back)
                
                if not df.empty:
                    # CSV 저장
                    filepath = collector.save_to_csv(df, symbol, interval)
                    
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
