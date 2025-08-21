#!/usr/bin/env python3
"""
Binance API 클라이언트
Binance API와 상호작용하기 위한 클라이언트 클래스입니다.
"""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging
from typing import Dict, Any, Optional
import json

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BinanceClient:
    """Binance API 클라이언트"""
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        """초기화"""
        self.base_url = "https://api.binance.com"
        self.api_key = api_key
        self.api_secret = api_secret
        
        # 세션 설정 (재시도 로직 포함)
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # 헤더 설정
        if self.api_key:
            self.session.headers.update({
                'X-MBX-APIKEY': self.api_key
            })
    
    def get_server_time(self) -> Optional[Dict[str, Any]]:
        """서버 시간 조회"""
        try:
            response = self.session.get(f"{self.base_url}/api/v3/time")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"서버 시간 조회 오류: {e}")
            return None
    
    def get_exchange_info(self) -> Optional[Dict[str, Any]]:
        """거래소 정보 조회"""
        try:
            response = self.session.get(f"{self.base_url}/api/v3/exchangeInfo")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"거래소 정보 조회 오류: {e}")
            return None
    
    def get_24hr_ticker(self, symbol: str = None) -> Optional[Dict[str, Any]]:
        """24시간 티커 정보 조회"""
        try:
            url = f"{self.base_url}/api/v3/ticker/24hr"
            if symbol:
                url += f"?symbol={symbol}"
            
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"24시간 티커 조회 오류: {e}")
            return None
    
    def get_klines(self, symbol: str, interval: str = "1m", 
                   limit: int = 500, start_time: int = None, 
                   end_time: int = None) -> Optional[list]:
        """K라인 데이터 조회"""
        try:
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            if start_time:
                params['startTime'] = start_time
            if end_time:
                params['endTime'] = end_time
            
            response = self.session.get(f"{self.base_url}/api/v3/klines", params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"K라인 데이터 조회 오류: {e}")
            return None
    
    def get_order_book(self, symbol: str, limit: int = 100) -> Optional[Dict[str, Any]]:
        """오더북 조회"""
        try:
            params = {
                'symbol': symbol,
                'limit': limit
            }
            
            response = self.session.get(f"{self.base_url}/api/v3/depth", params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"오더북 조회 오류: {e}")
            return None
    
    def get_recent_trades(self, symbol: str, limit: int = 500) -> Optional[list]:
        """최근 거래 내역 조회"""
        try:
            params = {
                'symbol': symbol,
                'limit': limit
            }
            
            response = self.session.get(f"{self.base_url}/api/v3/trades", params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"최근 거래 내역 조회 오류: {e}")
            return None
    
    def get_aggregate_trades(self, symbol: str, limit: int = 500, 
                           from_id: int = None, start_time: int = None, 
                           end_time: int = None) -> Optional[list]:
        """집계 거래 내역 조회"""
        try:
            params = {
                'symbol': symbol,
                'limit': limit
            }
            
            if from_id:
                params['fromId'] = from_id
            if start_time:
                params['startTime'] = start_time
            if end_time:
                params['endTime'] = end_time
            
            response = self.session.get(f"{self.base_url}/api/v3/aggTrades", params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"집계 거래 내역 조회 오류: {e}")
            return None
    
    def get_funding_rate(self, symbol: str = None, limit: int = 500) -> Optional[list]:
        """자금조달률 조회 (선물)"""
        try:
            params = {'limit': limit}
            if symbol:
                params['symbol'] = symbol
            
            response = self.session.get(f"{self.base_url}/fapi/v1/fundingRate", params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"자금조달률 조회 오류: {e}")
            return None
    
    def get_open_interest(self, symbol: str) -> Optional[Dict[str, Any]]:
        """미결제약정 조회 (선물)"""
        try:
            params = {'symbol': symbol}
            
            response = self.session.get(f"{self.base_url}/fapi/v1/openInterest", params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"미결제약정 조회 오류: {e}")
            return None
    
    def get_liquidation_orders(self, symbol: str = None, limit: int = 500) -> Optional[list]:
        """청산 주문 조회 (선물)"""
        try:
            params = {'limit': limit}
            if symbol:
                params['symbol'] = symbol
            
            response = self.session.get(f"{self.base_url}/fapi/v1/allForceOrders", params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"청산 주문 조회 오류: {e}")
            return None
    
    def test_connection(self) -> bool:
        """API 연결 테스트"""
        try:
            server_time = self.get_server_time()
            if server_time and 'serverTime' in server_time:
                logger.info(f"Binance API 연결 성공: {server_time['serverTime']}")
                return True
            return False
        except Exception as e:
            logger.error(f"API 연결 테스트 실패: {e}")
            return False


if __name__ == "__main__":
    # 테스트 코드
    client = BinanceClient()
    
    # 연결 테스트
    if client.test_connection():
        print("✅ Binance API 연결 성공")
        
        # 거래소 정보 조회
        exchange_info = client.get_exchange_info()
        if exchange_info:
            print(f"📊 거래소 정보: {len(exchange_info.get('symbols', []))}개 심볼")
        
        # BTCUSDT 24시간 티커 조회
        btc_ticker = client.get_24hr_ticker("BTCUSDT")
        if btc_ticker:
            print(f"📈 BTCUSDT 24시간 변화: {btc_ticker.get('priceChangePercent', 'N/A')}%")
    else:
        print("❌ Binance API 연결 실패")

