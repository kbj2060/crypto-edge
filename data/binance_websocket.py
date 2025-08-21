import json
import asyncio
import websockets
import threading
import time
from typing import Dict, List, Callable, Optional
from datetime import datetime
import logging

class BinanceWebSocket:
    """바이낸스 웹소켓 클라이언트 - 실시간 청산 데이터 및 Kline 데이터 수집"""
    
    def __init__(self, symbol: str = "ETHUSDT"):
        self.symbol = symbol.lower()
        self.ws_url = "wss://fstream.binance.com/ws"
        self.running = False
        self.callbacks = {
            'liquidation': [],
            'kline_3m': []  # 3분봉 Kline 콜백 추가
        }
        
        # 데이터 저장소
        self.liquidations = []
        self.price_history = []  # 가격 히스토리 추가
        
        # 설정
        self.max_liquidations = 1000  # 최대 저장 청산 데이터 수
        self.max_price_history = 1000  # 최대 저장 가격 데이터 수
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def add_callback(self, event_type: str, callback: Callable):
        """콜백 함수 등록"""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
    
    def remove_callback(self, event_type: str, callback: Callable):
        """콜백 함수 제거"""
        if event_type in self.callbacks:
            if callback in self.callbacks[event_type]:
                self.callbacks[event_type].remove(callback)
    
    async def connect_liquidation_stream(self):
        """청산 데이터 스트림 연결"""
        uri = f"{self.ws_url}/{self.symbol}@forceOrder"
        
        try:
            async with websockets.connect(uri) as websocket:
                self.logger.info(f"청산 스트림 연결됨: {self.symbol}")
                
                async for message in websocket:
                    if not self.running:
                        break
                    
                    try:
                        data = json.loads(message)
                        await self.process_liquidation(data)
                    except json.JSONDecodeError as e:
                        self.logger.error(f"JSON 파싱 오류: {e}")
                    except Exception as e:
                        self.logger.error(f"청산 데이터 처리 오류: {e}")
                        
        except Exception as e:
            self.logger.error(f"청산 스트림 연결 오류: {e}")
    
    async def connect_kline_3m_stream(self):
        """3분봉 Kline 스트림 연결"""
        uri = f"{self.ws_url}/{self.symbol}@kline_3m"
        
        try:
            async with websockets.connect(uri) as websocket:
                self.logger.info(f"3분봉 Kline 스트림 연결됨: {self.symbol}")
                
                async for message in websocket:
                    if not self.running:
                        break
                    
                    try:
                        data = json.loads(message)
                        await self.process_kline_3m(data)
                    except json.JSONDecodeError as e:
                        self.logger.error(f"JSON 파싱 오류: {e}")
                    except Exception as e:
                        self.logger.error(f"Kline 데이터 처리 오류: {e}")
                        
        except Exception as e:
            self.logger.error(f"3분봉 Kline 스트림 연결 오류: {e}")
    
    '''
    웹소켓 청산 데이터 처리
    {
        "e": "forceOrder",  // 이벤트 유형
        "E": 1713772800000, // 이벤트 시간
        "o": {
            "s": "BTCUSDT", // 심볼
            "S": "SELL",    // 방향
            "q": "0.001",   // 수량
            "p": "10000",   // 가격
            "T": 1713772800000 // 시간
        }
    }
    '''
    async def process_liquidation(self, data: Dict):
        """청산 데이터 처리"""
        try:
            if 'o' in data:  # 청산 이벤트
                # qty_usd 계산 (수량 × 가격)
                qty_usd = float(data['o']['q']) * float(data['o']['p'])
                
                liquidation = {
                    'timestamp': datetime.now(),
                    'symbol': data['o']['s'],
                    'side': data['o']['S'],  # BUY/SELL
                    'quantity': float(data['o']['q']),
                    'price': float(data['o']['p']),
                    'qty_usd': qty_usd,  # USD 기준 청산 금액
                    'time': data['o']['T']
                }
                
                # 콜백 실행
                # integrated_smart_trader.py 에서 _setup_callbacks 함수에서 청산 이벤트 처리
                for callback in self.callbacks['liquidation']:
                    try:
                        callback(liquidation)
                    except Exception as e:
                        self.logger.error(f"청산 콜백 실행 오류: {e}")
                                
        except Exception as e:
            self.logger.error(f"청산 데이터 처리 오류: {e}")
    
    async def process_kline_3m(self, data: Dict):
        """3분봉 Kline 데이터 처리"""
        try:
            if 'k' in data:  # Kline 이벤트
                kline = data['k']
                
                # 3분봉 마감 체크 (k.x == true)
                if kline.get('x', False):
                    # 가격 데이터 저장
                    price_data = {
                        'timestamp': datetime.now(),
                        'price': float(kline['c']),  # 종가
                        'open': float(kline['o']),
                        'high': float(kline['h']),
                        'low': float(kline['l']),
                        'close': float(kline['c']),
                        'volume': float(kline['v']),
                        'close_time': kline['t']  # 캔들 종료 시간
                    }
                    
                    # 가격 히스토리에 추가
                    self.price_history.append(price_data)
                    
                    # 최대 개수 제한
                    if len(self.price_history) > self.max_price_history:
                        self.price_history.pop(0)
                    
                    # 3분봉 마감 콜백 실행
                    for callback in self.callbacks['kline_3m']:
                        try:
                            callback(price_data)
                        except Exception as e:
                            self.logger.error(f"3분봉 Kline 콜백 실행 오류: {e}")
                                                            
        except Exception as e:
            self.logger.error(f"3분봉 Kline 데이터 처리 오류: {e}")
    
    async def start(self):
        """웹소켓 스트림 시작"""
        self.running = True
        self.logger.info("웹소켓 스트림 시작")
        
        # 여러 스트림을 동시에 실행
        tasks = [
            self.connect_liquidation_stream(),
            self.connect_kline_3m_stream(),  # 3분봉 Kline 스트림 추가
        ]
        
        await asyncio.gather(*tasks)
    
    def stop(self):
        """웹소켓 스트림 중지"""
        self.running = False
        self.logger.info("웹소켓 스트림 중지")
    
    def start_background(self):
        """백그라운드에서 웹소켓 실행"""
        def run_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.start())
        
        self.thread = threading.Thread(target=run_async, daemon=True)
        self.thread.start()
        self.logger.info("백그라운드 웹소켓 시작됨")
    
