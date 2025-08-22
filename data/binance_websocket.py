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
            'kline_1m': []   # 1분봉 Kline 콜백만 사용
        }
        
        # 데이터 저장소
        self.liquidations = []
        self.price_history = []  # 가격 히스토리 추가
        self.liquidation_bucket = []  # 청산 버킷 추가
        self.bucket_start_time = datetime.now()  # 버킷 시작 시간
        
        # 설정
        self.max_liquidations = 1000  # 최대 저장 청산 데이터 수
        self.max_price_history = 1000  # 최대 저장 가격 데이터 수
        
        # 전략 실행기 (나중에 설정)
        self.session_strategy = None
        self.advanced_liquidation_strategy = None
        
        # 1분봉 카운터 (3분봉 시뮬레이션용)
        self.minute_counter = 0
        
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
    
    def set_strategies(self, session_strategy=None, advanced_liquidation_strategy=None):
        """전략 실행기 설정"""
        self.session_strategy = session_strategy
        self.advanced_liquidation_strategy = advanced_liquidation_strategy
        self.logger.info("전략 실행기 설정 완료")
    
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
    
    async def connect_kline_1m_stream(self):
        """1분봉 Kline 스트림 연결"""
        uri = f"{self.ws_url}/{self.symbol}@kline_1m"
        
        try:
            async with websockets.connect(uri) as websocket:
                self.logger.info(f"1분봉 Kline 스트림 연결됨: {self.symbol}")
                
                async for message in websocket:
                    if not self.running:
                        break
                    
                    try:
                        data = json.loads(message)
                        await self.process_kline_1m(data)
                    except json.JSONDecodeError as e:
                        self.logger.error(f"JSON 파싱 오류: {e}")
                    except Exception as e:
                        self.logger.error(f"Kline 데이터 처리 오류: {e}")
                        
        except Exception as e:
            self.logger.error(f"1분봉 Kline 스트림 연결 오류: {e}")
    
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
                
                # 청산 버킷에 추가
                self.liquidations.append(liquidation)
                self.liquidation_bucket.append(liquidation)
                
                # 최대 개수 제한
                if len(self.liquidations) > self.max_liquidations:
                    self.liquidations.pop(0)
                
                # 콜백 실행
                for callback in self.callbacks['liquidation']:
                    try:
                        callback(liquidation)
                    except Exception as e:
                        self.logger.error(f"청산 콜백 실행 오류: {e}")
                                
        except Exception as e:
            self.logger.error(f"청산 데이터 처리 오류: {e}")
    
    async def process_kline_1m(self, data: Dict):
        """1분봉 Kline 데이터 처리 - 3분봉 시뮬레이션 포함"""
        try:
            if 'k' in data:  # Kline 이벤트
                kline = data['k']
                
                # 1분봉 마감 체크 (k.x == true)
                if kline.get('x', True):  # 마감된 캔들만
                    print(f"⏰ 1분봉 마감 감지: {datetime.now().strftime('%H:%M:%S')}")
                    
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
                    
                    # 1분봉 카운터 증가
                    self.minute_counter += 1
                    
                    # 청산 전략 실행 (매 1분마다)
                    self.liquidation_bucket = [
                        {
                            'timestamp': datetime.now(),
                            'symbol': 'ETHUSDT',
                            'side': 'SELL',  # 롱 청산
                            'quantity': 0.5,
                            'price': 3456.78,
                            'qty_usd': 1728.39,
                            'time': 1735123456789
                        },
                        {
                            'timestamp': datetime.now(),
                            'symbol': 'ETHUSDT',
                            'side': 'BUY',   # 숏 청산
                            'quantity': 1.2,
                            'price': 3457.12,
                            'qty_usd': 4148.54,
                            'time': 1735123459123
                        },
                        {
                            'timestamp': datetime.now(),
                            'symbol': 'ETHUSDT',
                            'side': 'SELL',  # 롱 청산
                            'quantity': 0.8,
                            'price': 3455.90,
                            'qty_usd': 2764.72,
                            'time': 1735123461456
                        },
                        {
                            'timestamp': datetime.now(),
                            'symbol': 'ETHUSDT',
                            'side': 'SELL',  # 롱 청산
                            'quantity': 2.1,
                            'price': 3454.33,
                            'qty_usd': 7254.09,
                            'time': 1735123463789
                        },
                        {
                            'timestamp': datetime.now(),
                            'symbol': 'ETHUSDT',
                            'side': 'BUY',   # 숏 청산
                            'quantity': 0.3,
                            'price': 3458.67,
                            'qty_usd': 1037.60,
                            'time': 1735123465012
                        },
                        {
                            'timestamp': datetime.now(),
                            'symbol': 'ETHUSDT',
                            'side': 'SELL',  # 롱 청산
                            'quantity': 1.7,
                            'price': 3453.21,
                            'qty_usd': 5870.46,
                            'time': 1735123467345
                        },
                        {
                            'timestamp': datetime.now(),
                            'symbol': 'ETHUSDT',
                            'side': 'BUY',   # 숏 청산
                            'quantity': 0.9,
                            'price': 3459.84,
                            'qty_usd': 3113.86,
                            'time': 1735123469678
                        },
                        {
                            'timestamp': datetime.now(),
                            'symbol': 'ETHUSDT',
                            'side': 'SELL',  # 롱 청산
                            'quantity': 3.5,
                            'price': 3452.90,
                            'qty_usd': 12085.15,
                            'time': 1735123471901
                        },
                        {
                            'timestamp': datetime.now(),
                            'symbol': 'ETHUSDT',
                            'side': 'BUY',   # 숏 청산
                            'quantity': 0.6,
                            'price': 3460.12,
                            'qty_usd': 2076.07,
                            'time': 1735123474234
                        },
                        {
                            'timestamp': datetime.now(),
                            'symbol': 'ETHUSDT',
                            'side': 'SELL',  # 롱 청산
                            'quantity': 1.4,
                            'price': 3451.75,
                            'qty_usd': 4832.45,
                            'time': 1735123476567
                        }
                    ]
                    if self.advanced_liquidation_strategy and self.liquidation_bucket:
                        try:
                            print(f"🎯 청산 전략 실행 시작... (버킷 크기: {len(self.liquidation_bucket)})")
                            
                            # 현재 가격 가져오기
                            current_price = float(kline['c'])
                            
                            key_levels = self.advanced_liquidation_strategy.calculate_key_levels(self.price_history)
                            opening_range = self.advanced_liquidation_strategy.calculate_opening_range(self.price_history)
                            vwap = self.advanced_liquidation_strategy.calculate_vwap(self.price_history)
                            vwap_std = self.advanced_liquidation_strategy.calculate_vwap_std(self.price_history)
                            atr = self.advanced_liquidation_strategy.calculate_atr(self.price_history)
                            
                            # 청산 전략 분석
                            signal = self.advanced_liquidation_strategy.analyze_bucket_liquidations(
                                self.liquidation_bucket, current_price, key_levels, opening_range, vwap, vwap_std, atr
                            )
                            
                            if signal:
                                print(f"⚡ 청산 신호 감지: {signal.get('action', 'UNKNOWN')} - {signal.get('tier', 'UNKNOWN')}")
                                # 여기서 신호 출력 로직 추가
                            else:
                                print(f"📊 청산 신호 없음")
                            
                            # 버킷 초기화
                            self.liquidation_bucket = []
                            self.bucket_start_time = datetime.now()
                            print(f"🔄 청산 버킷 초기화 완료")
                            
                        except Exception as e:
                            self.logger.error(f"청산 전략 실행 오류: {e}")
                    
                    # 세션 전략 실행 (3분마다 - 3분봉 시뮬레이션)
                    if self.minute_counter % 3 == 0:
                        if self.session_strategy:
                            try:
                                print(f"🎯 세션 전략 실행 시작... (3분봉 시뮬레이션)")
                                # 여기서 세션 전략 실행 로직 추가
                                # self.session_strategy.analyze_session(...)
                                print(f"✅ 세션 전략 실행 완료")
                            except Exception as e:
                                self.logger.error(f"세션 전략 실행 오류: {e}")
                    
                    # 1분봉 콜백 실행
                    for callback in self.callbacks['kline_1m']:
                        try:
                            callback(price_data)
                        except Exception as e:
                            self.logger.error(f"1분봉 Kline 콜백 실행 오류: {e}")
        
        except Exception as e:
            self.logger.error(f"1분봉 Kline 데이터 처리 오류: {e}")
    
    async def start(self):
        """웹소켓 스트림 시작"""
        self.running = True
        self.logger.info("웹소켓 스트림 시작")
        
        # 여러 스트림을 동시에 실행
        tasks = [
            self.connect_liquidation_stream(),
            self.connect_kline_1m_stream(),  # 1분봉 Kline 스트림 추가
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
    
