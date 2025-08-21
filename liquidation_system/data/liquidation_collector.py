#!/usr/bin/env python3
"""
실시간 청산 데이터 수집기
Binance WebSocket을 통해 실시간 청산 데이터를 수집하고 데이터베이스에 저장합니다.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
import websockets
import time
import signal
import sys
from pathlib import Path

# 상대 경로로 import
from .liquidation_database import LiquidationDatabase
from .binance_client import BinanceClient

# 로깅 설정 (파일에만, 콘솔은 청산 데이터만)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('liquidation_collector.log')
    ]
)
logger = logging.getLogger(__name__)


class LiquidationCollector:
    """실시간 청산 데이터 수집기"""
    
    def __init__(self, symbols: List[str] = None, db_path: str = "data/liquidations.db"):
        """초기화"""
        self.symbols = symbols or ['ETHUSDT']
        self.db = LiquidationDatabase(db_path)
        self.binance_client = BinanceClient()
        self.running = False
        self.websocket = None
        
        # 시그널 핸들러 설정
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """시그널 핸들러"""
        logger.info(f"시그널 {signum} 수신, 종료 중...")
        
        # 즉시 종료 플래그 설정
        self.running = False
        
        # 강제 종료를 위한 추가 처리
        import os
        if signum in [signal.SIGINT, signal.SIGTERM]:
            # 비동기 작업을 강제로 중단하기 위해 이벤트 루프 중단
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.stop()
            except:
                pass
            
            # 강제 종료
            os._exit(0)
    
    async def connect_websocket(self):
        """WebSocket 연결 - binance_websocket.py와 동일한 방식"""
        try:
            # Binance Futures 청산 스트림 URL (binance_websocket.py와 동일)
            if len(self.symbols) == 1:
                # 단일 심볼의 경우
                stream_url = f"wss://fstream.binance.com/ws/{self.symbols[0].lower()}@forceOrder"
            else:
                # 여러 심볼의 경우 
                stream_names = [f"{symbol.lower()}@forceOrder" for symbol in self.symbols]
                stream_url = f"wss://fstream.binance.com/ws/{'/'.join(stream_names)}"
            
            logger.info(f"WebSocket 연결 시도: {stream_url}")
            
            self.websocket = await websockets.connect(stream_url)
            logger.info("WebSocket 연결 성공")
            
        except Exception as e:
            logger.error(f"WebSocket 연결 실패: {e}")
            raise
    
    async def process_liquidation_event(self, event_data: Dict[str, Any]):
        """청산 이벤트 처리 - binance_websocket.py와 동일한 구조"""
        try:
            # binance_websocket.py와 동일한 구조: 'o' 키 확인
            if 'o' not in event_data:
                return
            
            liquidation_data = event_data['o']
            
            # 필수 필드 확인
            required_fields = ['s', 'S', 'q', 'p', 'T']
            if not all(field in liquidation_data for field in required_fields):
                return
            
            # binance_websocket.py와 동일한 방식으로 청산 이벤트 파싱
            symbol = liquidation_data['s']  # 심볼
            side = liquidation_data['S']    # 사이드 (BUY=숏청산, SELL=롱청산)
            quantity = float(liquidation_data['q'])  # 수량
            price = float(liquidation_data['p'])    # 청산 가격
            time_ms = liquidation_data['T']  # 타임스탬프
            
            # USDT 가치 계산
            usdt_value = quantity * price
            
            # 청산 방향성 해석 (websocket_handler.py와 동일한 방식)
            if side == 'SELL':
                liquidation_type = "롱 포지션 강제 청산"
                emoji = "📉"
            elif side == 'BUY':
                liquidation_type = "숏 포지션 강제 청산"
                emoji = "📈"
            else:
                liquidation_type = f"{side} 청산"
                emoji = "🔥"
            
            # 심볼에서 USDT 제거
            clean_symbol = symbol.replace('USDT', '')
            
            # websocket_handler.py와 동일한 출력 형식
            print(f"{emoji} {liquidation_type}: {quantity:.2f} {clean_symbol} (${usdt_value:,.0f}) @ ${price:.2f}")
            
            # 데이터베이스에 저장 (binance_websocket.py 구조와 호환)
            liquidation_event = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': price,
                'time': time_ms,
                'usdt_value': usdt_value
            }
            
            self.db.insert_liquidation_event(liquidation_event)
            
        except Exception as e:
            logger.error(f"청산 이벤트 처리 오류: {e}")
    
    async def listen_liquidations(self):
        """청산 데이터 수신 및 처리 - binance_websocket.py와 동일한 방식"""
        try:
            # binance_websocket.py와 동일한 방식으로 연결
            async for message in self.websocket:
                if not self.running:
                    break
                
                try:
                    # JSON 파싱
                    event_data = json.loads(message)
                    
                    # binance_websocket.py와 동일하게 'o' 키가 있는 청산 이벤트만 처리
                    await self.process_liquidation_event(event_data)
                    
                except json.JSONDecodeError as e:
                    logger.error(f"JSON 파싱 오류: {e}")
                except Exception as e:
                    logger.error(f"청산 데이터 처리 오류: {e}")
                    
        except Exception as e:
            logger.error(f"청산 스트림 연결 오류: {e}")
    
    async def start(self):
        """수집기 시작 - binance_websocket.py와 동일한 방식"""
        try:
            self.running = True
            logger.info("청산 데이터 수집기 시작")
            
            # WebSocket 연결
            await self.connect_websocket()
            
            # 청산 데이터 수신 시작 (binance_websocket.py와 동일한 방식)
            await self.listen_liquidations()
            
        except Exception as e:
            logger.error(f"수집기 시작 오류: {e}")
        finally:
            await self.stop()
    
    async def stop(self):
        """수집기 종료"""
        logger.info("청산 데이터 수집기 종료 중...")
        self.running = False
        
        # WebSocket 연결 종료
        if self.websocket:
            try:
                if self.websocket.state.name == 'OPEN':
                    await self.websocket.close()
            except Exception as e:
                logger.error(f"WebSocket 연결 종료 중 오류: {e}")
            finally:
                self.websocket = None
        
        logger.info("청산 데이터 수집기 종료 완료")
    
# get_status 메서드 제거 - 불필요한 상태 정보 제거


# MockLiquidationCollector 클래스 제거 - 실제 청산 데이터만 처리

# generate_test_liquidation 메서드 제거 - 실제 청산 데이터만 처리


# 독립 실행 코드 제거 - run.py를 통해서만 실행
