import json
import asyncio
import websockets
import threading
import time
import requests  # pip install requests 필요
from typing import Dict, List, Callable, Optional
from datetime import datetime, timedelta, timezone
import pandas as pd
import logging

# Global Indicator Manager import
from indicators.global_indicators import get_global_indicator_manager
# Time Manager import
from utils.time_manager import get_time_manager

class BinanceWebSocket:
    """바이낸스 웹소켓 클라이언트 - 실시간 청산 데이터 및 Kline 데이터 수집"""
    
    def __init__(self, symbol: str = "ETHUSDT"):
        """웹소켓 초기화"""
        self.symbol = symbol.lower()
        self.ws_url = "wss://fstream.binance.com/ws"
        self.running = False
        
        # 콜백 함수들
        self.callbacks = {
            'liquidation': [],
            'kline_1m': []   # 1분봉 Kline 콜백만 사용
        }
        
        # TimeManager 초기화
        self.time_manager = get_time_manager()
        
        # Global Indicator Manager 초기화
        self.global_manager = get_global_indicator_manager()
        
        # 데이터 저장소
        self.liquidations = []
        self.liquidation_bucket = []  # 청산 버킷 추가
        self.bucket_start_time = self.time_manager.get_current_time()  # 버킷 시작 시간
        
        # 설정
        self.max_liquidations = 1000  # 최대 저장 청산 데이터 수
        
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
        
        async with websockets.connect(uri) as websocket:
            self.logger.info(f"1분봉 Kline 스트림 연결됨: {self.symbol}")
            
            async for message in websocket:
                if not self.running:
                    break
                
                data = json.loads(message)
                await self.process_kline_1m(data)
    
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
                    'timestamp': self.time_manager.get_current_time(),
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
        if 'k' not in data:  # Kline 이벤트가 아니면 종료
            return
            
        kline = data['k']
        
        # 1분봉 마감 체크 (k.x == true)
        if not kline.get('x', True):  # 마감되지 않은 캔들이면 종료
            return
            
        print(f"⏰ 1분봉 마감 감지: {(self.time_manager.get_current_time() + timedelta(seconds=1)).strftime('%H:%M:%S')}")
        
        # 가격 데이터 생성 (1분봉은 DataManager에 추가하지 않음)
        price_data = self._create_price_data(kline)
        
        # 1분봉 데이터를 임시 저장 (3분봉 생성용)
        self._store_1min_data(price_data)
        
        # 1분봉 카운터 증가
        self.minute_counter += 1
        
        # 청산 전략 실행 (매 1분마다)
        if self.advanced_liquidation_strategy:
            await self._execute_liquidation_strategy()
        
        # 세션 전략 실행 (3분마다)
        if self.minute_counter % 3 == 0:
            # 3분봉 데이터 생성
            df_3m = self._create_3min_candle()
            if df_3m is None:
                return
            await self._execute_session_strategy(df_3m)
        
        # 1분봉 콜백 실행
        self._execute_kline_callbacks(price_data)
            
    
    def _create_price_data(self, kline: Dict) -> Dict:
        """가격 데이터 생성"""
        return {
            'open': float(kline['o']),
            'high': float(kline['h']),
            'low': float(kline['l']),
            'close': float(kline['c']),
            'volume': float(kline['v']),      # VWAP용: base volume (ETH)
            'quote_volume': float(kline['q']), # VPVR용: quote volume (USDT)
            'timestamp': kline['t']           # 캔들 종료 시간
        }
    
    def _store_1min_data(self, price_data: Dict):
        """1분봉 데이터를 임시 저장 (3분봉 생성용)"""
        try:
            # 최근 3개 1분봉 데이터 저장
            if not hasattr(self, '_recent_1min_data'):
                self._recent_1min_data = []
            
            self._recent_1min_data.append(price_data)
            
            # 최대 3개까지만 유지
            if len(self._recent_1min_data) > 3:
                self._recent_1min_data = self._recent_1min_data[-3:]
                
        except Exception as e:
            self.logger.error(f"1분봉 데이터 임시 저장 오류: {e}")
    
    async def _execute_liquidation_strategy(self):
        """청산 전략 실행"""
        print(f"🎯 청산 전략 실행 시작... (버킷 크기: {len(self.liquidation_bucket)})")
        
        # 청산 전략 분석
        signal = self.advanced_liquidation_strategy.analyze_bucket_liquidations(self.liquidation_bucket)
        
        if signal:
            print(f"⚡ 청산 신호 감지: {signal.get('action', 'UNKNOWN')} - {signal.get('tier', 'UNKNOWN')}")
        else:
            print(f"📊 청산 신호 없음")
        
        # 버킷 초기화
        self.liquidation_bucket = []
        self.bucket_start_time = self.time_manager.get_current_time()
        print(f"🔄 청산 버킷 초기화 완료")
            
    
    async def _execute_session_strategy(self, df_3m: pd.DataFrame):
        """세션 전략 실행"""
        if not self.session_strategy:
            return
        print(df_3m)
        # 글로벌 지표 업데이트
        self.global_manager.update_all_indicators(df_3m.reset_index().iloc[0])
        
        # 전략 분석에 필요한 데이터 수집
        strategy_data = self._collect_strategy_data()
        
        # 세션 전략 분석 실행
        session_signal = self.session_strategy.analyze_session_strategy(
            df_3m, strategy_data['key_levels'], self.time_manager.get_current_time()
        )
        
        # 신호 결과 출력
        self._print_session_signal(session_signal)
    
    def _create_3min_candle(self) -> Optional[pd.DataFrame]:
        """3분봉 데이터 생성 (저장된 1분봉 데이터 사용)"""
        try:
            # 저장된 1분봉 데이터 확인
            if not hasattr(self, '_recent_1min_data') or len(self._recent_1min_data) < 3:
                return None
            
            recent_3_candles = self._recent_1min_data[-3:]
            print(recent_3_candles)
            # 3분봉 데이터 생성 (OHLCV)
            df_3m = pd.DataFrame([{
                'open': float(recent_3_candles[0]['open']),
                'high': max(float(candle['high']) for candle in recent_3_candles),
                'low': min(float(candle['low']) for candle in recent_3_candles),
                'close': float(recent_3_candles[-1]['close']),
                'volume': sum(float(candle['volume']) for candle in recent_3_candles),
                'quote_volume': sum(float(candle['quote_volume']) for candle in recent_3_candles)
            }], index=[recent_3_candles[-1]['timestamp']])
            
            return df_3m
            
        except Exception as e:
            self.logger.error(f"3분봉 데이터 생성 오류: {e}")
            return None
    
    def _collect_strategy_data(self) -> Dict:
        """전략 분석에 필요한 데이터 수집"""
        strategy_data = {
            'key_levels': {},
            'opening_range': {},
            'vwap': 0.0,
            'vwap_std': 0.0,
            'atr': 0.0
        }
        
        try:
            # 키 레벨 (Daily Levels)
            daily_levels = self.global_manager.get_indicator('daily_levels')
            if daily_levels:
                prev_day_data = daily_levels.get_status()
                strategy_data['key_levels'] = {
                    'prev_day_high': prev_day_data.get('prev_day_high'),
                    'prev_day_low': prev_day_data.get('prev_day_low')
                }
            
            # Opening Range 정보
            try:
                session_config = self.time_manager.get_indicator_mode_config()
                opening_range = self.global_manager.get_indicator('opening_range').get_status()
                if session_config.get('use_session_mode'):
                    strategy_data['opening_range'] = {
                        'high': opening_range.get('high'),
                        'low': opening_range.get('low'),
                        'session_name': session_config.get('session_name', 'UNKNOWN'),
                        'session_start': session_config.get('session_start_time'),
                        'elapsed_minutes': session_config.get('elapsed_minutes', 0),
                        'session_status': session_config.get('session_status', 'UNKNOWN')
                    }
            except Exception:
                pass
            
            # VWAP 및 VWAP 표준편차
            vwap_indicator = self.global_manager.get_indicator('vwap')
            if vwap_indicator:
                vwap_status = vwap_indicator.get_status()
                strategy_data['vwap'] = vwap_status.get('current_vwap')
                strategy_data['vwap_std'] = vwap_status.get('current_vwap_std')
            
            # ATR
            atr_indicator = self.global_manager.get_indicator('atr')
            if atr_indicator:
                strategy_data['atr'] = atr_indicator.get_status().get('current_atr')
                
        except Exception as e:
            self.logger.error(f"전략 데이터 수집 오류: {e}")
        
        return strategy_data
    
    def _print_session_signal(self, session_signal: Optional[Dict]):
        """세션 전략 신호 결과 출력"""
        if not session_signal:
            print(f"📊 세션 전략 신호 없음")
            return
        
        print(f"🎯 세션 전략 신호: {session_signal.get('playbook', 'UNKNOWN')} {session_signal.get('side', 'UNKNOWN')} | {session_signal.get('stage', 'UNKNOWN')} | {session_signal.get('confidence', 0):.0%}")
        
        # Entry 신호인 경우 핵심 정보만
        if session_signal.get('stage') == 'ENTRY':
            entry_price = session_signal.get('entry_price', 0)
            stop_loss = session_signal.get('stop_loss', 0)
            take_profit = session_signal.get('take_profit1', 0)
            
            if entry_price and stop_loss and take_profit:
                risk = abs(entry_price - stop_loss)
                reward = abs(take_profit - entry_price)
                rr_ratio = reward / risk if risk > 0 else 0
                print(f"💰 진입: ${entry_price:.2f} | 손절: ${stop_loss:.2f} | 목표: ${take_profit:.2f} | R/R: {rr_ratio:.2f}")
    
    def _execute_kline_callbacks(self, price_data: Dict):
        """1분봉 Kline 콜백 실행"""
        for callback in self.callbacks['kline_1m']:
            try:
                callback(price_data)
            except Exception as e:
                self.logger.error(f"1분봉 Kline 콜백 실행 오류: {e}")
    
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
    
