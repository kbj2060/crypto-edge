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
from data.bucket_aggregator import BucketAggregator
from data.data_manager import get_data_manager
from indicators.global_indicators import get_atr, get_global_indicator_manager, get_vwap
# Time Manager import
from signals import vpvr_golden_strategy
from utils.time_manager import get_time_manager
# Binance Data Loader import
from data.binance_dataloader import BinanceDataLoader

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
        self.bucket_aggregator = BucketAggregator()
        
        # TimeManager 초기화
        self.time_manager = get_time_manager()
        
        # Global Indicator Manager 초기화
        self.global_manager = get_global_indicator_manager()
        
        self.data_manager = get_data_manager()

        # Binance Data Loader 초기화
        self.data_loader = BinanceDataLoader()
        
        # 데이터 저장소
        self.liquidation_bucket = []  # 청산 버킷 추가
        self.bucket_start_time = self.time_manager.get_current_time()  # 버킷 시작 시간
        
        # 설정
        self.max_liquidations = 1000  # 최대 저장 청산 데이터 수
        
        # 전략 실행기 (외부에서 주입받음 - 실행 엔진 역할)
        self.session_strategy = None
        self.advanced_liquidation_strategy = None
        self.vpvr_golden_strategy = None
        
        # 진행 중인 3분봉 데이터 관리
        self._recent_1min_data = []  # 최근 1분봉 데이터 (웹소켓으로 수집)
        self._first_3min_candle_closed = False  # 첫 3분봉 마감 여부 추적
        
    
    def add_callback(self, event_type: str, callback: Callable):
        """콜백 함수 등록"""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
    
    def remove_callback(self, event_type: str, callback: Callable):
        """콜백 함수 제거"""
        if event_type in self.callbacks:
            if callback in self.callbacks[event_type]:
                self.callbacks[event_type].remove(callback)
    
    def set_strategies(
            self, 
            session_strategy=None, 
            advanced_liquidation_strategy=None, 
            squeeze_momentum_strategy=None, 
            fade_reentry_strategy=None,
            liquidation_strategy=None,
            vpvr_golden_strategy=None
            ):
        """전략 실행기 설정 - 실행 엔진에서 외부 전략 인스턴스 수신"""
        try:
            # 전략 인스턴스 검증 및 설정
            if session_strategy is not None:
                self.session_strategy = session_strategy
                print(f"✅ 세션 전략 설정 완료: {type(session_strategy).__name__}")
            
            if advanced_liquidation_strategy is not None:
                self.advanced_liquidation_strategy = advanced_liquidation_strategy
                print(f"✅ 고급 청산 전략 설정 완료: {type(advanced_liquidation_strategy).__name__}")
            
            if squeeze_momentum_strategy is not None:
                self.squeeze_momentum_strategy = squeeze_momentum_strategy
                print(f"✅ SQUEEZE 모멘텀 전략 설정 완료: {type(squeeze_momentum_strategy).__name__}")
            
            if fade_reentry_strategy is not None:
                self.fade_reentry_strategy = fade_reentry_strategy
                print(f"✅ 페이드 리입 전략 설정 완료: {type(fade_reentry_strategy).__name__}")
                
            if vpvr_golden_strategy is not None:
                self.vpvr_golden_strategy = vpvr_golden_strategy
                print(f"✅ VPVR 골든 포켓 전략 설정 완료: {type(vpvr_golden_strategy).__name__}")
                
        except Exception as e:
            print(f"❌ 전략 설정 오류: {e}")
            import traceback
            traceback.print_exc()
    
    async def connect_liquidation_stream(self):
        """청산 데이터 스트림 연결"""
        uri = f"{self.ws_url}/{self.symbol}@forceOrder"
        
        try:
            async with websockets.connect(uri) as websocket:
                async for message in websocket:
                    if not self.running:
                        break
                    
                    try:
                        data = json.loads(message)
                        await self.process_liquidation(data)
                    except json.JSONDecodeError as e:
                        print(f"JSON 파싱 오류: {e}")
                    except Exception as e:
                        print(f"청산 데이터 처리 오류: {e}")
                        
        except Exception as e:
            print(f"청산 스트림 연결 오류: {e}")
    
    async def connect_kline_1m_stream(self):
        """1분봉 Kline 스트림 연결"""
        uri = f"{self.ws_url}/{self.symbol}@kline_1m"
        
        async with websockets.connect(uri) as websocket:
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
        if 'o' in data:  # 청산 이벤트
            # qty_usd 계산 (수량 × 가격)
            qty_usd = float(data['o']['q']) * float(data['o']['p'])
            
            liquidation = {
                'timestamp': self.time_manager.get_current_time(),
                'symbol': data['o']['s'],
                'side': data['o']['S'],  # BUY/SELL
                'size': float(data['o']['q']),
                'price': float(data['o']['p']),
                'qty_usd': qty_usd,  # USD 기준 청산 금액
                'time': data['o']['T']
            }
            
            # 청산 버킷에 추가
            self.liquidation_bucket.append(liquidation)
            self.bucket_aggregator.add_liquidation_event(liquidation)
            
            # 최대 개수 제한
            if len(self.liquidation_bucket) > self.max_liquidations:
                self.liquidation_bucket.pop(0)
            
            # 콜백 실행
            for callback in self.callbacks['liquidation']:
                try:
                    callback(liquidation)
                except Exception as e:
                    print(f"청산 콜백 실행 오류: {e}")
                                
    
    async def process_kline_1m(self, data: Dict):
        """1분봉 Kline 데이터 처리 - 3분봉 시뮬레이션 포함"""
        if 'k' not in data:  # Kline 이벤트가 아니면 종료
            return
        kline = data['k']
        
        # 1분봉 마감 체크 (k.x == true)
        if not kline.get('x', True):  # 마감되지 않은 캔들이면 종료
            return
        
        # 매 1분 세션 정보 업데이트트
        self.time_manager.update_session_status()

        print(f"\n⏰ OPEN TIME : {(self.time_manager.get_current_time() + timedelta(seconds=1)).strftime('%H:%M:%S')}")
        
        # 가격 데이터 생성 (1분봉은 DataManager에 추가하지 않음)
        price_data = self._create_price_data(kline)

        # 1분봉 데이터를 임시 저장 (3분봉 생성용)
        self._store_1min_data(price_data)
        
        # 세션 전략 실행 (정확한 3분봉 마감 시간에)
        if self._is_3min_candle_close():
            # 3분봉 데이터 생성
            if self.time_manager.is_session_active():
                self.session_strategy.on_session_open(self.time_manager.get_current_time())

            series_3m = await self._create_3min_candle()
            self.data_manager.update_with_candle(series_3m)
            self.global_manager.update_all_indicators(series_3m)

            self._execute_session_strategy()
            self._execute_fade_reentry_3m_strategy()
            self._execute_vpvr_golden_strategy()

        # SQUEEZE 모멘텀 전략 실행
        self._execute_fade_reentry_1m_strategy()
        self._execute_squeeze_momentum_1m_strategy(price_data)
        
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
    
    def _is_3min_candle_close(self) -> bool:
        """현재 시간이 3분봉 마감 시간인지 체크 (51분, 54분, 57분, 00분...)"""
        try:
            time.sleep(1)
            current_time = self.time_manager.get_current_time()
            current_minute = current_time.minute

            return current_minute % 3 == 0
        except Exception as e:
            print(f"3분봉 마감 시간 체크 오류: {e}")
            return False
    
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
            print(f"1분봉 데이터 임시 저장 오류: {e}")
    
    def _execute_fade_reentry_1m_strategy(self):
        """빠른 패스 전략 실행"""
        if not self.fade_reentry_strategy:
            return
        
        self.fade_reentry_strategy.on_bucket_close(self.liquidation_bucket)
    
    def _execute_fade_reentry_3m_strategy(self):
        """빠른 패스 전략 실행"""
        if not self.fade_reentry_strategy:
            return
        
        self.fade_reentry_strategy.on_kline_close_3m()

    def _execute_squeeze_momentum_1m_strategy(self, price_data: Dict):
        """SQUEEZE 모멘텀 전략 실행"""
        if not self.squeeze_momentum_strategy:
            return
        
        self.squeeze_momentum_strategy.on_bucket_close(self.liquidation_bucket)
        
        # 딕셔너리를 DataFrame으로 변환 (리스트로 감싸기)
        df_1m = pd.DataFrame([price_data])
        df_1m.set_index('timestamp', inplace=True)
            
        result = self.squeeze_momentum_strategy.on_kline_close_1m(df_1m)

        if result:
            print(f"🎯 [SQUEEZE] SQUEEZE 1M 전략 신호: {result['action']} {result['entry']} | {result['stop']} | {result['targets'][0]} {result['targets'][1]}")
        else:
            print(f"📊 [SQUEEZE] SQUEEZE 1M 전략 신호 없음")

    def _execute_session_strategy(self):
        """세션 전략 실행"""
        if not self.session_strategy:
            return
        
        df_3m = self.data_manager.get_latest_data(count=2)
        result = self.session_strategy.on_kline_close_3m(df_3m)

    def _execute_vpvr_golden_strategy(self):
        """VPVR 골든 포켓 전략 실행"""
        if not self.vpvr_golden_strategy:
            return
        
        # VPVRConfig 클래스의 인스턴스 생성 (괄호로 인스턴스화)
        config = self.vpvr_golden_strategy.VPVRConfig()
        df_3m = self.data_manager.get_latest_data(count=config.lookback_bars + 5)
        sig = self.vpvr_golden_strategy.evaluate(df_3m)

        if sig:
            print(f"🎯 [VPVR] VPVR 골든 포켓 전략 신호: {sig['action']} {sig['entry']} | {sig['stop']} | {sig['targets'][0]} {sig['targets'][1]} {sig['targets'][2]}")
        else:
            print(f"📊 [VPVR] VPVR 골든 포켓 전략 신호 없음")


    async def _create_3min_candle(self) -> Optional[pd.Series]:
        """3분봉 데이터 생성 (첫 3분봉 마감 시 API 사용, 이후 웹소켓으로 수집)"""
        try:
            # 1. 첫 3분봉 마감이면 바이낸스 API에서 데이터 가져오기
            if not self._first_3min_candle_closed:
                print("🔄 첫 3분봉 마감 - 바이낸스 API에서 3분봉 데이터 가져오는 중...")
                
                # 현재 시간 기준으로 마지막 완성된 3분봉 데이터 가져오기
                current_time = self.time_manager.get_current_time()
                
                # 현재 진행 중인 3분봉의 시작 시간 계산 (수정됨)
                current_minute = current_time.minute
                
                current_candle_start = current_time.replace(
                    minute=(current_minute // 3) * 3,
                    second=0, 
                    microsecond=0
                )
                
                # 마지막 완성된 3분봉은 현재 진행 중인 3분봉의 이전 3분봉
                # 예: 19:29분이면 19:24:00 ~ 19:26:59 UTC 3분봉을 가져와야 함
                last_completed_start = current_candle_start - timedelta(minutes=3)
                last_completed_end = current_candle_start - timedelta(seconds=1)  # 19:26:59
                # 바이낸스 API에서 마지막 완성된 3분봉 데이터 가져오기
                df_3m = self.data_loader.fetch_data(
                    interval=3,  # 3분봉 직접 요청
                    symbol=self.symbol.upper(),
                    start_time=last_completed_start,
                    end_time=last_completed_end
                )
                
                if df_3m is not None and not df_3m.empty:
                    # 가장 최근 3분봉 사용
                    latest_3m = pd.Series(df_3m.iloc[-1])
                    
                    # 3분봉 데이터를 Series로 변환
                    result_series = pd.Series({
                        'open': float(latest_3m['open']),
                        'high': float(latest_3m['high']),
                        'low': float(latest_3m['low']),
                        'close': float(latest_3m['close']),
                        'volume': float(latest_3m['volume']),
                        'quote_volume': float(latest_3m['quote_volume'])
                    }, name=latest_3m.name)  # timestamp를 name으로 설정
                
                    # 첫 3분봉 마감 완료 표시
                    self._first_3min_candle_closed = True
                    
                    self._recent_1min_data = []

                    return result_series
                else:
                    print("❌ 첫 3분봉 API 데이터 로드 실패")
                    return None
            
            # 웹소켓 데이터로 3분봉 생성
            if len(self._recent_1min_data) >= 3:
                recent_3_candles = self._recent_1min_data[-3:]
                
                # 3분봉 데이터 계산
                open_price = recent_3_candles[0]['open']
                high_price = max(candle['high'] for candle in recent_3_candles)
                low_price = min(candle['low'] for candle in recent_3_candles)
                close_price = recent_3_candles[-1]['close']
                total_volume = sum(candle['volume'] for candle in recent_3_candles)
                total_quote_volume = sum(candle['quote_volume'] for candle in recent_3_candles)
                
                # 🔧 수정: 사용된 1분봉 데이터의 마지막 시간을 기준으로 3분봉 마감 시간 계산
                last_1min_timestamp = self.time_manager.get_timestamp_datetime(recent_3_candles[-1]['timestamp'])
                
                # 3분봉 마감 시간 = 마지막 1분봉 시간 (이미 3분봉 구간의 마지막)
                # API 데이터와 동일한 형식으로 통일: XX:XX:00
                accurate_timestamp = last_1min_timestamp.replace(
                    second=0,
                    microsecond=0
                )
                
                # 3분봉 데이터를 Series로 생성
                result_series = pd.Series({
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': total_volume,
                    'quote_volume': total_quote_volume
                }, name=accurate_timestamp)
                
                return result_series
            
        except Exception as e:
            print(f"3분봉 데이터 생성 오류: {e}")
            return None
    
    def _print_session_strategy(self, session_signal: Optional[Dict]):
        """세션 전략 신호 결과 출력"""
        if not session_signal:
            print(f"📊 세션 전략 신호 없음")
            return
        
        print(f"🎯 세션 전략 신호: {session_signal.get('playbook', 'UNKNOWN')} {session_signal.get('side', 'UNKNOWN')} | {session_signal.get('stage', 'UNKNOWN')} | {session_signal.get('confidence', 0):.0%}")
        
        # Entry 신호인 경우 핵심 정보만
        if session_signal.get('stage') == 'ENTRY':
            entry_price = session_signal.get('entry_price')
            stop_loss = session_signal.get('stop_loss')
            take_profit = session_signal.get('take_profit1')
            
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
                print(f"1분봉 Kline 콜백 실행 오류: {e}")
    
    async def start(self):
        """웹소켓 스트림 시작"""
        self.running = True
        # 여러 스트림을 동시에 실행
        tasks = [
            self.connect_liquidation_stream(),
            self.connect_kline_1m_stream(),  # 1분봉 Kline 스트림 추가
        ]
        
        await asyncio.gather(*tasks)
    
    def stop(self):
        """웹소켓 스트림 중지"""
        self.running = False
    
    def start_background(self):
        """백그라운드에서 웹소켓 실행"""
        def run_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.start())
        
        self.thread = threading.Thread(target=run_async, daemon=True)
        self.thread.start()
    
