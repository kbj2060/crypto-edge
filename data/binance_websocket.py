import json
import asyncio
import websockets
import threading
import time
import requests  # pip install requests 필요
from typing import Dict, List, Callable, Optional
from datetime import datetime, timedelta, timezone
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
                    
                    # 가격 데이터 저장 (지표별 거래량 데이터 분리)
                    price_data = {
                        'timestamp': datetime.now(),
                        'price': float(kline['c']),  # 종가
                        'open': float(kline['o']),
                        'high': float(kline['h']),
                        'low': float(kline['l']),
                        'close': float(kline['c']),
                        'volume': float(kline['v']),      # VWAP용: base volume (ETH)
                        'quote_volume': float(kline['q']), # VPVR용: quote volume (USDT)
                        'trade_count': int(kline['n']),    # 거래 횟수
                        'close_time': kline['t']           # 캔들 종료 시간
                    }
                    
                    # 가격 히스토리에 추가
                    self.price_history.append(price_data)
                    
                    # 최대 개수 제한
                    if len(self.price_history) > self.max_price_history:
                        self.price_history.pop(0)
                    
                    # 1분봉 카운터 증가
                    self.minute_counter += 1
                    
                    # 청산 전략 실행 (매 1분마다)
                    if self.advanced_liquidation_strategy and self.liquidation_bucket:
                        try:
                            print(f"🎯 청산 전략 실행 시작... (버킷 크기: {len(self.liquidation_bucket)})")
                            
                            # 현재 가격 가져오기
                            current_price = float(kline['c'])
                            
                            # 글로벌 지표 시스템에서 지표 데이터 가져오기
                            try:
                                from indicators.global_indicators import get_global_indicator_manager
                                import pandas as pd
                                
                                global_manager = get_global_indicator_manager()
                                
                                # Daily Levels (어제 고가/저가)
                                daily_levels = global_manager.get_indicator('daily_levels')
                                key_levels = {}
                                if daily_levels and daily_levels.is_loaded():
                                    prev_day_data = daily_levels.get_prev_day_high_low()
                                    key_levels = {
                                        'prev_day_high': prev_day_data.get('high', 0),
                                        'prev_day_low': prev_day_data.get('low', 0)
                                    }
                                
                                # Opening Range (현재 세션 정보)
                                opening_range = {}
                                try:
                                    from indicators.opening_range import get_session_manager
                                    session_manager = get_session_manager()
                                    session_config = session_manager.get_indicator_mode_config()
                                    
                                    if session_config.get('use_session_mode'):
                                        opening_range = {
                                            'session_name': session_config.get('session_name', 'UNKNOWN'),
                                            'session_start': session_config.get('session_start_time'),
                                            'elapsed_minutes': session_config.get('elapsed_minutes', 0),
                                            'session_status': session_config.get('session_status', 'UNKNOWN')
                                        }
                                except Exception as e:
                                    print(f"⚠️ Opening Range 정보 가져오기 실패: {e}")
                                
                                # VWAP 및 VWAP 표준편차
                                vwap_indicator = global_manager.get_indicator('vwap')
                                vwap = 0.0
                                vwap_std = 0.0
                                if vwap_indicator:
                                    vwap_status = vwap_indicator.get_vwap_status()
                                    vwap = vwap_status.get('current_vwap', 0)
                                    vwap_std = vwap_status.get('current_vwap_std', 0)
                                
                                # ATR
                                atr_indicator = global_manager.get_indicator('atr')
                                atr = 0.0
                                if atr_indicator:
                                    atr = atr_indicator.get_atr()
                                
                                # price_data를 DataFrame으로 가공
                                # analyze_all_strategies 함수는 DataFrame을 기대하지만
                                # 웹소켓에서는 실시간 가격만 받으므로 단일 행 DataFrame 생성
                                price_data = pd.DataFrame({
                                    'timestamp': [datetime.now(timezone.utc)],
                                    'open': [float(kline['o'])],
                                    'high': [float(kline['h'])],
                                    'low': [float(kline['l'])],
                                    'close': [float(kline['c'])],
                                    'volume': [float(kline['v'])]  # 실제 거래량 사용
                                })
                                
                                print(f"📊 글로벌 지표 데이터 로드 완료:")
                                print(f"   📅 Key Levels: {key_levels}")
                                print(f"   🌅 Opening Range: {opening_range}")
                                print(f"   📊 VWAP: ${vwap:.2f}")
                                print(f"   📊 VWAP STD: ${vwap_std:.2f}")
                                print(f"   📊 ATR: {atr:.3f}")
                                print(f"   📈 Price Data: DataFrame 생성 완료 (행: {len(price_data)})")
                                
                            except Exception as e:
                                print(f"❌ 글로벌 지표 데이터 로드 실패: {e}")
                                # 기본값으로 설정
                                key_levels = {}
                                opening_range = {}
                                vwap = 0.0
                                vwap_std = 0.0
                                atr = 0.0
                                # 기본 price_data 생성
                                price_data = pd.DataFrame({
                                    'timestamp': [datetime.now(timezone.utc)],
                                    'open': [current_price],
                                    'high': [current_price],
                                    'low': [current_price],
                                    'close': [current_price],
                                    'volume': [0.0]
                                })
                            
                            # 청산 전략 분석 - analyze_all_strategies 호출
                            signal = self.advanced_liquidation_strategy.analyze_all_strategies(
                                price_data, key_levels, opening_range, vwap, vwap_std, atr
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
                                
                                # 3분봉 데이터 시뮬레이션 (1분봉 3개를 합쳐서 3분봉 생성)
                                if len(self.price_history) >= 3:
                                    recent_3_candles = self.price_history[-3:]
                                    
                                    # 3분봉 데이터 생성 (OHLCV)
                                    three_min_data = {
                                        'timestamp': recent_3_candles[-1]['timestamp'],
                                        'open': float(recent_3_candles[0]['open']),
                                        'high': max(float(candle['high']) for candle in recent_3_candles),
                                        'low': min(float(candle['low']) for candle in recent_3_candles),
                                        'close': float(recent_3_candles[-1]['close']),
                                        'volume': sum(float(candle['volume']) for candle in recent_3_candles)
                                    }
                                    
                                    print(f"   📊 3분봉 데이터 생성: O:{three_min_data['open']:.2f} H:{three_min_data['high']:.2f} L:{three_min_data['low']:.2f} C:{three_min_data['close']:.2f} V:{three_min_data['volume']:.2f}")
                                    
                                    # 글로벌 지표 시스템에서 지표 데이터 가져오기
                                    try:
                                        from indicators.global_indicators import get_global_indicator_manager
                                        import pandas as pd
                                        
                                        global_manager = get_global_indicator_manager()
                                        
                                        # 3분봉 데이터를 DataFrame으로 변환 (timezone 정보 포함)
                                        df_3m = pd.DataFrame([three_min_data])
                                        df_3m.set_index('timestamp', inplace=True)
                                        
                                        # 인덱스에 UTC timezone 정보 추가
                                        if df_3m.index.tz is None:
                                            df_3m.index = df_3m.index.tz_localize('UTC')
                                            print(f"   📊 DataFrame 인덱스에 UTC timezone 설정 완료")
                                        
                                        # 글로벌 지표 업데이트
                                        global_manager.update_all_indicators_with_candle(df_3m.iloc[0])
                                        
                                        print(f"   📊 3분봉 데이터 글로벌 지표 업데이트 완료")
                                        
                                        # 키 레벨 가져오기
                                        daily_levels = global_manager.get_indicator('daily_levels')
                                        key_levels = {}
                                        if daily_levels and daily_levels.is_loaded():
                                            prev_day_data = daily_levels.get_prev_day_high_low()
                                            key_levels = {
                                                'prev_day_high': prev_day_data.get('high', 0),
                                                'prev_day_low': prev_day_data.get('low', 0)
                                            }
                                        
                                        # Opening Range 정보 가져오기
                                        opening_range = {}
                                        try:
                                            from indicators.opening_range import get_session_manager
                                            session_manager = get_session_manager()
                                            session_config = session_manager.get_indicator_mode_config()
                                            
                                            if session_config.get('use_session_mode'):
                                                opening_range = {
                                                    'session_name': session_config.get('session_name', 'UNKNOWN'),
                                                    'session_start': session_config.get('session_start_time'),
                                                    'elapsed_minutes': session_config.get('elapsed_minutes', 0),
                                                    'session_status': session_config.get('session_status', 'UNKNOWN')
                                                }
                                        except Exception as e:
                                            print(f"   ⚠️ Opening Range 정보 가져오기 실패: {e}")
                                        
                                        # VWAP 및 VWAP 표준편차
                                        vwap_indicator = global_manager.get_indicator('vwap')
                                        vwap = 0.0
                                        vwap_std = 0.0
                                        if vwap_indicator:
                                            vwap_status = vwap_indicator.get_vwap_status()
                                            vwap = vwap_status.get('current_vwap', 0)
                                            vwap_std = vwap_status.get('current_vwap_std', 0)
                                        
                                        # ATR
                                        atr_indicator = global_manager.get_indicator('atr')
                                        atr = 0.0
                                        if atr_indicator:
                                            atr = atr_indicator.get_atr()
                                        
                                        print(f"   📊 글로벌 지표 데이터 로드 완료:")
                                        print(f"      📅 Key Levels: {key_levels}")
                                        print(f"      🌅 Opening Range: {opening_range}")
                                        print(f"      📊 VWAP: ${vwap:.2f}")
                                        print(f"      📊 VWAP STD: ${vwap_std:.2f}")
                                        print(f"      📊 ATR: {atr:.3f}")
                                        
                                        # 세션 전략 분석 실행 (고급 청산 전략과 동일한 방식)
                                        print(f"   📊 세션 전략에 전달할 DataFrame 정보:")
                                        print(f"      📊 인덱스 타입: {type(df_3m.index)}")
                                        print(f"      📊 인덱스 timezone: {df_3m.index.tz}")
                                        print(f"      📊 데이터 행 수: {len(df_3m)}")
                                        
                                        session_signal = self.session_strategy.analyze_session_strategy(
                                            df_3m, key_levels, datetime.now(timezone.utc)
                                        )
                                        
                                        if session_signal:
                                            print(f"   🎯 세션 전략 신호 감지!")
                                            print(f"      📚 플레이북: {session_signal.get('playbook', 'UNKNOWN')}")
                                            print(f"      🎯 신호 타입: {session_signal.get('signal_type', 'UNKNOWN')}")
                                            print(f"      ⚡ 액션: {session_signal.get('action', 'UNKNOWN')}")
                                            print(f"      🏆 등급: {session_signal.get('stage', 'UNKNOWN')}")
                                            print(f"      📊 신뢰도: {session_signal.get('confidence', 0):.0%}")
                                            print(f"      📝 이유: {session_signal.get('reason', 'N/A')}")
                                            
                                            # Entry 신호인 경우 추가 정보
                                            if session_signal.get('stage') == 'ENTRY':
                                                entry_price = session_signal.get('entry_price', 0)
                                                stop_loss = session_signal.get('stop_loss', 0)
                                                take_profit = session_signal.get('take_profit1', 0)
                                                if entry_price and stop_loss and take_profit:
                                                    risk = abs(entry_price - stop_loss)
                                                    reward = abs(take_profit - entry_price)
                                                    rr_ratio = reward / risk if risk > 0 else 0
                                                    print(f"      💰 진입가: ${entry_price:.2f}")
                                                    print(f"      🛑 손절가: ${stop_loss:.2f}")
                                                    print(f"      🎯 목표가: ${take_profit:.2f}")
                                                    print(f"      ⚖️  리스크/리워드: {rr_ratio:.2f}")
                                        else:
                                            print(f"   📊 세션 전략 신호 없음")
                                            
                                    except Exception as e:
                                        print(f"   ❌ 세션 전략 실행 오류: {e}")
                                        import traceback
                                        traceback.print_exc()
                                
                                print(f"✅ 세션 전략 실행 완료")
                            except Exception as e:
                                self.logger.error(f"세션 전략 실행 오류: {e}")
                                import traceback
                                traceback.print_exc()
                    
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
    
