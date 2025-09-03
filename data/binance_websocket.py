import json
import asyncio
import math
import websockets
import threading
import time
import requests  # pip install requests 필요
from typing import Any, Dict, List, Callable, Optional
from datetime import datetime, timedelta, timezone
import pandas as pd
import logging

# Global Indicator Manager import
from LLM_decider import LLMDecider
from data.bucket_aggregator import BucketAggregator
from data.data_manager import get_data_manager
from indicators.global_indicators import get_atr, get_daily_levels, get_global_indicator_manager, get_opening_range, get_vpvr, get_vwap
# Time Manager import
from signals import vpvr_golden_strategy
from utils.investing_crawler import fetch_us_high_events_today
from utils.telegram import send_telegram_message
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
            'kline_1m': [self.update_session_status]  # 1분봉 Kline 콜백만 사용
        }
        self.bucket_aggregator = BucketAggregator()
        self.time_manager = get_time_manager()
        self.global_manager = get_global_indicator_manager()
        self.data_manager = get_data_manager()
        self.data_loader = BinanceDataLoader()
        self.llm_decider = LLMDecider()

        # 데이터 저장소
        self.liquidation_bucket = []  # 청산 버킷 추가
        self.max_liquidations = 1000  # 최대 저장 청산 데이터 수
        
        # 전략 실행기 (외부에서 주입받음 - 실행 엔진 역할)
        self.session_strategy = None
        self.advanced_liquidation_strategy = None
        self.vpvr_golden_strategy = None
        self.bollinger_squeeze_strategy = None
        self.vwap_pinball_strategy = None
        self.ema_trend_15m_strategy = None
        self.orderflow_cvd_strategy = None
        
        # 진행 중인 3분봉 데이터 관리
        self._recent_1min_data = []  # 최근 1분봉 데이터 (웹소켓으로 수집)
        self._first_3min_candle_closed = False  # 첫 3분봉 마감 여부 추적
        self._session_activated = self.time_manager.is_session_active()
        self.signals = {}  # 딕셔너리로 변경: 시그널 이름을 키로 사용
        self.events = []


    def update_session_status(self, price_data: Dict):
        """세션 상태 업데이트"""
        self.time_manager.update_session_status()
        self._session_activated = self.time_manager.is_session_active()

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
        squeeze_momentum_strategy=None,
        fade_reentry_strategy=None,
        bollinger_squeeze_strategy=None,
        vpvr_golden_strategy=None,
        vwap_pinball_strategy=None,
        ema_trend_15m_strategy=None,
        orderflow_cvd_strategy=None,
        vol_spike_3m_strategy=None
    ):
        """전략 실행기 설정 - 실행 엔진에서 외부 전략 인스턴스 수신"""
        try:
            # 전략 인스턴스 검증 및 설정
            if session_strategy is not None:
                self.session_strategy = session_strategy
                print(f"✅ 세션 전략 설정 완료: {type(session_strategy).__name__}")
            
            if bollinger_squeeze_strategy is not None:
                self.bollinger_squeeze_strategy = bollinger_squeeze_strategy
                print(f"✅ 볼린저 스퀴즈 전략 설정 완료: {type(bollinger_squeeze_strategy).__name__}")
            
            if squeeze_momentum_strategy is not None:
                self.squeeze_momentum_strategy = squeeze_momentum_strategy
                print(f"✅ SQUEEZE 모멘텀 전략 설정 완료: {type(squeeze_momentum_strategy).__name__}")
            
            if fade_reentry_strategy is not None:
                self.fade_reentry_strategy = fade_reentry_strategy
                print(f"✅ 페이드 리입 전략 설정 완료: {type(fade_reentry_strategy).__name__}")
                
            if vpvr_golden_strategy is not None:
                self.vpvr_golden_strategy = vpvr_golden_strategy
                print(f"✅ VPVR 골든 포켓 전략 설정 완료: {type(vpvr_golden_strategy).__name__}")
                
            if vwap_pinball_strategy is not None:
                self.vwap_pinball_strategy = vwap_pinball_strategy
                print(f"✅ VWAP 피니언 전략 설정 완료: {type(vwap_pinball_strategy).__name__}")
                
            if ema_trend_15m_strategy is not None:
                self.ema_trend_15m_strategy = ema_trend_15m_strategy
                print(f"✅ EMA 트렌드 전략 설정 완료: {type(ema_trend_15m_strategy).__name__}")
            
            if orderflow_cvd_strategy is not None:
                self.orderflow_cvd_strategy = orderflow_cvd_strategy
                print(f"✅ ORDERFLOW CVD 전략 설정 완료: {type(orderflow_cvd_strategy).__name__}")
            
            if vol_spike_3m_strategy is not None:
                self.vol_spike_3m_strategy = vol_spike_3m_strategy
                print(f"✅ VOL SPIKE 3M 전략 설정 완료: {type(vol_spike_3m_strategy).__name__}")

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
    
    async def process_liquidation(self, data: Dict):
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
        """1분봉 Kline 데이터 처리 - 3분봉 포함"""
        if 'k' not in data:  # Kline 이벤트가 아니면 종료
            return
        kline = data['k']
        
        # 1분봉 마감 체크 (k.x == true)
        if not kline.get('x', True):  # 마감되지 않은 캔들이면 종료
            return
        
        # 웹소켓 59초에 마감
        time.sleep(2)

        print(f"\n⏰ OPEN TIME : {(self.time_manager.get_current_time()).strftime('%H:%M:%S')}")
        
        price_data = self._create_price_data(kline)
        self._store_1min_data(price_data)

        # 이벤트 차단 기간 체크
        is_event_blocking = self.is_in_event_blocking_period()
        
        # 세션 전략 실행 (정확한 3분봉 마감 시간에)
        if self._is_3min_candle_close():
            series_3m = await self._create_3min_candle()
            self.data_manager.update_with_candle(series_3m)
            self.global_manager.update_all_indicators(series_3m)

            # 이벤트 차단 기간이 아닐 때만 전략 신호 실행
            if not is_event_blocking:
                self._execute_session_strategy()
                self._execute_vpvr_golden_strategy()
                self._execute_bollinger_squeeze_strategy()
                self._execute_vwap_pinball_strategy()
                self._execute_ema_trend_15m_strategy()
                self._execute_fade_reentry_3m_strategy()
                self._execute_orderflow_cvd_strategy()
                self._execute_vol_spike_3m_strategy()
                
                decision = self.decide_trade_realtime(self.signals, leverage=20)
                self.print_decision_interpretation(decision)
                judge = self.llm_decider.decide(decision)
                print(decision, judge)
                if judge.get("decision") != "HOLD":
                    send_telegram_message(judge)
                self.signals = {}
            else:
                print("📊 이벤트 차단 기간: 데이터 업데이트만 수행, 전략 신호 차단")

        self._execute_fade_reentry_1m_strategy()
        self._execute_squeeze_momentum_1m_strategy(price_data)

        self._execute_kline_callbacks(price_data)

        if self.time_manager.is_midnight_time():
            self._load_daily_events()
        # self.ask_ai_decision(price_data)
    
    def important_event_occurred(self) -> bool:
        """중요 이벤트 발생 여부 체크"""
        return self.is_in_event_blocking_period()
    
    def _load_daily_events(self):
        """일일 이벤트 데이터 로드"""
        try:
            print("00시 발생. 오늘의 뉴스 불러오기")
            today = fetch_us_high_events_today(headless=False)
            event_times = [event['time'] for event in today]
            self.events.extend(event_times)
            print(f"📅 오늘의 이벤트 {len(event_times)}개 로드 완료")
        except Exception as e:
            print(f"❌ 일일 이벤트 로드 오류: {e}")
    
    def is_in_event_blocking_period(self) -> bool:
        """이벤트 발생 시간 ±30분 동안인지 체크"""
        if not self.events:
            return False
        
        current_time = self.time_manager.get_current_time()
        
        for event_time in self.events:
            # 이벤트 시간 ±30분 범위 체크
            event_start = event_time - timedelta(minutes=30)
            event_end = event_time + timedelta(minutes=30)
            
            if event_start <= current_time <= event_end:
                print(f"🚫 이벤트 차단 기간: {event_time.strftime('%H:%M')} ±30분 (현재: {current_time.strftime('%H:%M')})")
                return True
        
        return False
    
    def ask_ai_decision(self, price_data: Dict):
        atr = get_atr()
        vwap, vwap_std = get_vwap()
        prev_day_high, prev_day_low = get_daily_levels()
        high, low = get_opening_range()
        poc, hvn, lvn = get_vpvr()
        tech = {
            'atr': atr, 
            'vwap': vwap, 
            'vwap_std': vwap_std,
            'prev_day_high': prev_day_high,
            'prev_day_low': prev_day_low,
            'session_high': high,
            'session_low': low,
            'poc': poc,
            'hvn': hvn,
            'lvn': lvn
            }
        session = self.time_manager.get_current_session_info()
        session_info = {
            'session': session,
            'elapsed_minutes': session.elapsed_minutes,
            'remaining_minutes': session.remaining_minutes,
            'is_session_active': session.is_active
            }
        self._features.update({"session_info": session_info})
        self._features.update({"technical_indicators": tech})
        self._features.update({"current_price": price_data})
        self._features.update({"liquidation_bucket": self.liquidation_bucket})
        print(self._features)
        
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
            # time.sleep(1)
            current_time = self.time_manager.get_current_time()
            current_minute = current_time.minute

            return current_minute % 3 == 0
        except Exception as e:
            print(f"3분봉 마감 시간 체크 오류: {e}")
            return False
        
    def _is_15min_candle_close(self) -> bool:
        """현재 시간이 3분봉 마감 시간인지 체크 (51분, 54분, 57분, 00분...)"""
        try:
            # time.sleep(1)
            current_time = self.time_manager.get_current_time()
            current_minute = current_time.minute

            return current_minute % 15 == 0
        except Exception as e:
            print(f"15분봉 마감 시간 체크 오류: {e}")
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

    def _execute_vol_spike_3m_strategy(self):
        """볼륨 스파이크 전략 실행"""
        if not self.vol_spike_3m_strategy:
            return
        
        result = self.vol_spike_3m_strategy.on_kline_close_3m()

        if result:
            name = result.get('name', 'UNKNOWN')
            action = result.get('action', 'UNKNOWN')
            score = result.get('score', 0)
            confidence = result.get('confidence', 'LOW')
            timestamp = result.get('timestamp', self.time_manager.get_current_time())

            self.signals['VOL_SPIKE_3M'] = {'action': result.get('action', 'UNKNOWN'), 'score': result.get('score', 0), 'confidence': result.get('confidence', 'LOW'), 'timestamp': timestamp}
            print(f"🎯 [VOL_SPIKE_3M] 신호: {action} | 점수={score:.2f} | 신뢰도={confidence}")
        else:
            print(f"📊 [VOL_SPIKE_3M] 전략 신호 없음")

    def _execute_orderflow_cvd_strategy(self):
        """체결 불균형 근사 전략 실행"""
        if not self.orderflow_cvd_strategy:
            return
        
        result = self.orderflow_cvd_strategy.on_kline_close_3m()
        if result:
            name = result.get('name', 'UNKNOWN')
            action = result.get('action', 'UNKNOWN')
            score = result.get('score', 0)
            confidence = result.get('confidence', 'LOW')
            timestamp = result.get('timestamp', self.time_manager.get_current_time())

            self.signals['ORDERFLOW_CVD'] = {'action': result.get('action', 'UNKNOWN'), 'score': result.get('score', 0), 'confidence': result.get('confidence', 'LOW'), 'timestamp': timestamp}
            print(f"🎯 [ORDERFLOW_CVD] 신호: {action} | 점수={score:.2f} | 신뢰도={confidence}")
        else:
            print(f"📊 [ORDERFLOW_CVD] 전략 신호 없음")

    def _execute_ema_trend_15m_strategy(self):
        """EMA 트렌드 전략 실행 (15분봉)"""
        if not self.ema_trend_15m_strategy:
            return
        
        result = self.ema_trend_15m_strategy.on_kline_close_15m()
        if result:
            name = result.get('name', 'UNKNOWN')
            action = result.get('action', 'UNKNOWN')
            score = result.get('score', 0)
            confidence = result.get('confidence', 'LOW')
            timestamp = result.get('timestamp', self.time_manager.get_current_time())

            self.signals['EMA_TREND_15m'] = {'action': result.get('action', 'UNKNOWN'), 'score': result.get('score', 0), 'confidence': result.get('confidence', 'LOW'), 'timestamp': timestamp}
            print(f"🎯 [EMA_TREND_15m] 신호: {action} | 점수={score:.2f} | 신뢰도={confidence}")
        else:
            print(f"📊 [EMA_TREND_15m] 전략 신호 없음")


    def _execute_vwap_pinball_strategy(self):
        """VWAP 피니언 전략 실행"""
        if not self.vwap_pinball_strategy:
            return
        
        df_3m = self.data_manager.get_latest_data(count=4)
        result = self.vwap_pinball_strategy.on_kline_close_3m(df_3m)

        if result:
            action = result.get('action', 'UNKNOWN')
            entry = result.get('entry', 0)
            stop = result.get('stop', 0)
            targets = result.get('targets', [0, 0])
            score = result.get('score', 0)
            confidence = result.get('confidence', 'LOW')

            self.signals['VWAP'] = {'action': result.get('action', 'UNKNOWN'), 'score': result.get('score', 0), 'confidence': result.get('confidence', 'LOW'), 'entry': entry, 'stop': stop, 'timestamp': self.time_manager.get_current_time()}
            print(f"🎯 [VWAP PINBALL] 신호: {action} | 진입=${entry:.4f} | 손절=${stop:.4f} | 목표=${targets[0]:.4f}, ${targets[1]:.4f} | 점수={score:.2f} | 신뢰도={confidence}")
        else:
            print(f"📊 [VWAP PINBALL] 전략 신호 없음")

    def _execute_fade_reentry_1m_strategy(self):
        """빠른 패스 전략 실행"""
        if not self.fade_reentry_strategy:
            return
        
        self.fade_reentry_strategy.on_bucket_close(self.liquidation_bucket)

    def _execute_fade_reentry_3m_strategy(self):
        """페이드 리입 전략 실행 (3분봉)"""
        if not self.fade_reentry_strategy:
            return
        
        result = self.fade_reentry_strategy.on_kline_close_3m()

        if result:
            action = result.get('action', 'UNKNOWN')
            entry = result.get('entry', 0)
            stop = result.get('stop', 0)
            targets = result.get('targets', [0, 0])
            score = result.get('score', 0)
            confidence = result.get('confidence', 'LOW')
            self.signals['FADE'] = {'action': result.get('action', 'UNKNOWN'), 'score': result.get('score', 0), 'confidence': result.get('confidence', 'LOW'), 'entry': entry, 'stop': stop, 'timestamp': self.time_manager.get_current_time()}
            print(f"🎯 [FADE] 3M ENTRY 신호: {action} | 진입=${entry:.4f} | 손절=${stop:.4f} | 목표=${targets[0]:.4f}, ${targets[1]:.4f} | 점수={score:.2f} | 신뢰도={confidence}")
        else:
            print(f"📊 [FADE] 3M ENTRY 전략 신호 없음")

    def _execute_squeeze_momentum_1m_strategy(self, price_data: Dict):
        """SQUEEZE 모멘텀 전략 실행 (1분봉)"""
        if not self.squeeze_momentum_strategy:
            return
        
        try:
            # 1분 버킷 처리
            self.squeeze_momentum_strategy.on_bucket_close(self.liquidation_bucket)
            
            # 1분봉 Kline 처리
            df_1m = pd.DataFrame([price_data])
            df_1m.set_index('timestamp', inplace=True)
            
            result = self.squeeze_momentum_strategy.on_kline_close_1m(df_1m)
            

            if result:
                action = result.get('action', 'UNKNOWN')
                entry = result.get('entry', 0)
                stop = result.get('stop', 0)
                targets = result.get('targets', [0, 0])
                score = result.get('score', 0)  # 점수
                confidence = result.get('confidence', 'LOW')
                print(f"🎯 [SQUEEZE] 1M 신호: {action} | 진입=${entry:.4f} | 손절=${stop:.4f} | 목표=${targets[0]:.4f}, ${targets[1]:.4f} | 점수={score:.2f} | 신뢰도={confidence}")
                self.signals['LIQUIDATION_SQUEEZE'] = {'action': result.get('action', 'UNKNOWN'), 'score': result.get('score', 0), 'confidence': result.get('confidence', 'LOW'), 'entry': entry, 'stop': stop, 'timestamp': self.time_manager.get_current_time()}
            else:
                print(f"📊 [SQUEEZE] 1M 전략 신호 없음")
        except Exception as e:
            print(f"❌ [SQUEEZE] 1M 전략 실행 오류: {e}")

    def _execute_session_strategy(self):
        """세션 전략 실행"""
        if not self.session_strategy:
            return
        
        df_3m = self.data_manager.get_latest_data(count=2)
        result = self.session_strategy.on_kline_close_3m(df_3m, self._session_activated)
        
        # 전략 분석 결과 출력
        if result:
            stage = result.get('stage', 'UNKNOWN')
            action = result.get('action', 'UNKNOWN')
            entry = result.get('entry', 0)
            stop = result.get('stop', 0)
            targets = result.get('targets', [0, 0])
            score = result.get('score', 0)
            confidence = result.get('confidence', 'LOW')
            
            self.signals['SESSION'] = {'action': result.get('action', 'UNKNOWN'), 'score': result.get('score', 0), 'confidence': result.get('confidence', 'LOW'), 'entry': entry, 'stop': stop, 'timestamp': self.time_manager.get_current_time()}
            print(f"🎯 [SESSION] {stage} {action} | 진입=${entry:.4f} | 손절=${stop:.4f} | 목표=${targets[0]:.4f}, ${targets[1]:.4f} | 점수={score:.2f} | 신뢰도={confidence}")
        else:
            print(f"📊 [SESSION] 전략 신호 없음")

    def _execute_bollinger_squeeze_strategy(self):

        if not self.bollinger_squeeze_strategy:
            return
        
        result = self.bollinger_squeeze_strategy.evaluate()

        if result:
            action = result.get('action', 'UNKNOWN')
            entry = result.get('entry', 0)
            stop = result.get('stop', 0)
            targets = result.get('targets', [0, 0])
            score = result.get('score', 0)
            confidence = result.get('confidence', 'LOW')
            self.signals['BB_SQUEEZE'] = {'action': result.get('action', 'UNKNOWN'), 'score': result.get('score', 0), 'confidence': result.get('confidence', 'LOW'), 'entry': entry, 'stop': stop, 'timestamp': self.time_manager.get_current_time()}
            print(f"🎯 [BB Squeeze] 신호: {action} | 진입=${entry:.4f} | 손절=${stop:.4f} | 목표=${targets[0]:.4f}, ${targets[1]:.4f} | 점수={score:.2f} | 신뢰도={confidence}")
        else:
            print(f"📊 [BB Squeeze] 전략 신호 없음")

    def _execute_vpvr_golden_strategy(self):
        """VPVR 골든 포켓 전략 실행"""
        if not self.vpvr_golden_strategy:
            return
        
        # VPVRConfig 클래스의 인스턴스 생성 (괄호로 인스턴스화)
        config = self.vpvr_golden_strategy.VPVRConfig()
        df_3m = self.data_manager.get_latest_data(count=config.lookback_bars + 5)
        sig = self.vpvr_golden_strategy.evaluate(df_3m)
        
        # 전략 분석 결과 출력
        if sig:
            action = sig.get('action', 'UNKNOWN')
            entry = sig.get('entry', 0)
            stop = sig.get('stop', 0)
            targets = sig.get('targets', [0, 0])
            score = sig.get('score', 0)
            confidence = sig.get('confidence', 'LOW')
            self.signals['VPVR'] = {'action': sig.get('action', 'UNKNOWN'), 'score': sig.get('score', 0), 'confidence': sig.get('confidence', 'LOW') ,'entry': entry, 'stop': stop, 'timestamp': self.time_manager.get_current_time()}

            print(f"🎯 [VPVR] 골든 포켓 신호: {action} | 진입=${entry:.4f} | 손절=${stop:.4f} | 목표=${targets[0]:.4f}, ${targets[1]:.4f} | 점수={score:.2f} | 신뢰도={confidence}")
        else:
            print(f"📊 [VPVR] 골든 포켓 전략 신호 없음")

    async def _create_3min_candle(self) -> Optional[pd.Series]:
        """3분봉 데이터 생성 (첫 3분봉 마감 시 API 사용, 이후 웹소켓으로 수집)"""
        try:
            # 1. 첫 3분봉 마감이면 바이낸스 API에서 데이터 가져오기
            if not self._first_3min_candle_closed:
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
        
    def print_decision_interpretation(self, decision: dict) -> None:
        """
        decision: decide_trade_realtime(...) 반환값
        사람이 보기 쉽게 해석해서 출력합니다.
        """
        if not decision or not isinstance(decision, dict):
            print("⚠️ decision이 비어있거나 형식이 잘못되었습니다.")
            return

        action = decision.get("action", "HOLD")
        net_score = decision.get("net_score", 0.0)
        reason = decision.get("reason", "")
        raw = decision.get("raw", {})
        sizing = decision.get("sizing", {})
        recommended_scale = decision.get("recommended_trade_scale", 0.0)
        oppositions = decision.get("oppositions", [])
        agree_counts = decision.get("agree_counts", {"BUY": 0, "SELL": 0})
        meta = decision.get("meta", {})

        # compute per-strategy signed contributions (if possible)
        contributions = []
        for name, info in (raw.items() if isinstance(raw, dict) else []):
            try:
                act = (info.get("action") or "").upper()
                score = float(info.get("score") or 0.0)
                conf = float(info.get("conf_factor") or 0.6)
                weight = float(info.get("weight") or 0.0)
                sign = 0
                if act == "BUY":
                    sign = 1
                elif act == "SELL":
                    sign = -1
                contrib = sign * score * conf * weight
                contributions.append((name, contrib, act, score, conf, weight))
            except Exception:
                # best-effort fallback
                contributions.append((name, 0.0, info.get("action"), info.get("score"), info.get("confidence"), info.get("weight")))

        # sort by absolute contribution descending
        contributions_sorted = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)

        # Header
        print("────────────────────────────────────────────────────────")
        print(f"🕒 Decision @ {meta.get('timestamp_utc', 'unknown')}")
        print(f"▶ 추천 액션: {action}    |   net_score={net_score:.3f}    |   recommended_scale={recommended_scale:.3f}")
        print(f"▶ 이유: {reason}")
        print("────────────────────────────────────────────────────────")

        # Top contributors
        if contributions_sorted:
            print("전략별 기여 (큰 순):")
            for (name, contrib, act, score, conf, weight) in contributions_sorted:
                # format contribution sign and percent-ish
                sign_sym = "+" if contrib > 0 else ("-" if contrib < 0 else " ")
                print(f" - {name:12s} | action={str(act):5s} | score={score:.3f} conf={conf:.2f} weight={weight:.2f} | contrib={sign_sym}{abs(contrib):.4f}")
        else:
            print("전략별 정보가 없습니다.")

        # Conflicts and confirmations
        print("────────────────────────────────────────────────────────")
        print(f"확인수 (같은 방향, confirm threshold 이상): BUY={agree_counts.get('BUY',0)}  SELL={agree_counts.get('SELL',0)}")
        if oppositions:
            print("충돌(상반되는 강한 신호):")
            for nm, act, sc in oppositions:
                print(f" - {nm}: {act} (score={sc:.2f})")
        else:
            print("충돌 없음")

        # sizing / execution hint
        print("────────────────────────────────────────────────────────")
        print("포지션 사이징 / 권장 진입 정보:")
        qty = sizing.get("qty")
        risk_usd = sizing.get("risk_usd")
        entry = sizing.get("entry_used")
        stop = sizing.get("stop_used")
        print(f" - 권장 사이즈(스케일): {recommended_scale:.3f} (0..1 로 해석)")
        if qty is not None:
            print(f" - 권장 수량(qty): {qty:.4f}")
        else:
            print(f" - 권장 수량(qty): 계산 불가 (entry/stop 미확보)")
        print(f" - 리스크(달러): ${risk_usd}")
        if entry is not None and stop is not None:
            dist = abs(entry - stop)
            print(f" - entry={entry:.4f}  stop={stop:.4f}  (스탑거리={dist:.4f})")
        else:
            print(" - entry/stop 정보 부족 (신호 전략에서 제공되는 entry/stop 사용 권장)")

        # human guidance
        print("────────────────────────────────────────────────────────")
        if action == "HOLD":
            # if hold, explain top reasons why
            reasons = []
            # net too small
            if abs(net_score) < 0.35:
                reasons.append("net_score가 작음 (잡음일 가능성)")
            if oppositions:
                reasons.append("상반되는 강한 신호 존재")
            if reasons:
                print("권고: HOLD (보류). 이유들:")
                for r in reasons:
                    print(" -", r)
            else:
                print("권고: HOLD. 추가 확인 또는 더 강한 컨펌 대기.")
        else:
            # actionable suggestion
            print(f"권고: {action} — 실행 전 체크리스트:")
            # checklist items
            checklist = []
            # if any strong opposite exists -> warn
            if oppositions:
                checklist.append("상반되는 강한 신호 존재: 재확인 권장 (충돌 시 사이즈 축소)")
            # if recommended_scale small -> warn
            if recommended_scale < 0.35:
                checklist.append(f"권장 스케일이 작음 ({recommended_scale:.2f}) — 소량/스캘프 권장")
            # if confidence overall low (average conf factor small)
            avg_conf = 0.0
            if contributions_sorted:
                avg_conf = sum([c[4] for c in contributions_sorted]) / max(1.0, len(contributions_sorted))
            if avg_conf < 0.6:
                checklist.append("전반적 신뢰도 낮음(중·저) — 보수적 사이징 권장")
            # print checklist
            if checklist:
                for it in checklist:
                    print(" -", it)
            else:
                print(" - 조건 양호: 설정한 사이즈로 진입 고려 가능")

        print("────────────────────────────────────────────────────────")
        print("")  # blank line for spacing

    
    def decide_trade_realtime(
        self,
        signals: Dict[str, Dict[str, Any]],
        *,
        account_balance: float = 10000.0,
        base_risk_pct: float = 0.005,           # 기본 리스크: 계좌의 0.5%
        leverage: float = 20,                  # 선물 레버리지(노미널에 반영하길 원하면 조정)
        weights: Optional[Dict[str, float]] = None,
        open_threshold: float = 0.5,
        immediate_threshold: float = 0.75,
        confirm_threshold: float = 0.45,
        confirm_window_sec: int = 180,
        session_priority: bool = True,
        news_event: bool = False,
    ) -> Dict[str, Any]:
        """
        Realtime decision helper to be run every 3 minutes.
        signals: dict of signals, each signal dict should have:
        - name: str (e.g. 'SESSION','VPVR','VWAP PINBALL','SQUEEZE','FADE')
        - action: 'BUY' or 'SELL' (or None)
        - score: float between 0..1 (or None)
        - confidence: 'HIGH'/'MEDIUM'/'LOW' or None
        - entry: optional float (recommended entry price)
        - stop: optional float (recommended stop price)
        - timestamp: optional datetime
        Returns a dict with:
        - action: 'LONG'/'SHORT'/'HOLD'
        - net_score, reason, recommended_trade_scale (0..1),
        - sizing: qty, risk_usd, entry_used, stop_used (qty may be None if unusable)
        - raw: normalized component scores per-strategy
        """
        # default weights (can be tuned)
        default_weights = {
            "SESSION":             0.220,  # 세션 추세/오프닝
            "VWAP":                0.200,  # 리버전/페이드 핵심
            "FADE":                0.180,  # 청산 기반 스파이크
            "LIQUIDATION_SQUEEZE": 0.120,  # 청산 스퀴즈
            "VOL_SPIKE_3M":        0.090,  # 단기 변동성 급증
            "VPVR":                0.080,  # 거래량 지지/저항
            "ORDERFLOW_CVD":       0.060,  # 미세구조 확인
            "BB_SQUEEZE":          0.030,  # 변동성 예고
            "EMA_TREND_15M":       0.020   # 장기 추세 필터
        }

            
        if weights is None:
            weights = default_weights.copy()
        else:
            # ensure missing keys get defaults
            for k, v in default_weights.items():
                weights.setdefault(k, v)

        # normalize name helper
        def norm_name(n: str) -> str:
            s = n.strip().upper()
            # common aliases
            if "VWAP" in s:
                return "VWAP"
            if "VPVR" in s:
                return "VPVR"
            if "SESSION" in s:
                return "SESSION"
            if "LIQUIDATION_SQUEEZE" in s:
                return "LIQUIDATION_SQUEEZE"
            if "FADE" in s:
                return "FADE"
            if "BB_SQUEEZE" in s:  # Fixed comparison operator
                return "BB_SQUEEZE"
            if "ORDERFLOW_CVD" in s:  # Fixed comparison operator
                return "ORDERFLOW_CVD"
            if "EMA_TREND_15M" in s:  # Fixed comparison operator
                return "EMA_TREND_15M"
            if "VOL_SPIKE_3M" in s:  # Fixed comparison operator
                return "VOL_SPIKE_3M"
            return s

        now = self.time_manager.get_current_time()

        # confidence numeric mapping
        conf_map = {"HIGH": 1.0, "MEDIUM": 0.7, "LOW": 0.4, None: 0.6}

        # collect per-strategy signed weighted scores
        signed = {}
        raw = {}
        used_weight_sum = 0.0
        for name, s in signals.items():
            name = norm_name(name)  # 시그널 이름을 키로 사용
            action = (s.get("action")).upper()
            score = float(s.get("score"))
            conf = (s.get("confidence"))
            conf_factor = float(conf_map.get(conf))
            w = float(weights.get(name))
            # compute signed value
            sign = 0
            if action == "BUY":
                sign = 1
            elif action == "SELL":
                sign = -1
            val = sign * score * conf_factor * w
            signed[name] = val
            raw[name] = {
                "action": action if action else None,
                "score": score,
                "confidence": conf,
                "conf_factor": conf_factor,
                "weight": w,
                "entry": s.get("entry"),
                "stop": s.get("stop"),
                "timestamp": self.time_manager.get_current_time()
            }
            if w > 0:
                used_weight_sum += w

        # if no weights used -> hold
        if used_weight_sum <= 0:
            return {
                "action": "HOLD",
                "net_score": 0.0,
                "reason": "no recognized weighted strategies",
                "recommended_trade_scale": 0.0,
                "sizing": {"qty": None, "risk_usd": 0.0, "entry_used": None, "stop_used": None},
                "raw": raw
            }

        net = sum(signed.values()) / max(1e-9, used_weight_sum)  # roughly in -1..1

        # detect strong session override
        session_rec = raw.get("SESSION")
        session_override = False
        session_action = None
        if session_rec and session_priority:
            sess_act = session_rec.get("action")
            sess_score = float(session_rec.get("score") or 0.0)
            sess_conf = session_rec.get("confidence")
            if sess_act in ("BUY", "SELL") and sess_score >= immediate_threshold and sess_conf == "HIGH":
                # check opposing strong signals
                opp_strong = False
                for nm, r in raw.items():
                    if nm == "SESSION": continue
                    if r.get("action") and r.get("action") != sess_act and float(r.get("score") or 0.0) >= 0.60:
                        opp_strong = True
                        break
                if not opp_strong:
                    session_override = True
                    session_action = sess_act

        # confirmations: count other strategies in same direction with score >= confirm_threshold within time window
        agree_counts = {"BUY": 0, "SELL": 0}
        for nm, r in raw.items():
            act = r.get("action")
            if act not in ("BUY", "SELL"):
                continue
            sc = float(r.get("score") or 0.0)
            ts = r.get("timestamp")
            # time-based confirmation: if timestamp provided, ensure recency
            if ts is not None and isinstance(ts, datetime):
                if abs((now - ts).total_seconds()) > confirm_window_sec:
                    continue
            if sc >= confirm_threshold:
                agree_counts[act] += 1

        # conflict detection: opposing significant strategies
        oppositions = []
        for nm, r in raw.items():
            act = r.get("action")
            sc = float(r.get("score") or 0.0)
            if act in ("BUY", "SELL") and sc >= 0.5:
                oppositions.append((nm, act, sc))

        # compute recommended trade scale (0..1)
        # base_scale ~ proportional to |net| (net 0.75 -> scale 1)
        base_scale = min(1.0, max(0.0, abs(net) / 0.75))
        # conflict penalty
        if len(oppositions) >= 2:
            conflict_penalty = 0.25
        elif len(oppositions) == 1:
            conflict_penalty = 0.6
        else:
            conflict_penalty = 1.0
        # confidence multiplier: geometric mean of conf_factors among used strategies
        conf_factors = [r.get("conf_factor", 0.6) for nm, r in raw.items() if r.get("weight", 0) > 0]
        conf_mult = 0.6
        if conf_factors:
            prod = 1.0
            for f in conf_factors:
                prod *= f
            conf_mult = prod ** (1.0 / max(1, len(conf_factors)))
        recommended_scale = max(0.0, min(1.0, base_scale * conflict_penalty * conf_mult))

        # Final decision
        action = "HOLD"
        reason = []
        if session_override:
            action = "LONG" if session_action == "BUY" else "SHORT"
            reason.append(f"SESSION strong override (score={session_rec.get('score')}, conf={session_rec.get('confidence')})")
        else:
            if net >= open_threshold:
                action = "LONG"
                reason.append(f"net_score {net:.3f} >= open_threshold {open_threshold}")
            elif net <= -open_threshold:
                action = "SHORT"
                reason.append(f"net_score {net:.3f} <= -open_threshold {-open_threshold}")
            else:
                # conditional opening if confirmation present and net magnitude moderate
                if net > 0 and agree_counts["BUY"] >= 1 and net >= (open_threshold * 0.6):
                    action = "LONG"
                    reason.append(f"conditional LONG: net {net:.3f}, confirmations {agree_counts['BUY']}")
                elif net < 0 and agree_counts["SELL"] >= 1 and abs(net) >= (open_threshold * 0.6):
                    action = "SHORT"
                    reason.append(f"conditional SHORT: net {net:.3f}, confirmations {agree_counts['SELL']}")
                else:
                    action = "HOLD"
                    reason.append(f"net_score too small ({net:.3f}) or no confirmations")

        # Determine sizing: use primary signal entry/stop if available
        entry_used = None
        stop_used = None
        # priority for sizing: SESSION -> VPVR -> VWAP -> SQUEEZE -> FADE
        priority_order = [
            "SESSION",
            "VWAP",
            "FADE",
            "VOL_SPIKE_3M",
            "VPVR",
            "ORDERFLOW_CVD",
            "BB_SQUEEZE",
            "EMA_TREND_15M",
            "LIQUIDATION_SQUEEZE"
        ]       
        selected_strategy = None
        for pname in priority_order:
            r = raw.get(pname)
            if r and r.get("action") and r.get("action") in ("BUY", "SELL"):
                # prefer strategy that matches final action
                if action == "HOLD":
                    # choose first available to provide sizing suggestion
                    selected_strategy = pname
                    break
                if (action == "LONG" and r.get("action") == "BUY") or (action == "SHORT" and r.get("action") == "SELL"):
                    selected_strategy = pname
                    break
        if selected_strategy:
            r = raw.get(selected_strategy)
            entry_used = r.get("entry")
            stop_used = r.get("stop")

        # fallback: if no entry/stop from signals, try to infer using ATR if available
        if (entry_used is None or stop_used is None):
            # try to call get_atr() if present in global scope
            try:
                atr_val = float(get_atr())
                # if we have an approximate last price from signals, use last provided entry-like price
                any_price = None
                for nm, r in raw.items():
                    if r.get("entry") is not None:
                        any_price = float(r.get("entry"))
                        break
                if any_price is None:
                    # try to take entry from any signal
                    for nm, r in raw.items():
                        if r.get("score", 0) > 0:
                            any_price = r.get("entry") or r.get("stop")
                            if any_price is not None:
                                any_price = float(any_price); break
                if entry_used is None and any_price is not None:
                    entry_used = any_price
                if stop_used is None and any_price is not None:
                    # place stop at entry +/- 1.5*ATR (direction-based)
                    if atr_val is None or math.isnan(atr_val):
                        atr_val = max(1.0, 0.5 * abs(entry_used) * 0.001)  # tiny fallback
                    if action == "LONG":
                        stop_used = entry_used - 1.5 * atr_val
                    elif action == "SHORT":
                        stop_used = entry_used + 1.5 * atr_val
                    else:
                        stop_used = None
            except Exception:
                pass

        # compute qty given entry_used and stop_used
        qty = None
        risk_usd = account_balance * float(base_risk_pct)
        if entry_used is not None and stop_used is not None and entry_used != stop_used and action in ("LONG", "SHORT"):
            distance = abs(entry_used - stop_used)
            if distance > 0:
                # qty in base USD units (e.g. if contract is 1 USD price per unit)
                # For futures, user should convert to contract units according to their product
                qty = risk_usd / distance
                # apply recommended_scale as multiplier to qty
                qty = qty * recommended_scale * leverage
        else:
            qty = None

        sizing = {
            "qty": float(qty) if qty is not None else None,
            "risk_usd": round(float(risk_usd), 4),
            "entry_used": float(entry_used) if entry_used is not None else None,
            "stop_used": float(stop_used) if stop_used is not None else None,
            "recommended_scale": round(recommended_scale, 3)
        }

        # assemble readable reason
        reason_text = "; ".join(reason)

        return {
            "action": action,
            "net_score": round(net, 4),
            "raw": raw,
            "reason": reason_text,
            "recommended_trade_scale": round(recommended_scale, 3),
            "sizing": sizing,
            "oppositions": oppositions,
            "agree_counts": agree_counts,
            "meta": {"timestamp_utc": now.isoformat(), "used_weight_sum": used_weight_sum}
        }
    
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
    
