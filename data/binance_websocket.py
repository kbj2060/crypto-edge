import json
import asyncio
import websockets
import threading
from typing import Any, Dict, Callable, Optional
from datetime import datetime
import pandas as pd

# 리팩토링된 컴포넌트들
from data.strategy_executor import StrategyExecutor
from data.candle_creator import CandleCreator
from data.trade_decision_engine import TradeDecisionEngine
from data.event_manager import EventManager

# 기존 imports
from llm.LLM_decider import LLMDecider
from data.bucket_aggregator import BucketAggregator
from data.data_manager import get_data_manager
from indicators.global_indicators import get_global_indicator_manager
from utils.display_utils import print_decision_interpretation, print_llm_judgment
from utils.telegram import send_telegram_message
from utils.time_manager import get_time_manager
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
            'kline_3m': [self.update_session_status]
        }
        
        # 리팩토링된 컴포넌트들
        self.strategy_executor = StrategyExecutor()
        self.candle_creator = CandleCreator(symbol)
        self.decision_engine = TradeDecisionEngine()
        self.event_manager = EventManager()
        
        # 기존 매니저들
        self.time_manager = get_time_manager()
        self.global_manager = get_global_indicator_manager()
        self.data_manager = get_data_manager()
        self.data_loader = BinanceDataLoader()
        self.llm_decider = LLMDecider()

        # 데이터 저장소
        self.liquidation_bucket = []
        self.max_liquidations = 1000
        
        # 세션 상태
        self._session_activated = self.time_manager.is_session_active()
        self.queue = asyncio.Queue()


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
        bollinger_squeeze_strategy=None,
        vpvr_golden_strategy=None,
        ema_trend_15m_strategy=None,
        orderflow_cvd_strategy=None,
        rsi_divergence_strategy=None,
        ichimoku_strategy=None,
        vwap_pinball_strategy=None,
        vol_spike_strategy=None,
    ):
        """전략 실행기 설정"""
        self.strategy_executor.set_strategies(
            session_strategy=session_strategy,
            bollinger_squeeze_strategy=bollinger_squeeze_strategy,
            vpvr_golden_strategy=vpvr_golden_strategy,
            ema_trend_15m_strategy=ema_trend_15m_strategy,
            orderflow_cvd_strategy=orderflow_cvd_strategy,
            rsi_divergence_strategy=rsi_divergence_strategy,
            ichimoku_strategy=ichimoku_strategy,
            vwap_pinball_strategy=vwap_pinball_strategy,
            vol_spike_strategy=vol_spike_strategy,
        )
    
    async def connect_kline_3m_stream(self):
        """1분봉 Kline 스트림 연결"""
        uri = f"{self.ws_url}/{self.symbol}@kline_3m"
        
        async with websockets.connect(uri) as websocket:
            # 첫 시작 시 signals가 비어있으면 모든 지표 업데이트 및 전략 실행
            
            print("🚀 첫 시작 - 모든 지표 업데이트 및 전략 실행")
            await self._initialize_all_strategies()

            async for message in websocket:
                if not self.running:
                    break
                
                data = json.loads(message)
                await self.queue.put(("kline_3m", data))
                # await self.process_kline_3m(data)
    
    async def _initialize_all_strategies(self):
        """첫 시작 시 모든 지표 업데이트 및 전략 실행"""
        # 모든 전략 실행
        self.strategy_executor.execute_all_strategies(self._session_activated)
        print("✅ 모든 지표 및 전략 초기화 완료")

        signals = self.strategy_executor.get_signals()
        decision = self.decision_engine.decide_trade_realtime(signals, leverage=30)
        print_decision_interpretation(decision)


    async def worker(self):
        """큐에서 데이터를 소비하며 전략 실행 (오류 처리 포함)"""
        while self.running:
            try:
                event_type, data = await self.queue.get()

                if event_type == "kline_3m":
                    await self.process_kline_3m(data)
                    
            except Exception as e:
                print(f"❌ [Worker] 데이터 처리 오류: {e}")
                import traceback
                traceback.print_exc()
                # 오류 발생 시에도 계속 실행
                continue

    async def process_kline_3m(self, data: Dict):
        """1분봉 Kline 데이터 처리 - 3분봉 포함 (오류 처리 강화)"""
        try:
            if 'k' not in data: 
                return
            kline = data['k']

            if not kline.get('x', True): 
                return
            
            await asyncio.sleep(1)

            print(f"\n⏰ OPEN TIME : {(self.time_manager.get_current_time()).strftime('%H:%M:%S')}")
            
            price_data = self.candle_creator.create_price_data(kline)
            # self.candle_creator.store_1min_data(price_data)
            
        except Exception as e:
            print(f"❌ [ProcessKline] 1분봉 데이터 처리 오류: {e}")
            import traceback
            traceback.print_exc()
            return

        try:
            # 이벤트 차단 기간 체크
            is_event_blocking = self.event_manager.is_in_event_blocking_period()

            series_3m = await self.candle_creator.create_3min_candle()
            if series_3m is not None:
                self.data_manager.update_with_candle(series_3m)
                self.global_manager.update_all_indicators(series_3m)

                # 이벤트 차단 기간이 아닐 때만 전략 신호 실행
                if not is_event_blocking:
                    self.strategy_executor.execute_all_strategies(self._session_activated)
                    
                    signals = self.strategy_executor.get_signals()
                    decision = self.decision_engine.decide_trade_realtime(signals, leverage=20)
                    print_decision_interpretation(decision)

                    # series_3m이 있을 때만 candle_data 추가
                    decision["candle_data"] = series_3m.to_dict()
                    
                    #judge = await self.llm_decider.decide_async(decision)
                    #print_llm_judgment(judge)

                    action = decision.get("action")
                    net_score = decision.get("net_score")
                    
                    if action != "HOLD":
                        send_telegram_message(action, net_score)
                else:
                    print("📊 이벤트 차단 기간: 데이터 업데이트만 수행, 전략 신호 차단")

            self._execute_kline_callbacks(price_data)

            if self.time_manager.is_midnight_time():
                self.event_manager.load_daily_events()
                print(self.event_manager.get_events())
            
        except Exception as e:
            print(f"❌ [ProcessKline] 전략 실행 오류: {e}")
            import traceback
            traceback.print_exc()
    
    def important_event_occurred(self) -> bool:
        """중요 이벤트 발생 여부 체크"""
        return self.event_manager.important_event_occurred()
    
    def _execute_kline_callbacks(self, price_data: Dict):
        """1분봉 Kline 콜백 실행"""
        for callback in self.callbacks['kline_3m']:
            try:
                callback(price_data)
            except Exception as e:
                print(f"1분봉 Kline 콜백 실행 오류: {e}")
    
    async def start(self):
        """웹소켓 스트림 시작"""
        self.running = True
        # 여러 스트림을 동시에 실행
        tasks = [
            # self.connect_liquidation_stream(),
            self.connect_kline_3m_stream(),
            self.worker()  # 1분봉 Kline 스트림 추가
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
    
