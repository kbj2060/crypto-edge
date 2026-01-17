import json
import asyncio
from datetime import timezone
import websockets
import threading
from typing import Dict, Callable, Optional
from datetime import datetime

# 리팩토링된 컴포넌트들
from managers.strategy_executor import StrategyExecutor
from managers.candle_creator import CandleCreator
from managers.event_manager import EventManager
from engines.trade_decision_engine import TradeDecisionEngine

# 기존 imports
from managers.data_manager import get_data_manager
from indicators.global_indicators import get_all_indicators, get_atr, get_daily_levels, get_global_indicator_manager, get_vpvr, get_vwap
from utils.display_utils import print_decision_interpretation
from utils.telegram import send_telegram_message
from managers.time_manager import get_time_manager
from utils.decision_logger import get_decision_logger
from managers.binance_dataloader import BinanceDataLoader
from managers.binance_trader import BinanceTrader
from managers.trade_executor import TradeExecutor

class BinanceWebSocket:
    """바이낸스 웹소켓 클라이언트 - 실시간 청산 데이터 및 Kline 데이터 수집"""
    
    def __init__(
        self,
        symbol: str = "ETHUSDT",
        strategy_executor: Optional[StrategyExecutor] = None,
        enable_trading: bool = False,
        simulation_mode: bool = True,
        demo: bool = False
    ):
        """
        웹소켓 초기화
        
        Args:
            symbol: 거래 심볼
            strategy_executor: 전략 실행기
            enable_trading: 실제 거래 활성화 여부
            simulation_mode: 시뮬레이션 모드 (실제 주문 실행 안 함)
        """
        self.symbol = symbol.lower()
        # 웹소켓 URL 설정
        if demo:
            self.ws_url = "wss://demo-stream.binance.com/ws"  # Demo Trading 웹소켓 (Spot)
        else:
            self.ws_url = "wss://fstream.binance.com/ws"  # 메인넷 웹소켓
        self.running = False
        self.enable_trading = enable_trading
        self.simulation_mode = simulation_mode
        self.demo = demo
        
        # 콜백 함수들
        self.callbacks = {
            'liquidation': [],
            'kline_3m': []
        }
        
        # 리팩토링된 컴포넌트들
        self.strategy_executor = strategy_executor or StrategyExecutor()
        self.candle_creator = CandleCreator(symbol)
        self.decision_engine = TradeDecisionEngine()
        self.event_manager = EventManager()
        
        # 기존 매니저들
        self.time_manager = get_time_manager()
        self.global_manager = get_global_indicator_manager()
        self.data_manager = get_data_manager()
        self.data_loader = BinanceDataLoader()
        self.decision_logger = get_decision_logger(symbol)
        
        # 거래 실행기 초기화
        self.trade_executor = None
        if self.enable_trading:
            try:
                binance_trader = BinanceTrader(
                    simulation_mode=self.simulation_mode,
                    demo=self.demo,
                    use_futures=False  # Spot 거래 사용 (LONG=매수, SHORT=매도)
                )
                self.trade_executor = TradeExecutor(
                    binance_trader=binance_trader,
                    symbol=symbol.upper()
                )
                if simulation_mode:
                    mode_str = '시뮬레이션'
                elif self.demo:
                    mode_str = 'Demo Trading'
                else:
                    mode_str = '실제 거래'
                print(f"✅ 거래 실행기 초기화 완료 ({mode_str} 모드)")
            except Exception as e:
                print(f"⚠️ 거래 실행기 초기화 실패: {e}")
                print("   거래 없이 계속 실행됩니다.")

        # 데이터 저장소
        self.liquidation_bucket = []
        self.max_liquidations = 1000
        
        # 세션 상태 (더 이상 사용하지 않음)
        self._session_activated = False
        self.queue = asyncio.Queue()

        # 카운트다운 태스크
        self.countdown_task = None


    def add_callback(self, event_type: str, callback: Callable):
        """콜백 함수 등록"""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
    
    def remove_callback(self, event_type: str, callback: Callable):
        """콜백 함수 제거"""
        if event_type in self.callbacks:
            if callback in self.callbacks[event_type]:
                self.callbacks[event_type].remove(callback)
    
    async def connect_kline_3m_stream(self):
        """3분봉 Kline 스트림 연결"""
        uri = f"{self.ws_url}/{self.symbol}@kline_3m"
        
        async with websockets.connect(uri) as websocket:
            # 첫 시작 시 signals가 비어있으면 모든 지표 업데이트 및 전략 실행
            print("🚀 첫 시작 - 모든 지표 업데이트 및 전략 실행")
            await self._initialize_all_strategies()

            # 3분봉 카운트다운 시작
            self.countdown_task = asyncio.create_task(self._countdown_to_next_3min_candle())

            async for message in websocket:
                if not self.running:
                    break
                
                data = json.loads(message)
                await self.queue.put(("kline_3m", data))
    
    async def _countdown_to_next_3min_candle(self):
        """다음 3분봉까지 남은 시간 카운트다운"""
        try:
            while self.running:
                current_time = datetime.now(timezone.utc)
                current_minute = current_time.minute
                
                # 다음 3분봉까지 남은 초 계산
                next_3min_minute = ((current_minute // 3) + 1) * 3
                if next_3min_minute >= 60:
                    next_3min_minute = 0
                    next_3min_time = current_time.replace(hour=current_time.hour + 1, minute=0, second=0, microsecond=0)
                else:
                    next_3min_time = current_time.replace(minute=next_3min_minute, second=0, microsecond=0)
                
                remaining_seconds = int((next_3min_time - current_time).total_seconds())
                
                if remaining_seconds > 0:
                    print(f"\r⏳ 다음 3분봉까지 {remaining_seconds:3d}초 남음...", end="", flush=True)
                    await asyncio.sleep(1)
                else:
                    break
                    
        except asyncio.CancelledError:
            # 카운트다운이 취소되면 정상적으로 종료
            pass
        except Exception as e:
            print(f"\n❌ 카운트다운 오류: {e}")
    
    async def _initialize_all_strategies(self):
        """첫 시작 시 모든 지표 업데이트 및 전략 실행"""
        self.strategy_executor.execute_all_strategies()
        signals = self.strategy_executor.get_signals()
        decision = self.decision_engine.decide_trade_realtime(signals)
        print("✅ 모든 지표 및 전략 초기화 완료")
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
        """3분봉 Kline 데이터 처리 (오류 처리 강화)"""
        if not data.get('k', {}).get('x', True):
            return
            
        kline = data['k']
        
        await asyncio.sleep(1)

        print(f"\n⏰ OPEN TIME : {(self.time_manager.get_current_time()).strftime('%H:%M:%S')}")
        
        # 3분봉이 완성되었으므로 카운트다운 재시작
        if self.countdown_task and not self.countdown_task.done():
            self.countdown_task.cancel()
            
        self.countdown_task = asyncio.create_task(self._countdown_to_next_3min_candle())
        
        price_data = self.candle_creator.create_price_data(kline)
        series_3m = self.candle_creator.create_3min_series(price_data)

        self.data_manager.update_with_candle(series_3m)

        if self.candle_creator.is_candle_close("15m"):
            self.data_manager.update_with_candle_15m()
        
        if self.candle_creator.is_candle_close("1h"):
            self.data_manager.update_with_candle_1h()

        self.global_manager.update_all_indicators(series_3m)
        self.strategy_executor.execute_all_strategies()

        signals = self.strategy_executor.get_signals()
        decision = self.decision_engine.decide_trade_realtime(signals)

        # decision이 None이거나 비어있으면 스킵
        if not decision or not isinstance(decision, dict):
            print("⚠️ decision 생성 실패: decision이 None이거나 비어있습니다.")
            return

        indicators = get_all_indicators()
        signals.update({'timestamp': price_data['timestamp'], 'indicators': indicators})

        # Decision 로그에 저장
        # self.decision_logger.log_decision(decision)
        
        # decision이 유효한 경우에만 출력
        try:
            print_decision_interpretation(decision)
        except Exception as e:
            print(f"⚠️ decision 출력 실패: {e}")

        # Meta-Guided Consensus 구조: final_decision에서 action 확인
        final_decision = decision.get("final_decision")
        if not final_decision or not isinstance(final_decision, dict):
            # final_decision이 없으면 스킵
            return
        
        action = final_decision.get("action", "HOLD")
        
        if action and action != "HOLD":
            try:
                send_telegram_message(decision)
            except Exception as e:
                print(f"⚠️ 텔레그램 메시지 전송 실패: {e}")
        
        # 거래 실행 (활성화된 경우)
        if self.enable_trading and self.trade_executor:
            # final_decision이 없거나 비어있으면 스킵
            if not final_decision or not isinstance(final_decision, dict):
                return
            
            action = final_decision.get("action", "HOLD")
            if action != "HOLD":
                # 메타 라벨링 결과 확인
                meta_labeling = final_decision.get("meta", {}).get("meta_labeling", {})
                should_execute = meta_labeling.get("should_execute", False)
                
                if should_execute:
                    try:
                        self.trade_executor.execute_decision(final_decision)
                    except Exception as e:
                        print(f"❌ 거래 실행 오류: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    probability = meta_labeling.get("probability", 0.0)
                    print(f"⚠️ 메타 라벨링에 의해 거래 차단 (확률: {probability:.1%})")

        self._execute_kline_callbacks(price_data)

        if self.time_manager.is_midnight_time():
            self.event_manager.load_daily_events()
            print(self.event_manager.get_events())

    def important_event_occurred(self) -> bool:
        """중요 이벤트 발생 여부 체크"""
        return self.event_manager.important_event_occurred()
    
    def _execute_kline_callbacks(self, price_data: Dict):
        """3분봉 Kline 콜백 실행"""
        for callback in self.callbacks['kline_3m']:
            try:
                callback(price_data)
            except Exception as e:
                print(f"3분봉 Kline 콜백 실행 오류: {e}")
    
    async def start(self):
        """웹소켓 스트림 시작"""
        self.running = True

        tasks = [
            self.connect_kline_3m_stream(),
            self.worker() 
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