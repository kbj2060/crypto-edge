#!/usr/bin/env python3
"""
실시간 청산 데이터 기반 자동 트레이더
바이낸스 웹소켓을 통해 실시간 청산 데이터를 수집하고 신호를 생성합니다.
"""

import asyncio
import time
import datetime
from typing import Dict, Any, Optional
from data.binance_websocket import BinanceWebSocket
from signals.liquidation_strategy import LiquidationStrategy, LiquidationConfig
from signals.timing_strategy import TimingStrategy, TimingConfig

class RealtimeLiquidationTrader:
    """실시간 청산 데이터 기반 자동 트레이더"""
    
    def __init__(self, symbol: str = "ETHUSDT"):
        self.symbol = symbol
        self.running = False
        
        # 웹소켓 클라이언트
        self.websocket = BinanceWebSocket(symbol)
        
        # 전략 설정
        self.liquidation_cfg = LiquidationConfig(
            min_liquidation_count=3,
            min_liquidation_value=100000.0,
            buy_liquidation_ratio=0.7,
            sell_liquidation_ratio=0.7,
            volume_spike_threshold=2.0
        )
        
        self.timing_cfg = TimingConfig(
            entry_confidence_min=0.3,
            entry_rr_min=0.2,
            entry_score_threshold=0.4
        )
        
        # 전략 인스턴스
        self.liquidation_strategy = LiquidationStrategy(self.liquidation_cfg)
        self.timing_strategy = TimingStrategy(self.timing_cfg)
        
        # 콜백 등록
        self._setup_callbacks()
        
        # 통계
        self.signal_count = 0
        self.last_signal_time = None
    
    def _setup_callbacks(self):
        """웹소켓 콜백 설정"""
        self.websocket.add_callback('liquidation', self._on_liquidation)
        self.websocket.add_callback('volume', self._on_volume_spike)
        self.websocket.add_callback('price', self._on_price_update)
    
    def _on_liquidation(self, liquidation_data: Dict):
        """청산 이벤트 콜백"""
        print(f"🔥 청산 감지: {liquidation_data['side']} {liquidation_data['quantity']:.2f} @ ${liquidation_data['price']:.2f}")
        
        # 실시간 신호 분석
        self._analyze_realtime_signal()
    
    def _on_volume_spike(self, volume_data: Dict):
        """거래량 급증 콜백"""
        print(f"📈 거래량 급증: {volume_data['ratio']:.1f}x @ ${volume_data['price']:.2f}")
        
        # 실시간 신호 분석
        self._analyze_realtime_signal()
    
    def _on_price_update(self, price_data: Dict):
        """가격 업데이트 콜백"""
        # 가격 변동이 클 때만 출력
        if len(self.websocket.price_history) >= 2:
            prev_price = self.websocket.price_history[-2]['price']
            current_price = price_data['price']
            change_pct = ((current_price - prev_price) / prev_price) * 100
            
            if abs(change_pct) > 0.5:  # 0.5% 이상 변동 시
                print(f"💰 가격 변동: ${prev_price:.2f} → ${current_price:.2f} ({change_pct:+.2f}%)")
    
    def _analyze_realtime_signal(self):
        """실시간 신호 분석"""
        try:
            # 현재 가격과 ATR 가져오기
            if not self.websocket.price_history:
                return
            
            current_price = self.websocket.price_history[-1]['price']
            
            # ATR 계산 (간단한 변동성 계산)
            if len(self.websocket.price_history) >= 14:
                prices = [p['price'] for p in self.websocket.price_history[-14:]]
                price_changes = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
                atr = sum(price_changes) / len(price_changes)
            else:
                atr = current_price * 0.02  # 기본값
            
            # 청산 통계 분석
            liquidation_stats = self.websocket.get_liquidation_stats(5)  # 최근 5분
            volume_analysis = self.websocket.get_volume_analysis(3)     # 최근 3분
            
            # 청산 신호 분석
            signal = self.liquidation_strategy.analyze_liquidation_signal(
                liquidation_stats, volume_analysis, current_price, atr
            )
            
            if signal:
                self._process_signal(signal)
            
            # 시장 심리 출력
            self._print_market_sentiment(liquidation_stats, volume_analysis)
            
        except Exception as e:
            print(f"❌ 실시간 신호 분석 오류: {e}")
    
    def _process_signal(self, signal: Dict):
        """신호 처리"""
        try:
            action = signal.get('action')
            confidence = signal.get('confidence', 0)
            risk_reward = signal.get('risk_reward', 0)
            entry_price = signal.get('entry_price', 0)
            stop_loss = signal.get('stop_loss', 0)
            take_profit1 = signal.get('take_profit1', 0)
            take_profit2 = signal.get('take_profit2', 0)
            reason = signal.get('reason', '')
            
            # 신호 출력
            if action == "BUY":
                print(f"\n📈 BUY 신호 - {datetime.datetime.now().strftime('%H:%M:%S')}")
                print(f"💰 ${entry_price:.2f} | 📊 {confidence:.1%} | ⚖️ {risk_reward:.1f}")
                print(f"🛑 ${stop_loss:.2f} | 💎 ${take_profit1:.2f} | 💎 ${take_profit2:.2f}")
                print(f"🔍 {reason}")
            elif action == "SELL":
                print(f"\n📉 SELL 신호 - {datetime.datetime.now().strftime('%H:%M:%S')}")
                print(f"💰 ${entry_price:.2f} | 📊 {confidence:.1%} | ⚖️ {risk_reward:.1f}")
                print(f"🛑 ${stop_loss:.2f} | 💎 ${take_profit1:.2f} | 💎 ${take_profit2:.2f}")
                print(f"🔍 {reason}")
            
            # 타이밍 분석
            timing_analysis = self.timing_strategy.analyze_entry_timing(signal)
            
            if timing_analysis and timing_analysis.get('action') in ['BUY', 'SELL']:
                # 포지션 오픈
                position_id = self.timing_strategy.open_position(timing_analysis)
                print(f"🚀 포지션 오픈: {position_id}")
                
                # 통계 업데이트
                self.signal_count += 1
                self.last_signal_time = datetime.datetime.now()
                
                # 포지션 요약
                position_summary = self.timing_strategy.get_position_summary()
                if position_summary['active_positions'] > 0:
                    print(f"📊 활성 포지션: {position_summary['active_positions']}개 | 💰 일일 손익: {position_summary['daily_pnl']:.4f}")
            
        except Exception as e:
            print(f"❌ 신호 처리 오류: {e}")
    
    def _print_market_sentiment(self, liquidation_stats: Dict, volume_analysis: Dict):
        """시장 심리 출력"""
        sentiment = self.liquidation_strategy.get_market_sentiment(liquidation_stats, volume_analysis)
        
        # 중요한 변화가 있을 때만 출력
        if (liquidation_stats['total_count'] > 0 or 
            volume_analysis['volume_ratio'] > 1.5):
            
            print(f"\n📊 시장 심리 - {datetime.datetime.now().strftime('%H:%M:%S')}")
            print(f"🔥 청산: {liquidation_stats['total_count']}개 (BUY: {liquidation_stats['buy_ratio']:.1%}, SELL: {liquidation_stats['sell_ratio']:.1%})")
            print(f"📈 거래량: {volume_analysis['volume_trend']} ({volume_analysis['volume_ratio']:.1f}x)")
            print(f"🎯 종합: {sentiment['overall_sentiment']}")
    
    def start(self):
        """트레이더 시작"""
        print(f"🚀 {self.symbol} 실시간 청산 트레이더 시작!")
        print(f"📊 청산 임계값: {self.liquidation_cfg.min_liquidation_count}개, ${self.liquidation_cfg.min_liquidation_value:,.0f}")
        print(f"📈 거래량 임계값: {self.liquidation_cfg.volume_spike_threshold}x")
        print(f"⏰ 신호 쿨다운: {self.liquidation_strategy.signal_cooldown}")
        print("=" * 60)
        
        self.running = True
        
        # 웹소켓 백그라운드 시작
        self.websocket.start_background()
        
        # 메인 루프
        try:
            while self.running:
                # 통계 출력 (1분마다)
                if (not self.last_signal_time or 
                    datetime.datetime.now() - self.last_signal_time > datetime.timedelta(minutes=1)):
                    
                    self._print_status()
                    time.sleep(60)
                else:
                    time.sleep(1)
                    
        except KeyboardInterrupt:
            print("\n⏹️ 사용자에 의해 중지됨")
        finally:
            self.stop()
    
    def _print_status(self):
        """상태 출력"""
        liquidation_stats = self.websocket.get_liquidation_stats(5)
        volume_analysis = self.websocket.get_volume_analysis(3)
        
        print(f"\n📊 상태 업데이트 - {datetime.datetime.now().strftime('%H:%M:%S')}")
        print(f"🔥 최근 5분 청산: {liquidation_stats['total_count']}개 (${liquidation_stats['total_value']:,.0f})")
        print(f"📈 거래량 트렌드: {volume_analysis['volume_trend']} ({volume_analysis['volume_ratio']:.1f}x)")
        print(f"🎯 총 신호: {self.signal_count}개")
        
        if self.last_signal_time:
            time_since = datetime.datetime.now() - self.last_signal_time
            print(f"⏰ 마지막 신호: {time_since.total_seconds():.0f}초 전")
    
    def stop(self):
        """트레이더 중지"""
        self.running = False
        self.websocket.stop()
        print("🛑 실시간 청산 트레이더 중지됨")

def main():
    """메인 함수"""
    trader = RealtimeLiquidationTrader("ETHUSDT")
    trader.start()

if __name__ == "__main__":
    main()
