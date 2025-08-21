#!/usr/bin/env python3
"""
웹소켓 콜백 핸들러
"""

import datetime
from typing import Dict, Callable
from data.binance_websocket import BinanceWebSocket


class WebSocketHandler:
    """웹소켓 콜백 핸들러"""
    
    def __init__(self, websocket: BinanceWebSocket):
        self.websocket = websocket
        self.callbacks = {}
    
    def setup_callbacks(self, callbacks: Dict[str, Callable]):
        """콜백 설정"""
        for event_type, callback in callbacks.items():
            self.websocket.add_callback(event_type, callback)
            self.callbacks[event_type] = callback
    
    def on_liquidation(self, liquidation_data: Dict, print_density_func, analyze_liquidation_func):
        """청산 이벤트 콜백"""
        # 간단한 한 줄 출력
        side = liquidation_data['side']
        quantity = liquidation_data['quantity']
        price = liquidation_data['price']
        value = quantity * price
        
        # 청산 방향성 해석
        if side == 'SELL':
            liquidation_type = "롱 포지션 강제 청산"
            emoji = "📉"
        elif side == 'BUY':
            liquidation_type = "숏 포지션 강제 청산"
            emoji = "📈"
        else:
            liquidation_type = f"{side} 청산"
            emoji = "🔥"
        
        print(f"{emoji} {liquidation_type}: {quantity:.2f} ETH (${value:,.0f}) @ ${price:.2f}")
        
        # 현재 호가 ±3% 범위 청산 밀도 분석 출력
        print_density_func()
        
        # 실시간 청산 신호 분석
        analyze_liquidation_func()
    
    def on_volume_spike(self, volume_data: Dict, volume_buffer: list, last_summary_time: datetime.datetime,
                            summary_cooldown: int, print_summary_func, analyze_liquidation_func):
        """거래량 급증 콜백"""
        # 거래량 급증을 버퍼에 추가
        volume_buffer.append({
            'timestamp': datetime.datetime.now(),
            'data': volume_data
        })
        
        # 30초마다 요약 출력
        now = datetime.datetime.now()
        if (not last_summary_time or 
            (now - last_summary_time).total_seconds() >= summary_cooldown):
                
            print_summary_func(volume_buffer)  # volume_buffer 매개변수 전달
            last_summary_time = now
            volume_buffer.clear()
        
        # 실시간 청산 신호 분석
        analyze_liquidation_func()
        
        return last_summary_time
    
    def on_price_update(self, price_data: Dict, analyze_technical_func):
        """가격 업데이트 콜백"""
        # 가격 변동이 클 때만 출력 (스캘핑용으로 더 민감하게)
        if len(self.websocket.price_history) >= 2:
            prev_price = self.websocket.price_history[-2]['price']
            current_price = price_data['price']
            change_pct = ((current_price - prev_price) / prev_price) * 100
            
            if abs(change_pct) > 0.1:  # 0.2%에서 0.1%로 낮춤 (스캘핑용)
                print(f"💰 가격 변동: ${prev_price:.2f} → ${current_price:.2f} ({change_pct:+.2f}%)")
                # 큰 가격 변동 시에만 실시간 기술적 분석
                analyze_technical_func()
    
    def on_kline(self, kline_data: Dict, analyze_technical_func):
        """1분봉 K라인 업데이트 콜백"""
        # K라인이 닫힐 때(x=True)만 분석
        if kline_data.get('x', False):
            analyze_technical_func()
