#!/usr/bin/env python3
"""
통합 스마트 트레이더 핵심 컴포넌트
"""

from managers.binance_websocket import BinanceWebSocket
from config.integrated_config import IntegratedConfig



class TraderCore:
    """트레이더 핵심 컴포넌트 관리"""
    
    def __init__(self, config: IntegratedConfig, strategy_executor=None):
        self.config = config
        
        # TimeManager 초기화
        from managers.time_manager import get_time_manager
        self.time_manager = get_time_manager()
        
        # 핵심 컴포넌트 초기화
        self.websocket = BinanceWebSocket(self.config.symbol, strategy_executor)
    
    def start_websocket(self):
        """웹소켓 시작"""
        self.websocket.start_background()
    
    def stop_websocket(self):
        """웹소켓 중지"""
        self.websocket.stop()
    
    def get_websocket(self) -> BinanceWebSocket:
        """웹소켓 인스턴스 반환"""
        return self.websocket
    