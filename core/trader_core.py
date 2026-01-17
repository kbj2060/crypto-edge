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
        # 거래 활성화 여부는 config에서 가져오거나 기본값 사용
        enable_trading = getattr(config, 'enable_trading', False)
        simulation_mode = getattr(config, 'simulation_mode', True)
        use_demo = getattr(config, 'use_demo', False)
        
        self.websocket = BinanceWebSocket(
            self.config.symbol, 
            strategy_executor,
            enable_trading=enable_trading,
            simulation_mode=simulation_mode,
            demo=use_demo
        )
    
    def start_websocket(self):
        """웹소켓 시작"""
        self.websocket.start_background()
    
    def stop_websocket(self):
        """웹소켓 중지"""
        self.websocket.stop()
    
    def get_websocket(self) -> BinanceWebSocket:
        """웹소켓 인스턴스 반환"""
        return self.websocket
    