#!/usr/bin/env python3
"""
통합 스마트 트레이더 핵심 컴포넌트
"""

import threading
import pandas as pd
from typing import Dict, Any
from data.binance_websocket import BinanceWebSocket
from signals.integrated_strategy import IntegratedStrategy
from signals.timing_strategy import TimingStrategy
from config.integrated_config import IntegratedConfig
from data.loader import build_df


class DataLoader:
    """데이터 로더 클래스"""
    
    def load_klines(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        """K라인 데이터 로드"""
        return build_df(symbol, interval, limit)


class TraderCore:
    """트레이더 핵심 컴포넌트 관리"""
    
    def __init__(self, config: IntegratedConfig):
        self.config = config
        self.running = False
        
        # 핵심 컴포넌트 초기화
        self._init_core_components()
        
        # 스레드 초기화
        self._init_threads()
    
    def _init_core_components(self):
        """핵심 컴포넌트 초기화"""
        self.websocket = BinanceWebSocket(self.config.symbol)
        self.integrated_strategy = IntegratedStrategy(self.config)
        self.timing_strategy = TimingStrategy(self.integrated_strategy.timing_cfg)
        self.data_loader = DataLoader()
    
    def _init_threads(self):
        """스레드 초기화"""
        self.hybrid_thread = None
        self.websocket_thread = None
    
    def start_websocket(self):
        """웹소켓 시작"""
        self.websocket.start_background()
    
    def stop_websocket(self):
        """웹소켓 중지"""
        self.websocket.stop()
    
    def get_websocket(self) -> BinanceWebSocket:
        """웹소켓 인스턴스 반환"""
        return self.websocket
    
    def get_integrated_strategy(self) -> IntegratedStrategy:
        """통합 전략 인스턴스 반환"""
        return self.integrated_strategy
    
    def get_timing_strategy(self) -> TimingStrategy:
        """타이밍 전략 인스턴스 반환"""
        return self.timing_strategy
    
    def get_data_loader(self) -> DataLoader:
        """데이터 로더 인스턴스 반환"""
        return self.data_loader
