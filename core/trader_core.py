#!/usr/bin/env python3
"""
통합 스마트 트레이더 핵심 컴포넌트
"""

import pandas as pd
from data.binance_websocket import BinanceWebSocket
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
        
        # 핵심 컴포넌트 초기화
        self.websocket = BinanceWebSocket(self.config.symbol)
        self.data_loader = DataLoader()
    
    def start_websocket(self):
        """웹소켓 시작"""
        self.websocket.start_background()
    
    def stop_websocket(self):
        """웹소켓 중지"""
        self.websocket.stop()
    
    def get_websocket(self) -> BinanceWebSocket:
        """웹소켓 인스턴스 반환"""
        return self.websocket
    
    def get_data_loader(self) -> DataLoader:
        return self.data_loader
