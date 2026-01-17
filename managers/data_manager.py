#!/usr/bin/env python3
"""
Data Manager - 중앙 데이터 관리
- 1000개 캔들 데이터 중앙 관리
- 모든 지표들이 공통으로 사용할 데이터 제공
- 싱글톤 패턴으로 전역 접근 가능
"""

import pandas as pd
import threading
from typing import Optional
from datetime import datetime, timedelta
from config.integrated_config import IntegratedConfig
from managers.time_manager import get_time_manager
from managers.binance_dataloader import BinanceDataLoader


class DataManager:
    """중앙 데이터 관리 클래스 (싱글톤 패턴)"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(DataManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        # 이미 초기화된 경우 중복 초기화 방지
        if hasattr(self, '_initialized'):
            return
        
        self.data = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume'])  # 3분봉 데이터
        self._data_loaded = False
        self.time_manager = get_time_manager()
        self.dataloader = BinanceDataLoader()
        
        self._initialized = True

    
    def load_initial_data(self, symbol: str = 'ETHUSDT', df_3m: Optional[pd.DataFrame] = None, df_15m: Optional[pd.DataFrame] = None, df_1h: Optional[pd.DataFrame] = None) -> bool:
        """초기 데이터 로딩 (전날 00시부터 현재까지)"""
        try:
            # 전날 00시부터 현재까지 데이터 가져오기
            current_time = self.time_manager.get_current_time()
            yesterday_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)

            if df_3m is None:
                # 3분봉 데이터 직접 가져오기 (긴 기간은 자동으로 여러 번에 나누어 요청)
                df_3m = self.dataloader.fetch_data(
                    interval="3m",
                    symbol=symbol,
                    start_time=yesterday_start,
                    end_time=current_time
                )
                
                df_15m = self.dataloader.fetch_data(
                    interval="15m",
                    symbol=symbol,
                    limit=400
                )
                
                df_1h = self.dataloader.fetch_data(
                    interval="1h",
                    symbol=symbol,
                    limit=300
                )
                
                # 가져온 데이터를 인스턴스 변수에 할당
                if df_3m is not None and not df_3m.empty:
                    self.data = df_3m.copy()
                if df_15m is not None and not df_15m.empty:
                    self.data_15m = df_15m.copy()
                if df_1h is not None and not df_1h.empty:
                    self.data_1h = df_1h.copy()
                
                self._data_loaded = True
                return True
            else:   
                self.data = df_3m.copy()
                self.data_15m = df_15m.copy()
                self.data_1h = df_1h.copy()
                
                self._data_loaded = True
                return True
        except Exception:
            return False
    
    def update_with_candle(self, candle_data: pd.Series) -> None:
        """새로운 캔들 데이터로 업데이트 (실시간 용) - 최적화 버전"""
        try:
            if candle_data is None:
                return
            
            # Series에서 직접 값 가져오기
            open_price = float(candle_data['open'])
            high_price = float(candle_data['high'])
            low_price = float(candle_data['low'])
            close_price = float(candle_data['close'])
            volume = float(candle_data['volume'])
            quote_volume = float(candle_data['quote_volume'])
            
            timestamp = candle_data.name
            
            # 최적화: pd.concat 대신 직접 loc로 추가 (더 빠름)
            if timestamp in self.data.index:
                # 기존 행 업데이트
                self.data.loc[timestamp, 'open'] = open_price
                self.data.loc[timestamp, 'high'] = high_price
                self.data.loc[timestamp, 'low'] = low_price
                self.data.loc[timestamp, 'close'] = close_price
                self.data.loc[timestamp, 'volume'] = volume
                self.data.loc[timestamp, 'quote_volume'] = quote_volume
            else:
                # 새 행 추가 (더 빠른 방법)
                self.data.loc[timestamp] = {
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume,
                    'quote_volume': quote_volume
                }
                
                # 데이터 크기 제한 (메모리 관리)
                if len(self.data) > 2000:
                    # 가장 오래된 데이터 제거
                    self.data = self.data.tail(1500)
            

            #print(f"✅ 3분봉 데이터 업데이트 완료: {self.time_manager.get_current_time().strftime('%H:%M:%S')}")
        except Exception:
            pass

    def update_with_candle_15m(self, symbol: str = 'ETHUSDT', data: pd.DataFrame = None) -> None:
        # 웹소켓으로 받은 데이터가 아닌 api 로 1개만 받아 추가
        if data is None:
            new = self.dataloader.fetch_data(interval="15m", symbol=symbol, limit=1)
        else:
            new = data
            
        # 최적화: pd.concat 대신 직접 loc로 추가
        if new is not None and not new.empty:
            for timestamp, row in new.iterrows():
                if timestamp in self.data_15m.index:
                    # 기존 행 업데이트
                    self.data_15m.loc[timestamp] = row
                else:
                    # 새 행 추가
                    self.data_15m.loc[timestamp] = row
                    
                    # 데이터 크기 제한
                    if len(self.data_15m) > 400:
                        self.data_15m = self.data_15m.tail(400)

        #print(f"✅ 15분봉 데이터 업데이트 완료: {self.time_manager.get_current_time().strftime('%H:%M:%S')}")

    def update_with_candle_1h(self, symbol: str = 'ETHUSDT', data: pd.DataFrame = None) -> None:
        # 최적화: pd.concat 대신 직접 loc로 추가
        if data is None:
            new = self.dataloader.fetch_data(interval="1h", symbol=symbol, limit=1)
        else:
            new = data
            
        # 최적화: pd.concat 대신 직접 loc로 추가
        if new is not None and not new.empty:
            for timestamp, row in new.iterrows():
                if timestamp in self.data_1h.index:
                    # 기존 행 업데이트
                    self.data_1h.loc[timestamp] = row
                else:
                    # 새 행 추가
                    self.data_1h.loc[timestamp] = row
                    
                    # 데이터 크기 제한
                    if len(self.data_1h) > 300:
                        self.data_1h = self.data_1h.tail(300)

        #print(f"✅ 1시간봉 데이터 업데이트 완료: {self.time_manager.get_current_time().strftime('%H:%M:%S')}")
    
    def get_dataframe(self) -> pd.DataFrame:
        """전체 3분봉 데이터를 DataFrame으로 반환"""
        try:
            if self.data.empty:
                return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume'])
            
            # DataFrame을 직접 반환
            return self.data.copy()
            
        except Exception:
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume'])
    
    def get_init_data(self) -> pd.DataFrame:
        """초기 데이터를 DataFrame으로 반환"""
        try:
            config = IntegratedConfig()
            if self.data.empty:
                return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume'])
            return self.data.head(config.agent_start_idx).copy()
        except Exception:
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume'])
    
    def get_latest_data(self, count: int = 1) -> pd.DataFrame:
        """최신 3분봉 데이터를 DataFrame으로 반환"""
        try:
            if self.data.empty:
                return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume'])
            
            latest_df = self.data.tail(count).copy()
            return latest_df
                
        except Exception:
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume'])
    
    def get_latest_data_15m(self, count: int = 1) -> pd.DataFrame:
        """최신 15분봉 데이터를 DataFrame으로 반환"""
        try:
            if self.data_15m.empty:
                return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume'])
            latest_df = self.data_15m.tail(count).copy()
            return latest_df
        except Exception:
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume'])

    def get_latest_data_1h(self, count: int = 1) -> pd.DataFrame:
        """최신 1시간봉 데이터를 DataFrame으로 반환"""
        try:
            if self.data_1h.empty:
                return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume'])
            latest_df = self.data_1h.tail(count).copy()
            return latest_df
        except Exception:
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume'])

    def get_data_range(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """특정 시간 범위의 3분봉 데이터를 DataFrame으로 반환"""
        try:
            if self.data.empty:
                return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume'])
            
            # DataFrame 인덱스로 시간 범위 필터링
            mask = (self.data.index >= start_time) & (self.data.index <= end_time)
            filtered_df = self.data[mask]

            return filtered_df.copy()
            
        except Exception:
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume'])
    
    def is_ready(self) -> bool:
        """데이터가 준비되었는지 확인"""
        return self._data_loaded and len(self.data) >= 10
    
    def clear(self) -> None:
        """모든 데이터 초기화"""
        self.data = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume'])
        self._data_loaded = False

# 전역 DataManager 인스턴스 생성 함수
def get_data_manager() -> DataManager:
    """전역 DataManager 인스턴스 반환"""
    return DataManager()
