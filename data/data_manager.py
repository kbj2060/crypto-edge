#!/usr/bin/env python3
"""
Data Manager - 중앙 데이터 관리
- 1000개 캔들 데이터 중앙 관리
- 모든 지표들이 공통으로 사용할 데이터 제공
- 싱글톤 패턴으로 전역 접근 가능
"""

import pandas as pd
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone, timedelta
from collections import deque
from utils.time_manager import get_time_manager
from data.binance_dataloader import BinanceDataLoader


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

    
    def load_initial_data(self, symbol: str = 'ETHUSDT') -> bool:
        """초기 데이터 로딩 (전날 00시부터 현재까지)"""
        try:
            # 전날 00시부터 현재까지 데이터 가져오기
            current_time = self.time_manager.get_current_time()
            yesterday_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)

            # 3분봉 데이터 직접 가져오기 (긴 기간은 자동으로 여러 번에 나누어 요청)
            df_3m = self.dataloader.fetch_data(
                interval=3,
                symbol=symbol,
                start_time=yesterday_start,
                end_time=current_time
            )
            
            if df_3m is not None and not df_3m.empty:
                self.data = df_3m.copy()
                
                # 마지막 3분봉 타임스탬프 설정
                if not self.data.empty:
                    pass
                
                self._data_loaded = True
                return True
            else:
                return False
        
        except Exception as e:
            return False
    
    def update_with_candle(self, candle_data: pd.Series) -> None:
        """새로운 캔들 데이터로 업데이트 (실시간 용)"""
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
                        
            # 새로운 캔들 데이터를 DataFrame에 추가
            new_row = pd.DataFrame([{
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume,
                'quote_volume': quote_volume
            }], index=[candle_data.name])
            
            # 기존 데이터에 추가 (인덱스 중복 방지)
            self.data = pd.concat([self.data, new_row], ignore_index=False)
            
            # 중복된 인덱스 제거 (최신 데이터 유지)
            self.data = self.data[~self.data.index.duplicated(keep='last')]
            
            # 최대 1000개 캔들 유지
            if len(self.data) > 1000:
                self.data = self.data.tail(1000)

            print(f"✅ 3분봉 데이터 업데이트 완료: {datetime.now(timezone.utc).strftime('%H:%M:%S')}")
        except Exception as e:
            pass

    
    def get_dataframe(self) -> pd.DataFrame:
        """전체 3분봉 데이터를 DataFrame으로 반환"""
        try:
            if self.data.empty:
                return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume'])
            
            # DataFrame을 직접 반환
            return self.data.copy()
            
        except Exception as e:
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume'])
    
    def get_latest_data(self, count: int = 1) -> pd.DataFrame:
        """최신 3분봉 데이터를 DataFrame으로 반환"""
        try:
            if self.data.empty:
                return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume'])
            
            latest_df = self.data.tail(count).copy()
            return latest_df
                
        except Exception as e:
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume'])
    
    def get_data_range(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """특정 시간 범위의 3분봉 데이터를 DataFrame으로 반환"""
        try:
            if self.data.empty:
                return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume'])
            
            # 시간대 변환
            if start_time.tzinfo is None:
                start_time = self.time_manager.convert_to_utc(start_time)
            if end_time.tzinfo is None:
                end_time = self.time_manager.convert_to_utc(end_time)
            
            # DataFrame 인덱스로 시간 범위 필터링
            mask = (self.data.index >= start_time) & (self.data.index <= end_time)
            filtered_df = self.data[mask]
            
            return filtered_df.copy()
            
        except Exception as e:
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
