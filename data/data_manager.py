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
        
        print(f"🚀 DataManager 싱글톤 초기화: 3분봉 캔들 관리")
        
    
    def load_initial_data(self, symbol: str = 'ETHUSDT') -> bool:
        """초기 데이터 로딩 (전날 00시부터 현재까지)"""
        try:
            print("📊 DataManager: 초기 데이터 로딩 시작...")
            
            # 전날 00시부터 현재까지 데이터 가져오기
            current_time = self.time_manager.get_current_time()
            yesterday_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
            
            print(f"📊 데이터 기간: 전날 00시({yesterday_start.strftime('%Y-%m-%d %H:%M')}) ~ 현재({current_time.strftime('%Y-%m-%d %H:%M')})")
            print(f"🎯 목표: 전체 기간 3분봉 데이터 로딩")
            
            # 3분봉 데이터 직접 가져오기 (긴 기간은 자동으로 여러 번에 나누어 요청)
            df_3m = self.dataloader.fetch_data(
                interval=3,
                symbol=symbol,
                start_time=yesterday_start,
                end_time=current_time
            )
            
            if df_3m is not None and not df_3m.empty:
                print(f"✅ DataManager: {len(df_3m)}개 3분봉 데이터 로드 성공")
                self.data = df_3m.copy()
                
                # 마지막 3분봉 타임스탬프 설정
                if not self.data.empty:
                    print(f"📊 마지막 3분봉 시간: {self.data.index[-1].strftime('%H:%M')}")
                
                self._data_loaded = True
                return True
            else:
                print("❌ DataManager: 3분봉 데이터 로드 실패")
                return False
                
        except Exception as e:
            print(f"❌ DataManager 초기 로딩 오류: {e}")
            return False
    

    
    def update_with_candle(self, candle_data: pd.Series) -> None:
        """새로운 캔들 데이터로 업데이트 (실시간 용)"""
        try:
            # TimeManager를 사용하여 timestamp 추출 및 정규화
            timestamp = self.time_manager.extract_and_normalize_timestamp(candle_data)
            
            # 새로운 캔들 데이터를 DataFrame에 추가
            new_row = pd.DataFrame([{
                'open': candle_data['open'],
                'high': candle_data['high'],
                'low': candle_data['low'],
                'close': candle_data['close'],
                'volume': candle_data['volume'],
                'quote_volume': candle_data['quote_volume']
            }], index=[timestamp])
            
            self.data = pd.concat([self.data, new_row], ignore_index=False)
            
            # 최대 1000개 캔들 유지
            if len(self.data) > 1000:
                self.data = self.data.tail(1000)
                
        except Exception as e:
            print(f"❌ DataManager 캔들 업데이트 오류: {e}")

    
    def get_dataframe(self) -> pd.DataFrame:
        """전체 3분봉 데이터를 DataFrame으로 반환"""
        try:
            if self.data.empty:
                return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume'])
            
            # DataFrame을 직접 반환
            return self.data.copy()
            
        except Exception as e:
            print(f"❌ DataManager DataFrame 반환 오류: {e}")
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume'])
    
    def get_latest_data(self, count: int = 1) -> pd.DataFrame:
        """최신 3분봉 데이터를 DataFrame으로 반환"""
        try:
            if self.data.empty:
                return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume'])
            
            latest_df = self.data.tail(count).copy()
            return latest_df
                
        except Exception as e:
            print(f"❌ DataManager 최신 데이터 조회 오류: {e}")
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
            print(f"❌ DataManager 시간 범위 조회 오류: {e}")
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume'])
    
    def is_ready(self) -> bool:
        """데이터가 준비되었는지 확인"""
        return self._data_loaded and len(self.data) >= 10
    
    def clear(self) -> None:
        """모든 데이터 초기화"""
        self.data = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume'])
        self._data_loaded = False
        print("🔄 DataManager: 모든 데이터 초기화 완료")


# 전역 DataManager 인스턴스 생성 함수
def get_data_manager() -> DataManager:
    """전역 DataManager 인스턴스 반환"""
    return DataManager()
