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
            
        self.max_candles = 1000
        self.data = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume'])  # DataFrame으로 초기화
        self._data_loaded = False
        self.time_manager = get_time_manager()
        self.dataloader = BinanceDataLoader()
        self._initialized = True
        
        print(f"🚀 DataManager 싱글톤 초기화: 최대 {self.max_candles}개 캔들 관리")
    
    def load_initial_data(self, symbol: str = 'ETHUSDT') -> bool:
        """초기 데이터 로딩 (전날 00시부터 현재까지)"""
        try:
            print("📊 DataManager: 초기 데이터 로딩 시작...")
            
            # 전날 00시부터 현재까지 데이터 가져오기
            current_time = datetime.now(timezone.utc)
            yesterday_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
            
            print(f"📊 데이터 기간: 전날 00시({yesterday_start.strftime('%Y-%m-%d %H:%M')}) ~ 현재({current_time.strftime('%Y-%m-%d %H:%M')})")
            print(f"🎯 목표: {self.max_candles}개 캔들 데이터 로딩")
            
            df = self.dataloader.fetch_3m_data(
                symbol=symbol,
                start_time=yesterday_start,
                end_time=current_time,
                limit=self.max_candles
            )
            
            if df is not None and not df.empty:
                print(f"✅ DataManager: {len(df)}개 데이터 로드 성공")
                # DataFrame을 직접 저장
                self.data = df.copy()
                self._data_loaded = True
                return True
            else:
                print("❌ DataManager: 데이터 로드 실패")
                return False
                
        except Exception as e:
            print(f"❌ DataManager 초기 로딩 오류: {e}")
            return False
    
    def update_with_candle(self, candle_data: Dict[str, Any]) -> None:
        """새로운 캔들 데이터로 업데이트 (실시간 용)"""
        try:
            # 데이터 검증
            required_fields = ['close_time', 'open', 'high', 'low', 'close', 'volume', 'quote_volume']
            if not all(field in candle_data for field in required_fields):
                print(f"⚠️ DataManager: 필수 필드 누락 - {required_fields}")
                return
            
            # timestamp를 UTC로 변환
            timestamp = candle_data['close_time']
            if isinstance(timestamp, (int, float)):
                # 밀리초 타임스탬프인 경우 datetime으로 변환
                timestamp = pd.to_datetime(timestamp, unit='ms', utc=True)
                candle_data['close_time'] = timestamp
            elif timestamp.tzinfo is None:
                # timezone이 없는 datetime인 경우 UTC로 변환
                timestamp = self.time_manager.convert_to_utc(timestamp)
                candle_data['close_time'] = timestamp
            
            # 새로운 캔들 데이터를 DataFrame에 추가
            new_row = pd.DataFrame([candle_data], index=[timestamp])
            self.data = pd.concat([self.data, new_row], ignore_index=False)
            
            # 최대 캔들 수 제한
            if len(self.data) > self.max_candles:
                self.data = self.data.tail(self.max_candles)
            
        except Exception as e:
            print(f"❌ DataManager 캔들 업데이트 오류: {e}")
    
    def get_dataframe(self) -> pd.DataFrame:
        """전체 데이터를 DataFrame으로 반환"""
        try:
            if self.data.empty:
                return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume'])
            
            # DataFrame을 직접 반환
            return self.data.copy()
            
        except Exception as e:
            print(f"❌ DataManager DataFrame 반환 오류: {e}")
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume'])
    
    def get_latest_data(self, count: int = 1) -> Optional[Dict[str, Any]]:
        """최신 캔들 데이터 반환"""
        try:
            if self.data.empty:
                return None
            
            if count == 1:
                # 마지막 행을 딕셔너리로 변환
                last_row = self.data.iloc[-1]
                return {
                    'timestamp': self.data.index[-1],
                    'open': float(last_row['open']),
                    'high': float(last_row['high']),
                    'low': float(last_row['low']),
                    'close': float(last_row['close']),
                    'volume': float(last_row['volume']),
                    'quote_volume': float(last_row['quote_volume'])
                }
            else:
                # 마지막 N개 행을 딕셔너리 리스트로 변환
                latest_data = []
                for i in range(min(count, len(self.data))):
                    idx = -(i + 1)
                    row = self.data.iloc[idx]
                    latest_data.append({
                        'timestamp': self.data.index[idx],
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': float(row['volume']),
                        'quote_volume': float(row['quote_volume'])
                    })
                return latest_data
                
        except Exception as e:
            print(f"❌ DataManager 최신 데이터 조회 오류: {e}")
            return None
    
    def get_data_range(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """특정 시간 범위의 데이터 반환"""
        try:
            if self.data.empty:
                return []
            
            # 시간대 변환
            if start_time.tzinfo is None:
                start_time = self.time_manager.convert_to_utc(start_time)
            if end_time.tzinfo is None:
                end_time = self.time_manager.convert_to_utc(end_time)
            
            # DataFrame 인덱스로 시간 범위 필터링
            mask = (self.data.index >= start_time) & (self.data.index <= end_time)
            filtered_df = self.data[mask]
            
            # 딕셔너리 리스트로 변환
            filtered_data = []
            for timestamp, row in filtered_df.iterrows():
                filtered_data.append({
                    'timestamp': timestamp,
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume']),
                    'quote_volume': float(row['quote_volume'])
                })
            
            return filtered_data
            
        except Exception as e:
            print(f"❌ DataManager 시간 범위 조회 오류: {e}")
            return []
    
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
