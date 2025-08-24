#!/usr/bin/env python3
"""
Data Indicator - 최근 3분봉 데이터 관리
- 최근 200개 3분봉 데이터 유지
- FIFO 방식으로 새로운 데이터 추가/오래된 데이터 제거
- 실시간 데이터 업데이트 지원
"""

import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from collections import deque
from utils.time_manager import get_time_manager


class DataIndicator:
    """최근 3분봉 데이터를 관리하는 지표 클래스"""
    
    def __init__(self, max_candles: int = 1000):
        """
        DataIndicator 초기화
        
        Args:
            max_candles: 최대 유지할 캔들 개수 (기본값: 480)
        """
        self.max_candles = max_candles
        self.data = deque(maxlen=max_candles)  # FIFO 큐로 최근 데이터 유지
        self._initialized = False
        self.time_manager = get_time_manager()  # TimeManager 통일
        
        print(f"🚀 Data Indicator 초기화: 최대 {max_candles}개 캔들 유지")
    
    def update_with_candle(self, candle_data: Dict[str, Any]) -> None:
        """
        새로운 3분봉 데이터로 업데이트
        
        Args:
            candle_data: 3분봉 캔들 데이터 {
                'timestamp': datetime,
                'open': float,
                'high': float,
                'low': float,
                'close': float,
                'volume': float
            }
        """
        try:
            # 데이터 검증
            required_fields = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(field in candle_data for field in required_fields):
                print(f"⚠️ Data Indicator: 필수 필드 누락 - {required_fields}")
                return
            
            # timestamp를 UTC로 변환 (TimeManager 사용)
            timestamp = candle_data['timestamp']
            if timestamp.tzinfo is None:
                timestamp = candle_data['timestamp'] = self.time_manager.convert_to_utc(timestamp)
            
            # 새로운 캔들 데이터 추가 (FIFO 방식)
            self.data.append(candle_data)
            
            # 초기화 상태 업데이트
            if not self._initialized and len(self.data) >= 10:  # 최소 10개 캔들
                self._initialized = True
                print(f"✅ Data Indicator: 초기화 완료 ({len(self.data)}개 캔들)")
            
        except Exception as e:
            print(f"❌ Data Indicator 업데이트 오류: {e}")
    
    def update_with_dataframe(self, df: pd.DataFrame) -> None:
        """DataFrame을 한 번에 처리하여 DataIndicator에 삽입"""
        try:
            if df is None or df.empty:
                print("⚠️ Data Indicator: 빈 DataFrame")
                return
            
            print(f"🚀 Data Indicator 벌크 삽입: {len(df)}개 캔들")
            
            # 기존 데이터 초기화 후 벌크 데이터 삽입
            self.data.clear()
            bulk_data = []
            for ts, row in df.iterrows():
                # timestamp가 index인 경우와 column인 경우 모두 처리
                if isinstance(ts, (str, pd.Timestamp)):
                    timestamp = self.time_manager.convert_to_utc(ts) if ts.tzinfo is None else ts
                else:
                    # timestamp가 column에 있는 경우
                    timestamp = self.time_manager.convert_to_utc(row.get('timestamp', ts)) if row.get('timestamp', ts).tzinfo is None else row.get('timestamp', ts)
                
                bulk_data.append({
                    'timestamp': timestamp,
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume'])
                })
                
            self.data.extend(bulk_data)
            self._initialized = True

        except Exception as e:
            print(f"❌ Data Indicator 벌크 업데이트 오류: {e}")
    

    
    def get_dataframe(self) -> pd.DataFrame:
        """
        전체 데이터를 DataFrame으로 반환 (통일된 구조)
        
        Returns:
            pandas DataFrame (columns: open, high, low, close, volume, index: timestamp)
        """
        try:
            if not self.data:
                return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
            
            # 데이터를 DataFrame으로 변환
            df_data = []
            for candle in self.data:
                df_data.append({
                    'open': float(candle['open']),
                    'high': float(candle['high']),
                    'low': float(candle['low']),
                    'close': float(candle['close']),
                    'volume': float(candle['volume'])
                })
            
            df = pd.DataFrame(df_data, index=[candle['timestamp'] for candle in self.data])
            df.index.name = 'timestamp'
            
            return df
            
        except Exception as e:
            print(f"❌ Data Indicator DataFrame 변환 오류: {e}")
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
    
    def get_status(self) -> Dict[str, Any]:
        """
        Data Indicator 상태 정보 반환
        
        Returns:
            상태 정보 딕셔너리
        """
        try:
            if not self.data:
                return {
                    'is_initialized': False,
                    'candles_count': 0,
                    'max_candles': self.max_candles,
                    'data_range': None,
                    'latest_timestamp': None
                }
            
            latest_timestamp = self.data[-1]['timestamp']
            earliest_timestamp = self.data[0]['timestamp']
            
            return {
                'is_initialized': self._initialized,
                'candles_count': len(self.data),
                'max_candles': self.max_candles,
                'data_range': {
                    'start': earliest_timestamp,
                    'end': latest_timestamp,
                    'duration_hours': (latest_timestamp - earliest_timestamp).total_seconds() / 3600
                },
                'latest_timestamp': latest_timestamp,
                'is_full': len(self.data) >= self.max_candles
            }
            
        except Exception as e:
            print(f"❌ Data Indicator 상태 조회 오류: {e}")
            return {}
    
    def get_latest_data(self, count: int = 1) -> Optional[Dict[str, Any]]:
        """최신 캔들 데이터 반환 (웹소켓에서 사용)"""
        try:
            if not self.data:
                return None
            
            if count == 1:
                return self.data[-1]
            else:
                return list(self.data)[-count:]
                
        except Exception as e:
            print(f"❌ Data Indicator 최신 데이터 조회 오류: {e}")
            return None
