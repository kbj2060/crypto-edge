#!/usr/bin/env python3
"""
시간 관리자 (Time Manager)
- UTC 시간 통일 관리
- 시간대 변환
- 기본적인 시간 유틸리티
- 싱글톤 패턴으로 구현
"""

import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Optional, Union

class TimeManager:
    """시간 관리자 - 싱글톤 패턴"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls, target_time: Optional[datetime] = None):
        if cls._instance is None:
            cls._instance = super(TimeManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, target_time: Optional[datetime] = None):
        if not self._initialized:
            self._initialized = True
            self._timezone = timezone.utc
            self.target_time = target_time if target_time is not None else datetime.now(self._timezone)

    def normalize_minute(self, target_time: datetime) -> datetime:
        """초 반올림해서 분에 +1 해주는 함수"""
        try:
            # 30초 이상이면 다음 분으로 반올림
            if target_time.second >= 30:
                return target_time + timedelta(minutes=1)
            return target_time
        except Exception as e:
            print(f"분 정규화 오류: {e}")
            return target_time
    
    # =============================================================================
    # 기본 시간 관리 메서드
    # =============================================================================
    def update_with_candle(self, candle_data: pd.Series):
        """새로운 캔들로 업데이트"""
        self.target_time = self.ensure_utc(candle_data.name)
    
    def get_current_time(self) -> datetime:
        """현재 시간을 UTC로 반환"""
        # return datetime.now(self._timezone)
        return self.target_time
    
    def ensure_utc(self, dt: datetime) -> datetime:
        """datetime을 UTC로 변환 (이미 UTC면 그대로 반환)"""
        if dt.tzinfo is None:
            return dt.replace(tzinfo=self._timezone)
        elif dt.tzinfo != self._timezone:
            return dt.astimezone(self._timezone)
        return dt
    
    def format_datetime(self, dt: datetime, format_str: str = "%Y-%m-%d %H:%M UTC") -> str:
        """datetime을 지정된 형식의 문자열로 변환"""
        dt_utc = self.ensure_utc(dt)
        return dt_utc.strftime(format_str)
    
    # =============================================================================
    # Timestamp 유틸리티 메서드
    # =============================================================================
    
    def get_timestamp_datetime(self, timestamp: Union[datetime, int, float, None]) -> datetime:
        """timestamp를 datetime 타입으로 변환"""
        try:
            if isinstance(timestamp, datetime):
                return timestamp
            elif isinstance(timestamp, (int, float)):
                return datetime.fromtimestamp(timestamp/1000, tz=timezone.utc)
            else:
                return datetime.now(timezone.utc)
        except Exception:
            return datetime.now(timezone.utc)
    
    def is_midnight_time(self) -> bool:
        """밤 12시인지 확인"""
        current_time = self.get_current_time()
        return current_time.hour == 0 and current_time.minute == 0

# 전역 TimeManager 인스턴스
_global_time_manager: Optional[TimeManager] = None

def get_time_manager(target_time: Optional[datetime] = None) -> TimeManager:
    """전역 TimeManager 인스턴스 반환 (싱글톤 패턴)"""
    global _global_time_manager

    if _global_time_manager is None:
        _global_time_manager = TimeManager(target_time )
    
    return _global_time_manager