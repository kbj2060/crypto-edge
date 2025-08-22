#!/usr/bin/env python3
"""
타임스탬프 유틸리티 함수들
다양한 타임스탬프 형식 간 변환을 처리
"""

from datetime import datetime
from typing import Union


def get_timestamp_int(timestamp: Union[datetime, int, float, None]) -> int:
    """timestamp를 int 타입으로 변환"""
    try:
        if isinstance(timestamp, datetime):
            return int(timestamp.timestamp())
        elif isinstance(timestamp, (int, float)):
            return int(timestamp)
        else:
            return 0
    except Exception:
        return 0


def get_timestamp_datetime(timestamp: Union[datetime, int, float, None]) -> datetime:
    """timestamp를 datetime 타입으로 변환"""
    try:
        if isinstance(timestamp, datetime):
            return timestamp
        elif isinstance(timestamp, (int, float)):
            return datetime.fromtimestamp(timestamp)
        else:
            return datetime.now()
    except Exception:
        return datetime.now()


def get_current_timestamp_int() -> int:
    """현재 시간을 int timestamp로 반환"""
    return int(datetime.now().timestamp())


def get_current_timestamp_datetime() -> datetime:
    """현재 시간을 datetime으로 반환"""
    return datetime.now()
