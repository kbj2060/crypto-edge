#!/usr/bin/env python3
"""
어제 3분봉 데이터의 high, low만 가져오는 간단한 클래스
Note: 어제 데이터는 공용 데이터와 별개이므로 개별 API 호출 유지
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import pandas as pd

from data.data_manager import get_data_manager
from utils.time_manager import get_time_manager


class DailyLevels:
    """어제 3분봉 데이터의 high, low만 관리하는 간단한 클래스"""
    
    def __init__(self):
        self.time_manager = get_time_manager()
        self.prev_day_high = 0.0
        self.prev_day_low = 0.0
        
        # 자동으로 데이터 로드
        self._initialize_levels()
    
    def _initialize_levels(self):
        # high, low만 계산
        df = self.get_data()

        self.prev_day_high = float(df['high'].max())
        self.prev_day_low = float(df['low'].min())

    def get_data(self) -> pd.DataFrame:
        """OR 시간 정보 반환"""
        data_manager = get_data_manager()
        
        if not data_manager.is_ready():
            print("⚠️ DataManager가 준비되지 않았습니다")
            return {}
        
        utc_now = datetime.now(timezone.utc)
        prev_day = utc_now - timedelta(days=1)
        
        start_time = prev_day.replace(hour=0, minute=0, second=0, microsecond=0)
        end_time = prev_day.replace(hour=23, minute=59, second=59, microsecond=999999)

        start_utc = self.time_manager.ensure_utc(start_time)
        end_utc = self.time_manager.ensure_utc(end_time)

        df = data_manager.get_data_range(start_utc, end_utc)

        return df.copy()
    
    def get_status(self) -> Dict[str, float]:
        """어제 고가/저가 반환"""
        return {
            'prev_day_high':self.prev_day_high,
            'prev_day_low':self.prev_day_low
            }
