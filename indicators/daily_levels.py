#!/usr/bin/env python3
"""
어제 3분봉 데이터의 high, low만 가져오는 간단한 클래스
Note: 어제 데이터는 공용 데이터와 별개이므로 개별 API 호출 유지
"""

from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import pandas as pd

from managers.data_manager import get_data_manager
from utils.time_manager import get_time_manager


class DailyLevels:
    """어제 3분봉 데이터의 high, low만 관리하는 간단한 클래스"""
    
    def __init__(self, target_time: Optional[datetime] = None):
        self.time_manager = get_time_manager()
        self.prev_day_high = 0.0
        self.prev_day_low = 0.0
        self.last_update_date = None  # 마지막 업데이트 날짜 저장
        self.target_time = target_time if target_time is not None else self.time_manager.get_current_time()

        # 자동으로 데이터 로드
        self._initialize_levels()
    
    def _is_new_day(self, data_now: datetime = None) -> bool:
        """하루가 바뀌었는지 확인"""
        current_date = data_now.date()
        return self.last_update_date != current_date
    
    def _initialize_levels(self):
        # high, low만 계산
        df = self.get_data()
        self.last_update_date = self.target_time.date()

        if not df.empty:
            self.prev_day_high = float(df['high'].max())
            self.prev_day_low = float(df['low'].min())
        else:
            self.prev_day_high = 0.0
            self.prev_day_low = 0.0
        
    
    def update_with_candle(self, candle_data: pd.Series):
        """새로운 캔들로 업데이트 (하루가 바뀌면 데이터 갱신)"""
        try:
            # 하루가 바뀌었는지 확인
            self.target_time = self.time_manager.ensure_utc(candle_data.name)
            
            if self._is_new_day(candle_data.name):
                print("🔄 새로운 날이 되었습니다. Daily Levels 데이터를 갱신합니다.")
                data_now = candle_data.name
                df = self.get_data()
                self.prev_day_high = float(df['high'].max())
                self.prev_day_low = float(df['low'].min())
                self.last_update_date = data_now.date()

        except Exception as e:
            print(f"❌ Daily Levels 업데이트 오류: {e}")
    
    def get_data(self) -> pd.DataFrame:
        """OR 시간 정보 반환"""
        data_manager = get_data_manager()
        
        if not data_manager.is_ready():
            print("⚠️ DataManager가 준비되지 않았습니다")
            return {}
        
        prev_day = self.target_time - timedelta(days=1)
        
        start_time = prev_day.replace(hour=0, minute=0, second=0, microsecond=0)
        end_time = prev_day.replace(hour=23, minute=59, second=59, microsecond=999999)

        start_utc = self.time_manager.ensure_utc(start_time)
        end_utc = self.time_manager.ensure_utc(end_time)
        df = data_manager.get_data_range(start_utc, end_utc)

        if self.target_time is not None:
            mask = (df.index >= start_time) & (df.index <= end_time)
            df_mask = df[mask].copy()
        else:
            df_mask = df.copy()

        return df_mask
    
    def get_status(self) -> Dict[str, Any]:
        """어제 고가/저가 및 업데이트 정보 반환"""
        return {
            'prev_day_high': self.prev_day_high,
            'prev_day_low': self.prev_day_low,
            'last_update_date': self.last_update_date.isoformat() if self.last_update_date else None,
        }
