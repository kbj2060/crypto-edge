#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ATR (Average True Range) 지표
- 3분봉 실시간 ATR 계산
- Wilder's smoothing 사용
- 세션과 관계없이 연속 롤링 계산
"""

from typing import Dict, Optional
import datetime as dt
from collections import deque

import pandas as pd
from datetime import datetime, timedelta
from data.data_manager import get_data_manager
from utils.time_manager import get_time_manager

class ATR3M:
    """3분봉 실시간 ATR 관리 클래스 - 연속 롤링 방식"""
    
    def __init__(self, length: int = 14, max_candles: int = 100, init_data: Optional[pd.DataFrame] = None, target_time: Optional[datetime] = None):
        self.length = length
        self.max_candles = max_candles
        
        # 캔들 데이터 저장 (롤링 윈도우)
        self.candles = []
        self.true_ranges = []  # deque 대신 list 사용
        
        # ATR 값
        self.current_atr = 0.0
        self.last_update_time = None

        self.time_manager = get_time_manager()

        self._initialize_atr(init_data, target_time)

    
    def _initialize_atr(self, target_time: Optional[datetime] = None):
        df = self.get_data(target_time)

        self.current_atr = self.calculate_atr_from_dataframe(df)

    def get_data(self, target_time: Optional[datetime] = None) -> pd.DataFrame:
        """OR 시간 정보 반환"""
        data_manager = get_data_manager()
        target_time = target_time if target_time is not None else self.time_manager.get_current_time()
        df = data_manager.get_data_range(target_time - timedelta(minutes=self.max_candles * 3), target_time)
        return df.copy()
    
    def update_with_candle(self, candle_data: pd.Series):
        """새로운 3분봉으로 ATR 업데이트 - 연속 롤링"""

        if hasattr(candle_data, 'name') and candle_data.name is not None:
            timestamp = candle_data.name
        elif hasattr(candle_data, 'index') and len(candle_data.index) > 0:
            timestamp = candle_data.index[0]
        else:
            # 기본값으로 현재 시간 사용
            timestamp = dt.datetime.now(dt.timezone.utc)

        # 캔들 데이터 저장
        candle_df = pd.DataFrame([{
            'high': float(candle_data['high'].item()),
            'low': float(candle_data['low'].item()),
            'close': float(candle_data['close'].item())
        }], index=[timestamp])

        self.candles.append(candle_df)

        # 최대 캔들 개수 제한
        if len(self.candles) > self.max_candles:
            self.candles = self.candles[-self.max_candles:]
        
        # True Range 계산 (최소 2개 캔들 필요)
        if len(self.candles) >= 2:
            current = self.candles[-1].iloc[0]  # DataFrame에서 첫 번째 행 추출
            previous = self.candles[-2].iloc[0]  # DataFrame에서 첫 번째 행 추출
            
            # True Range 계산
            tr1 = current['high'] - current['low']
            tr2 = abs(current['high'] - previous['close'])
            tr3 = abs(current['low'] - previous['close'])
            
            true_range = max(tr1, tr2, tr3)
            self.true_ranges.append(true_range)
            
            # 최대 true_ranges 개수 제한
            if len(self.true_ranges) > self.length:
                self.true_ranges = self.true_ranges[-self.length:]
            
            # ATR 계산
            self._calculate_atr()
            self.last_update_time = dt.datetime.now(dt.timezone.utc)
    
    def _calculate_atr(self):
        """Wilder's smoothing으로 ATR 계산"""
        if not self.true_ranges:
            return
        
        if len(self.true_ranges) == 1:
            self.current_atr = self.true_ranges[0]
        else:
            prev_atr = self.current_atr
            current_tr = self.true_ranges[-1]
            self.current_atr = ((self.length - 1) * prev_atr + current_tr) / self.length
    
    def calculate_atr_from_dataframe(self, df: pd.DataFrame) -> float:
        """
        100개의 최근 데이터프레임에서 ATR 계산
        
        Args:
            df: OHLCV 데이터프레임 (high, low, close 컬럼 필요)
            period: ATR 기간 (기본값: 14)
            
        Returns:
            float: 계산된 ATR 값
        """
        if df.empty or len(df) < self.length:
            return 0.0
        
        try:
            # 최근 200개 데이터만 사용
            recent_df = df.copy()
            
            # True Range 계산
            high_low = recent_df['high'] - recent_df['low']
            high_close_prev = abs(recent_df['high'] - recent_df['close'].shift(1))
            low_close_prev = abs(recent_df['low'] - recent_df['close'].shift(1))
            
            # True Range는 세 값 중 최대값
            true_ranges_list = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
            
            # 첫 번째 값은 NaN이므로 제거
            true_ranges_list = true_ranges_list.dropna()
            
            if len(true_ranges_list) < self.length:
                return 0.0
            
            # list 초기화
            self.true_ranges.clear()
            
            # Wilder's smoothing으로 ATR 계산
            self.current_atr = true_ranges_list.iloc[:self.length].mean()  # 초기값은 단순 평균
            
            # list에 값들 추가
            for i in range(self.length):
                self.true_ranges.append(true_ranges_list.iloc[i])
            
            # 나머지 값들로 smoothing
            for i in range(self.length, len(true_ranges_list)):
                self.current_atr = ((self.length - 1) * self.current_atr + true_ranges_list.iloc[i]) / self.length
                self.true_ranges.append(true_ranges_list.iloc[i])
            
            return float(self.current_atr)
            
        except Exception as e:
            return 0.0

    def is_ready(self) -> bool:
        """ATR 계산 준비 여부"""
        return len(self.true_ranges) >= 1
    
    def is_mature(self) -> bool:
        """ATR이 충분한 데이터로 성숙했는지 여부"""
        return len(self.true_ranges) >= self.length
    
    def get_candles_count(self) -> int:
        """현재 저장된 캔들 개수 반환"""
        return len(self.candles)
    
    def get_status(self) -> Dict[str, any]:
        """ATR 상태 정보 반환"""
        return {
            'atr': self.current_atr,
            'is_ready': self.is_ready(),
            'is_mature': self.is_mature(),
            'candles_count': self.get_candles_count(),
            'true_ranges_count': len(self.true_ranges),
            'length': self.length,
            'max_candles': self.max_candles,
            'last_update': self.last_update_time.isoformat() if self.last_update_time else None
        }
    
    def reset(self):
        """데이터 초기화"""
        self.candles.clear()
        self.true_ranges.clear()
        self.current_atr = 0.0
        self.last_update_time = None
        print("🔄 ATR3M 데이터 초기화 완료")
