#!/usr/bin/env python3
"""
일일 가격 레벨 지표
- 전일 고가/저가/종가/시가
- 당일 시가
- 최근 스윙 고/저 (피벗)
"""

import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Tuple


def get_daily_levels(df: pd.DataFrame, current_time: datetime) -> Dict[str, float]:
    """
    일일 가격 레벨 계산
    
    Args:
        df: 3분봉 OHLCV 데이터 (timestamp, open, high, low, close, volume)
        current_time: 현재 시간 (UTC)
    
    Returns:
        Dict[str, float]: 일일 레벨 정보
    """
    try:
        if df.empty:
            return {}
        
        # UTC 기준으로 전일 00:00 계산
        current_utc = current_time.replace(tzinfo=timezone.utc)
        yesterday_start = current_utc.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
        yesterday_end = yesterday_start + timedelta(days=1)
        
        # 전일 데이터 필터링
        yesterday_data = df[
            (df['timestamp'] >= int(yesterday_start.timestamp() * 1000)) &
            (df['timestamp'] < int(yesterday_end.timestamp() * 1000))
        ]
        
        if yesterday_data.empty:
            return {}
        
        # 전일 레벨 계산
        daily_levels = {
            'prev_day_high': float(yesterday_data['high'].max()),
            'prev_day_low': float(yesterday_data['low'].min())
        }
        
        return daily_levels
        
    except Exception as e:
        print(f"❌ 일일 레벨 계산 오류: {e}")
        return {}


def get_swing_levels(df: pd.DataFrame, lookback: int = 20) -> Dict[str, float]:
    """
    최근 스윙 고/저 (피벗) 계산
    
    Args:
        df: 3분봉 OHLCV 데이터
        lookback: 피벗 계산용 룩백 기간 (기본: 20봉)
    
    Returns:
        Dict[str, float]: 스윙 레벨 정보
    """
    try:
        if len(df) < lookback:
            return {}
        
        recent_data = df.tail(lookback)
        
        # 고점/저점 찾기 (3봉 연속 패턴)
        highs = []
        lows = []
        
        for i in range(1, len(recent_data) - 1):
            # 고점: 양쪽보다 높은 봉
            if (recent_data['high'].iloc[i] > recent_data['high'].iloc[i-1] and 
                recent_data['high'].iloc[i] > recent_data['high'].iloc[i+1]):
                highs.append(recent_data['high'].iloc[i])
            
            # 저점: 양쪽보다 낮은 봉
            if (recent_data['low'].iloc[i] < recent_data['low'].iloc[i-1] and 
                recent_data['low'].iloc[i] < recent_data['low'].iloc[i+1]):
                lows.append(recent_data['low'].iloc[i])
        
        swing_levels = {}
        
        if highs:
            swing_levels['recent_swing_high'] = float(max(highs[-3:]))  # 최근 3개 고점 중 최고
        if lows:
            swing_levels['recent_swing_low'] = float(min(lows[-3:]))   # 최근 3개 저점 중 최저
        
        return swing_levels
        
    except Exception as e:
        print(f"❌ 스윙 레벨 계산 오류: {e}")
        return {}


def calculate_all_daily_levels(df: pd.DataFrame, current_time: datetime) -> Dict[str, float]:
    """
    모든 일일 레벨을 한번에 계산
    
    Args:
        df: 3분봉 OHLCV 데이터
        current_time: 현재 시간 (UTC)
    
    Returns:
        Dict[str, float]: 모든 일일 레벨 정보
    """
    daily_levels = get_daily_levels(df, current_time)
    swing_levels = get_swing_levels(df)
    
    # 통합
    all_levels = {**daily_levels, **swing_levels}
    
    return all_levels
