#!/usr/bin/env python3
"""
오프닝 레인지 (Opening Range) 지표
- 유럽 오픈: 07:00 UTC
- 미국 오픈: 13:30 UTC
- OR 30분 완성 후 신호 허용
"""

import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Tuple


def get_session_open_time(current_time: datetime) -> Tuple[datetime, str]:
    """
    현재 시간 기준으로 가장 가까운 세션 오픈 시간 계산
    
    Args:
        current_time: 현재 시간 (UTC)
    
    Returns:
        Tuple[datetime, str]: (세션 오픈 시간, 세션 이름)
    """
    try:
        current_utc = current_time.replace(tzinfo=timezone.utc)
        
        # 오늘 날짜의 세션 시간들
        today = current_utc.date()
        
        europe_open = datetime.combine(today, datetime.min.time().replace(hour=7, minute=0), tzinfo=timezone.utc)
        us_open = datetime.combine(today, datetime.min.time().replace(hour=13, minute=30), tzinfo=timezone.utc)
        
        # 어제 날짜의 세션 시간들 (자정을 넘긴 경우)
        yesterday = today - timedelta(days=1)
        yesterday_europe_open = datetime.combine(yesterday, datetime.min.time().replace(hour=7, minute=0), tzinfo=timezone.utc)
        yesterday_us_open = datetime.combine(yesterday, datetime.min.time().replace(hour=13, minute=30), tzinfo=timezone.utc)
        
        # 가능한 세션 시간들
        session_times = [
            (yesterday_europe_open, "EUROPE"),
            (yesterday_us_open, "US"),
            (europe_open, "EUROPE"),
            (us_open, "US")
        ]
        
        # 현재 시간보다 이전이면서 가장 가까운 세션 찾기
        valid_sessions = [(time, name) for time, name in session_times if time <= current_utc]
        
        if not valid_sessions:
            # 모든 세션이 미래인 경우 (새벽 시간대)
            return (europe_open, "EUROPE")
        
        # 가장 가까운 세션 반환
        closest_session = max(valid_sessions, key=lambda x: x[0])
        return closest_session
        
    except Exception as e:
        print(f"❌ 세션 오픈 시간 계산 오류: {e}")
        # 기본값: 유럽 오픈
        today = current_time.date()
        default_open = datetime.combine(today, datetime.min.time().replace(hour=7, minute=0), tzinfo=timezone.utc)
        return (default_open, "EUROPE")


def is_session_active(current_time: datetime) -> bool:
    """
    현재 활성 세션이 있는지 확인
    
    Args:
        current_time: 현재 시간 (UTC)
    
    Returns:
        bool: 활성 세션 존재 여부
    """
    try:
        current_utc = current_time.replace(tzinfo=timezone.utc)
        
        # 오늘 날짜의 세션 시간들
        today = current_utc.date()
        
        europe_open = datetime.combine(today, datetime.min.time().replace(hour=7, minute=0), tzinfo=timezone.utc)
        europe_close = datetime.combine(today, datetime.min.time().replace(hour=13, minute=30), tzinfo=timezone.utc)
        us_open = datetime.combine(today, datetime.min.time().replace(hour=13, minute=30), tzinfo=timezone.utc)
        us_close = datetime.combine(today + timedelta(days=1), datetime.min.time().replace(hour=7, minute=0), tzinfo=timezone.utc)
        
        # 유럽 세션: 07:00-13:30 UTC
        is_europe_active = europe_open <= current_utc < europe_close
        
        # 미국 세션: 13:30-07:00 UTC (다음날)
        is_us_active = us_open <= current_utc < us_close
        
        return is_europe_active or is_us_active
        
    except Exception as e:
        print(f"❌ 세션 활성 상태 확인 오류: {e}")
        return False


def get_current_session_info(current_time: datetime) -> Dict[str, any]:
    """
    현재 세션 정보 반환
    
    Args:
        current_time: 현재 시간 (UTC)
    
    Returns:
        Dict: 현재 세션 정보
    """
    try:
        current_utc = current_time.replace(tzinfo=timezone.utc)
        
        if not is_session_active(current_utc):
            return {
                'is_active': False,
                'current_session': None,
                'session_open_time': None,
                'session_close_time': None,
                'elapsed_minutes': 0,
                'remaining_minutes': 0,
                'status': 'NO_SESSION'
            }
        
        # 오늘 날짜의 세션 시간들
        today = current_utc.date()
        
        europe_open = datetime.combine(today, datetime.min.time().replace(hour=7, minute=0), tzinfo=timezone.utc)
        europe_close = datetime.combine(today, datetime.min.time().replace(hour=13, minute=30), tzinfo=timezone.utc)
        us_open = datetime.combine(today, datetime.min.time().replace(hour=13, minute=30), tzinfo=timezone.utc)
        us_close = datetime.combine(today + timedelta(days=1), datetime.min.time().replace(hour=7, minute=0), tzinfo=timezone.utc)
        
        # 유럽 세션 활성
        if europe_open <= current_utc < europe_close:
            elapsed_minutes = (current_utc - europe_open).total_seconds() / 60
            remaining_minutes = (europe_close - current_utc).total_seconds() / 60
            
            return {
                'is_active': True,
                'current_session': 'EUROPE',
                'session_open_time': europe_open,
                'session_close_time': europe_close,
                'elapsed_minutes': elapsed_minutes,
                'remaining_minutes': remaining_minutes,
                'status': 'EUROPE_ACTIVE'
            }
        
        # 미국 세션 활성
        elif us_open <= current_utc < us_close:
            elapsed_minutes = (current_utc - us_open).total_seconds() / 60
            remaining_minutes = (us_close - current_utc).total_seconds() / 60
            
            return {
                'is_active': True,
                'current_session': 'US',
                'session_open_time': us_open,
                'session_close_time': us_close,
                'elapsed_minutes': elapsed_minutes,
                'remaining_minutes': remaining_minutes,
                'status': 'US_ACTIVE'
            }
        
        return {
            'is_active': False,
            'current_session': None,
            'session_open_time': None,
            'session_close_time': None,
            'elapsed_minutes': 0,
            'remaining_minutes': 0,
            'status': 'UNKNOWN'
        }
        
    except Exception as e:
        print(f"❌ 현재 세션 정보 확인 오류: {e}")
        return {
            'is_active': False,
            'current_session': None,
            'session_open_time': None,
            'session_close_time': None,
            'elapsed_minutes': 0,
            'remaining_minutes': 0,
            'status': 'ERROR'
        }


def is_or_completed(current_time: datetime, session_open_time: datetime, or_minutes: int = 30) -> bool:
    """
    오프닝 레인지가 완성되었는지 확인
    
    Args:
        current_time: 현재 시간 (UTC)
        session_open_time: 세션 오픈 시간 (UTC)
        or_minutes: OR 완성에 필요한 분 (기본: 30분)
    
    Returns:
        bool: OR 완성 여부
    """
    try:
        current_utc = current_time.replace(tzinfo=timezone.utc)
        session_utc = session_open_time.replace(tzinfo=timezone.utc)
        
        # 세션 오픈 후 경과 시간
        elapsed_minutes = (current_utc - session_utc).total_seconds() / 60
        
        return elapsed_minutes >= or_minutes
        
    except Exception as e:
        print(f"❌ OR 완성 확인 오류: {e}")
        return False


def calculate_opening_range(df: pd.DataFrame, 
                          session_open_time: datetime,
                          or_minutes: int = 30) -> Dict[str, any]:
    """
    오프닝 레인지 계산
    
    Args:
        df: 3분봉 OHLCV 데이터 (timestamp, open, high, low, close, volume)
        session_open_time: 세션 오픈 시간 (UTC)
        or_minutes: OR 계산 기간 (기본: 30분)
    
    Returns:
        Dict: 오프닝 레인지 정보
    """
    try:
        if df.empty:
            return {}
        
        session_utc = session_open_time.replace(tzinfo=timezone.utc)
        or_end_time = session_utc + timedelta(minutes=or_minutes)
        
        # OR 기간의 데이터 필터링
        or_start_timestamp = int(session_utc.timestamp() * 1000)
        or_end_timestamp = int(or_end_time.timestamp() * 1000)
        
        or_data = df[
            (df['timestamp'] >= or_start_timestamp) & 
            (df['timestamp'] <= or_end_timestamp)
        ]
        
        if or_data.empty:
            return {}
        
        # OR 고/저 계산
        or_high = float(or_data['high'].max())
        or_low = float(or_data['low'].min())
        or_open = float(or_data['open'].iloc[0])
        or_close = float(or_data['close'].iloc[-1])
        
        # OR 범위
        or_range = or_high - or_low
        or_mid = (or_high + or_low) / 2
        
        # OR 확장 레벨
        or_extension_high = or_high + or_range
        or_extension_low = or_low - or_range
        
        result = {
            'session_open_time': session_utc.isoformat(),
            'session_name': "EUROPE" if session_utc.hour == 7 else "US",
            'or_start': session_utc.isoformat(),
            'or_end': or_end_time.isoformat(),
            'or_minutes': or_minutes,
            'or_high': or_high,
            'or_low': or_low,
            'or_open': or_open,
            'or_close': or_close,
            'or_range': or_range,
            'or_mid': or_mid,
            'or_extension_high': or_extension_high,
            'or_extension_low': or_extension_low,
            'candle_count': len(or_data),
            'is_completed': True  # OR 기간이 끝난 데이터로 계산했으므로 항상 True
        }
        
        return result
        
    except Exception as e:
        print(f"❌ 오프닝 레인지 계산 오류: {e}")
        return {}


def get_current_or_status(df: pd.DataFrame, 
                         current_time: datetime,
                         or_minutes: int = 30) -> Dict[str, any]:
    """
    현재 OR 상태 확인
    
    Args:
        df: 3분봉 OHLCV 데이터
        current_time: 현재 시간 (UTC)
        or_minutes: OR 완성에 필요한 분
    
    Returns:
        Dict: 현재 OR 상태
    """
    try:
        # 세션 오픈 시간 계산
        session_open_time, session_name = get_session_open_time(current_time)
        
        # OR 완성 여부 확인
        is_completed = is_or_completed(current_time, session_open_time, or_minutes)
        
        if not is_completed:
            # OR 미완성 상태
            elapsed_minutes = (current_time - session_open_time).total_seconds() / 60
            remaining_minutes = max(0, or_minutes - elapsed_minutes)
            
            return {
                'session_open_time': session_open_time.isoformat(),
                'session_name': session_name,
                'or_minutes': or_minutes,
                'elapsed_minutes': elapsed_minutes,
                'remaining_minutes': remaining_minutes,
                'is_completed': False,
                'can_trade': False
            }
        
        # OR 완성 상태 - 전체 OR 계산
        or_data = calculate_opening_range(df, session_open_time, or_minutes)
        or_data['can_trade'] = True
        
        return or_data
        
    except Exception as e:
        print(f"❌ 현재 OR 상태 확인 오류: {e}")
        return {}


def get_or_breakout_levels(opening_range: Dict[str, any], 
                          current_price: float,
                          tolerance_pct: float = 0.05) -> Dict[str, any]:
    """
    OR 돌파 레벨 계산
    
    Args:
        opening_range: 오프닝 레인지 데이터
        current_price: 현재 가격
        tolerance_pct: 돌파 확인용 허용 오차 (기본: 0.05%)
    
    Returns:
        Dict: 돌파 레벨 정보
    """
    try:
        if not opening_range or not opening_range.get('is_completed'):
            return {}
        
        or_high = opening_range.get('or_high', 0)
        or_low = opening_range.get('or_low', 0)
        tolerance = current_price * tolerance_pct
        
        # 돌파 상태 확인
        is_above_or = current_price > or_high + tolerance
        is_below_or = current_price < or_low - tolerance
        
        # 돌파 거리
        distance_above = current_price - or_high if is_above_or else 0
        distance_below = or_low - current_price if is_below_or else 0
        
        # 돌파 강도 (OR 범위 대비)
        or_range = opening_range.get('or_range', 1)
        strength_above = distance_above / or_range if or_range > 0 else 0
        strength_below = distance_below / or_range if or_range > 0 else 0
        
        result = {
            'or_high': or_high,
            'or_low': or_low,
            'current_price': current_price,
            'is_above_or': is_above_or,
            'is_below_or': is_below_or,
            'distance_above': distance_above,
            'distance_below': distance_below,
            'strength_above': strength_above,
            'strength_below': strength_below,
            'breakout_direction': 'ABOVE' if is_above_or else 'BELOW' if is_below_or else 'INSIDE'
        }
        
        return result
        
    except Exception as e:
        print(f"❌ OR 돌파 레벨 계산 오류: {e}")
        return {}
