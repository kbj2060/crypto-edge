#!/usr/bin/env python3
"""
오프닝 레인지 (Opening Range) 지표
- 유럽 오픈: 07:00 UTC
- 미국 오픈: 13:30 UTC
- OR 30분 완성 후 신호 허용
"""

import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass

# =============================================================================
# 상수 정의
# =============================================================================

# 세션 시간 상수
EUROPE_OPEN_HOUR = 7
EUROPE_OPEN_MINUTE = 0
US_OPEN_HOUR = 13
US_OPEN_MINUTE = 30

# 세션 이름 상수
SESSION_EUROPE = "EUROPE"
SESSION_US = "US"

# OR 기본 설정
DEFAULT_OR_MINUTES = 30

# 상태 상수
STATUS_NO_SESSION = "NO_SESSION"
STATUS_EUROPE_ACTIVE = "EUROPE_ACTIVE"
STATUS_US_ACTIVE = "US_ACTIVE"
STATUS_UNKNOWN = "UNKNOWN"
STATUS_ERROR = "ERROR"

# 돌파 허용 오차 기본값
DEFAULT_BREAKOUT_TOLERANCE_PCT = 0.05

# =============================================================================
# 타입 정의
# =============================================================================

@dataclass
class SessionInfo:
    """세션 정보 데이터 클래스"""
    is_active: bool
    current_session: Optional[str]
    session_open_time: Optional[datetime]
    session_close_time: Optional[datetime]
    elapsed_minutes: float
    remaining_minutes: float
    status: str

@dataclass
class OpeningRangeData:
    """오프닝 레인지 데이터 클래스"""
    session_open_time: str
    session_name: str
    or_start: str
    or_end: str
    or_minutes: int
    or_high: float
    or_low: float
    or_open: float
    or_close: float
    or_range: float
    or_mid: float
    or_extension_high: float
    or_extension_low: float
    candle_count: int
    is_completed: bool

@dataclass
class BreakoutLevels:
    """OR 돌파 레벨 데이터 클래스"""
    or_high: float
    or_low: float
    current_price: float
    is_above_or: bool
    is_below_or: bool
    distance_above: float
    distance_below: float
    strength_above: float
    strength_below: float
    breakout_direction: str

# =============================================================================
# 유틸리티 함수
# =============================================================================

def _create_session_time(date: datetime.date, hour: int, minute: int) -> datetime:
    """세션 시간 생성 헬퍼 함수"""
    return datetime.combine(date, datetime.min.time().replace(hour=hour, minute=minute), tzinfo=timezone.utc)

def _calculate_elapsed_minutes(current_time: datetime, session_time: datetime) -> float:
    """경과 시간 계산 (분 단위)"""
    return (current_time - session_time).total_seconds() / 60

def _calculate_remaining_minutes(current_time: datetime, session_time: datetime, total_minutes: int) -> float:
    """남은 시간 계산 (분 단위)"""
    elapsed = _calculate_elapsed_minutes(current_time, session_time)
    return max(0, total_minutes - elapsed)

# =============================================================================
# 세션 관리 함수
# =============================================================================

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
        today = current_utc.date()
        yesterday = today - timedelta(days=1)
        
        # 가능한 세션 시간들 생성
        session_times = [
            (_create_session_time(yesterday, EUROPE_OPEN_HOUR, EUROPE_OPEN_MINUTE), SESSION_EUROPE),
            (_create_session_time(yesterday, US_OPEN_HOUR, US_OPEN_MINUTE), SESSION_US),
            (_create_session_time(today, EUROPE_OPEN_HOUR, EUROPE_OPEN_MINUTE), SESSION_EUROPE),
            (_create_session_time(today, US_OPEN_HOUR, US_OPEN_MINUTE), SESSION_US)
        ]
        
        # 현재 시간보다 이전이면서 가장 가까운 세션 찾기
        valid_sessions = [(time, name) for time, name in session_times if time <= current_utc]
        
        if not valid_sessions:
            # 모든 세션이 미래인 경우 (새벽 시간대)
            return (_create_session_time(today, EUROPE_OPEN_HOUR, EUROPE_OPEN_MINUTE), SESSION_EUROPE)
        
        # 가장 가까운 세션 반환
        closest_session = max(valid_sessions, key=lambda x: x[0])
        return closest_session
        
    except Exception as e:
        print(f"❌ 세션 오픈 시간 계산 오류: {e}")
        # 기본값: 유럽 오픈
        today = current_time.date()
        default_open = _create_session_time(today, EUROPE_OPEN_HOUR, EUROPE_OPEN_MINUTE)
        return (default_open, SESSION_EUROPE)


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
        today = current_utc.date()
        
        # 세션 시간 계산
        europe_open = _create_session_time(today, EUROPE_OPEN_HOUR, EUROPE_OPEN_MINUTE)
        europe_close = _create_session_time(today, US_OPEN_HOUR, US_OPEN_MINUTE)
        us_open = _create_session_time(today, US_OPEN_HOUR, US_OPEN_MINUTE)
        us_close = _create_session_time(today + timedelta(days=1), EUROPE_OPEN_HOUR, EUROPE_OPEN_MINUTE)
        
        # 세션 활성 상태 확인
        is_europe_active = europe_open <= current_utc < europe_close
        is_us_active = us_open <= current_utc < us_close
        
        return is_europe_active or is_us_active
        
    except Exception as e:
        print(f"❌ 세션 활성 상태 확인 오류: {e}")
        return False


def get_current_session_info(current_time: datetime) -> SessionInfo:
    """
    현재 세션 정보 반환
    
    Args:
        current_time: 현재 시간 (UTC)
    
    Returns:
        SessionInfo: 현재 세션 정보
    """
    try:
        current_utc = current_time.replace(tzinfo=timezone.utc)
        
        if not is_session_active(current_utc):
            return SessionInfo(
                is_active=False,
                current_session=None,
                session_open_time=None,
                session_close_time=None,
                elapsed_minutes=0.0,
                remaining_minutes=0.0,
                status=STATUS_NO_SESSION
            )
        
        today = current_utc.date()
        
        # 세션 시간 계산
        europe_open = _create_session_time(today, EUROPE_OPEN_HOUR, EUROPE_OPEN_MINUTE)
        europe_close = _create_session_time(today, US_OPEN_HOUR, US_OPEN_MINUTE)
        us_open = _create_session_time(today, US_OPEN_HOUR, US_OPEN_MINUTE)
        us_close = _create_session_time(today + timedelta(days=1), EUROPE_OPEN_HOUR, EUROPE_OPEN_MINUTE)
        
        # 유럽 세션 활성
        if europe_open <= current_utc < europe_close:
            elapsed_minutes = _calculate_elapsed_minutes(current_utc, europe_open)
            remaining_minutes = _calculate_remaining_minutes(current_utc, europe_open, 
                                                          (europe_close - europe_open).total_seconds() / 60)
            
            return SessionInfo(
                is_active=True,
                current_session=SESSION_EUROPE,
                session_open_time=europe_open,
                session_close_time=europe_close,
                elapsed_minutes=elapsed_minutes,
                remaining_minutes=remaining_minutes,
                status=STATUS_EUROPE_ACTIVE
            )
        
        # 미국 세션 활성
        elif us_open <= current_utc < us_close:
            elapsed_minutes = _calculate_elapsed_minutes(current_utc, us_open)
            remaining_minutes = _calculate_remaining_minutes(current_utc, us_open,
                                                          (us_close - us_open).total_seconds() / 60)
            
            return SessionInfo(
                is_active=True,
                current_session=SESSION_US,
                session_open_time=us_open,
                session_close_time=us_close,
                elapsed_minutes=elapsed_minutes,
                remaining_minutes=remaining_minutes,
                status=STATUS_US_ACTIVE
            )
        
        return SessionInfo(
            is_active=False,
            current_session=None,
            session_open_time=None,
            session_close_time=None,
            elapsed_minutes=0.0,
            remaining_minutes=0.0,
            status=STATUS_UNKNOWN
        )
        
    except Exception as e:
        print(f"❌ 현재 세션 정보 확인 오류: {e}")
        return SessionInfo(
            is_active=False,
            current_session=None,
            session_open_time=None,
            session_close_time=None,
            elapsed_minutes=0.0,
            remaining_minutes=0.0,
            status=STATUS_ERROR
        )

# =============================================================================
# 오프닝 레인지 계산 함수
# =============================================================================

def is_or_completed(current_time: datetime, session_open_time: datetime, or_minutes: int = DEFAULT_OR_MINUTES) -> bool:
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
        
        elapsed_minutes = _calculate_elapsed_minutes(current_utc, session_utc)
        return elapsed_minutes >= or_minutes
        
    except Exception as e:
        print(f"❌ OR 완성 확인 오류: {e}")
        return False


def calculate_opening_range(df: pd.DataFrame, 
                          session_open_time: datetime,
                          or_minutes: int = DEFAULT_OR_MINUTES) -> Dict[str, Any]:
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
        
        # OR 범위 및 중간값
        or_range = or_high - or_low
        or_mid = (or_high + or_low) / 2
        
        # OR 확장 레벨
        or_extension_high = or_high + or_range
        or_extension_low = or_low - or_range
        
        # 세션 이름 결정
        session_name = SESSION_EUROPE if session_utc.hour == EUROPE_OPEN_HOUR else SESSION_US
        
        result = {
            'session_open_time': session_utc.isoformat(),
            'session_name': session_name,
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
                         or_minutes: int = DEFAULT_OR_MINUTES) -> Dict[str, Any]:
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
            elapsed_minutes = _calculate_elapsed_minutes(current_time, session_open_time)
            remaining_minutes = _calculate_remaining_minutes(current_time, session_open_time, or_minutes)
            
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


def get_or_breakout_levels(opening_range: Dict[str, Any], 
                          current_price: float,
                          tolerance_pct: float = DEFAULT_BREAKOUT_TOLERANCE_PCT) -> Dict[str, Any]:
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
        
        # 돌파 방향 결정
        if is_above_or:
            breakout_direction = 'ABOVE'
        elif is_below_or:
            breakout_direction = 'BELOW'
        else:
            breakout_direction = 'INSIDE'
        
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
            'breakout_direction': breakout_direction
        }
        
        return result
        
    except Exception as e:
        print(f"❌ OR 돌파 레벨 계산 오류: {e}")
        return {}

# =============================================================================
# 세션 관리 클래스
# =============================================================================

class SessionManager:
    """
    세션 관리 클래스 - indicator들을 위한 중앙화된 세션 상태 관리
    """
    
    def __init__(self):
        self.current_session_info: Optional[SessionInfo] = None
        self.last_update_time: Optional[datetime] = None
        self.session_history: Dict[str, Dict[str, Any]] = {}
    
    def update_session_status(self, current_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        현재 시간 기준으로 세션 상태 업데이트
        
        Args:
            current_time: 현재 시간 (None이면 datetime.now() 사용)
        
        Returns:
            Dict: 세션 상태 정보
        """
        try:
            if current_time is None:
                current_time = datetime.now(timezone.utc)
            
            # 세션 정보 업데이트
            self.current_session_info = get_current_session_info(current_time)
            self.last_update_time = current_time
            
            # 세션 전환 이력 저장
            session_id = f"{current_time.strftime('%Y%m%d')}_{self.current_session_info.current_session or 'NONE'}"
            
            if session_id not in self.session_history:
                self.session_history[session_id] = {
                    'session_name': self.current_session_info.current_session,
                    'start_time': self.current_session_info.session_open_time,
                    'end_time': self.current_session_info.session_close_time,
                    'first_seen': current_time,
                    'status': self.current_session_info.status
                }
            
            return self.current_session_info.__dict__
            
        except Exception as e:
            print(f"❌ 세션 상태 업데이트 오류: {e}")
            return {
                'is_active': False,
                'current_session': None,
                'status': STATUS_ERROR
            }
    
    def get_session_status(self) -> Dict[str, Any]:
        """
        현재 세션 상태 반환 (캐시된 정보)
        
        Returns:
            Dict: 세션 상태 정보
        """
        if self.current_session_info is None:
            return self.update_session_status()
        
        return self.current_session_info.__dict__
    
    def is_session_active(self) -> bool:
        """
        현재 세션이 활성 상태인지 확인
        
        Returns:
            bool: 세션 활성 여부
        """
        session_info = self.get_session_status()
        return session_info.get('is_active', False)
    
    def get_current_session_name(self) -> Optional[str]:
        """
        현재 세션 이름 반환
        
        Returns:
            str: 세션 이름 ('EUROPE', 'US') 또는 None
        """
        session_info = self.get_session_status()
        return session_info.get('current_session')
    
    def get_session_open_time(self) -> Optional[datetime]:
        """
        현재 세션 시작 시간 반환
        
        Returns:
            datetime: 세션 시작 시간 또는 None
        """
        session_info = self.get_session_status()
        session_open = session_info.get('session_open_time')
        
        if session_open:
            if isinstance(session_open, str):
                return datetime.fromisoformat(session_open.replace('Z', '+00:00'))
            return session_open
        
        return None
    
    def should_use_session_mode(self) -> bool:
        """
        indicator가 세션 모드를 사용해야 하는지 판단
        
        Returns:
            bool: 세션 모드 사용 여부
        """
        return self.is_session_active()
    
    def get_session_elapsed_minutes(self) -> float:
        """
        현재 세션 경과 시간 (분)
        
        Returns:
            float: 경과 시간 (분)
        """
        session_info = self.get_session_status()
        return session_info.get('elapsed_minutes', 0.0)
    
    def get_indicator_mode_config(self) -> Dict[str, Any]:
        """
        indicator들이 사용할 모드 설정 정보 반환
        
        Returns:
            Dict: 모드 설정 정보
        """
        session_info = self.get_session_status()
        
        return {
            'use_session_mode': self.should_use_session_mode(),
            'session_name': self.get_current_session_name(),
            'session_start_time': self.get_session_open_time(),
            'elapsed_minutes': self.get_session_elapsed_minutes(),
            'session_status': session_info.get('status', STATUS_UNKNOWN),
            'mode': 'session' if self.should_use_session_mode() else 'lookback'
        }
    
    def get_session_history(self) -> Dict[str, Dict[str, Any]]:
        """
        세션 이력 반환
        
        Returns:
            Dict: 세션 이력
        """
        return self.session_history.copy()
    
    def get_next_session_start(self, current_session_start: datetime) -> datetime:
        """
        현재 세션 시작 시간 기준으로 다음 세션 시작 시간 반환
        
        Args:
            current_session_start: 현재 세션 시작 시간
            
        Returns:
            datetime: 다음 세션 시작 시간
        """
        try:
            # 현재 세션 시작 시간을 UTC로 변환
            if current_session_start.tzinfo is None:
                current_session_start = current_session_start.replace(tzinfo=timezone.utc)
            
            # 현재 날짜
            current_date = current_session_start.date()
            
            # 다음 날짜
            next_date = current_date + timedelta(days=1)
            
            # 세션 시작 시간들 (UTC)
            europe_session = datetime.combine(current_date, datetime.min.time().replace(hour=EUROPE_OPEN_HOUR, minute=EUROPE_OPEN_MINUTE), tzinfo=timezone.utc)
            us_session = datetime.combine(current_date, datetime.min.time().replace(hour=US_OPEN_HOUR, minute=US_OPEN_MINUTE), tzinfo=timezone.utc)
            
            next_europe_session = datetime.combine(next_date, datetime.min.time().replace(hour=EUROPE_OPEN_HOUR, minute=EUROPE_OPEN_MINUTE), tzinfo=timezone.utc)
            next_us_session = datetime.combine(next_date, datetime.min.time().replace(hour=US_OPEN_HOUR, minute=US_OPEN_MINUTE), tzinfo=timezone.utc)
            
            # 현재 세션 시작 시간 이후의 모든 세션 시작 시간들
            future_sessions = [
                europe_session,
                us_session,
                next_europe_session,
                next_us_session
            ]
            
            # 현재 세션 시작 시간보다 이후인 것 중 가장 이른 시간
            future_sessions = [s for s in future_sessions if s > current_session_start]
            
            if not future_sessions:
                # 미래 세션이 없으면 24시간 후 반환
                return current_session_start + timedelta(days=1)
            
            return min(future_sessions)
            
        except Exception as e:
            print(f"⚠️ 다음 세션 시작 시간 계산 오류: {e}")
            # 오류 시 24시간 후 반환
            return current_session_start + timedelta(days=1)

# =============================================================================
# 전역 인스턴스 관리
# =============================================================================

# 전역 세션 매니저 인스턴스
_global_session_manager: Optional[SessionManager] = None

def get_session_manager() -> SessionManager:
    """
    전역 세션 매니저 인스턴스 반환 (싱글톤 패턴)
    
    Returns:
        SessionManager: 세션 매니저 인스턴스
    """
    global _global_session_manager
    
    if _global_session_manager is None:
        _global_session_manager = SessionManager()
    
    return _global_session_manager
