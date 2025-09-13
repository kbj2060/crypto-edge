#!/usr/bin/env python3
"""
세션 관리자 (Session Manager)
- 유럽: 07:00–15:30 UTC
- 미국: 13:30–20:00 UTC
- UTC 시간 통일 관리
- 세션 시간 계산 및 관리
- 싱글톤 패턴으로 구현
"""

import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, Tuple, List, Union
from dataclasses import dataclass

# 세션 시간 상수
EUROPE_OPEN_HOUR = 7
EUROPE_OPEN_MINUTE = 0
EUROPE_CLOSE_HOUR = 15
EUROPE_CLOSE_MINUTE = 30

US_OPEN_HOUR = 13
US_OPEN_MINUTE = 30
US_CLOSE_HOUR = 20
US_CLOSE_MINUTE = 0

# 세션 이름 상수
SESSION_EUROPE = "EUROPE"
SESSION_US = "US"

# 상태 상수
STATUS_NO_SESSION = "NO_SESSION"
STATUS_EUROPE_ACTIVE = "EUROPE_ACTIVE"
STATUS_US_ACTIVE = "US_ACTIVE"
STATUS_UNKNOWN = "UNKNOWN"
STATUS_ERROR = "ERROR"

@dataclass
class SessionInfo:
    """세션 정보 데이터 클래스"""
    is_active: bool
    current_session: Optional[str]
    session_open_time: Optional[datetime]
    session_close_time: Optional[datetime]
    session_date: Optional[datetime.date]  # 세션 날짜 추가
    elapsed_minutes: float
    remaining_minutes: float
    status: str

@dataclass
class SessionTimeInfo:
    """세션 시간 정보 (TimeManager용)"""
    session_name: str
    open_time: datetime
    close_time: datetime
    session_date: datetime.date  # 세션 날짜 추가
    elapsed_minutes: float
    remaining_minutes: float
    is_active: bool

class SessionManager:
    """세션 관리자 - 싱글톤 패턴"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SessionManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._initialized = True
            self._timezone = timezone.utc
            self.current_session_info: Optional[SessionInfo] = None
            self.last_update_time: Optional[datetime] = None
            self.session_history: Dict[str, Dict[str, Any]] = {}
        
            # 세션 시간 정보를 미리 저장
            self._session_times_cache: Dict[str, Dict[str, datetime]] = {}
            self._last_cache_update_date: Optional[datetime.date] = None
            
            # 초기 세션 시간 계산
            self.update_session()
                
    def _get_cached_session_times(self, target_time: Optional[datetime] = None) -> Dict[str, datetime]:
        """캐시된 세션 시간 반환"""
        if target_time is None:
            return self._session_times_cache['today']
        
        if target_time.date() == self.get_current_time().date():
            return self._session_times_cache['today']
        elif target_time.date() == self.get_current_time().date() - timedelta(days=1):
            return self._session_times_cache['yesterday']
        else:
            # 캐시에 없는 날짜는 실시간 계산
            return self._calculate_session_times_for_date(target_time.date())
    
    def _calculate_session_times_for_date(self, target_date: datetime.date) -> Dict[str, datetime]:
        """특정 날짜의 세션 시간을 실시간으로 계산"""
        return {
            'europe_open': self.create_session_time(target_date, EUROPE_OPEN_HOUR, EUROPE_OPEN_MINUTE),
            'europe_close': self.create_session_time(target_date, EUROPE_CLOSE_HOUR, EUROPE_CLOSE_MINUTE),
            'us_open': self.create_session_time(target_date, US_OPEN_HOUR, US_OPEN_MINUTE),
            'us_close': self.create_session_time(target_date, US_CLOSE_HOUR, US_CLOSE_MINUTE)
        }
    
    # =============================================================================
    # 기본 시간 관리 메서드
    # =============================================================================
    
    def get_current_time(self) -> datetime:
        """현재 시간을 UTC로 반환"""
        return datetime.now(self._timezone)
    
    def ensure_utc(self, dt: datetime) -> datetime:
        """datetime을 UTC로 변환 (이미 UTC면 그대로 반환)"""
        if dt.tzinfo is None:
            return dt.replace(tzinfo=self._timezone)
        elif dt.tzinfo != self._timezone:
            return dt.astimezone(self._timezone)
        return dt
    
    def create_session_time(self, date: datetime.date, hour: int, minute: int) -> datetime:
        """세션 시간 생성 (UTC)"""
        return datetime.combine(date, datetime.min.time().replace(hour=hour, minute=minute), tzinfo=self._timezone)
    
    def format_datetime(self, dt: datetime, format_str: str = "%Y-%m-%d %H:%M UTC") -> str:
        """datetime을 지정된 형식의 문자열로 변환"""
        dt_utc = self.ensure_utc(dt)
        return dt_utc.strftime(format_str)
    
    
    # =============================================================================
    # 세션 관리 메서드
    # =============================================================================
    
    def get_all_session_times(self, target_time: Optional[datetime] = None) -> List[Tuple[datetime, str, str]]:
        """
        모든 세션 시간 반환 (시간순 정렬)
        
        Returns:
            List[Tuple[datetime, str, str]]: (세션 시간, 세션 이름, 날짜) 리스트
        """
        return self._session_times_cache.get('all_sessions')
    
    def get_current_session_info(self, target_time: Optional[datetime] = None) -> SessionTimeInfo:
        """현재 세션 정보 반환 (TimeManager 스타일)"""
        current_time = target_time if target_time is not None else self.get_current_time()
        session_times = self._get_cached_session_times(current_time)
        
        # 유럽 세션 활성 확인
        if session_times.get('europe_open') and session_times.get('europe_close') and session_times['europe_open'] <= current_time < session_times['europe_close']:
            elapsed = self._calculate_elapsed_minutes(current_time, session_times['europe_open'])
            remaining = self._calculate_remaining_minutes(current_time, session_times['europe_open'], 
                                                      (session_times['europe_close'] - session_times['europe_open']).total_seconds() / 60)
            
            return SessionTimeInfo(
                session_name=SESSION_EUROPE,
                open_time=session_times['europe_open'],
                close_time=session_times['europe_close'],
                session_date=session_times['europe_open'].date(), # 세션 날짜 추가
                elapsed_minutes=elapsed,
                remaining_minutes=remaining,
                is_active=True
            )
        
        # 미국 세션 활성 확인
        elif session_times.get('us_open') and session_times.get('us_close') and session_times['us_open'] <= current_time < session_times['us_close']:
            elapsed = self._calculate_elapsed_minutes(current_time, session_times['us_open'])
            remaining = self._calculate_remaining_minutes(current_time, session_times['us_open'],
                                                      (session_times['us_close'] - session_times['us_open']).total_seconds() / 60)
            
            return SessionTimeInfo(
                session_name=SESSION_US,
                open_time=session_times['us_open'],
                close_time=session_times['us_close'],
                session_date=session_times['us_open'].date(), # 세션 날짜 추가
                elapsed_minutes=elapsed,
                remaining_minutes=remaining,
                is_active=True
            )
        
        # 세션 외 시간
        return SessionTimeInfo(
            session_name="NONE",
            open_time=None,
            close_time=None,
            session_date=None, # 세션 날짜 추가
            elapsed_minutes=0.0,
            remaining_minutes=0.0,
            is_active=False
        )
    
    def get_session_info(self, target_time: Optional[datetime] = None) -> SessionInfo:
        """현재 세션 정보 반환 (opening_range.py 스타일)"""
        current_time = target_time if target_time is not None else self.get_current_time()
        
        current_utc = self.ensure_utc(current_time)
        session_times = self._get_cached_session_times(current_utc)
        
        # 세션 활성 상태 직접 확인
        is_europe_active = session_times.get('europe_open') and session_times.get('europe_close') and session_times['europe_open'] <= current_utc < session_times['europe_close']
        is_us_active = session_times.get('us_open') and session_times.get('us_close') and session_times['us_open'] <= current_utc < session_times['us_close']
        
        if not (is_europe_active or is_us_active):
            return SessionInfo(
                is_active=False,
                current_session=None,
                session_open_time=None,
                session_close_time=None,
                session_date=None, # 세션 날짜 추가
                elapsed_minutes=0.0,
                remaining_minutes=0.0,
                status=STATUS_NO_SESSION
            )
        
        if session_times.get('us_open') and session_times.get('us_close') and session_times['us_open'] <= current_utc < session_times['us_close']:
            elapsed_minutes = self._calculate_elapsed_minutes(current_utc, session_times['us_open'])
            remaining_minutes = self._calculate_remaining_minutes(current_utc, session_times['us_open'],
                                                                (session_times['us_close'] - session_times['us_open']).total_seconds() / 60)
            
            return SessionInfo(
                is_active=True,
                current_session=SESSION_US,
                session_open_time=session_times['us_open'],
                session_close_time=session_times['us_close'],
                session_date=session_times['us_open'].date(),
                elapsed_minutes=elapsed_minutes,
                remaining_minutes=remaining_minutes,
                status=STATUS_US_ACTIVE
            )

        # 유럽 세션 활성
        elif session_times.get('europe_open') and session_times.get('europe_close') and session_times['europe_open'] <= current_utc < session_times['europe_close']:
            elapsed_minutes = self._calculate_elapsed_minutes(current_utc, session_times['europe_open'])
            remaining_minutes = self._calculate_remaining_minutes(current_utc, session_times['europe_open'], 
                                                                (session_times['europe_close'] - session_times['europe_open']).total_seconds() / 60)
            
            return SessionInfo(
                is_active=True,
                current_session=SESSION_EUROPE,
                session_open_time=session_times['europe_open'],
                session_close_time=session_times['europe_close'],
                session_date=session_times['europe_open'].date(),
                elapsed_minutes=elapsed_minutes,
                remaining_minutes=remaining_minutes,
                status=STATUS_EUROPE_ACTIVE
            )
        
        return SessionInfo(
            is_active=False,
            current_session=None,
            session_open_time=None,
            session_close_time=None,
            session_date=None, # 세션 날짜 추가
            elapsed_minutes=0.0,
            remaining_minutes=0.0,
            status=STATUS_UNKNOWN
        )
    
    def is_session_active(self, target_time: Optional[datetime] = None) -> bool:
        """현재 활성 세션이 있는지 확인"""
        target_time = target_time if target_time is not None else self.get_current_time()
        session_times = self._get_cached_session_times(target_time)

        # 세션 활성 상태 확인
        is_europe_active = session_times.get('europe_open') and session_times.get('europe_close') and session_times['europe_open'] <= target_time < session_times['europe_close']
        is_us_active = session_times.get('us_open') and session_times.get('us_close') and session_times['us_open'] <= target_time < session_times['us_close']
        
        return is_europe_active or is_us_active

    
    def get_previous_session_open(self, target_time: Optional[datetime] = None) -> Tuple[datetime, str]:
        """과거 바로 이전 세션의 오픈 시간과 이름 반환"""
        current_time = target_time if target_time is not None else self.get_current_time()
        
        current_utc = self.ensure_utc(current_time)
        all_sessions = self._session_times_cache.get('all_sessions')
        
        # 현재 시간보다 이전이면서 가장 가까운 세션 찾기
        past_sessions = [(time, name) for time, name, date in all_sessions if time <= current_utc]
        
        if not past_sessions:
            # 모든 세션이 미래인 경우 (새벽 시간대)
            today = current_utc.date()
            return (self.create_session_time(today, EUROPE_OPEN_HOUR, EUROPE_OPEN_MINUTE), SESSION_EUROPE)
        
        # 가장 가까운 세션 반환
        return max(past_sessions, key=lambda x: x[0])
    
    def get_previous_session_close(self, target_time: Optional[datetime] = None) -> Optional[datetime]:
        """과거 바로 이전 세션의 종료 시간 반환"""
        current_time = target_time if target_time is not None else self.get_current_time()
        
        current_utc = self.ensure_utc(current_time)
        all_sessions = self._session_times_cache.get('all_sessions', [])
        
        # 현재 시간 이전의 세션 중 가장 늦은 시간
        past_sessions = [s for s in all_sessions if s[0] < current_utc]
        
        if not past_sessions:
            return None
        
        # 가장 늦은 세션의 종료 시간 반환
        latest_session = max(past_sessions, key=lambda x: x[0])
        session_name = latest_session[1]
        session_date = latest_session[2]
        
        if session_name == SESSION_EUROPE:
            return self._session_times_cache.get(session_date, {}).get('europe_close')
        else:  # SESSION_US
            return self._session_times_cache.get(session_date, {}).get('us_close')

    
    # =============================================================================
    # 세션 상태 관리 메서드 (SessionManager 스타일)
    # =============================================================================
    
    def update_session(self, target_time: Optional[datetime] = None) -> Dict[str, Any]:
        """세션 시간 캐시와 세션 상태를 통합 업데이트"""
        target_time = target_time if target_time is not None else self.get_current_time()
        # 1. 세션 시간 캐시 업데이트
        target_date = target_time.date()
        
        # 캐시가 최신이면 업데이트하지 않음
        if (self._last_cache_update_date and 
            self._last_cache_update_date == target_date):
            pass  # 캐시는 최신이므로 건너뛰기
        else:
            # 오늘과 어제의 세션 시간 계산
            today = target_date
            yesterday = today - timedelta(days=1)
            
            # 오늘 세션 시간
            self._session_times_cache['today'] = {
                'europe_open': self.create_session_time(today, EUROPE_OPEN_HOUR, EUROPE_OPEN_MINUTE),
                'europe_close': self.create_session_time(today, EUROPE_CLOSE_HOUR, EUROPE_CLOSE_MINUTE),
                'us_open': self.create_session_time(today, US_OPEN_HOUR, US_OPEN_MINUTE),
                'us_close': self.create_session_time(today, US_CLOSE_HOUR, US_CLOSE_MINUTE)
            }
            
            # 어제 세션 시간
            self._session_times_cache['yesterday'] = {
                'europe_open': self.create_session_time(yesterday, EUROPE_OPEN_HOUR, EUROPE_OPEN_MINUTE),
                'europe_close': self.create_session_time(yesterday, EUROPE_CLOSE_HOUR, EUROPE_CLOSE_MINUTE),
                'us_open': self.create_session_time(yesterday, US_OPEN_HOUR, US_OPEN_MINUTE),
                'us_close': self.create_session_time(yesterday, US_CLOSE_HOUR, US_CLOSE_MINUTE)
            }
            
            # all_sessions에 모든 세션 정보 저장 (시간순 정렬)
            self._session_times_cache['all_sessions'] = [
                (self._session_times_cache['yesterday']['europe_open'], SESSION_EUROPE, 'yesterday'),
                (self._session_times_cache['yesterday']['us_open'], SESSION_US, 'yesterday'),
                (self._session_times_cache['today']['europe_open'], SESSION_EUROPE, 'today'),
                (self._session_times_cache['today']['us_open'], SESSION_US, 'today')
            ]
            
            # 시간순으로 정렬
            self._session_times_cache['all_sessions'].sort(key=lambda x: x[0])
            
            self._last_cache_update_date = target_date
        
        # 2. 세션 정보 업데이트
        self.current_session_info = self.get_session_info(target_time)
        self.last_update_time = target_time
        
        # 3. 세션 전환 이력 저장 (날짜와 세션 이름으로 고유 ID 생성)
        if self.current_session_info.current_session:
            session_id = f"{self.current_session_info.session_date}_{self.current_session_info.current_session}"
            
            if session_id not in self.session_history:
                self.session_history[session_id] = {
                    'session_name': self.current_session_info.current_session,
                    'session_date': self.current_session_info.session_date,
                    'start_time': self.current_session_info.session_open_time,
                    'end_time': self.current_session_info.session_close_time,
                    'first_seen': target_time,
                    'last_seen': target_time,
                    'status': self.current_session_info.status,
                    'elapsed_minutes': self.current_session_info.elapsed_minutes
                }
            else:
                # 기존 세션 정보 업데이트
                self.session_history[session_id]['last_seen'] = target_time
                self.session_history[session_id]['elapsed_minutes'] = self.current_session_info.elapsed_minutes
        
        return self.current_session_info.__dict__
            
    def get_session_status(self, target_time: Optional[datetime] = None) -> Dict[str, Any]:
        """현재 세션 상태 반환 (캐시된 정보)"""
        current_time = target_time if target_time is not None else self.get_current_time()
        
        if self.current_session_info is None:
            return self.update_session(current_time)
        
        return self.current_session_info.__dict__
    
    def get_current_session_name(self, target_time: Optional[datetime] = None) -> Optional[str]:
        """현재 세션 이름 반환"""
        session_info = self.get_session_status(target_time)
        return session_info.get('current_session')
    
    def get_session_open_time_from_status(self, target_time: Optional[datetime] = None) -> Optional[datetime]:
        """현재 세션 시작 시간 반환 (상태에서)"""
        session_info = self.get_session_status(target_time)
        session_open = session_info.get('session_open_time')
        
        if session_open:
            if isinstance(session_open, str):
                return datetime.fromisoformat(session_open.replace('Z', '+00:00'))
            return session_open
        
        return None
    
    def should_use_session_mode(self, target_time: Optional[datetime] = None) -> bool:
        """indicator가 세션 모드를 사용해야 하는지 판단"""
        return self.is_session_active(target_time)
    
    def get_session_elapsed_minutes(self, target_time: Optional[datetime] = None) -> float:
        """현재 세션 경과 시간 (분)"""
        session_info = self.get_session_status(target_time)
        return session_info.get('elapsed_minutes', 0.0)
    
    def get_indicator_mode_config(self, target_time: Optional[datetime] = None) -> Dict[str, Any]:
        """indicator들이 사용할 모드 설정 정보 반환"""
        session_info = self.get_session_status(target_time)
        
        return {
            'use_session_mode': self.should_use_session_mode(target_time),
            'session_name': self.get_current_session_name(target_time),
            'session_start_time': self.get_session_open_time_from_status(target_time),
            'elapsed_minutes': self.get_session_elapsed_minutes(target_time),
            'session_status': session_info.get('status', STATUS_UNKNOWN),
            'mode': 'session' if self.should_use_session_mode(target_time) else 'lookback'
        }
    
    # =============================================================================
    # 유틸리티 메서드
    # =============================================================================
    
    def _calculate_elapsed_minutes(self, current_time: datetime, session_time: datetime) -> float:
        """경과 시간 계산 (분 단위)"""
        return (current_time - session_time).total_seconds() / 60
    
    def _calculate_remaining_minutes(self, current_time: datetime, session_time: datetime, total_minutes: int) -> float:
        """남은 시간 계산 (분 단위)"""
        elapsed = self._calculate_elapsed_minutes(current_time, session_time)
        return max(0, total_minutes - elapsed)

# 전역 SessionManager 인스턴스
_global_session_manager: Optional[SessionManager] = None

def get_session_manager() -> SessionManager:
    """전역 SessionManager 인스턴스 반환 (싱글톤 패턴)"""
    global _global_session_manager
    
    if _global_session_manager is None:
        _global_session_manager = SessionManager()
    
    return _global_session_manager

# =============================================================================
# 호환성을 위한 별칭 함수들
# =============================================================================

def get_current_session_info(current_time: Optional[datetime] = None) -> SessionInfo:
    """opening_range.py 호환성을 위한 별칭"""
    return get_session_manager().get_session_info(current_time)

def is_session_active(current_time: Optional[datetime] = None) -> bool:
    """opening_range.py 호환성을 위한 별칭"""
    return get_session_manager().is_session_active(current_time)
