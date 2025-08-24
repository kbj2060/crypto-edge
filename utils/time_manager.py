#!/usr/bin/env python3
"""
통합 시간 관리자 (Integrated Time Manager)
- 유럽: 07:00–15:30 UTC
- 미국: 13:30–20:00 UTC
- UTC 시간 통일 관리
- 세션 시간 계산 및 관리
- 시간대 변환
- 싱글톤 패턴으로 구현
"""

import pytz
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

class TimeManager:
    """통합 시간 관리자 - 싱글톤 패턴"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TimeManager, cls).__new__(cls)
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
            self._update_session_times_cache()
            
            print("🕐 TimeManager 초기화 완료")
    
    def _update_session_times_cache(self):
        """세션 시간 캐시 업데이트"""
        try:
            current_date = self.get_current_time().date()
            
            # 캐시가 최신이면 업데이트하지 않음
            if (self._last_cache_update_date and 
                self._last_cache_update_date == current_date):
                return
            
            # 오늘과 어제의 세션 시간 계산
            today = current_date
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
            
            self._last_cache_update_date = current_date
            print(f"📅 세션 시간 캐시 업데이트 완료: {current_date}")
            
        except Exception as e:
            print(f"❌ 세션 시간 캐시 업데이트 오류: {e}")
    
    def _get_cached_session_times(self, target_date: Optional[datetime.date] = None) -> Dict[str, datetime]:
        """캐시된 세션 시간 반환"""
        self._update_session_times_cache()
        
        if target_date is None:
            return self._session_times_cache['today']
        
        if target_date == self.get_current_time().date():
            return self._session_times_cache['today']
        elif target_date == self.get_current_time().date() - timedelta(days=1):
            return self._session_times_cache['yesterday']
        else:
            # 캐시에 없는 날짜는 실시간 계산
            return self._calculate_session_times_for_date(target_date)
    
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
    
    def format_current_time(self, format_str: str = "%Y-%m-%d %H:%M UTC") -> str:
        """현재 시간을 지정된 형식의 문자열로 변환"""
        return self.format_datetime(self.get_current_time(), format_str)
    
    # =============================================================================
    # 세션 관리 메서드
    # =============================================================================
    
    def get_session_times(self, target_date: Optional[datetime.date] = None) -> Dict[str, datetime]:
        """특정 날짜의 세션 시간들 반환 (캐시 사용)"""
        return self._get_cached_session_times(target_date)
    
    def get_all_session_times(self) -> List[Tuple[datetime, str, str]]:
        """
        모든 세션 시간 반환 (시간순 정렬)
        
        Returns:
            List[Tuple[datetime, str, str]]: (세션 시간, 세션 이름, 날짜) 리스트
        """
        self._update_session_times_cache()
        return self._session_times_cache['all_sessions']
    
    def get_current_session_info(self) -> SessionTimeInfo:
        """현재 세션 정보 반환 (TimeManager 스타일)"""
        current_time = self.get_current_time()
        session_times = self.get_session_times()
        
        # 유럽 세션 활성 확인
        if session_times['europe_open'] <= current_time < session_times['europe_close']:
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
        elif session_times['us_open'] <= current_time < session_times['us_close']:
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
    
    def get_session_info(self, current_time: Optional[datetime] = None) -> SessionInfo:
        """현재 세션 정보 반환 (opening_range.py 스타일)"""
        if current_time is None:
            current_time = self.get_current_time()
        
        try:
            current_utc = self.ensure_utc(current_time)
            
            if not self._is_session_active(current_utc):
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
            
            today = current_utc.date()
            session_times = self.get_session_times(today)
            
            # 유럽 세션 활성
            if session_times['europe_open'] <= current_utc < session_times['europe_close']:
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
            
            # 미국 세션 활성
            elif session_times['us_open'] <= current_utc < session_times['us_close']:
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
            
        except Exception as e:
            print(f"❌ 현재 세션 정보 확인 오류: {e}")
            return SessionInfo(
                is_active=False,
                current_session=None,
                session_open_time=None,
                session_close_time=None,
                session_date=None, # 세션 날짜 추가
                elapsed_minutes=0.0,
                remaining_minutes=0.0,
                status=STATUS_ERROR
            )
    
    def _is_session_active(self, current_time: datetime) -> bool:
        """현재 활성 세션이 있는지 확인"""
        try:
            session_times = self.get_session_times(current_time.date())
            
            # 세션 활성 상태 확인
            is_europe_active = session_times['europe_open'] <= current_time < session_times['europe_close']
            is_us_active = session_times['us_open'] <= current_time < session_times['us_close']
            
            return is_europe_active or is_us_active
            
        except Exception as e:
            print(f"❌ 세션 활성 상태 확인 오류: {e}")
            return False
    
    def get_previous_session_open(self, current_time: Optional[datetime] = None) -> Tuple[datetime, str]:
        """과거 바로 이전 세션의 오픈 시간과 이름 반환"""
        if current_time is None:
            current_time = self.get_current_time()
        
        current_utc = self.ensure_utc(current_time)
        all_sessions = self.get_all_session_times()
        
        # 현재 시간보다 이전이면서 가장 가까운 세션 찾기
        past_sessions = [(time, name) for time, name, date in all_sessions if time <= current_utc]
        
        if not past_sessions:
            # 모든 세션이 미래인 경우 (새벽 시간대)
            today = current_utc.date()
            return (self.create_session_time(today, EUROPE_OPEN_HOUR, EUROPE_OPEN_MINUTE), SESSION_EUROPE)
        
        # 가장 가까운 세션 반환
        return max(past_sessions, key=lambda x: x[0])
    
    def get_previous_session_close(self, current_time: Optional[datetime] = None) -> Optional[datetime]:
        """과거 바로 이전 세션의 종료 시간 반환"""
        if current_time is None:
            current_time = self.get_current_time()
        
        current_utc = self.ensure_utc(current_time)
        all_sessions = self.get_all_session_times()
        
        # 현재 시간 이전의 세션 중 가장 늦은 시간
        past_sessions = [s for s in all_sessions if s[0] < current_utc]
        
        if not past_sessions:
            return None
        
        # 가장 늦은 세션의 종료 시간 반환
        latest_session = max(past_sessions, key=lambda x: x[0])
        session_name = latest_session[1]
        session_date = latest_session[2]
        
        if session_name == SESSION_EUROPE:
            return self._session_times_cache[session_date]['europe_close']
        else:  # SESSION_US
            return self._session_times_cache[session_date]['us_close']
    
    # 호환성을 위한 별칭
    def get_session_open_time(self, current_time: Optional[datetime] = None) -> Tuple[datetime, str]:
        """get_previous_session_open의 별칭 (호환성)"""
        return self.get_previous_session_open(current_time)
    
    def get_previous_session_end_time(self, current_time: Optional[datetime] = None) -> Optional[datetime]:
        """get_previous_session_close의 별칭 (호환성)"""
        return self.get_previous_session_close(current_time)
    
    def get_previous_session_end(self, current_time: Optional[datetime] = None) -> Optional[datetime]:
        """get_previous_session_close의 별칭 (호환성)"""
        return self.get_previous_session_close(current_time)
    
    def get_next_session_start(self, current_time: Optional[datetime] = None) -> datetime:
        """다음 세션 시작 시간 반환 (캐시 사용)"""
        if current_time is None:
            current_time = self.get_current_time()
        
        current_time = self.ensure_utc(current_time)
        all_sessions = self.get_all_session_times()
        
        # 현재 시간 이후의 세션 중 가장 이른 시간
        future_sessions = [s for s in all_sessions if s[0] > current_time]
        
        if not future_sessions:
            # 미래 세션이 없으면 24시간 후 반환
            return current_time + timedelta(days=1)
        
        return min(future_sessions, key=lambda x: x[0])[0]
    
    # =============================================================================
    # 세션 상태 관리 메서드 (SessionManager 스타일)
    # =============================================================================
    
    def update_session_status(self, current_time: Optional[datetime] = None) -> Dict[str, Any]:
        """현재 시간 기준으로 세션 상태 업데이트"""
        try:
            if current_time is None:
                current_time = self.get_current_time()
            
            # 세션 정보 업데이트
            self.current_session_info = self.get_session_info(current_time)
            self.last_update_time = current_time
            
            # 세션 전환 이력 저장 (날짜와 세션 이름으로 고유 ID 생성)
            if self.current_session_info.current_session:
                session_id = f"{self.current_session_info.session_date}_{self.current_session_info.current_session}"
                
                if session_id not in self.session_history:
                    self.session_history[session_id] = {
                        'session_name': self.current_session_info.current_session,
                        'session_date': self.current_session_info.session_date,
                        'start_time': self.current_session_info.session_open_time,
                        'end_time': self.current_session_info.session_close_time,
                        'first_seen': current_time,
                        'last_seen': current_time,
                        'status': self.current_session_info.status,
                        'elapsed_minutes': self.current_session_info.elapsed_minutes
                    }
                else:
                    # 기존 세션 정보 업데이트
                    self.session_history[session_id]['last_seen'] = current_time
                    self.session_history[session_id]['elapsed_minutes'] = self.current_session_info.elapsed_minutes
            
            return self.current_session_info.__dict__
            
        except Exception as e:
            print(f"❌ 세션 상태 업데이트 오류: {e}")
            return {
                'is_active': False,
                'current_session': None,
                'status': STATUS_ERROR
            }
    
    def get_session_status(self) -> Dict[str, Any]:
        """현재 세션 상태 반환 (캐시된 정보)"""
        if self.current_session_info is None:
            return self.update_session_status()
        
        return self.current_session_info.__dict__
    
    def is_session_active(self) -> bool:
        """현재 세션이 활성 상태인지 확인"""
        session_info = self.get_session_status()
        return session_info.get('is_active', False)
    
    def get_current_session_name(self) -> Optional[str]:
        """현재 세션 이름 반환"""
        session_info = self.get_session_status()
        return session_info.get('current_session')
    
    def get_session_open_time_from_status(self) -> Optional[datetime]:
        """현재 세션 시작 시간 반환 (상태에서)"""
        session_info = self.get_session_status()
        session_open = session_info.get('session_open_time')
        
        if session_open:
            if isinstance(session_open, str):
                return datetime.fromisoformat(session_open.replace('Z', '+00:00'))
            return session_open
        
        return None
    
    def should_use_session_mode(self) -> bool:
        """indicator가 세션 모드를 사용해야 하는지 판단"""
        return self.is_session_active()
    
    def get_session_elapsed_minutes(self) -> float:
        """현재 세션 경과 시간 (분)"""
        session_info = self.get_session_status()
        return session_info.get('elapsed_minutes', 0.0)
    
    def get_indicator_mode_config(self) -> Dict[str, Any]:
        """indicator들이 사용할 모드 설정 정보 반환"""
        session_info = self.get_session_status()
        
        return {
            'use_session_mode': self.should_use_session_mode(),
            'session_name': self.get_current_session_name(),
            'session_start_time': self.get_session_open_time_from_status(),
            'elapsed_minutes': self.get_session_elapsed_minutes(),
            'session_status': session_info.get('status', STATUS_UNKNOWN),
            'mode': 'session' if self.should_use_session_mode() else 'lookback'
        }
    
    def get_session_history(self) -> Dict[str, Dict[str, Any]]:
        """세션 이력 반환"""
        return self.session_history.copy()
    
    def get_session_by_date(self, target_date: datetime.date) -> Optional[Dict[str, Any]]:
        """
        특정 날짜의 세션 정보 반환
        
        Args:
            target_date: 대상 날짜
            
        Returns:
            Dict: 해당 날짜의 세션 정보 또는 None
        """
        for session_id, session_data in self.session_history.items():
            if session_data.get('session_date') == target_date:
                return session_data
        return None
    
    def get_sessions_in_date_range(self, start_date: datetime.date, end_date: datetime.date) -> Dict[str, Dict[str, Any]]:
        """
        특정 기간의 세션 정보 반환
        
        Args:
            start_date: 시작 날짜
            end_date: 종료 날짜
            
        Returns:
            Dict: 해당 기간의 세션 정보들
        """
        result = {}
        for session_id, session_data in self.session_history.items():
            session_date = session_data.get('session_date')
            if session_date and start_date <= session_date <= end_date:
                result[session_id] = session_data
        return result
    
    def get_latest_session_info(self) -> Optional[Dict[str, Any]]:
        """
        가장 최근 세션 정보 반환
        
        Returns:
            Dict: 가장 최근 세션 정보 또는 None
        """
        if not self.session_history:
            return None
        
        latest_session = max(self.session_history.values(), 
                           key=lambda x: x.get('last_seen', datetime.min))
        return latest_session
    
    def get_cache_status(self) -> Dict[str, Any]:
        """
        캐시 상태 정보 반환
        
        Returns:
            Dict: 캐시 상태 정보
        """
        return {
            'last_update_date': self._last_cache_update_date,
            'cache_keys': list(self._session_times_cache.keys()),
            'today_sessions': {
                'europe_open': self.format_datetime(self._session_times_cache.get('today', {}).get('europe_open')),
                'europe_close': self.format_datetime(self._session_times_cache.get('today', {}).get('europe_close')),
                'us_open': self.format_datetime(self._session_times_cache.get('today', {}).get('us_open')),
                'us_close': self.format_datetime(self._session_times_cache.get('today', {}).get('us_close'))
            } if 'today' in self._session_times_cache else {},
            'yesterday_sessions': {
                'europe_open': self.format_datetime(self._session_times_cache.get('yesterday', {}).get('europe_open')),
                'europe_close': self.format_datetime(self._session_times_cache.get('yesterday', {}).get('europe_close')),
                'us_open': self.format_datetime(self._session_times_cache.get('yesterday', {}).get('us_open')),
                'us_close': self.format_datetime(self._session_times_cache.get('yesterday', {}).get('us_close'))
            } if 'yesterday' in self._session_times_cache else {},
            'all_sessions_count': len(self._session_times_cache.get('all_sessions', []))
        }
    
    def force_cache_update(self):
        """캐시 강제 업데이트"""
        self._last_cache_update_date = None
        self._update_session_times_cache()
        print("🔄 세션 시간 캐시 강제 업데이트 완료")
    
    # =============================================================================
    # Timestamp 유틸리티 메서드 (timestamp_utils.py 통합)
    # =============================================================================
    
    def get_timestamp_int(self, timestamp: Union[datetime, int, float, None]) -> int:
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
    
    def get_timestamp_datetime(self, timestamp: Union[datetime, int, float, None]) -> datetime:
        """timestamp를 datetime 타입으로 변환"""
        try:
            if isinstance(timestamp, datetime):
                return timestamp
            elif isinstance(timestamp, (int, float)):
                return datetime.fromtimestamp(timestamp, tz=timezone.utc)
            else:
                return self.get_current_time()
        except Exception:
            return self.get_current_time()
    
    def get_current_timestamp_int(self) -> int:
        """현재 시간을 int timestamp로 반환"""
        current_time = self.get_current_time()
        return int(current_time.timestamp())
    
    def get_current_timestamp_datetime(self) -> datetime:
        """현재 시간을 datetime으로 반환"""
        return self.get_current_time()
    
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

# 전역 TimeManager 인스턴스
_global_time_manager: Optional[TimeManager] = None

def get_time_manager() -> TimeManager:
    """전역 TimeManager 인스턴스 반환 (싱글톤 패턴)"""
    global _global_time_manager
    
    if _global_time_manager is None:
        _global_time_manager = TimeManager()
    
    return _global_time_manager

# =============================================================================
# 호환성을 위한 별칭 함수들
# =============================================================================

def get_session_manager() -> TimeManager:
    """SessionManager 호환성을 위한 별칭 (TimeManager 반환)"""
    return get_time_manager()

def get_current_session_info(current_time: Optional[datetime] = None) -> SessionInfo:
    """opening_range.py 호환성을 위한 별칭"""
    return get_time_manager().get_session_info(current_time)

def is_session_active(current_time: Optional[datetime] = None) -> bool:
    """opening_range.py 호환성을 위한 별칭"""
    if current_time is None:
        return get_time_manager().is_session_active()
    return get_time_manager()._is_session_active(current_time)

def get_session_open_time(current_time: Optional[datetime] = None) -> Tuple[datetime, str]:
    """opening_range.py 호환성을 위한 별칭"""
    if current_time is None:
        current_time = get_time_manager().get_current_time()
    return get_time_manager().get_session_open_time(current_time)
