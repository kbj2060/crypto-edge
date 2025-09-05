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
    session_date: Optional[datetime.date]
    elapsed_minutes: float
    remaining_minutes: float
    status: str

@dataclass
class SessionTimeInfo:
    """세션 시간 정보 (TimeManager용)"""
    session_name: str
    open_time: datetime
    close_time: datetime
    session_date: datetime.date
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
    
    def normalize_minute(self, current_time: datetime) -> datetime:
        """초 반올림해서 분에 +1 해주는 함수"""
        try:
            # 30초 이상이면 다음 분으로 반올림
            if current_time.second >= 30:
                return current_time + timedelta(minutes=1)
            return current_time
        except Exception as e:
            print(f"분 정규화 오류: {e}")
            return current_time
    
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
        """특정 날짜의 세션 시간들 반환"""
        if target_date is None:
            target_date = self.get_current_time().date()
        
        return {
            'europe_open': self.create_session_time(target_date, EUROPE_OPEN_HOUR, EUROPE_OPEN_MINUTE),
            'europe_close': self.create_session_time(target_date, EUROPE_CLOSE_HOUR, EUROPE_CLOSE_MINUTE),
            'us_open': self.create_session_time(target_date, US_OPEN_HOUR, US_OPEN_MINUTE),
            'us_close': self.create_session_time(target_date, US_CLOSE_HOUR, US_CLOSE_MINUTE)
        }
    
    def get_current_session_info(self) -> SessionTimeInfo:
        """현재 세션 정보 반환 (TimeManager 스타일)"""
        current_time = self.get_current_time()
        session_times = self.get_session_times()
        
        # 미국 세션 우선 확인 (겹치는 시간대에서 미국 세션 우선)
        if session_times['us_open'] <= current_time < session_times['us_close']:
            elapsed = self._calculate_elapsed_minutes(current_time, session_times['us_open'])
            remaining = self._calculate_remaining_minutes(current_time, session_times['us_open'], 
                                                        (session_times['us_close'] - session_times['us_open']).total_seconds() / 60)
            
            return SessionTimeInfo(
                session_name=SESSION_US,
                open_time=session_times['us_open'],
                close_time=session_times['us_close'],
                session_date=session_times['us_open'].date(),
                elapsed_minutes=elapsed,
                remaining_minutes=remaining,
                is_active=True
            )
        
        # 유럽 세션 활성 확인
        elif session_times['europe_open'] <= current_time < session_times['europe_close']:
            elapsed = self._calculate_elapsed_minutes(current_time, session_times['europe_open'])
            remaining = self._calculate_remaining_minutes(current_time, session_times['europe_open'], 
                                                        (session_times['europe_close'] - session_times['europe_open']).total_seconds() / 60)
            
            return SessionTimeInfo(
                session_name=SESSION_EUROPE,
                open_time=session_times['europe_open'],
                close_time=session_times['europe_close'],
                session_date=session_times['europe_open'].date(),
                elapsed_minutes=elapsed,
                remaining_minutes=remaining,
                is_active=True
            )
        
        # 세션 외 시간
        return SessionTimeInfo(
            session_name="NONE",
            open_time=None,
            close_time=None,
            session_date=None,
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
            session_times = self.get_session_times(current_utc.date())
            
            # 미국 세션 우선 확인 (겹치는 시간대에서 미국 세션 우선)
            if session_times['us_open'] <= current_utc < session_times['us_close']:
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
            
            # 유럽 세션 활성 확인
            elif session_times['europe_open'] <= current_utc < session_times['europe_close']:
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
            
            # 세션 외 시간
            return SessionInfo(
                is_active=False,
                current_session=None,
                session_open_time=None,
                session_close_time=None,
                session_date=None,
                elapsed_minutes=0.0,
                remaining_minutes=0.0,
                status=STATUS_NO_SESSION
            )
            
        except Exception as e:
            print(f"❌ 현재 세션 정보 확인 오류: {e}")
            return SessionInfo(
                is_active=False,
                current_session=None,
                session_open_time=None,
                session_close_time=None,
                session_date=None,
                elapsed_minutes=0.0,
                remaining_minutes=0.0,
                status=STATUS_ERROR
            )
    
    def is_session_active(self) -> bool:
        """현재 세션이 활성 상태인지 확인"""
        session_info = self.get_session_info()
        return session_info.is_active
    
    def get_current_session_name(self) -> Optional[str]:
        """현재 세션 이름 반환"""
        session_info = self.get_session_info()
        return session_info.current_session
    
    def get_session_open_time_from_status(self) -> Optional[datetime]:
        """현재 세션 시작 시간 반환 (상태에서)"""
        session_info = self.get_session_info()
        return session_info.session_open_time
    
    def get_session_elapsed_minutes(self) -> float:
        """현재 세션 경과 시간 (분)"""
        session_info = self.get_session_info()
        return session_info.elapsed_minutes
    
    def update_session_status(self, current_time: Optional[datetime] = None) -> Dict[str, Any]:
        """현재 시간 기준으로 세션 상태 업데이트"""
        try:
            if current_time is None:
                current_time = self.get_current_time()
            
            if self.current_session_info is None:
                self.current_session_info = self.get_session_info(current_time)
                return self.current_session_info.__dict__
            
            # 세션 정보 업데이트
            prev_session_info = self.current_session_info
            self.current_session_info = self.get_session_info(current_time)

            if self.current_session_info.current_session != prev_session_info.current_session:
                print(f"세션 상태 업데이트: {self.current_session_info.current_session}")

            self.last_update_time = current_time
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
    
    def should_use_session_mode(self) -> bool:
        """indicator가 세션 모드를 사용해야 하는지 판단"""
        return self.is_session_active()
    
    def get_indicator_mode_config(self) -> Dict[str, Any]:
        """indicator들이 사용할 모드 설정 정보 반환"""
        session_info = self.get_session_info()
        
        return {
            'use_session_mode': self.should_use_session_mode(),
            'session_name': self.get_current_session_name(),
            'session_start_time': self.get_session_open_time_from_status(),
            'elapsed_minutes': self.get_session_elapsed_minutes(),
            'session_status': session_info.status,
            'mode': 'session' if self.should_use_session_mode() else 'lookback'
        }
    
    # =============================================================================
    # Timestamp 유틸리티 메서드
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
    
    def is_midnight_time(self) -> bool:
        """밤 12시인지 확인"""
        current_time = self.get_current_time()
        return current_time.hour == 0 and current_time.minute == 0
    
    def extract_and_normalize_timestamp(self, candle_data: Any) -> datetime:
        """
        캔들 데이터에서 timestamp를 추출하여 정규화된 UTC datetime으로 반환
        
        Args:
            candle_data: Series, DataFrame, Dict 등 다양한 형태의 캔들 데이터
            
        Returns:
            datetime: 정규화된 UTC datetime
        """
        try:
            timestamp = None
            
            # 1. Series의 name 속성 사용 (가장 우선)
            if hasattr(candle_data, 'name') and candle_data.name is not None:
                timestamp = candle_data.name
            
            # 2. index가 있는 경우 첫 번째 인덱스 사용
            elif hasattr(candle_data, 'index') and len(candle_data.index) > 0:
                timestamp = candle_data.index[0]
            
            # 3. timestamp 키가 있는 경우
            elif hasattr(candle_data, 'get') and candle_data.get('timestamp'):
                timestamp = candle_data.get('timestamp')
            
            # 4. 기본값으로 현재 시간 사용
            if timestamp is None:
                return self.get_current_time()
            
            # timestamp가 이미 datetime 객체인 경우
            if isinstance(timestamp, datetime):
                # timezone이 없으면 UTC로 설정
                if timestamp.tzinfo is None:
                    return timestamp.replace(tzinfo=timezone.utc)
                return timestamp
            
            # timestamp가 다른 타입인 경우 변환
            if hasattr(timestamp, 'to_pydatetime'):
                # pandas Timestamp인 경우
                dt = timestamp.to_pydatetime()
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            
            # 숫자인 경우 (밀리초 타임스탬프)
            if isinstance(timestamp, (int, float)):
                # 13자리면 밀리초, 10자리면 초 단위
                if len(str(int(timestamp))) == 13:
                    return datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
                else:
                    return datetime.fromtimestamp(timestamp, tz=timezone.utc)
            
            # 문자열인 경우
            if isinstance(timestamp, str):
                try:
                    dt = pd.to_datetime(timestamp, utc=True)
                    return dt.to_pydatetime()
                except:
                    pass
            
            # 모든 변환 실패 시 현재 시간 반환
            return self.get_current_time()
            
        except Exception as e:
            print(f"⚠️ TimeManager timestamp 추출 오류: {e}, 현재 시간 사용")
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
    return get_time_manager().get_session_info(current_time).is_active