"""
Opening Range (OR) 지표 모듈

주요 기능:
- 세션 시작 후 지정된 시간(기본 30분) 동안의 고가/저가 계산
- OR 완성 여부 확인
- OR 상태 정보 제공
- 공용 데이터를 사용한 효율적인 계산
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from utils.session_manager import get_session_manager
from managers.data_manager import get_data_manager
from utils.time_manager import get_time_manager

# 기본 설정값
DEFAULT_OR_MINUTES = 30


class OpeningRange:
    """
    Opening Range 계산 및 관리 클래스
    
    세션 시작 후 지정된 시간 동안의 고가/저가를 계산하고 관리합니다.
    공용 데이터를 사용하여 효율적으로 OR을 계산합니다.
    """
    
    def __init__(self, or_minutes: int = DEFAULT_OR_MINUTES, symbol: str = "ETHUSDC", target_time: Optional[datetime] = None):
        """
        OpeningRange 초기화
        
        Args:
            or_minutes: OR 완성에 필요한 분 (기본: 30분)
        """
        self.symbol = symbol
        self.or_minutes = or_minutes
        self.session_manager = get_session_manager()
        self.time_manager = get_time_manager()
        self.data_manager = get_data_manager()
        self._current_session_start = None
        self._or = {}
        self.target_time = target_time if target_time is not None else self.time_manager.get_current_time()

        self.is_initialized = self._initialize_or()
        
        print(f"🚀 OpeningRange 초기화 완료 (OR 분: {or_minutes}분)")
        
    def _initialize_or(self):
        """OR 계산"""
        current_session_start = self._get_or_time()

        if current_session_start <= self.target_time <= current_session_start + timedelta(minutes=self.or_minutes):
            self.calculate_opening_range(
                current_session_start, 
                self.target_time
                )

        if current_session_start:
            self.calculate_opening_range(
                current_session_start, 
                current_session_start + timedelta(minutes=self.or_minutes)
                )
        else:
            prev_session_close = self.session_manager.get_previous_session_close()
            self.calculate_opening_range(
                prev_session_close, 
                prev_session_close + timedelta(minutes=self.or_minutes)
                )
        return True

    def _get_or_time(self):
        """세션 상태 초기화"""
        try:
            is_active = self.session_manager.is_session_active()

            if is_active:
                session_open_time = self.session_manager.get_current_session_info().open_time
            else:
                session_open_time = self.session_manager.get_previous_session_close()

            return session_open_time
        
        except Exception as e:
            print(f"❌ 세션 상태 초기화 오류: {e}")
            return None

    
    def get_status(self) -> Dict[str, Any]:
        """간단한 OR 데이터 반환"""
        return self._or.copy() if self._or else {}
    
    def update_with_candle(self, candle_data: pd.Series):
        """새로운 캔들로 업데이트 (호환성용)"""
        try:
            self.target_time = self.time_manager.ensure_utc(candle_data.name)
            self.is_initialized = self._initialize_or()
        except Exception as e:
            print(f"❌ OR 업데이트 오류: {e}")
            
    def get_data(self, start_time: datetime, end_time: datetime) ->  pd.DataFrame:
        """OR 시간 정보 반환"""
        try:
            data_manager = get_data_manager()
            if not data_manager.is_ready():
                print("⚠️ DataManager가 준비되지 않았습니다")
                return pd.DataFrame()  # 빈 DataFrame 반환
            
            # UTC 시간으로 변환
            start_utc = self.time_manager.ensure_utc(start_time)
            end_utc = self.time_manager.ensure_utc(end_time)
            
            # DataManager에서 지정된 기간 데이터 가져오기
            or_data = data_manager.get_data_range(start_utc, end_utc)
            return or_data if or_data is not None else pd.DataFrame()
        except Exception as e:
            print(f"❌ OR 데이터 가져오기 오류: {e}")
            return pd.DataFrame()  # 빈 DataFrame 반환

    def calculate_opening_range(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """
        지정된 시간 범위로 DataManager에서 데이터를 가져와서 OR 계산
        
        Args:
            start_time: 시작 시간
            end_time: 종료 시간
            
        Returns:
            Dict: OR 정보
        """
        try:
            df = self.get_data(start_time, end_time)

            if df is not None and not df.empty:
                start_utc = self.time_manager.ensure_utc(start_time)
                end_utc = self.time_manager.ensure_utc(end_time)

                or_high = float(df['high'].max())
                or_low = float(df['low'].min())
                
                # 결과 저장
                self._or = {
                    'start_time': start_utc.isoformat(),
                    'end_time': end_utc.isoformat(),
                    'or_minutes': self.or_minutes,
                    'high': or_high,
                    'low': or_low,
                    'candle_count': len(df),
                    'is_completed': True,
                    'calculation_time': self.time_manager.get_current_time().isoformat()
                }
                
                return self._or
            else:
                # 빈 OR 데이터라도 반환하여 None 오류 방지
                self._or = {
                    'start_time': self.target_time.isoformat(),
                    'end_time': self.target_time.isoformat(),
                    'or_minutes': self.or_minutes,
                    'high': None,
                    'low': None,
                    'candle_count': 0,
                    'is_completed': False,
                    'calculation_time': self.time_manager.get_current_time().isoformat()
                }
                return self._or
            
        except Exception as e:
            print(f"❌ OR 데이터 계산 오류: {e}")
            # 오류 시에도 빈 OR 데이터 반환
            self._or = {
                'start_time': self.target_time.isoformat(),
                'end_time': self.target_time.isoformat(),
                'or_minutes': self.or_minutes,
                'high': None,
                'low': None,
                'candle_count': 0,
                'is_completed': False,
                'calculation_time': self.time_manager.get_current_time().isoformat()
            }
            return self._or