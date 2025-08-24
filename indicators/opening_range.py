"""
Opening Range (OR) 지표 모듈

주요 기능:
- 세션 시작 후 지정된 시간(기본 30분) 동안의 고가/저가 계산
- OR 완성 여부 확인
- OR 상태 정보 제공
- 공용 데이터를 사용한 효율적인 계산
"""

import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional
from utils.time_manager import get_time_manager
from data.data_manager import get_data_manager

# 기본 설정값
DEFAULT_OR_MINUTES = 30


class OpeningRange:
    """
    Opening Range 계산 및 관리 클래스
    
    세션 시작 후 지정된 시간 동안의 고가/저가를 계산하고 관리합니다.
    공용 데이터를 사용하여 효율적으로 OR을 계산합니다.
    """
    
    def __init__(self, or_minutes: int = DEFAULT_OR_MINUTES, symbol: str = "ETHUSDT"):
        """
        OpeningRange 초기화
        
        Args:
            or_minutes: OR 완성에 필요한 분 (기본: 30분)
        """
        self.symbol = symbol
        self.or_minutes = or_minutes
        self.time_manager = get_time_manager()
        self._or = {}
        self._current_session_start = None

        self._initialize_or()
        
        print(f"🚀 OpeningRange 초기화 완료 (OR 분: {or_minutes}분)")
        
    def _initialize_or(self):
        """OR 계산"""
        current_session_start = self._get_or_time()
        self.calculate_opening_range(
            current_session_start + timedelta(seconds=1), 
            current_session_start + timedelta(minutes=self.or_minutes)
            )
        
    def _get_or_time(self):
        """세션 상태 초기화"""
        try:
            current_time = self.time_manager.get_current_time()
            session_open_time, session_name = self.time_manager.get_session_open_time(current_time)
            
            if session_open_time:
                print(f"🌅 현재 세션 활성화: {session_name} 세션")
                current_session_start = session_open_time
                
                if self.is_or_completed(current_time, session_open_time):
                    print(f"✅ 현재 세션 OR 완성됨: {session_name} 세션")
                else:
                    elapsed = self.time_manager._calculate_elapsed_minutes(current_time, session_open_time)
                    remaining = self.or_minutes - elapsed
                    print(f"⏳ 현재 세션 OR 진행 중: {elapsed:.1f}분 경과, {remaining:.1f}분 남음")
            else:
                # 직전 세션 확인
                prev_session = self.time_manager.get_previous_session_open(current_time)
                if prev_session[0]:
                    prev_start, prev_name = prev_session
                    print(f"🌙 현재 세션 비활성: 직전 세션({prev_name}) OR 사용")
                    if self.is_or_completed(current_time, prev_start):
                        print(f"✅ 직전 세션 OR 완성됨: {prev_name} 세션")
                        current_session_start = prev_start
                else:
                    print("⚠️ 활성 세션 없음")
                    current_session_start = None

            return current_session_start
        
        except Exception as e:
            print(f"❌ 세션 상태 초기화 오류: {e}")
            return None
    
    def is_or_completed(self, current_time: datetime, session_start: datetime) -> bool:
        """OR 완성 여부 확인"""
        try:
            current_utc = self.time_manager.ensure_utc(current_time)
            session_utc = self.time_manager.ensure_utc(session_start)
            elapsed_minutes = self.time_manager._calculate_elapsed_minutes(current_utc, session_utc)
            return elapsed_minutes >= self.or_minutes
        except Exception:
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """간단한 OR 데이터 반환"""
        return self._or.copy() if self._or else {}
    
    def update_with_candle(self, candle_data: pd.Series):
        """새로운 캔들로 업데이트 (호환성용)"""
        # TODO: 실시간 업데이트 구현
        pass

    def get_data(self, start_time: datetime, end_time: datetime) ->  pd.DataFrame:
        """OR 시간 정보 반환"""
        data_manager = get_data_manager()
        if not data_manager.is_ready():
            print("⚠️ DataManager가 준비되지 않았습니다")
            return {}
        
        # UTC 시간으로 변환
        start_utc = self.time_manager.ensure_utc(start_time)
        end_utc = self.time_manager.ensure_utc(end_time)
        
        print(f"📊 DataManager에서 OR 데이터 계산 시작")
        print(f"📊 요청 기간: {start_utc} ~ {end_utc}")
        
        # DataManager에서 지정된 기간 데이터 가져오기
        or_data = data_manager.get_data_range(start_utc, end_utc)
        return or_data
    
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

            if not df.empty:
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
                
                print(f"✅ OR 데이터 계산 완료: {or_high:.2f}~{or_low:.2f} ({len(df)}개 캔들)")
                return self._or
            else:
                print(f"⚠️ 지정된 기간에 해당하는 데이터가 없습니다: {start_utc} ~ {end_utc}")
                return {}
            
        except Exception as e:
            print(f"❌ OR 데이터 계산 오류: {e}")
            return {}