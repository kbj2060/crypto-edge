from typing import List
from datetime import datetime, timedelta
from utils.time_manager import get_time_manager
from utils.investing_crawler import fetch_us_high_events_today


class EventManager:
    """이벤트 관리를 담당하는 클래스"""
    
    def __init__(self):
        self.time_manager = get_time_manager()
        self.events: List[datetime] = []

    def load_daily_events(self):
        """일일 이벤트 데이터 로드"""
        try:
            print("00시 발생. 오늘의 뉴스 불러오기")
            today = fetch_us_high_events_today(headless=False)
            event_times = [event['time'] for event in today]
            self.events.extend(event_times)
            print(f"📅 오늘의 이벤트 {len(event_times)}개 로드 완료")
        except Exception as e:
            print(f"❌ 일일 이벤트 로드 오류: {e}")

    def is_in_event_blocking_period(self) -> bool:
        """이벤트 발생 시간 ±30분 동안인지 체크"""
        current_time = self.time_manager.get_current_time()
        
        for event_time in self.events:
            # 이벤트 시간 ±30분 범위 체크
            event_start = event_time - timedelta(minutes=30)
            event_end = event_time + timedelta(minutes=30)
            
            if event_start <= current_time <= event_end:
                print(f"🚫 이벤트 차단 기간: {event_time.strftime('%H:%M')} ±30분 (현재: {current_time.strftime('%H:%M')})")
                return True
        
        return False

    def important_event_occurred(self) -> bool:
        """중요 이벤트 발생 여부 체크"""
        return self.is_in_event_blocking_period()

    def get_events(self) -> List[datetime]:
        """현재 이벤트 목록 반환"""
        return self.events.copy()
