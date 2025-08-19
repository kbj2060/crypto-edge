#!/usr/bin/env python3
"""
트레이더 유틸리티 함수들
"""

import datetime
from typing import Optional


def get_next_5min_candle_time() -> datetime.datetime:
    """다음 5분봉 시간 계산"""
    now = datetime.datetime.now()
    minutes_to_next = 5 - (now.minute % 5)
    if minutes_to_next == 5:
        minutes_to_next = 0
    
    next_candle = now.replace(second=0, microsecond=0)
    if minutes_to_next > 0:
        next_candle = next_candle + datetime.timedelta(minutes=minutes_to_next)
    
    return next_candle


def calculate_current_atr(price_history: list) -> Optional[float]:
    """현재 ATR 계산"""
    try:
        if len(price_history) >= 14:
            prices = [p['price'] for p in price_history[-14:]]
            price_changes = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
            atr = sum(price_changes) / len(price_changes)
            return atr
    except Exception:
        pass
    return None


def format_time_delta(time_delta: datetime.timedelta) -> str:
    """시간 차이를 읽기 쉬운 형태로 포맷"""
    total_seconds = int(time_delta.total_seconds())
    
    if total_seconds < 60:
        return f"{total_seconds}초"
    elif total_seconds < 3600:
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes}분 {seconds}초"
    else:
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        return f"{hours}시간 {minutes}분"


def calculate_price_change_percentage(prev_price: float, current_price: float) -> float:
    """가격 변화율 계산"""
    if prev_price == 0:
        return 0.0
    return ((current_price - prev_price) / prev_price) * 100


def is_significant_price_change(prev_price: float, current_price: float, threshold: float = 0.1) -> bool:
    """중요한 가격 변화인지 확인"""
    change_pct = calculate_price_change_percentage(prev_price, current_price)
    return abs(change_pct) > threshold
