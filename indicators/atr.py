#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ATR (Average True Range) 지표
- 3분봉 실시간 ATR 계산
- Wilder's smoothing 사용
- 세션과 관계없이 연속 롤링 계산
"""

from typing import Dict
from datetime import datetime, timezone
from collections import deque


class ATR3M:
    """3분봉 실시간 ATR 관리 클래스 - 연속 롤링 방식"""
    
    def __init__(self, length: int = 14, max_candles: int = 100):
        self.length = length
        self.max_candles = max_candles
        
        # 캔들 데이터 저장 (롤링 윈도우)
        self.candles = deque(maxlen=max_candles)
        self.true_ranges = deque(maxlen=length)
        
        # ATR 값
        self.current_atr = 0.0
        self.last_update_time = None
        
        print(f"🚀 ATR3M 초기화 완료 (기간: {length}, 연속 롤링 모드)")
    
    def update_with_candle(self, candle_data: Dict[str, any]):
        """새로운 3분봉으로 ATR 업데이트 - 연속 롤링"""
        try:
            # 캔들 데이터 저장
            candle = {
                'high': float(candle_data['high']),
                'low': float(candle_data['low']),
                'close': float(candle_data['close'])
            }
            self.candles.append(candle)
            
            # True Range 계산 (최소 2개 캔들 필요)
            if len(self.candles) >= 2:
                current = self.candles[-1]
                previous = self.candles[-2]
                
                # True Range 계산
                tr1 = current['high'] - current['low']
                tr2 = abs(current['high'] - previous['close'])
                tr3 = abs(current['low'] - previous['close'])
                
                true_range = max(tr1, tr2, tr3)
                self.true_ranges.append(true_range)
                
                # ATR 계산
                self._calculate_atr()
                self.last_update_time = datetime.now(timezone.utc)
                
        except Exception as e:
            print(f"❌ ATR 업데이트 오류: {e}")
    
    def _calculate_atr(self):
        """Wilder's smoothing으로 ATR 계산"""
        if not self.true_ranges:
            return
        
        if len(self.true_ranges) == 1:
            self.current_atr = self.true_ranges[0]
        else:
            prev_atr = self.current_atr
            current_tr = self.true_ranges[-1]
            self.current_atr = ((self.length - 1) * prev_atr + current_tr) / self.length
    
    def get_atr(self) -> float:
        """ATR 값 반환"""
        return self.current_atr
    
    def is_ready(self) -> bool:
        """ATR 계산 준비 여부"""
        return len(self.true_ranges) >= 1
    
    def is_mature(self) -> bool:
        """ATR이 충분한 데이터로 성숙했는지 여부"""
        return len(self.true_ranges) >= self.length
    
    def get_candles_count(self) -> int:
        """현재 저장된 캔들 개수 반환"""
        return len(self.candles)
    
    def get_status(self) -> Dict[str, any]:
        """ATR 상태 정보 반환"""
        return {
            'current_atr': self.current_atr,
            'is_ready': self.is_ready(),
            'is_mature': self.is_mature(),
            'candles_count': self.get_candles_count(),
            'true_ranges_count': len(self.true_ranges),
            'length': self.length,
            'max_candles': self.max_candles,
            'last_update': self.last_update_time.isoformat() if self.last_update_time else None
        }
    
    def reset(self):
        """데이터 초기화"""
        self.candles.clear()
        self.true_ranges.clear()
        self.current_atr = 0.0
        self.last_update_time = None
        print("🔄 ATR3M 데이터 초기화 완료")
