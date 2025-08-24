#!/usr/bin/env python3
"""
어제 3분봉 데이터의 high, low만 가져오는 간단한 클래스
Note: 어제 데이터는 공용 데이터와 별개이므로 개별 API 호출 유지
"""

from data.binance_dataloader import BinanceDataLoader
from typing import Dict


class DailyLevels:
    """어제 3분봉 데이터의 high, low만 관리하는 간단한 클래스"""
    
    def __init__(self, symbol: str = "ETHUSDT", auto_load: bool = True):
        self.dataloader = BinanceDataLoader()
        self.symbol = symbol
        self.prev_day_high = 0.0
        self.prev_day_low = 0.0
        self._loaded = False
        
        # 자동으로 데이터 로드
        if auto_load:
            self.fetch_prev_day_levels(symbol)
    
    def fetch_prev_day_levels(self, symbol: str = "ETHUSDT") -> bool:
        """어제 데이터에서 high, low만 가져오기"""
        try:
            df = self.dataloader.fetch_prev_day_3m(symbol)
            
            if df is None or df.empty:
                print("❌ 어제 3분봉 데이터 로드 실패")
                return False
            
            # high, low만 계산
            self.prev_day_high = float(df['high'].max())
            self.prev_day_low = float(df['low'].min())
            self._loaded = True
            
            print(f"✅ 어제 레벨 로드 완료: 고가 ${self.prev_day_high:.2f}, 저가 ${self.prev_day_low:.2f}")
            return True
            
        except Exception as e:
            print(f"❌ 어제 레벨 로드 오류: {e}")
            return False
    
    def get_prev_day_high_low(self) -> Dict[str, float]:
        """어제 고가/저가 반환"""
        return {
            'high':self.prev_day_high, 
            'low':self.prev_day_low
            }
    
    def is_loaded(self) -> bool:
        """데이터 로드 여부 확인"""
        return self._loaded
