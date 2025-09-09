from typing import Dict, Any, Optional
from datetime import datetime
import pandas as pd
from utils.time_manager import get_time_manager


class CandleCreator:
    """3분봉 데이터 처리를 담당하는 클래스 (웹소켓에서 직접 3분봉 수신)"""
    
    def __init__(self, symbol: str = "ETHUSDC"):
        self.symbol = symbol
        self.time_manager = get_time_manager()

    def create_price_data(self, kline: Dict) -> Dict:
        """웹소켓에서 받은 3분봉 데이터를 가격 데이터로 변환"""
        return {
            'open': float(kline['o']),
            'high': float(kline['h']),
            'low': float(kline['l']),
            'close': float(kline['c']),
            'volume': float(kline['v']),
            'quote_volume': float(kline['q']),
            'timestamp': kline['t']
        }


    def is_candle_close(self, interval: str) -> bool:
        """현재 시간이 캔들 마감 시간인지 체크"""
        try:
            if interval == "3m":
                current_time = self.time_manager.get_current_time()
                current_time = self.time_manager.normalize_minute(current_time)
                return current_time.minute % 3 == 0
            elif interval == "15m":
                current_time = self.time_manager.get_current_time()
                current_time = self.time_manager.normalize_minute(current_time)
                return current_time.minute % 15 == 0
            elif interval == "1h":
                current_time = self.time_manager.get_current_time()
                current_time = self.time_manager.normalize_minute(current_time)
                return current_time.minute % 60 == 0
        except Exception as e:
            print(f"{interval} 마감 시간 체크 오류: {e}")
            return False

    def create_3min_series(self, price_data: Dict) -> pd.Series:
        """가격 데이터를 3분봉 Series로 변환"""
        timestamp = self.time_manager.get_timestamp_datetime(price_data['timestamp'])
        
        return pd.Series({
            'open': price_data['open'],
            'high': price_data['high'],
            'low': price_data['low'],
            'close': price_data['close'],
            'volume': price_data['volume'],
            'quote_volume': price_data['quote_volume']
        }, name=timestamp)
