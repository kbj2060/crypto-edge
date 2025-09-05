from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
from data.binance_dataloader import BinanceDataLoader
from utils.time_manager import get_time_manager


class CandleCreator:
    """3분봉 생성을 담당하는 클래스"""
    
    def __init__(self, symbol: str = "ETHUSDT"):
        self.symbol = symbol
        self.data_loader = BinanceDataLoader()
        self.time_manager = get_time_manager()
        
        # 1분봉 데이터 저장소
        self._recent_1min_data = []
        self._first_3min_candle_closed = False

    def store_1min_data(self, price_data: Dict):
        """1분봉 데이터를 임시 저장 (3분봉 생성용)"""
        try:
            self._recent_1min_data.append(price_data)
            
            # 최대 3개까지만 유지
            if len(self._recent_1min_data) > 3:
                self._recent_1min_data = self._recent_1min_data[-3:]
                
        except Exception as e:
            print(f"1분봉 데이터 임시 저장 오류: {e}")

    def is_3min_candle_close(self) -> bool:
        """현재 시간이 3분봉 마감 시간인지 체크"""
        try:
            current_time = self.time_manager.get_current_time()
            current_minute = current_time.minute
            return current_minute % 3 == 0
        except Exception as e:
            print(f"3분봉 마감 시간 체크 오류: {e}")
            return False

    async def create_3min_candle(self) -> Optional[pd.Series]:
        """3분봉 데이터 생성"""
        try:
            # 첫 3분봉 마감이면 바이낸스 API에서 데이터 가져오기
            if not self._first_3min_candle_closed:
                return await self._create_first_3min_candle_from_api()
            
            # 웹소켓 데이터로 3분봉 생성
            return self._create_3min_candle_from_websocket()
            
        except Exception as e:
            print(f"3분봉 데이터 생성 오류: {e}")
            return None

    async def _create_first_3min_candle_from_api(self) -> Optional[pd.Series]:
        """첫 3분봉을 API에서 가져와서 생성"""
        current_time = self.time_manager.get_current_time()
        
        # 현재 진행 중인 3분봉의 시작 시간 계산
        current_candle_start = self._calculate_current_3min_candle_start(current_time)
        
        # 마지막 완성된 3분봉 시간 범위 계산
        last_completed_start, last_completed_end = self._calculate_last_completed_3min_range(current_candle_start)
        
        # 바이낸스 API에서 마지막 완성된 3분봉 데이터 가져오기
        df_3m = self.data_loader.fetch_data(
            interval="3m",
            symbol=self.symbol.upper(),
            start_time=last_completed_start,
            end_time=last_completed_end
        )
        
        if df_3m is not None and not df_3m.empty:
            # API 데이터를 Series로 변환
            result_series = self._convert_api_data_to_series(df_3m)
            
            # 첫 3분봉 마감 완료 표시 및 초기화
            self._mark_first_3min_candle_completed()
            
            return result_series
        else:
            print("❌ 첫 3분봉 API 데이터 로드 실패")
            return None

    def _create_3min_candle_from_websocket(self) -> Optional[pd.Series]:
        """웹소켓 데이터로 3분봉 생성"""
        if len(self._recent_1min_data) < 3:
            return None
            
        recent_3_candles = self._recent_1min_data[-3:]
        
        # 3분봉 OHLCV 데이터 계산
        ohlcv_data = self._calculate_3min_ohlcv(recent_3_candles)
        
        # 3분봉 마감 시간 계산
        accurate_timestamp = self._calculate_3min_timestamp(recent_3_candles)
        
        # 3분봉 데이터를 Series로 생성
        result_series = pd.Series(ohlcv_data, name=accurate_timestamp)
        
        return result_series

    def _calculate_current_3min_candle_start(self, current_time) -> datetime:
        """현재 진행 중인 3분봉의 시작 시간 계산"""
        current_minute = current_time.minute
        
        return current_time.replace(
            minute=(current_minute // 3) * 3,
            second=0, 
            microsecond=0
        )

    def _calculate_last_completed_3min_range(self, current_candle_start) -> tuple[datetime, datetime]:
        """마지막 완성된 3분봉의 시간 범위 계산"""
        last_completed_start = current_candle_start - timedelta(minutes=3)
        last_completed_end = current_candle_start - timedelta(seconds=1)
        
        return last_completed_start, last_completed_end

    def _convert_api_data_to_series(self, df_3m: pd.DataFrame) -> pd.Series:
        """API 데이터를 Series로 변환"""
        latest_3m = pd.Series(df_3m.iloc[-1])
        
        return pd.Series({
            'open': float(latest_3m['open']),
            'high': float(latest_3m['high']),
            'low': float(latest_3m['low']),
            'close': float(latest_3m['close']),
            'volume': float(latest_3m['volume']),
            'quote_volume': float(latest_3m['quote_volume'])
        }, name=latest_3m.name)

    def _mark_first_3min_candle_completed(self):
        """첫 3분봉 마감 완료 표시 및 초기화"""
        self._first_3min_candle_closed = True
        self._recent_1min_data = []

    def _calculate_3min_ohlcv(self, recent_3_candles: list) -> dict:
        """3분봉 OHLCV 데이터 계산"""
        return {
            'open': recent_3_candles[0]['open'],
            'high': max(candle['high'] for candle in recent_3_candles),
            'low': min(candle['low'] for candle in recent_3_candles),
            'close': recent_3_candles[-1]['close'],
            'volume': sum(candle['volume'] for candle in recent_3_candles),
            'quote_volume': sum(candle['quote_volume'] for candle in recent_3_candles)
        }

    def _calculate_3min_timestamp(self, recent_3_candles: list) -> datetime:
        """3분봉 마감 시간 계산"""
        last_1min_timestamp = self.time_manager.get_timestamp_datetime(recent_3_candles[-1]['timestamp'])
        
        return last_1min_timestamp.replace(
            second=0,
            microsecond=0
        )

    def create_price_data(self, kline: Dict) -> Dict:
        """가격 데이터 생성"""
        return {
            'open': float(kline['o']),
            'high': float(kline['h']),
            'low': float(kline['l']),
            'close': float(kline['c']),
            'volume': float(kline['v']),
            'quote_volume': float(kline['q']),
            'timestamp': kline['t']
        }
