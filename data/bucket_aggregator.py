import datetime
import requests
from typing import List, Dict, Optional
from dataclasses import dataclass
from utils.time_manager import get_time_manager


@dataclass
class BucketConfig:
    external_server_url: str = "http://158.180.82.65"       # μ/σ 추정 4h
    symbol: str = "ETHUSDT"             # 최근 비어있지 않은 버킷 신선도
    external_api_key: str = "1234567890"                 # 주문 틱

@dataclass
class LiquidationEvent:
    """청산 이벤트 데이터 클래스"""
    timestamp: datetime.datetime
    symbol: str
    side: str  # 'BUY' (숏 청산) 또는 'SELL' (롱 청산)
    size: float
    price: float
    qty_usd: Optional[float] = None
    
    def __post_init__(self):
        if self.qty_usd is None:
            self.qty_usd = self.size * self.price


class BucketAggregator:
    """버킷 데이터를 집계하고 관리하는 클래스"""
    
    def __init__(self, config=BucketConfig()):
        self.config = config
        self.time_manager = get_time_manager()
        self._liquidation_buckets = []
        self._bucket_timeframe = datetime.timedelta(minutes=5)  # 5분 버킷
        
    def add_liquidation_event(self, event):
        """청산 이벤트를 적절한 버킷에 추가"""
        try:
            # 이벤트를 LiquidationEvent 객체로 변환 (통일된 형식)
            if not isinstance(event, LiquidationEvent):
                # 데이터 검증
                size = event.get('size')
                price = event.get('price')
                if size is None or price is None:
                    return False
                
                event = LiquidationEvent(
                    timestamp=event.get('timestamp'),
                    symbol=event.get('symbol', self.config.symbol),
                    side=event.get('side'),
                    size=size,
                    price=price
                )
            
            self._liquidation_buckets.append(event)

        except Exception as e:
            return None
    
    def clear_old_buckets(self):
        """오래된 버킷 데이터 정리"""
        try:
            sec = BaseLiqConfig().recency_sec
            cutoff_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=sec)
            
            original_size = len(self._liquidation_buckets)
            self._liquidation_buckets = [e for e in self._liquidation_buckets if e.timestamp >= cutoff_time]
            removed_count = original_size - len(self._liquidation_buckets)
        
        except Exception as e:
            pass
    
    def load_external_data(self):
        """외부 서버에서 청산 데이터 가져오기"""
            # 외부 API 엔드포인트에서 최근 24시간 청산 데이터 가져오기
        external_server_url = getattr(self.config, 'external_server_url', None)
        if not external_server_url:
            return []
        
        # 엔드포인트 구성
        external_api_url = f"{external_server_url.rstrip('/')}/liquidations"
        
        # API 요청 헤더 (인증이 필요한 경우)
        headers = {}
        if hasattr(self.config, 'external_api_key'):
            headers['Authorization'] = f'Bearer {self.config.external_api_key}'
        
        # 외부 서버에서 데이터 가져오기
        response = requests.get(external_api_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # 응답 내용 확인 및 디버깅
        response_text = response.text.strip()
        if not response_text:
            return []

        # 외부 데이터를 내부 형식으로 변환
        external_data = response.json()
        liquidation_data = []
        
        # 응답 구조 확인 및 데이터 추출
        if isinstance(external_data, list):
            data_items = external_data
        else:
            return []
        
        if not data_items:
            return []
        
        for item in data_items:
            # 타임스탬프 처리 (타임 매니저 사용)
            timestamp = item.get('timestamp')
            
            # 타임 매니저로 안전하게 타임스탬프 변환
            utc_dt = self.time_manager.get_timestamp_datetime(timestamp)
            
            # 변환된 데이터 생성 (통일된 형식)
            converted_data = {
                'timestamp': utc_dt,
                'symbol': item.get('symbol', self.config.symbol),
                'side': item.get('side'),
                'size': item.get('size'),
                'price': item.get('price'),
                'qty_usd': item.get('size') * item.get('price')
            }
            
            liquidation_data.append(converted_data)
        
        # 데이터 품질 검증 및 개선
        long_count = sum(1 for item in liquidation_data if item.get('side') == 'SELL')
        short_count = sum(1 for item in liquidation_data if item.get('side') == 'BUY')
        
        return liquidation_data
    
    def get_bucket(self):
        return self._liquidation_buckets
