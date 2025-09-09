from dataclasses import dataclass, field
from typing import List

@dataclass
class IntegratedConfig:
    """통합 설정"""
    
    def __init__(self):
        # 기본 설정
        self.symbol: str = "ETHUSDC"
        
        # 세션 기반 전략 설정
        self.enable_session_strategy = True
        self.session_timeframe = "3m"
        
        # 고급 청산 전략 설정
        self.enable_advanced_liquidation = True
        self.adv_liq_symbol = "ETHUSDC"
        
        # 외부 서버 설정
        self.external_server_url = "http://158.180.82.65"  # 외부 서버 URL
        self.external_api_key = None  # 외부 서버 API 키 (필요시 설정)
        self.initial_data_hours = 24  # 초기 데이터 로딩 시 몇 시간 전까지 가져올지